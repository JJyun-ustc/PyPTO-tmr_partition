# Fixed QKV ready-batch experiment

[Chinese](../../zh-cn/user/how-to/use-qkv-ready-batch.md)

This A2A3 prototype starts at Shared commit `d8f997fab` and reuses the side
queues, task target ABI, six-thread topology and range dispatch infrastructure
from `7ac2fb88c`. Its model orchestration uses ordinary Shared submissions;
no Auto directive or old online placement score participates in this study.

## Select and build

`SIMPLER_TMR_QKV_BATCH=0` selects Shared (also the default); `=1` selects QKV.
Both modes load the same runtime binary and compute kernels. This experimental
switch implements the requested same-binary comparison. Use six active AICPU
threads so both sides exist: five S and one O. On the measured CPU pool this
is S0/S1 on CPUs 2/3, S2/S3/S4 on CPUs 4/5/6, O on CPU 7.

Build from this worktree, with its own system-site-packages venv. Install the
project's declared dependencies and CPU torch first. Use the parent checkout's `.claude/skills/testing/load-env.sh` when available;
otherwise use the explicit two-job limits below.
The runtime compiler internally calls `multiprocessing.cpu_count()`, so the
wrapper below also caps that legacy entry point.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install nanobind==2.15.0 scikit-build-core==1.1.2 cmake==4.4.2 ninja==1.13.2
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install cloudpickle==3.1.3 PyYAML==6.0.3 pytest==9.2.0
export PYPTO_BUILD_JOBS=2 PYPTO_TEST_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL="$PYPTO_BUILD_JOBS" MAKEFLAGS="-j$PYPTO_BUILD_JOBS"
pip install --no-build-isolation -e . --config-settings=build.targets=_task_interface
PYTHONPATH=python:. python -c \
  'import multiprocessing,os,runpy; multiprocessing.cpu_count=lambda:int(os.environ["PYPTO_BUILD_JOBS"]); runpy.run_module("simpler_setup.build_runtimes",run_name="__main__")' \
  --platforms a2a3 --profiling-dfx 1
cmake -S tests/ut/cpp -B tests/ut/cpp/build
cmake --build tests/ut/cpp/build --parallel "$PYPTO_BUILD_JOBS" \
  --target test_a2a3_tmr_scheduler_state test_a2a3_tmr_cluster_partitioner \
  test_a2a3_tmr_task_state test_a2a3_tmr_wiring
ctest --test-dir tests/ut/cpp/build --output-on-failure \
  -R '^test_a2a3_tmr_(scheduler_state|cluster_partitioner|task_state|wiring)$'
```

## Ownership and progress

Only the winner of `READY_CLAIMED` can submit to the 64-entry pending MPMC
queue. The early-release helper's successful NONE->DISPATCHED CAS certifies
that no early stager owns the task. A separate pre-load is insufficient:
a concurrent early candidate could claim the state between load and CAS.

Eligible paths are ordinary pure AIC/AIV tasks with no predicate, explicit
side, Auto scope, sync-start requirement or prior block claims. A task wider
than either side's static capacity goes directly to Shared. DUMMY/MIX and
special paths retain their existing handling. The pending queue owns the
slot after a successful push; a failed push changes no queue ownership and
immediately routes to Shared. READY_CLAIMED rejects duplicate notifications.

S0 nonblockingly pops the currently published prefix (1..8 entries), makes
one decision batch, and resumes its scheduler loop. This runs before idle
checks and before the sync-drain `continue`, so low load and sync drain do
not prevent service. Allocation publishes queues only; it takes no compute
resource and does not join or bypass a sync rendezvous. Queued tasks retain
normal lifecycle references and cannot be reclaimed while pending. The drain
barrier waits only for peer acknowledgment/staging, not pending allocation:
insufficient capacity advances the drain attempt and returns to the polling
loop; sufficient capacity stages and reopens the gate. Ready sync tasks have
no unfinished producers. Early sync candidates require all producer blocks
to be published before entering that protocol, so an unassigned pending
producer cannot be a prerequisite of an early sync drain. Final
queues use the existing queue capacity/lifetime invariant. Assignment has one
final publication, and ordinary range requeues never enter the allocator.

A side task stays on that side. Same-side S threads use the existing
`claim_block_range()` CAS and requeue any remainder to the same queue. The
prototype has no migration, cross-side pickup or task splitting policy.
Original realtime dispatch checks still decide whether resources can be used.

## Frozen formula

There are three permanent K/V entries for every (side, pure resource) pair:
queue backlog, running/pending occupancy, and idle-capacity pressure. Thus
there are 12 fixed entries, with only three unmasked for any candidate row.
No queue scan or in-flight task sample is needed.

For candidate side c and resource r:

```text
Q = [side_sign, resource_sign, blocks/capacity, age, 1, 1], d=6
K_j = [side_sign, resource_sign, occupancy_j, type_j/2,
       is_queue_j, is_running_j]
V_j = [occupancy_j, is_running_j * occupancy_j]
S_j = dot(Q,K_j)/sqrt(6) + (is_running_j ? 0.25 : 0)
Z = stable masked_softmax(S) * V
attention_pressure = 0.5 * Z[0] + 0.5 * Z[1]
score = (queued_blocks + batch_reserved + blocks)/capacity + attention_pressure
```

Type indices are queue=0, running=1, idle=2. Wait age is elapsed time since
pending insertion divided by 1ms and capped at one. Occupancies respectively
mean queued_blocks/capacity, occupied_slots/(2*capacity), and
1-idle_cores/capacity. Queue backlog is separately present in the base term;
softmax cannot erase its magnitude. Both terms are dimensionless. Running
release pressure estimates occupied execution slots, not remaining time.

The common Shared pure-resource backlog is apportioned in proportion to
static capacity; side queues use their actual block ledgers. Each S publishes
one packed atomic capacity/idle/pending word per resource every 64 loops from
its own CoreTracker. Occupied slots = 2*capacity - 2*idle - pending_free;
IDLE cores have two empty slots and are not in the PENDING eligibility mask.
MIX, sync, strict and early work all affect these hardware masks. O derives
static capacities from the existing round-robin assignment and never reads a
possibly uninitialized CoreTracker. Dynamic summaries start unavailable.
The allocator reads each word once; this is an approximate asynchronous
snapshot, not a globally synchronized hardware instant.

S0 owns cumulative assigned blocks. Ready Shared publication and successful
range claims update per-writer monotone ledgers; no new FIN global RMW exists.
Taken counts may lead a sampled publication, so estimated queue differences
are clamped at zero. Dispatch itself remains exact.

Ready order is preserved inside a batch. Each selected side immediately adds
the task's blocks to its resource-specific batch reservation; K/V and queue
snapshot stay fixed for the batch. Exact score ties prefer fewer batch-reserved
blocks, then side 0. Incompatible sides are excluded. Snapshot unavailable,
no compatible side or elapsed 50us budget all publish Shared. The deadline is
checked before and after scoring, so a late result loses publication rights;
mandatory fallback publication itself is outside the scoring budget.

## Deterministic evidence and accounting

`QkvFormula.KvStateOverridesLowerBaseLoadAndBatchChangesNextChoice` uses
capacity 8 on both sides. Empty idle sides select 0 then 1 for two one-block
tasks. With side 1 backlog=1, the base term prefers side 0. Making side 0
fully occupied and non-idle changes QKV's choice to side 1; restoring its K/V
returns the choice to side 0. This tests a case where attention overturns the
base term, rather than merely agreeing with it.

`QKV_SUMMARY` records ordinary candidate tasks, assigned tasks per side,
assigned/claimed blocks, remaining pending depth and decision cycles.
The candidate count is measured before the static-width check; subtract
`fallback_oversize` for the width-compatible eligible count. `QKV_BATCHES`
records natural sizes 1..8 and fallback counts in the order full, snapshot,
no-side, budget, oversize. Detailed per-batch timing is emitted only with
chip swimlane level 4; normal timing uses fixed logarithmic-ns bins plus
aggregate/max cycles. Exact batch timing measures snapshot, score and queue
publication, excluding the pending pop and reporting overhead. At most 4096
exact samples are retained; count and histogram remain uncapped.

## Reproduce the comparison

The runner verifies the physical card using `npu-smi` and CANN's SoC ini.
Physical card 4 is ACL logical device 0 in the measured container. The normal
repository shell precheck assumes card 0 and an older board-query schema;
the runner uses the actual card and verifies `Short_SoC_version=Ascend910B`.

```bash
python examples/a2a3/tensormap_and_ringbuffer/benchmark_qkv/run_study.py \
  --output results/qkv-rerun --card 4 --device 0 --phase validate
python examples/a2a3/tensormap_and_ringbuffer/benchmark_qkv/run_study.py \
  --output results/qkv-rerun --card 4 --device 0 --phase measure
```

Where available, wrap the entire study in `task-submit --device ... --run ...`;
the measured container has no task-submit, and results record this limitation.
The runner refuses existing run directories and changed frozen sources or
binaries. It runs both full output/K/V goldens, then separate diagnostics,
then six alternating-order time periods, five formal samples per mode per
period. Every fresh process has one retained but excluded first warmup round.
All 72 timing rounds (60 formal plus 12 warmups) are preserved. The 37 actually
loaded kernel code fingerprints must match across every invocation. Device
wall p95 uses linear interpolation at `(n-1)*0.95`. No diagnosis round enters
the formal sample set. Formal runs use log level `info` for cold aggregate
feedback, with swimlane disabled; `timing` suppresses those aggregates.

From the runtime root, aggregate feedback and run board boundary regressions:

```bash
python examples/a2a3/tensormap_and_ringbuffer/benchmark_qkv/analyze_results.py results/qkv-rerun
python examples/a2a3/tensormap_and_ringbuffer/benchmark_qkv/run_regressions.py \
  --output results/qkv-regressions-rerun --card 4 --device 0
```

Published evidence is in the main branch at
[official-results/qkv-ready-batch-20261125](https://github.com/JJyun-ustc/PyPTO-tmr_partition/tree/main/official-results/qkv-ready-batch-20261125).
See the [English report](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/docs/en/experiments/tmr-cluster/qkv-ready-batch.md)
and [Chinese mechanism](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/docs/zh/experiments/tmr-cluster/algorithm-qkv-ready-batch.md).
