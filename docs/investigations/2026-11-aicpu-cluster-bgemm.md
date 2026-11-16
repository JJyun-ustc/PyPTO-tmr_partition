# Automatic AICPU cluster placement on the official BGEMM benchmark

**Date:** 2026-11-10
**Verdict:** Keep the mechanism opt-in for research; defer automatic enablement.

## Question

Does balancing queue pressure and cross-cluster WAIT edges improve the real
TensorMap and RingBuffer BGEMM benchmark on a six-thread 910B2 setup?

## Initial implementation, 2026-11-10

The complete PyPTO repository was checked out in a separate top-level Git
worktree, including its pinned `runtime`, `libbacktrace`, and `msgpack-c`
submodules. The AICPU controller was brought into that `runtime` worktree.
The default active count remains four; paired benchmark cases explicitly
request six threads. Both policies use the same built runtime binaries,
orchestration, AIC/AIV kernels, graph parameters, and deterministic input
tensors. The automatic policy calls `set_aicpu_cluster_auto()` at AIC and AIV
submissions; the shared policy retains the normal queue path.

The two paired workloads are:

| Family | AIC + AIV tasks | Group structure |
| --- | ---: | --- |
| Case0 | 1000 | 250 independent groups, `grid_k=2`, 128×128 tiles, loop 4 |
| Scale | 4096 | 512 independent groups, `grid_k=4`, 64×64 tiles, loop 1 |

Every policy passed a golden-output run. Timing used 30 rounds per policy,
two 15-round batches in reversed order, with chip swimlane disabled. The
`device_wall` STRACE marker is the measured end-to-end on-NPU time. SHA256 of
all three TMR runtime binaries was unchanged across the suite. One separate
swimlane capture per policy and separate dep-gen captures supplied diagnostic
counts; these runs were excluded from the timing medians.

The local result files are under the top-level worktree's `official-results/`
directory. The analysis command is:

```bash
python examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm/analyze_cluster_placement.py \
  ../official-results/performance --outputs outputs
```

## Result

Measured on a 910B2 exposed as physical NPU 1, container device 0:

| Workload | Shared median / p95 ms | Automatic median / p95 ms | Median time increase |
| --- | ---: | ---: | ---: |
| Case0 | 1.041 / 1.126 | 1.467 / 1.545 | 40.88% |
| Scale | 4.824 / 4.855 | 6.385 / 6.476 | 32.36% |

The separate diagnostic captures had complete records: 1000/4096 unique
tasks, the same number of ready-wait events, and no stream drops. The
dep-gen graph contained 750/3584 WAIT edges. Actual cross-cluster edges
were **327→283** for Case0 and **1459→1381** for Scale (shared→automatic).
Mean ready-wait in those separate captures was **17.18→6.06 µs** for Case0
and **2.79→2.65 µs** for Scale. Thus fewer cuts and lower sampled mean queue
wait did not translate into a shorter whole run.

The median orchestrator span grew by approximately 0.413 ms for 1000 tasks
and 1.549 ms for 4096 tasks, around 0.4 µs per task. Scheduler spans grew
similarly because graph construction and scheduling overlap. An independent
diagnostic sample measured only the orchestrator's submit decision hook at
roughly 0.14 µs per automatic task; that counter excludes scope polling,
feedback collection, queue operations, and downstream effects. These
numbers locate the regression in the control path but do not isolate a
single causal instruction or exclude workload effects.

## Decision and limits

The mechanism is functional and verified on an official mixed AIC/AIV
benchmark, but these two workloads provide no performance case for making
it the default. Keep automatic placement as an explicit C++ task request.
The controller does not model AIC/AIV execution duration, so it cannot price
heavy and light kernels directly. Reconsider default enablement only after
profiling and reducing the per-task control cost, then repeat same-binary,
same-topology comparisons on more than one workload family. Paged attention
is a useful next candidate because its dependency pattern differs from
BGEMM. The previous 6144-node synthetic study used another physical NPU
and is not pooled with these results.

## Cumulative-sample revision, 2026-11-13

The revision removes the four-closed-scope learning window. Each S publishes
cumulative per-resource counters, so arrivals from old scopes are still
consumed. Coefficient fitting uses bounded effective sample aggregates,
minimum evidence, at least 256 intervening submissions, and a 100 µs minimum
interval after the first fit. Scope entry no longer triggers polling.

O-to-S publication and S-to-S dependency handoff have distinct source/target
matrices. Their samples require an empty queue and idle-resource evidence,
and are not remote-minus-local congested queue means. Blocked reservations
only influence cold-start future-work balancing. The scored queue term uses
ready-publication and dispatch counters; the common predecessor term is
replaced by the maximum applicable source-to-target handoff estimate. A
shape-specific warm-up gate selects dependency affinity plus capacity balance
before enough observations and a fitted coefficient exist. See the
[current model contract](../user/how-to/use-aicpu-cluster-placement.md).

Four golden runs passed, as did the original four-thread Bgemm64 case,
automatic DUMMY chain, non-sync 4/16/24/48/96-block AIV regression, and
automatic sync-start fallback. Nine new pure controller tests cover delayed
samples, fitting cadence, queue/channel separation, ready accounting,
source/target scoring, warm-up, and concurrent snapshots. The existing
affinity target also passed, together with 140 swimlane-converter and topology
header Python tests.

Fresh same-binary measurements, 30 rounds each, with the same reversed
15-round batch ordering:

| Workload | Shared median / p95 ms | Automatic median / p95 ms | Median time increase |
| --- | ---: | ---: | ---: |
| Case0 | 1.065 / 1.160 | 1.506 / 1.569 | 41.38% |
| Scale | 4.857 / 4.907 | 6.923 / 6.965 | 42.53% |

The revision fixes the learning and scoring mechanisms but still provides no
BGEMM speedup. These measurements are a separate suite; the earlier cut and
queue-wait capture statistics above are not reused as revision diagnostics.
New logs and SHA256 files are under `../official-results/v2/` relative to the
runtime checkout. Analyze them with:

```bash
python examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm/analyze_cluster_placement.py \
  ../official-results/v2/performance
```

The analyzer also validates revision-2 learning reports from the golden logs:
Case0 used 256 cold-start and 744 scored decisions; Scale used 256 and 3840.
O-to-S coefficients and cross-cluster S-to-S coefficients were updated in
both workloads. Per-coefficient update counts respect the submission bound;
the time-bound gate has a deterministic unit test. Unlike the initial Case0
capture, the cross-cluster handoff coefficients no longer remain entirely at
their priors. Channels without enough evidence still retain priors.

One initial shared-control run stalled before the user-kernel cache was
isolated. Fresh user-kernel artifacts, separate old/new controls, subsequent
golden runs, and all timing batches passed. The original failure is retained
under `ClusterScaleShared-initial-failed.log`; its cause is not established,
and none of its timing is used in the performance table. This limits any
claim that the validation proves all execution paths free of stalls.

## Control-path tracing and bounded optimizations, 2026-11-13

Stage totals are collected only at the existing scheduler-phase swimlane
level. The orchestrator records poll, queue-depth read, predecessor scan,
resource-load read, choose/commit, and dependency wiring/ready publication.
Schedulers record resource-feedback publication, pickup observation, and the
individual ready-queue passes. Reports are printed once at quiescent teardown.
Normal timing runs have stage tracing disabled.

The Scale diagnostic capture before optimization measured:

| O work, 4096 submissions | Total ms |
| --- | ---: |
| Poll and coefficient work, amortized | 0.156 |
| Queue-depth reads | 0.214 |
| Predecessor scan | 0.203 |
| Resource and counter reads | 0.342 |
| Score and commit | 0.266 |

These exclusive stages total 1.181 ms and include timestamp overhead. The
existing full-submit capture totaled 4.760 ms for automatic versus 2.832 ms
for shared placement. AIC kernel totals were 4.073 versus 4.095 ms, and AIV
totals 3.140 versus 3.201 ms. Kernel totals run across many cores and must not
be added to wall time. The similar computation and larger orchestrator submit
span locate the main incremental work in the serial submission/control path;
they do not identify a pure hardware communication latency.

The initial S strict/shared passes accumulated 2.412/2.057 ms across five
threads, despite those queues being empty in the automatic BGEMM workload.
Skipping staging on an empty snapshot reduced these totals to 0.292/0.265 ms.
A concurrent arrival can be delayed to the next loop; sync-start staging is
unchanged. Dispatch ownership and strict cluster constraints are preserved.
These S totals overlap each other and O execution, so their 3.9 ms reduction
does not imply a 3.9 ms wall-time gain.

Each variant below uses two reversed 15-round batches per policy, stage
tracing off, and unchanged runtime hashes within that suite:

| Variant | Scale shared / automatic median ms | Case0 shared / automatic median ms |
| --- | ---: | ---: |
| Instrumented baseline | 4.738 / 6.783 | 1.042 / 1.499 |
| Skip empty ready-queue layers | 4.706 / 6.700 | 1.047 / 1.497 |
| Also batch learner snapshots | 4.902 / 6.727 | 1.049 / 1.490 |
| Empty layers + isolated ready counters; no batching | 4.937 / 6.768 | 1.060 / 1.467 |

Batching learner snapshots did not give a stable additional wall-time gain and
was removed. Publication cells for distinct cluster/resource pairs now occupy
separate cache lines instead of sharing one line. Case0 improved modestly in
that variant, while Scale did not; this does not establish a general speedup
or prove false sharing is the dominant cause. Shared-control medians also
vary between suites, so small between-variant differences are inconclusive.
The retained changes remove avoidable queue work and separate ownership
traffic, but automatic placement remains substantially slower than shared.

The final diagnostic variant's Scale submit totals were 4.756 ms automatic
and 3.126 ms shared. The wiring/ready-publication substage was 1.377 versus
0.862 ms; the other five automatic stages totaled 1.161 ms. Thus replacing
the scoring formula alone, or only reducing coefficient-update frequency,
cannot remove all of the extra cost. The next larger design opportunity is
to amortize placement across bounded dependent groups and reduce repeated
shared-state reads/ready-counter updates. These are proposals, not validated
performance claims. A whole large scope should not be pinned blindly: group
budgets and idle-resource escape must remain available.

Local raw data is under `../official-results/v3/`. Diagnostic and performance
summaries can be regenerated separately:

```bash
python examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm/analyze_control_costs.py \
  ../official-results/v3/isolated
python examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm/summarize_control_study.py \
  ../official-results/v3
```

### O-to-S, S-to-S and orchestrator location

O-to-S publishes descriptors/dependency links and ready tasks. S-to-S releases
successor dependencies and may publish their ready state. The adaptive system
also has S-to-O feedback traffic. All use shared structures; a role name alone
does not determine message cost. The publication and dependency estimators
include different preparation work, fanout work and receiver polling. In
BGEMM the former is predominantly AIC work and the latter AIV work, so those
learned coefficients cannot establish an apples-to-apples latency ranking.
The measured O submission bottleneck is not evidence that a single O-to-S
hardware transfer is intrinsically slower than S-to-S.

S-to-S notifications can sit directly on a dependency chain; O-to-S cost can
instead limit the unique producer's graph-generation throughput. Which matters
more depends on ready parallelism and the critical path, not just edge count.

Current topology is two S in cluster 0 and O plus three S in cluster 1. The O
location can affect publication, feedback and cache ownership traffic. This
board exposes only two user AICPU in cluster 0 and four in cluster 1: moving O
across clusters with six active threads changes the S split from 2:3 to 1:4,
and also changes the per-cluster AI Core resource allocation. Such a move
cannot isolate the effect of O location. No position-comparison results are
used here, per the user's instruction. Temporary affinity source/binaries
were restored and the position-specific sample entry was removed.

### Revised placement model

The next source revision scores a candidate as quiet handoff cost plus half of
the estimated ready-queue pressure, one quarter of the resource-pressure
proxy, and a 128 ns prior for each known remote WAIT predecessor beyond the
one represented by the handoff term, if the selected handoff is remote. The
source-to-destination O publication and S dependency matrices
remain separate. Automatic single-block stealing still provides an idle-side
escape.

Quiet receiver pickup wait is now accumulated as a separate per-shape,
per-destination baseline. The queue coefficient subtracts this baseline from
its depth-normalized ready-wait sample before smoothing, so the same quiet
receiver wait is less likely to be charged by both the handoff and queue
terms. The subtraction is approximate because the queue sample is normalized
by depth while the quiet baseline is not. The fixed scoring weights are
policy priors. Board timing for this revision follows below.

### Four-term scoring measurement, 2026-11-14

The revised model above was built and measured from the complete official
worktree on the container's logical device 0 (physical Ascend 910B2 device 1).
The paired six-thread Case0 and Scale workloads each passed one golden-output
run. Timing then used two 15-round batches in reversed case order, with chip
swimlane disabled. Each policy produced 30 valid `device_wall` STRACE samples;
the three TMR runtime binary hashes matched before and after the suite.

| Workload | Shared median / p95 ms | Automatic median / p95 ms | Median increase |
| --- | ---: | ---: | ---: |
| Case0 | 1.063 / 1.153 | 1.421 / 1.513 | 33.69% |
| Scale | 5.036 / 5.090 | 6.605 / 6.651 | 31.15% |

The orchestrator-span medians were 0.985/1.343 ms for Case0 and 4.981/6.546 ms
for Scale (shared/automatic). Scheduler spans overlap the orchestrator spans
and must not be added to them. The relative gap is numerically smaller than
in prior suites, but those suites had different baselines and are not a
controlled before/after comparison. The current result still gives no BGEMM
speedup from automatic placement.

The board had an AIC RAS health alarm (`80C98001`) before and after this run.
No NPU process was reported at preflight, but `task-submit` is unavailable here,
so the suite ran without an exclusive device lock. The repository's automated
architecture precheck failed because it hardcodes physical card 0; an
equivalent board query against physical card 1 confirmed 910B2. Per-task
placement, fitted coefficients, and cut counts were not captured in this
timing suite, so no claim is made about why the score changed or how many
cross-cluster edges it selected.

Raw host logs, device logs, hashes, and computed timing summary are under
`../official-results/v4-model/` relative to this runtime checkout.

### Bounded future reservations, 2026-11-14

The scored policy adds a separate future-reservation term:

```text
budget[c] = capacity[c], or scheduler_count[c] before capacity publication
F[v,c] = min(blocked[c] + (new_task_blocks if pending WAIT else 0), budget[c])
         / budget[c]
score_new(v,c) = score_four_term(v,c) + 0.25 * queue_ns[c] * F[v,c]
```

The calculation uses existing O assignments and ready-publication counters.
It adds no shared counters, fanin traversal, or ready-path work. Ready tasks
leave the blocked reservation through the existing publication ledger; neither
free nor idle resources are reduced by reservations. New pending tasks reserve
their block count once, irrespective of fanin count. The term saturates at one
capacity wave per side, so it cannot distinguish deeper blocked backlogs after
both sides saturate. The quarter weight and capacity-wave horizon are policy
choices, not fitted release-time or kernel-duration estimates.

The original four-term binaries were measured first, then the revised AICPU
and host runtime were rebuilt in this worktree. Each variant passed all four
golden checks and used two reversed 15-round batches per policy, with swimlane
off. Runtime hashes stayed unchanged within each variant, and the AIC/AIV
compute-code hashes matched across both variants. O remained on CPU 7;
there was no orchestrator-position comparison.

| Workload / variant | Shared median / p95 ms | Automatic median / p95 ms | Automatic vs shared |
| --- | ---: | ---: | ---: |
| Case0 / original | 1.066 / 1.157 | 1.415 / 1.500 | +32.79% |
| Case0 / future reservation | 1.050 / 1.146 | 1.419 / 1.485 | +35.07% |
| Scale / original | 5.034 / 5.080 | 6.615 / 6.666 | +31.41% |
| Scale / future reservation | 4.978 / 5.000 | 6.633 / 6.669 | +33.25% |

Automatic medians changed by +0.24% and +0.28% between builds. These small
differences do not establish a regression attributable to the reservation term:
Shared controls also shifted, builds were sequential, and the device was
unlocked. They show no BGEMM speedup from this experiment. Separate revised
golden diagnostics recorded 256 cold and 744/3840 learned choices for
Case0/Scale, confirming both workloads entered the revised scored path.
Those diagnostics used info logging and are excluded from timing statistics;
their placement counts are not treated as measurements of timing-run cuts.

The controller target passed all 14 tests, including capacity normalization,
multi-block bounds, a blocked common-producer fanout, and transfer from future
reservation to ready pressure. The NPU still reported `80C98001` before and
after the suite; no running NPU processes were reported at preflight.
`task-submit` was absent. The architecture script again queried unavailable
physical card 0; physical card 1 and its CANN SoC configuration were checked
directly as 910B2 / `Ascend910B`.

Raw logs, both source snapshots, the implementation patch, binary hashes,
per-batch samples and learning diagnostics are under
`../official-results/v5-future/`. Its `summary.json` contains both variants;
the revised suite can also be analyzed with `analyze_cluster_placement.py`
against `../official-results/v5-future/reserved/performance`.

### Fixed calibration, no cross-pickup and snapshot batches, 2026-11-16

This revision replaces shape-maturity checks and repeated online updates with
a fixed 256-placement cold prefix. Before placement 257, O reads one coherent
snapshot of cumulative pickup samples, fits eligible channels once and freezes
the coefficients. Sparse or racing channels retain priors. The remaining
16/32-sample guards protect individual fits and never delay the fixed phase
transition. There is no rolling window, EWMA, update time gate or later retry.
Ordinary latency sampling stops after calibration; resource feedback continues.

Cross-cluster pickup and its move counters are removed. Automatic and strict
work share the same target-side queues; separate automatic queues and their
staging passes are removed. Each side's initial assignments must equal its
actual dispatches for the single-block BGEMM cases.

Every eight automatic placements define a snapshot epoch. Each shape reads
queue depths, publication/dispatch counts and resource state once on its first
use in the epoch. O adds new tasks locally to blocked or ready pressure in
block units. Concurrent S progress becomes visible at the next refresh.
These are advisory, per-shape snapshots, not an atomic global view.

This implements snapshot batching, not buffered parallel scoring. The current
synchronous submit API immediately returns tensors used to generate dependent
nodes. Generation, dependency handling and scoring still execute on O; S
execution overlaps them. No extra scheduler is borrowed for placement, no
task waits for a batch to fill, and scope exit needs no new flush protocol.
True parallel planning with deferred publication would require a different
submit/buffering protocol and is not measured by this revision.

The container now exposes logical device 0 on **physical 910B2 card 4**, rather
than the physical card 1 used for v4/v5. Card 4 reported healthy before and
after the suite. Both the unmodified v5 baseline and this revision were freshly
measured on card 4; previous-card measurements are not pooled. The topology
remained CPUs 2–7 with O fixed on CPU 7, and there was no O-position comparison.
The default precheck queried unavailable card 0; a direct card-4 board query
and CANN configuration confirmed 910B2 / Ascend910B. No running NPU processes
were reported at preflight, but task-submit was absent, so timing was unlocked.

Each build passed four separate golden checks, followed by two reversed
15-round batches per policy with swimlane disabled. All 30 device_wall samples
per case are retained, including the first round of each process. Runtime
hashes matched before and after each suite; compute-code hashes matched across
builds. The two builds were measured sequentially.

| Workload / variant | Shared median / p95 ms | Automatic median / p95 ms | Automatic vs shared |
| --- | ---: | ---: | ---: |
| Case0 / v5 baseline | 1.048 / 1.142 | 1.419 / 1.497 | +35.42% |
| Case0 / simplified | 1.047 / 1.139 | 1.197 / 1.323 | +14.32% |
| Scale / v5 baseline | 4.963 / 4.998 | 6.629 / 6.686 | +33.58% |
| Scale / simplified | 4.913 / 4.950 | 5.809 / 5.867 | +18.23% |

Automatic medians decreased by 15.66% for Case0 and 12.37% for Scale. Both
reversed batches show the same direction: Case0 automatic batch medians were
1.414/1.422 ms before and 1.193/1.207 ms after; Scale was 6.648/6.593 ms before
and 5.792/5.843 ms after. Shared controls shifted by approximately -0.09% and
-0.99%. This is evidence of an improvement for the combined revision on this
board, but neither a per-change attribution nor a general speedup over Shared.
Independent ablations and exclusive-device runs have not been performed.

Separate revised golden diagnostics recorded:

| Case | Initial assignments = dispatched per side | Cold / scored decisions | Calibrations | Snapshots |
| --- | --- | --- | ---: | ---: |
| Case0 | 327 / 673 | 256 / 744 | 1 | 250 |
| Scale | 1437 / 2659 | 256 / 3840 | 1 | 1024 |

Both AIC and AIV occur in every epoch, producing two snapshots per eight
nodes. These info-logging runs are excluded from timing and do not establish
timing-run cut counts or optimal load balance.

Validation passed 17 controller tests, 32 scheduler-state tests and 45 wiring
tests. Coverage includes fixed phase transition without shape evidence, frozen
weights, sparse/racing calibration channels, local batch reservations,
multi-block units, shared versus targeted routing, and sampling shutdown with
ready accounting intact. Runtime-only unit targets received a disabled
swimlane stub, and their synthetic descriptors explicitly select shared
placement rather than zero-initializing the new target field to strict side 0.
Four-thread Bgemm64, automatic 4/16/24/48/96-block AIV, automatic sync-start
fallback and the automatic DUMMY chain also passed device golden checks.

Source snapshots, the implementation patch, build/test logs, raw device/host
logs, hashes and per-batch samples are under
`../official-results/v6-simplified-20261116/`. Its `summary.json` contains both
variants. The analyzer accepts historical version-2 and fixed-calibration
version-3 reports and checks per-side assignment/dispatch equality for v3.
