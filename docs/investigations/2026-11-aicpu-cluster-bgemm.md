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

### 有界分组、ready 延迟绑定、单写者摘要与受限领取，2026-11-16

本轮以当前 v6 源码为新鲜基线，围绕 O 的逐节点控制成本分阶段实现四项改动。
完整状态机、压力口径及阈值见
[当前机制说明](../user/how-to/use-aicpu-cluster-placement.md)。

#### 实现与实验分阶段

| 阶段目录 | 累积引入的行为 |
| --- | --- |
| `baseline` | 当前 v6：逐节点完整评分，8 节点快照，最终侧在生成时固定，无跨侧领取 |
| `groups` | 依赖前驱识别小组，最多 8 节点、32 次放置的继承范围、64 槽 O 本地表 |
| `deferred` | 生成只记偏好，ready 时按最后释放前驱实际侧及队列拥塞轻量最终绑定 |
| `summary` | 64 次放置才刷新 O 快照；S 每 64 循环发布压缩资源字；单写者分片账本；普通轮关闭逐节点决策计时 |
| `steal` | 独立单 block 自动队列，每 32 循环可检查，远端 6/8 条阈值，一次一个，尝试后 128 循环冷却 |
| `final` | 相同四项组合，补上来源时间戳缺失的回归修复、格式整理及本地记账接口清理后重新测量 |

这些是累积阶段，不是四因素独立消融。`summary` 同时改变反馈布局、刷新频率、
本地预留归并和普通轮计时开销，不能把它的收益单独归因于某一个字段压缩。
固定 256 次放置校准和原评分权重保留；新反馈时序会改变采样及拟合结果。

任务 payload 的自动元数据保持 40 字节，tensor/scalar 偏移不变；组号与 scope 在其中
压缩表示。新增自动队列使用 DeviceArena 布局，host、AICPU 与 AICore 三端从当前
worktree 一起重建。严格/不可领取队列与可领取自动队列分开，不从混合队列弹出后再筛选。

`preferred`、`queued`、`actual` 三种侧别分别记录。生成预留由 O 累加，首次 ready
同时撤销偏好预留并按最终队列入账；派发永远扣原队列，即便执行 S 位于另一侧。
重入队不重复入账，完成/CONSUMED 不重复扣账。每个 S 以及 O 各有自己的摘要行，
不存在多个 S 高频修改同一个全侧计数器。摘要读中变化时沿用旧行，不阻塞调度。

#### 设备与测量口径

本环境此次枚举为 **physical/logical device 0**，910B2，PCIe `0000:C1:00.0`。
这是本轮重新采集的结果，不与此前 physical card 1/4 的历史样本合并；枚举变化本身
也不能证明物理硬件是否更换。预检前后设备健康均为 OK，预检未见运行中的 NPU 进程。
`task-submit` 仍不存在，因此计时未持有设备独占锁。

仓库架构脚本预期 `Chip Name=Ascend910` 与独立 `NPU Name=B2`，当前驱动直接输出
`Chip Name=910B2` 而没有 `NPU Name`，脚本不能解析。已保留脚本失败输出，并通过
成功的 `npu-smi info -t board -i 0 -c 0` 与 CANN
`Ascend910B2.ini` 的 `Short_SoC_version=Ascend910B` 核实。
所有阶段保持六线程，允许 CPU 为 2–7，O 固定在 CPU 7，未做 O 位置对照。

每阶段先单独跑四个黄金校验，再对四个配对用例各跑两批反序的 15 轮。
计时关闭 swimlane、使用 `device_wall`，保留全部 30 个样本，包括各进程首轮。
每阶段 Shared/Automatic 共用一套二进制，并核验前后 SHA256 相同。
所有阶段的 AIC/AIV 计算代码 FNV 哈希均分别为
`0x212c407e5c39dda2` / `0x4c0e49fc6b5f2f9f`。
另外每阶段独立执行带 swimlane level 4 的黄金诊断；诊断轮不进入计时统计。

初次基线采集在后续 orchestration 编译时遇到正在修改的头文件，编译失败；该批保存在
`baseline/invalid-build-overlap`，完全排除。随后恢复基线源码、保持每个阶段源码不变直到
整个测试完成，重新取得完整基线。各阶段源码、日志、capture 副本均独立保存。

#### 各阶段结果

下表为全部 30 轮的中位数，单位 ms；括号为 Automatic 相对同阶段 Shared 的变化。

| 阶段 | Case0 Shared | Case0 Automatic | Scale Shared | Scale Automatic |
| --- | ---: | ---: | ---: | ---: |
| v6 基线 | 1.0443 | 1.1994（+14.85%） | 4.9412 | 5.7833（+17.04%） |
| 仅加组复用 | 1.0460 | 1.1792（+12.74%） | 4.8845 | 5.4011（+10.58%） |
| 再加延迟绑定 | 1.0528 | 1.2247（+16.33%） | 4.9221 | 5.4856（+11.45%） |
| 再加低频摘要 | 1.0169 | 0.9836（−3.28%） | 4.8099 | 3.9837（−17.18%） |
| 再加受限领取 | 1.0518 | 1.0161（−3.39%） | 4.9652 | 4.0872（−17.68%） |
| 最终代码复核 | 1.0504 | 1.0125（−3.60%） | 4.8201 | 3.9146（−18.79%） |

最终 p95：Case0 Shared/Automatic 为 **1.1602 / 1.1340 ms**，
Scale 为 **4.8575 / 3.9705 ms**。
相对本轮 v6 的 Automatic 中位数，最终降低 **15.58% / 32.31%**。

观察与限制：

- 组复用对 Scale 有改善；单独叠加延迟绑定没有显示收益。
- 明显变化出现在第三阶段的组合，O 与 S 的共享反馈访问显著下降。
- 领取前后相对 Shared 的优势相近，不能证明领取本身带来额外加速。
  `steal` 到 `final` 的小幅差别也不能单独归因于来源修复：构建顺序、设备漂移和拟合样本都变化。
- Case0 的优势较小；顺序构建且设备无独占锁，不能据此宣称所有短图都有稳定收益。
- 本图生成仍由 O 串行执行；没有缓冲等待小组凑齐，也没有把完整评分搬到 S 或并行评分。

#### O 与 S 的控制成本

以下来自独立的、包含计时器开销的单次诊断，单位 ms。不同线程重叠；O 细分阶段属于
submit 的一部分；S observe、ready-binding 可嵌在其他阶段中，**不能相加还原 wall time**。

| 指标 | Case0 v6 | Case0 最终 | Scale v6 | Scale 最终 |
| --- | ---: | ---: | ---: | ---: |
| O 全部 `orch_submit` 区间之和 | 1.5920 | 1.3700 | 6.9959 | 5.2279 |
| O 放置与 wire/ready 细分阶段之和 | 0.4812 | 0.3021 | 2.3435 | 1.1031 |
| 全部 S 四类派发遍历区间之和 | 3.4501 | 3.1300 | 17.0965 | 13.1123 |
| 全部 S 资源反馈区间之和 | 0.1229 | 0.0105 | 0.5878 | 0.0431 |
| 全部 S 新增 ready-binding 区间之和 | 未单列 | 0.1234 | 未单列 | 0.4025 |
| 全部 S 领取检查区间之和 | 无 | 0.0167 | 无 | 0.0822 |

四类派发包括 sync、strict、auto、shared；循环总次数也会随整体执行时间变化，不能将
累计区间差异解释成单次派发的纯成本差。最大单线程派发遍历累计为 Case0
0.9149→0.7936 ms、Scale 4.5399→3.2313 ms。

最终 S 反馈最大单线程区间之和为 Case0 **0.0036 ms**、Scale **0.0149 ms**；
ready-binding 对应最大单线程为 **0.0354 / 0.1678 ms**。
ready-binding 数值包含 ready stamping、账本、采样条件和计时器，不能解释成纯选择规则耗时。
领取检查在该两次诊断中没有触发实际借用，表中仍记录了检查成本。

#### 状态、侧别和领取证据

最终诊断的入口完整评分/继承次数为 Case0 **500/500**，Scale **2048/2048**。
快照从 v6 的 **250/1024** 降到 **16/64**；实际只对组入口使用的 shape 刷新。
仍为 256 次冷放置，且校准恰好一次。

最终诊断的偏好侧分配/实际执行侧任务数分别为：

| 用例 | 偏好侧 0/1 | 实际侧 0/1 |
| --- | --- | --- |
| Case0 | 128 / 872 | 127 / 873 |
| Scale | 73 / 4023 | 105 / 3991 |

偏好不再等于执行位置。当前结果也显著偏向 O 所在的 cluster 1，不能将性能改善解释成
两侧按 2:3 完美均衡；低频摘要与阈值提供的是调整能力，并非强制均衡保证。

最终普通 Case0 黄金轮中，五个 S 各成功跨侧领取一个任务，共 **5 个 block**；
该轮正确性通过。独立的高开销诊断轮则没有触发领取，因此没有拿零领取的诊断轮
推断普通计时轮的领取次数。对带版本 4 账本的黄金轮/诊断轮检查：
`preferred == released`、`queued == taken`、各 shape 总 preferred/queued/actual 相等。
允许每侧 queued 与 actual 不等；不能沿用 v6 的“每侧 assigned == started”检查。

#### 回归与可复现材料

8 个相关 C++ 目标、共 **190 个测试**通过。新增测试覆盖组大小/年龄/scope/槽复用、
版本竞争沿用旧摘要、偏好侧与队列侧及执行侧分离、队列侧扣账、领取阈值、严格/多 block
不可领取以及校准结束后仍保留实际来源。另修复了老 TaskState 合成 fixture 缺失 task
描述导致 ready 路由空指针的问题，显式设置 Shared 哨兵 -1。

来源时间戳缺失用例先复现失败，再修复：采样之前启动的共享生产者可能给出 0 时间戳，
这不代表 O 发起 ready。`origin-before-fix.log` 保留失败证据；修复后全部目标通过。

板上额外 **15 个黄金回归用例**通过，覆盖自动 AIV 多 block 4/16/24/48/96、
MIX 多 block、自动 sync-start 共享回退、严格跨侧向量依赖、早派发生产者，以及自动/严格
DUMMY、长链与 dense fanout/fanin。源码已放到
`tests/st/a2a3/tensormap_and_ringbuffer/cluster_partition_study/`。
仿真运行库构建通过；仿真场景 kernel 编译因环境缺少 `g++-15` 失败，未声称仿真回归通过。
没有运行整个 runtime 的全量系统测试。

材料位于 `../official-results/v7-control-20261116/`：

- `summary.json`：六个阶段的 30 轮样本、中位数/p95、成本及状态账本。
- 每阶段 `source/`、`run/`、`diagnostics/`、`captures/`：源码、原始 host/device 日志、独立图与时间线。
- `implementation.patch`：相对本轮 v6 基线的改动；`verified-unit.log` 与 `regressions/`：验证记录。
- `plan.json`：分阶段阈值；架构失败输出、板卡查询与 SHA256 一起保留。

在 runtime 自身的项目虚拟环境中，按本机测试资源上限构建并测量当前版本：

```bash
source .venv/bin/activate
source ../.claude/skills/testing/load-env.sh
python examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm/run_control_stage.py \
  ../official-results/<new-study-directory> --jobs "$PYPTO_BUILD_JOBS"
```

输出目录必须全新，避免覆盖证据；当前驱动示例默认查询 physical 0。
仅重新测已有二进制可运行 `reproduce_cluster_study.py --physical-card <id> --output <new-dir>`。
比较本次所有阶段用 `analyze_control_study.py ../official-results/v7-control-20261116`。

### 删除跨侧领取与评分敏感性，2026-11-17

本轮先隔离删除 v7 的跨侧领取调度路径，再比较评分项及另一套容量波数模型。
最终默认选择是 **保留本侧自动/严格分队布局的三项评分**：

```text
score[c] = C[c] + 0.5*queue_ns[c]*B[c]/S_count[c]
           + 0.25*queue_ns[c]*min(future[c]+new_wait_blocks,capacity[c])/capacity[c]
```

资源附加项权重由 0.25 置零，额外远端 WAIT 惩罚由 128ns 置零；排队 0.5 和有界
未来预留 0.25 保留。它们是当前候选中的保守默认，不是理论最优或全图通用最优。
分组、ready 时轻量绑定、64 节点摘要周期、固定256节点校准保留。

跨侧检查循环不再运行，也不能从另一侧队列领取。独立自动队列仍供本侧 S 使用。
保留原队列和历史诊断存储布局是为了不把不必要的布局改动混入已验证版本；旧领取
检查/尝试槽位不驱动调度，应保持零。普通任务 ready 后不再具有跨侧补救机会。

#### 环境与方法

当前 physical/logical device 0，910B2，健康 OK，O 固定 CPU 7，六线程拓扑不变。
预检未见其他 NPU 进程，仍无 task-submit，所有计时无独占锁。
架构脚本仍不能解析驱动直接输出的 Chip Name=910B2，已保留失败输出，并通过成功
板卡查询和 CANN Ascend910B2.ini 核实，不把解析失败解释成不匹配。
本轮不直接合并11月16日的计时样本。

每个阶段先四个独立黄金校验，然后每策略/每用例30轮，两批15轮反序。
计时关闭诊断，保留进程首轮；每阶段 Shared 与 Automatic 使用同二进制，核对前后SHA。
AIC/AIV 计算代码哈希与各阶段一致。单独的 level4 黄金轮采集 O/S 成本。
源码切换会更新时间戳，避免恢复较老文件时 CMake 误用较新的旧对象。

#### 权重与替代模型对照

Q、R、F 下表表示各压力项前面的乘数，X 为额外远端边惩罚。
除 baseline 外，各候选均不执行跨侧领取，首先保持原队列布局和采样不变。

| 阶段目录 | Q | R | X(ns) | F | Case0 Auto ms | Scale Auto ms | Auto相对Shared Case0/Scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline，有领取v7 | 0.5 | 0.25 | 128 | 0.25 | 1.0186 | 3.9166 | −2.28% / −18.43% |
| no-pickup，仅删领取 | 0.5 | 0.25 | 128 | 0.25 | 1.0192 | 3.8588 | −1.99% / −19.58% |
| queue-future，最终默认 | 0.5 | 0 | 0 | 0.25 | 1.0012 | 3.8144 | −3.49% / −19.91% |
| queue-one | 1 | 0 | 0 | 0.25 | 1.0176 | 3.9395 | −1.88% / −17.56% |
| future-zero | 0.5 | 0 | 0 | 0 | 1.0008 | 3.8683 | −3.55% / −18.13% |
| unit-prices | 1 | 0 | 0 | 1 | 1.0416 | 4.6765 | +0.21% / −2.35% |
| waves，波数候选 | — | — | — | — | 0.9013 | 3.9863 | −13.75% / −19.83% |

仅删除领取，Case0基本不变，Scale自动中位数降低约1.48%；同阶段Shared基本稳定。
这支持减少控制路径，但不能据此断言领取必然显著扰乱模型。
在三项模型中，把排队系数从0.5增为1，或把未来项删除，均没有比默认组合更稳妥。
同时把排队与未来项设为1明显损害Scale。没有做全参数网格或统计意义上的最优求解。

#### 128ns 无法由这组图标定

feature-probe 是独立构建、只跑诊断的旧评分探针，不进入性能统计。
它记录校准后完整评分入口的特征：Case0 372次、Scale 1920次，
**待完成WAIT次数均为0，额外远端边项的非零次数均为0**。
存在依赖的后继主要继承小组；在这些运行中，完整评分入口没有覆盖这个参数的适用场景。
因此无法通过BGEMM推断128ns应该调成多少，也不能把它删除后的结果推广到所有动态图。

探针中 C 项的两侧平均价格为 Case0 1500/250ns，Scale 1500/500ns。
这些是包含接收发现等待和限幅的拟合价格，不是纯硬件通信时延；其偏向 O 所在侧的作用
很强。评分项之间有重叠、状态也有滞后，不能把各经验项都设成1就当作物理依据更充分。

#### 容量波数候选与未采用的清理

容量波数方案不使用ns价格参与决策：

```text
P[c] = (B[c] + min(future[c],K[c]) + U[c] + new_blocks) / K[c]
U[c] = K[c] - clamp(free_dispatch[c],0,K[c])，未知容量时取0
```

K为兼容计算容量，未知时用S数。先选最低P；对已知WAIT来源较多的一侧，或无WAIT时
O所在侧，最多容忍额外一波压力以保持亲和。1024只是定点数精度。
一波亲和预算仍是策略参数，不是无参数或时间预测模型。

waves-clean 又移除了该模型不用的学习/采样并合并队列，校验通过，但不能只根据
相对Shared收益选它：Shared二进制自身也变化。随后合并队列的三项版本
merged-three-price 出现 Case0 0.9717ms、Scale 4.0231ms；Scale相对已测三项分队版回退。
因此这两个清理版本保留为实验，不作为默认，源代码已恢复 queue-future 的被测版本。

#### 同原二进制交错复核

为排查顺序构建漂移，复用已测原二进制和匹配源码，在同一轮驱动中按
no-pickup→queue-future→waves-clean，再反序执行；每用例仍30轮。
interleaved-recheck 原始标签 final 指 waves-clean，映射见 recheck-labels.json。

| 版本 | Case0 Shared median ms | Case0 Auto median/p95 ms | Scale Shared median ms | Scale Auto median/p95 ms |
| --- | ---: | ---: | ---: | ---: |
| 仅删领取、旧评分 | 1.0440 | 1.0113 / 1.1493 | 4.7973 | 3.8752 / 3.9346 |
| 三项分队版，默认 | 1.0419 | 0.9922 / 1.1143 | 4.7954 | 3.8154 / 3.8670 |
| 波数清理候选 | 1.0736 | 0.8970 / 1.0310 | 4.9587 | 3.9443 / 4.0269 |

三项版相对仅删领取：自动中位数降低 **1.89% / 1.54%**，p95降低 **3.05% / 1.72%**；
各自同二进制Shared的差异很小。相对同二进制Shared，默认自动快 **4.78% / 20.44%**。
波数候选的Case0更快，但Scale绝对时间比三项版慢约3.38%；Shared本身也慢，不能仅
靠相对Shared比例把它选为默认。保守选择不表示波数机制在所有场景都更差。

#### 额外配置与控制成本

另设Heavy（512节点，tile128，loop8，grid_k2）和Deep（1024节点，tile64，loop2，
grid_k8），没有用它们调上表的主候选。它们仍属于BGEMM家族，不是跨图形泛化验证。
每项同样独立黄金校验、30轮反序计时。

| 版本 | Heavy Shared/Auto median ms | Deep Shared/Auto median ms |
| --- | ---: | ---: |
| 仅删领取、旧评分 | 0.6364 / 0.6895 | 1.2318 / 1.1050 |
| 三项分队版 | 0.6326 / 0.6865 | 1.2346 / 1.0957 |
| 波数候选 | 0.6373 / 0.6820 | 1.2889 / 1.0557 |

三项版没有明显损害这两项中位数，但Heavy自动仍慢于Shared，不能宣称全面加速。
波数候选Heavy的p95为0.8229ms，高于旧评分0.7839ms；清理后降到0.7973ms，
尾部风险也参与了判断，未只挑最好一轮或最小中位数。

O/S 控制成本来自独立插桩轮，不与普通device_wall相加。最终三项版的成本见
queue-future/control-costs.json；probe增加特征计数，其耗时不用于直接比较控制成本。
跨侧检查已关闭；S的ready绑定、单写者记账和一次校准仍存在，不能把旧控制开销都视为消失。

#### 验证与材料

默认恢复到 queue-future 的源码和原二进制，检查原哈希一致；没有把其他候选的二进制
当作默认使用。旧资源/额外边项的测试更新为三项模型预期，并保留组、账本、并发、
固定校准与ready来源回归。默认版的最终结果记录在 selected-unit.log 和 selected-regressions/；8个目标共190项
C++测试、15个板上黄金回归通过，普通/诊断/回归共检查28行账本一致性。
运行库也针对仿真构建；仿真kernel仍缺g++-15，没有声称完整仿真系统测试通过。

所有材料位于 `../official-results/v8-no-pickup-20261117/`：
summary.json、plan.json、selection.json、features.json、interleaved-recheck/、各阶段
源码/哈希/黄金与计时日志、独立诊断、holdout/，以及默认版验证记录。
原二进制存档仅供本机复核，不应随源代码发布上传。全部实验保持O位置不变。
