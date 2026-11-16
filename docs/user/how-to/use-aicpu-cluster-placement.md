# Use AICPU cluster placement on a2a3

The a2a3 TensorMap and RingBuffer runtime supports shared, explicit and automatic
AICPU scheduler-cluster placement. On the observed 910B2 six-thread pool,
cluster 0 has S0/S1 on CPUs 2/3; cluster 1 has S2/S3/S4 on CPUs 4/5/6 and
one orchestrator O on CPU 7. The default thread count remains four. Host
probing supplies cluster IDs. O can publish ready work directly to either side.

```cpp
CoreTaskArgs args;
args.set_aicpu_cluster_auto();
rt_submit_aiv_task(kernel_id, args);
```

No hint uses shared queues; `set_aicpu_cluster(0/1)` is strict placement.
Python graph generation does not insert automatic hints. The automatic choice
is made after WAIT reduction, before dependency wiring or ready publication.
The graph is generated online; scope boundaries do not trigger whole-graph
partitioning or execution barriers.

## Blocks, capacities and queues

A block is one logical execution instance of a task. A task may have several
blocks. AIC/AIV capacity counts compatible cores; MIX capacity counts owned
AI Core groups; DUMMY capacity is one per scheduler. A MIX block can involve
multiple compute cores. These resource groups are distinct from AICPU clusters.
Block backlog divided by capacity estimates resource waves, not execution time
or occupied resources; kernels can have different durations.

Automatic and strict tasks use the same cluster-specific per-shape queues.
There is no separate automatic queue and no cross-cluster pickup. Shared tasks
use global per-shape queues. Schedulers service their own targeted queues and
the shared queues. A targeted multi-block task can dispatch in partial ranges
and requeue the remainder on its original side. DUMMY and failed-predicate
tasks use the appropriate shared or targeted DUMMY queue.

Automatic multi-block sync-start requests retain the shared sync-start path;
strict multi-block sync-start targets are rejected. Targeted tasks cannot use
global early dispatch to bypass placement. The initial target is immutable.
A shared producer can still publish its actual pickup side for later consumers.
Placement does not copy tensor data between clusters.

## Fixed calibration prefix

The first 256 automatic placements use cold-start balancing and predecessor
affinity. Before decision 257, O takes one nonblocking, versioned snapshot of
S's cumulative first-pickup observations and fits each eligible channel once.
The transition depends only on the submission count, not shape maturity.

Queue fitting requires 32 observations; publication, dependency and quiet
pickup channels require 16. Missing, sparse or racing channels retain priors;
there is no retry or extension of the prefix. Quiet pickup is fitted before
subtracting it from the normalized queue observation. Fitted observations
are bounded to 20–32000 ns and 0.5–2 times their priors, then assigned directly.
There is no EWMA, rolling window, elapsed-time update gate or repeated polling.

S's ordinary latency sampling stops after calibration is frozen. Resource
feedback and publication/dispatch accounting continue. Late completions cannot
change frozen coefficients. Diagnostics can still record ready-wait spans.
Every runtime execution resets calibration, counts and parameters; no training
state is carried between benchmark rounds. Runs of at most 256 automatic
placements stay entirely in cold start.

Each of AIC, AIV, MIX and DUMMY has its own coefficients. Queue priors are
2000 ns on both sides; O-to-S publication and S-to-S dependency matrices both
start at 250 ns on the diagonal and 750 ns off diagonal. Quiet pickup starts
unknown. These are control priors, not hardware communication measurements.

## Eight-node snapshot batches

O still generates, scores and publishes each task synchronously so that submit
can immediately return output tensor handles to subsequent task generation.
It does not wait to accumulate eight tasks and does not run parallel scoring.
O does not wait for a submitted kernel to finish before generating the next
node, except for existing backpressure or explicit synchronization.

Every eight automatic decisions define a snapshot epoch. Each resource shape
reads queues, resources and publication/dispatch counters once on its first
appearance in that epoch. Later nodes of that shape reuse the snapshot. O adds
each newly assigned task locally to blocked pressure if it has pending WAIT
predecessors, otherwise to ready pressure. It adds block units, not queue
entries. This prevents all batch nodes from seeing an unchanged empty queue.

S execution progress and blocked-to-ready transitions within an epoch become
visible at the next snapshot. Different shapes take their snapshots lazily,
so snapshots are advisory, not atomic across threads or shapes. Batching saves
shared reads at the cost of freshness. All S threads remain available for
scheduling rather than being borrowed to run placement calculations.

## State and scores

At snapshot refresh, with shape omitted:

```text
blocked[c] = max(0, assigned[c] - ready_published[c])
ready_auto[c] = max(0, ready_published[c] - dispatched[c])
depth[c] = targeted_queue_entries[c]
           + shared_queue_entries * S_count[c] / total_S_count
B[c] = max(ready_auto[c], depth[c])
```

Only automatic work enters the cumulative block ledger. Targeted and shared
queue depth also reflect nonautomatic work. Queue entries approximate blocks
for multi-block tasks. Publication is counted once at first readiness;
dispatch is counted after publishing each dispatched block range, not at
completion or CONSUMED. There are no move counters. Within an epoch, local
new-task reservations are added to this snapshot until refresh.

Cold start chooses the smallest score and can favor the side with more known
WAIT predecessors when its score is at most 256 above the minimum:

```text
cold_score[c] = 1024 * (blocked[c] + B[c] + new_task_blocks) / capacity[c]
```

Unknown capacity falls back to scheduler count. After the fixed prefix:

```text
Q[c] = B[c] / S_count[c]
R[c] = 2*(capacity[c]-free_dispatch[c])/capacity[c]
       + (capacity[c]-idle[c])/capacity[c]
C[v,c] = publish_ns[O_cluster][c]                         if no pending WAIT
       = max(applicable dependency_ns[source][c])       otherwise
X[v,c] = known remote pending WAIT count,
         minus one if C equals the remote dependency coefficient and X > 0
budget[c] = capacity[c], or S_count[c] if capacity is unknown
F[v,c] = min(blocked[c] + (new_task_blocks if pending WAIT else 0), budget[c])
         / budget[c]
score[v,c] = C[v,c] + 0.5*queue_ns[c]*Q[c]
             + 0.25*queue_ns[c]*R[c] + 128 ns*X[v,c]
             + 0.25*queue_ns[c]*F[v,c]
```

Unknown predecessor locations include both possible source channels in the
maximum, without extra structural charge. Free includes IDLE and dispatchable
PENDING resources; the score clamps free/idle to capacity. Unknown capacity
uses one queue-cost unit before the resource term's quarter weight. Equal
scores favor the lower cumulative assignment count normalized by S count.

Future pressure is capped at queue_ns/4 per side and never marks a resource
busy. Both sides can saturate, so it is a soft preference, not a balance
constraint or a common-predecessor release model. Fixed weights are policy
choices. The score predicts neither kernel duration nor predecessor remaining
time and does not account for the full cost of the placement implementation.

## Calibration measurements

Each logical automatic task supplies at most one observation at first resource
preparation, including multi-block and requeued tasks. If same-shape work was
already queued at ready publication, the queue sample is:

```text
queue_sample = ready_to_first_prepare_ns * S_count / (S_count + ready_depth)
```

Quiet handoff samples require an empty same-shape queue, idle-resource evidence
at publication, a real IDLE slot at pickup, ready-to-pickup delay at most 5 us,
source preparation at most 20 us, and a known source cluster. They measure
source preparation plus receiver pickup wait. The receiver wait also supplies
a quiet baseline subtracted during queue fitting. The subtraction is approximate
because the queue observation is depth-normalized but the baseline is not.

O publication starts near ready stamping, not at the beginning of full task
submission. S dependency handoff starts at the releasing producer's fanout
walk. These estimates include visibility, polling and fanout effects and do
not isolate hardware cross-cluster latency. Per-S versioned buckets keep the
single calibration snapshot coherent without blocking dispatch.

## Benchmark evidence

The paired six-thread BGEMM cases use identical graph parameters, kernels and
deterministic inputs. Check golden outputs separately, then compare repeated
tracing-disabled device_wall medians and p95; do not select the fastest round.
Different runtime builds need fresh Shared controls. See the
[experiment history](../../investigations/2026-11-aicpu-cluster-bgemm.md), including
negative results from earlier adaptive-learning and cross-pickup variants.

At scheduler-phase swimlane level, quiescent teardown prints O stages and S
queue/feedback stages. These include instrumentation cost and overlap across
threads; do not add them to reconstruct wall time. Version-3 final reports
include fixed calibration count, batch size, snapshot count and per-side
assignments/dispatches. With no cross-cluster pickup, assignments must equal
actual dispatches on each side for these single-block benchmark graphs.
