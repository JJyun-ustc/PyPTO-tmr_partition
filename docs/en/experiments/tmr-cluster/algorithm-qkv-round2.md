# QKV round two: mechanism and experiment protocol

[Chinese](../../../zh/experiments/tmr-cluster/algorithm-qkv-round2.md) · [Results](qkv-round2.md)

Think of the two AICPU clusters as two stations. Shared publishes ready work to a
common waiting area. The first QKV prototype inserts a 64-task allocation queue
before that area; S0 alone removes up to eight currently available tasks and
chooses a station. A singleton is processed immediately. This extra service can
itself delay useful work, even when its side-selection calculation is inexpensive.

## Measure the extra waiting area first

The diagnostic records ready, allocation-queue removal, final-queue publication,
first dispatch, start proxy, first/final FIN and completion processing. The stages
separate centralized waiting from allocator service and waiting after side binding.
Allocator service is split into pop, snapshot, score and publication. S0's adjacent
dispatch gaps are sampled separately. Detailed records are emitted only after
scheduling quiesces. Their cold logging cost can dominate diagnostic device_wall;
none of these runs enters formal timing.

Two short controls isolate mechanisms. The attention-off control keeps pending,
S0 and sequential batch reservations. A second diagnostic retains centralized
service and base-load decisions but publishes to Shared. Neither control is a
candidate policy or a parameter search.

The improved policy bypasses the allocation queue when summaries are missing,
stale, or neither side reports enough immediate compatible capacity. This trades
prototype coverage for reduced centralized waiting. Coverage and fallbacks must
therefore accompany every performance claim.

## Algebraically identical scoring

Each candidate has three permanent entries: queued work, running/pending
occupancy, and unavailable capacity. Masking keeps only the candidate side and
resource channel. Let `u=blocks/capacity`, `q=queued/capacity`, `o=occupied_ratio`,
`h=1-idle/capacity`, and `a=min(wait_ms,1)`. Subtracting the common
`3/sqrt(6)` from the original logits gives:

```text
Lqueue = u*q/sqrt(6)
Lrun   = u*o/sqrt(6) + a/(2*sqrt(6)) + 1/4
Lidle  = u*h/sqrt(6) + a/sqrt(6) - 1/sqrt(6)
pressure = softmax(L) · [q/2, o, h/2]
score = (queued + batch_reserved + blocks)/capacity + pressure
```

The age-independent terms can be cached for matching candidates within a batch;
one age exponential and its square recover the remaining weights. This changes
neither the fixed mapping nor its attention contribution. Tests compare numerical
pressure and all eight sequential choices over 2,000 deterministic random states.
High in-flight pressure can reverse a lower base-load preference. With identical
sides, the first task chooses the lower side ID and its reservation makes the
next task prefer the other side. Reservations update immediately, in ready order.

## Ownership and resource matching

A successful pending enqueue transfers sole final-publication ownership to S0.
Full pending queues go directly to Shared. Snapshot, compatibility and budget
failures after dequeue also publish exactly once to Shared. Already-bound tasks
are never migrated. The original block-range claiming and final dispatch checks
remain authoritative. S0 services pending before the synchronous-drain continue,
so allocation cannot stop merely because a sync task is draining the machine.

Each S publishes local resource words with a timestamp and odd/even version.
Readers distinguish uninitialized state, age greater than 200 microseconds and
inconsistent versions. Snapshots are estimates, not hardware reservations.
Pure AIC and AIV use separate block ledgers. Ordinary MIX gets separate channels
for active masks 3, 5, 6 and 7. A group counts only when every required core in that
same AIC/AIV0/AIV1 group is immediately usable or accepts pending work. The code
reuses CoreTracker's matching/classification logic; aggregate AIC/AIV minima do
not establish compatible-group capacity.

Only ordinary whole MIX tasks fitting either side enter this allocator. Oversize,
strict placement, special predicates, sync-start, DUMMY and stager-owned tasks
retain their protocols. Qwen's forty fused-attention MIX tasks are synchronous,
so ordinary MIX coverage is validated with independent scenes and short chains.
DUMMY remains inline; sampled wait, processing time, successor count and serving
S quantify CPU pressure and ready fanout. Early-dispatch diagnostics distinguish
candidates, successful enqueue, queue-full fallback, doorbell release and sampled
held slots. Candidate-minus-hit includes rejections other than queue fullness.

## Calibration, real-action training and counterfactual replay

Actual AICore start is unavailable. Start proxies explicitly identify register
publication, observed pending promotion or gated doorbell release. A FIN-only
fallback is censored and excluded from runtime supervision. FIN itself is observed
by AICPU, so proxy runtime includes polling and synchronization effects.

Calibration groups kernel IDs, launch block count, resource mask and sync spec.
Only training runs fit the relation between proxy runtime and sequence length.
Queue/reservation depth and in-flight occupancy estimate base startup waiting.
The table is an approximate time model; it is not a measured per-block kernel
execution-time oracle.

The same actual-action rows fit ridge residual regression and a lightweight
learned single-head Q/K/V model. Both predict waiting/runtime residuals relative
to this base model. Complete runs at sequence lengths 512/1750 train the models;
2625 validates and controls early stopping; 3500 is held out. Normalization fits
training rows only. Repeated layers are never randomly split between partitions.
Only the action actually executed receives a real label.

Resource-calendar DAG replay generates separate candidate-side labels. Changing
an action reschedules compatible cores, resource releases and successor ready
events; recorded ready times are not reused as counterfactual labels. MIX reserves
all active cores within one group, sync cohorts reserve a common start, and DUMMY
uses sampled CPU service. Replay fidelity and its unmodeled instruction/pending
protocol costs are reported independently from real-action holdout errors.

The predeclared gate requires attention holdout composite MAE at least 2% below
ridge, neither target worse, and batch-eight/16-candidate CPU shadow p95 at most
50 microseconds. Exported native inference is checked against PyTorch. Passing
host gates still requires an actual AICPU shadow budget check before dispatch.
Failed gates retain all weights, errors and failure cases and exclude the trained
policy from formal action comparisons.

## Actual learned dispatch

Mode 4 retains improved admission, snapshot checks, fallback and sequential batch
reservations. Twelve explicit features project to a four-dimensional Q; three
five-dimensional resource tokens project to four-dimensional K/V. A single head
plus a linear skip predicts waiting/runtime residuals, with 136 learned parameters.
The microsecond score is `max(0, base_wait + residual_wait) +
max(0, base_runtime + residual_runtime)`. Missing kernel/launch/mask calibration
uses the fixed-formula fallback and is counted. The exported table targets only
Qwen seq_len=3500. Runtime weights and calibration remain immutable.

Actual heldout-quality and AICPU shadow gates passed before mode 4 dispatch.
Six separately collected complete replays each contain 11,166 tasks/23,605 edges,
with submitted/traced counts and every endpoint checked. Sync cohorts use distinct
physical groups. Missing fast-path records in earlier traces are not invented as
training labels. The original-QKV row means its preserved formula/routing in the
unified second-round binary, rather than a bitwise rerun of the old binary. Fast
direct-publication paths retain their existing Shared protocol.

## Frozen board comparison

Output plus all forty layers of K/V cache must pass golden, followed by short
chain, ordinary MIX, fallback and sync-coexistence checks. Calibration, weights,
parameters and source/binary hashes freeze before formal timing. Six periods
rotate and reverse policy order. Every fresh process retains six rounds: the
first is warm-up and the next five are formal, giving thirty samples per policy.
Shared is remeasured in the same round. All policies must load the same thirty-seven
compute-kernel fingerprints, topology, build and input. Reports retain median,
p95, coverage, natural batch histogram, fallback reasons, decision costs and
separate diagnostic waits, including regressions.

Commands and raw data: `results/qkv-round2-20261125/REPRODUCE.md`.
The final report in that directory is authoritative for measured outcomes.


[Reproduction and raw evidence](../../../../official-results/qkv-round2-20261125/REPRODUCE.md) · [Frozen runtime source](https://github.com/JJyun-ustc/PyPTO-tmr_partition/tree/8ebbff77377175bbbdc8d8f7252897f3eb73e6f8)
