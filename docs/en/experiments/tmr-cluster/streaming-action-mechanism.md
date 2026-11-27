# Streaming mechanism and executed scope

## Ownership and visibility

O publishes immutable per-launch task metadata. Successor submission and completed dependency registration receive separate monotonically increasing epochs. S captures the published epoch before reading successor records. Unknown, incomplete registration, usable bounded joins, successor-index overflow and excessive fanin have distinct states. Task locations may also be unknown; that yields neutral locality.

There are three logical lanes per pure resource: neutral, prefer side 0, prefer side 1. The two new preference lanes have dedicated bounded storage; legacy automatic side queues remain separate. Queue observations contain a ticket and immutable metadata-record index. S rechecks slot sequence/dequeue position before accepting the observation. Only exact-head CAS consumption grants the queue item, and the original task/block-range CAS still grants actual compute work. A generation check rejects a mismatched record. Observation alone never reserves a slot or task.

The implemented board path examines the three heads at the existing ordinary scheduling opportunity and consumes the oldest ready head. Existing scope, sync, strict-side, predicate, DUMMY, early-staging and MIX protocols remain authoritative. The new policy's intended eligibility is ordinary unclaimed single-block AIC/AIV only. Other work continues to occupy the actual resource trackers.

All calibration classes in this build remain neutral. Therefore the board exercises metadata publication, safe observation and common B claiming, but no preference enqueue, alternative action comparison, active resource-calendar override or learned residual. C/D are guarded placeholders sharing B behavior. The analytical evaluator is implemented and tested in the CPU reference and native unit tests; it is **not wired into active NPU dispatch**. Do not infer C/D effectiveness from these runs.

## Reference two-action calculation

The CPU/native reference compares default U first against alternative T first from exactly the same snapshot. Both actions process every compared task once, updating compatible slot release times after each placement. Continuation is work conserving: choose the earliest compatible start, not a later faster-side finish. A real available slot is required for the first action.

Only registered visible joins within the fixed index/fanin budget may replace dominated task-completion endpoints. Other affected endpoints remain in the maximum. The gain is the difference between these two local maxima. Completed dependency costs are not charged again. Conditional remaining time uses the historical tail surviving the elapsed time, rather than median duration minus elapsed time.

The strict gate requires 30 independent action-calibration blocks and gain greater than optimistic-error allowance + incremental cost + 0.5 µs. The admission table is empty: the factual placement contrasts do not establish a stable producer-locality benefit or calibrate the complete two-action intervention. No synthetic label is represented as a real board counterfactual. Neural training/promotion is not authorized by these missing labels.

## What the experiments can establish

The four real diamond cases verify early/late submission visibility and fast/slow sibling FIN ordering under Shared. CPU identical-prefix/different-future cases verify no future leakage and recompute releases and successor-ready after changed actions. Synthetic perfect priors are validation fixtures, not measured prediction accuracy.

The custom reuse calibration uses the same GM addresses for both arms and drains each arm before proceeding. The original Qwen 33→34 first-tile calibration controls producer/consumer sides while leaving all 37 compute kernels unchanged. Physical-core identity, S identity and optional occupancy matching reduce confounding; this remains a combined placement intervention under decode background, not an isolated transfer measurement.

AICore records span post-ACK kernel entry to post-kernel/pre-FIN. CPU publication/pending-promotion→FIN-observed proxy families are retained separately. Selected records satisfy causal timestamp inequalities on the common 50 MHz counter scale. Full AICore task/block counts are checked against the independent complete baseline trace for every profiled launch.

The formal experiment can quantify the cost of B's metadata and queue observation infrastructure relative to Shared. It cannot attribute that cost to queue operations alone, because B also enables task-record registration. It cannot quantify active C/D benefit, join-induced action ranking changes, or learned QKV versus linear net benefit. Those are explicit remaining gaps.

[Results / 结果](streaming-action-results.md) · [Reproduction / 复现](streaming-action-reproduce.md)
