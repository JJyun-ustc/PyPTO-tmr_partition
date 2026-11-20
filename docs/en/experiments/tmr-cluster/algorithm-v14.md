# Orchestrator block quotas (v14)

The orchestrator assigns disjoint, immutable block ranges to two scheduler
sides. Ready schedulers publish the assigned ranges without choosing a side.
The WAIT completion path does not collect consumer affinity votes, and there
is no cross-side pickup. Creation-time predecessor affinity remains available.

## Score and allocation

For resource shape r and side c, pressure in equivalent waves is
P[c] = (Q[c] + I[c] + 0.5 B[c]) / C[c]. Q is ready but unclaimed blocks,
I is claimed but not finished blocks, B is assigned but not ready blocks,
and C is the compatible capacity. All counts are specific to the resource shape.
Affinity adds H[c] = 0.5 (1 - a[c]) waves. Unknown affinity is one half
on each side. Scores use double arithmetic with a scale of 256; prices are
policy assumptions, not measured execution or communication durations.

For N blocks, O evaluates allocations x to side zero and N-x to side one.
It minimizes max(P[c] + H[c] + n[c]/C[c]) over nonempty sides, plus a fixed
0.125 wave penalty when both sides receive work. The constant-size candidate
set is {0, N, floor(x*), ceil(x*)}, where x* is the clipped equal-load solution.
Only feasible sides receive work. Predicated tasks remain on one side;
sync-start and explicitly pinned tasks retain their original dispatch rules.
The new allocation contributes its full marginal block count; the half-weight
future discount applies only to older reservations.

O immediately adds every allocation to its cumulative assigned counts.
At first ready, one writer adds each side's quota to its ready ledger exactly
once. This moves work out of future reservations. Claiming moves work from
Q to I; it does not erase running pressure. Each scheduler locally accumulates
six retirement units per logical block (6, 3, or 2 per finished subtask), using
single-writer relaxed atomic stores, without a new inter-thread fetch-add.
These counters track compute FIN, not asynchronous post-FIN work.

## Feedback and dispatch

Each S updates its capacity every 64 scheduler iterations. Each side's leader
aggregates only that side's S writers and publishes one versioned snapshot
covering all resource shapes. Ordinary S threads no longer read the two
summaries. O reads them every 64 task creations, makes one snapshot attempt,
and retains its previous view if publication overlaps the read.
O's own ready row is read directly and excluded from S summaries.
Cumulative assigned counts are current O state; snapshots replace old views
rather than adding cumulative counts repeatedly. Signed differences are
clamped only when forming pressure. Feedback is asynchronous and can be stale.

Each Auto task stores a quota and range cursor for each side. Side zero owns
[0,n0), side one owns [n0,N). The MPMC queues contain separate pointer entries,
not intrusive task nodes, so both can refer to the same task safely. A side
atomically claims only its own range and republishes its queue entry before
preparing the claimed blocks. Task completion still uses the existing shared
subtask completion counter and releases successors once, after all blocks.
The quota metadata replaces WAIT vote storage without growing TaskPayload.

## Validation and limits

The reproducible comparison entry is
runtime/examples/a2a3/tensormap_and_ringbuffer/benchmark_o_quota/run_study.py.
It compares independently built v14, v8, and frozen upstream Shared with six
AICPU threads, all six version permutations, five rounds per process, all first
rounds retained, and separate correctness and diagnostic runs. Compute code
hashes and runtime/source fingerprints must match their frozen manifests.

Pure CPU tests cover allocation, capacity ratios, reservation/ready separation,
running pressure, snapshot replacement, O row exclusion, concurrent range
coverage, dual-queue publication, and once-only ready accounting. Board golden
runs and original multiblock AIV/MIX/sync regressions validate integration.

Equivalent blocks do not model different kernel durations. MIX interference
with pure AIC/AIV tasks is not fully represented by separate shape ledgers.
Non-Auto work is not included in this pressure ledger. A side cannot borrow
another side's remaining quota if a prediction becomes stale. Task-level
predecessor affinity does not describe which tensor region a block accesses.
A split is permitted only under the existing independent-block task contract;
barrier cohorts must declare sync-start and bypass splitting.

See [the v14 comparison](o-quota.md) for measured results and diagnostic limits.
