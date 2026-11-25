# Fixed QKV ready-batch scheduling versus Shared

[Chinese](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/docs/zh/experiments/tmr-cluster/qkv-ready-batch.md) · [Mechanism and reproduction](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/docs/en/experiments/tmr-cluster/algorithm-qkv-ready-batch.md)

The fixed CPU QKV prototype completes implementation, build, correctness checks,
on-device golden validation and one Shared comparison. It regresses on this workload.
No weight sweep, batch-size grid, training, NPU offload or additional policy ablation was run.

| Qwen device_wall | Shared | QKV | Change |
| --- | ---: | ---: | ---: |
| Formal samples | 30 | 30 | — |
| Median | 36.697130 ms | 39.307130 ms | **+7.11%** |
| p95 | 36.829344 ms | 39.462639 ms | **+7.15%** |

All six period medians regress: +7.06%, +7.27%, +7.56%, +7.23%, +6.70%, +6.70%.
These results do not establish a causal explanation for the regression.

## Source, workload and consistency

The runtime branch is `runtime-qkv-ready-batch`. The measurement starts from Shared
`d8f997fab`, reusing side queues, range claims and the six-thread topology from
`7ac2fb88c`. Model orchestration uses ordinary Shared submissions; the old Auto
placement scorer is inactive. Shared and QKV select `SIMPLER_TMR_QKV_BATCH=0/1`
in the same locally built runtime binary.

Measured on 2026-11-25 UTC, Ascend 910B2 physical card 4 / ACL logical device 0.
Every formal round reports CPUs `[2,3,4,5,6,7]`: S0/S1 on side 0, S2/S3/S4 on
side 1 and O on CPU 7. Torch/OMP/MKL/OpenBLAS and compilation use two threads/workers.
Qwen uses batch 16, 40 layers, sequence 3500, maximum sequence 5500 and seed 1234.
The repository fixture repeats layer-0 weights across layers and computes hidden-to-hidden output.

Both output and **all 40 layers of K/V cache** golden checks pass at the repository
rtol=0.05/atol=0.1. All formal processes load the same **37 compute-kernel code
fingerprints** and orchestration hash `0xd0f1f953f424f7e4`. All 540 measured runtime
source/binary fingerprints agree before and after timing. Build flags are C++17,
`-O3 -g`, DFX=1 and other profiling=0. Publication adds license comments and docs;
[post-measurement changes](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/official-results/qkv-ready-batch-20261125/post-measurement-changes.json)
separate those edits from the original measured source patch and frozen hashes.

## Coverage, feedback and validation

Across the 30 formal QKV rounds:

- 298,830 eligible tasks; **288,522 assigned (96.55%)**, split 111,343 / 177,179 between sides.
- **44,793 natural batches**, including **40,325 multi-task batches** and **31,414 batches of eight**.
  Sizes 1..8: `[4468,3520,1210,1303,908,1314,656,31414]`.
- **10,308 Shared fallbacks**: 10,152 pending-queue-full and 156 unavailable-snapshot;
  no-compatible-side, budget and oversize are zero in these Qwen formal rounds.
  The board multiblock regression exercises oversize fallback, CPU tests exercise no-side,
  and a separate golden run records one budget fallback.
- Mean decision time **5.283 us/batch**, maximum **15.40 us**, over 44,793 batches;
  mean cumulative allocator time **7.889 ms/round**. This is CPU work and is not an
  additive decomposition of device_wall. The separate diagnostic has 1,487 exact
  samples: median **6.060 us**, p95 **7.634 us**. It is excluded from formal timing.

Assigned and claimed block totals agree and pending ends empty for every formal run.
Dispatch asserts that the executing side matches the selected queue. The CPU scheduler
suite passes 43 cases, including eight QKV cases; task-state, wiring and reused
infrastructure targets also pass. Four board golden groups cover sync coexistence,
ordinary multiblock, DUMMY and early dispatch. Sync coexistence assigns all six singleton
batches, demonstrating progress under low load and alongside sync tasks.
The deterministic formula case proves both that changed K/V can reverse the lower-base-load
choice and that the first assignment changes the second choice through batch reservations.

## Collection protocol and retained attempts

Six periods alternate Shared→QKV and QKV→Shared. Every fresh process has one retained,
excluded warmup followed by five formal samples. All **72 ordinary rounds** survive
(60 formal + 12 warmups); the first formal sample is retained. p95 interpolates at
`(n-1)*0.95`. Formal runs use `info` for cold aggregate counters and swimlane=0;
detailed per-batch diagnostics are disabled.

The initial `timing`-level attempt suppressed aggregate feedback and failed collection
validation. Both modes' 12 round durations are retained under
`attempts/measurement-timing-loglevel/`, excluded as a whole, then collection restarts
with matching logging in both modes. Strategy, parameters, runtime binary and compute
kernels do not change. The invalid ACL device-4 attempt and Shared verification before
the initialization-race correction are retained too. No task-submit/exclusive device lock
is available in the measured container; before/after device queries are retained.

Public evidence omits compiled binaries, venvs/caches and generic driver logs, following
the repository's publication convention. Every sample, failed collection attempt,
golden/CPU/board regression result, QKV diagnostic capture and fingerprint is retained.
Logs/patches and files over 500 KiB are gzip-compressed. Machine-local paths are replaced
with named placeholders; the manifest records original and published SHA-256 values.
Read compressed text with `gzip -dc <file>.gz`. The full original local archive remains unchanged.

## Next investigation and evidence

Investigate **S0 allocator service cost and pending-queue wait** next. High coverage,
frequent natural batches, about 7.9 ms allocator work per round and about 3.4% queue-full
fallback motivate this hypothesis; they do not prove it causes the 7.11% regression.
This round stops at code and data delivery.

[All evidence](https://github.com/JJyun-ustc/PyPTO-tmr_partition/tree/main/official-results/qkv-ready-batch-20261125),
[formal samples/statistics](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/official-results/qkv-ready-batch-20261125/summary.json),
[feedback](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/official-results/qkv-ready-batch-20261125/feedback.json),
[audit/period medians](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/official-results/qkv-ready-batch-20261125/audit.json),
[publication manifest](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/official-results/qkv-ready-batch-20261125/publication-manifest.json).
