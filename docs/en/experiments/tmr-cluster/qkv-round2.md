# QKV round two — one-page conclusion

[Chinese](../../../zh/experiments/tmr-cluster/qkv-round2.md) · [Mechanism](algorithm-qkv-round2.md)

| Policy | n | Median ms | p95 ms | Median vs Shared | p95 vs Shared |
|---|---:|---:|---:|---:|---:|
| Shared | 30 | 36.830900 | 37.007384 | +0.00% | +0.00% |
| Original QKV | 30 | 40.004970 | 40.195601 | +8.62% | +8.62% |
| Improved QKV | 30 | 37.668440 | 37.791733 | +2.27% | +2.12% |
| Learned QKV | 30 | 37.615719 | 37.809745 | +2.13% | +2.17% |


Thirty formal samples plus six retained warm-ups per policy. Same 910B2 (physical 4 / ACL 0), B16, 40 layers, seq=3500, seed=1234; one binary, six interleaved periods. Positive changes mean slower. All policies pass output and complete forty-layer K/V golden; all 37 loaded compute kernels match round one. No exclusive device lock is held.

1. **The main original cost is centralized pending/S0 waiting.** Attention-off barely changes short-control wall time; centralized service with Shared publication remains slow. Original/improved sampled pending means are 49.04/9.94 µs, with different coverage. Controls do not provide an additive causal decomposition. Pop/snapshot/score/publication and S0 dispatch gaps are retained.
2. **Ordinary MIX and feedback work, with a coverage bottleneck.** The sync-coexistence test actually assigns 20/23 ordinary MIX tasks. A 1,024-task short chain assigns only 0/1008 eligible tasks, largely falling back on missing/stale summaries. Qwen's forty MIX tasks remain synchronous attention, so MIX support adds no direct Qwen coverage. Improved admission reduces central service when immediate capacity is absent, at the cost of more Shared fallback.
3. **Better learned-QKV prediction does not establish a net gain over linear scheduling.** Composite MAE on 12,361 actual heldout actions is 7.616 → 6.526 (-14.31%) µs. AICPU batch-eight/16-candidate shadow p95 is QKV/linear 4.655/1.637 µs. Learned dispatch passes its gates; learned versus improved median changes by -0.14%. Linear dispatch is not formally benchmarked. Non-tie replay ranking agreement is equal for the two models, showing no attention advantage.

| Policy | Eligible | Assigned | Coverage | Multi-task batches | Fallback | Decision mean µs (batches) |
|---|---:|---:|---:|---:|---:|---:|
| Original QKV | 298830 | 288134 | 96.42% | 40281 | 10696 | 6.202 (43950) |
| Improved QKV | 298831 | 33553 | 11.23% | 9627 | 265278 | 3.900 (13615) |
| Learned QKV | 298830 | 33750 | 11.29% | 9585 | 265080 | 5.169 (13222) |


Decision means start after dequeue; pop is sampled separately. Start labels are register/promotion/doorbell proxies, FIN is AICPU-observed, and actual AICore start is unavailable. Each complete replay has 11,166 tasks/23,605 edges, no dangling endpoints, and recomputes releases and successor readiness after an intervention. Real supervision and replay labels remain separate. Failed and superseded attempts are archived.

**One next check: S0 service gaps and delayed summary refresh.** Hold weights fixed and explain pending waits plus short-MIX fallbacks. This round ends with code and data delivery.

[Detailed report](../../../../official-results/qkv-round2-20261125/REPORT.en.md) · [Commands](../../../../official-results/qkv-round2-20261125/REPRODUCE.md) · [Samples](../../../../official-results/qkv-round2-20261125/measure/summary.json) · [Feedback](../../../../official-results/qkv-round2-20261125/coverage.json) · [Models](../../../../official-results/qkv-round2-20261125/models/) · [Implementation diff](../../../../official-results/qkv-round2-20261125/runtime-and-tests.patch)


[Publication manifest](../../../../official-results/qkv-round2-20261125/publication-manifest.json) · [Audit](../../../../official-results/qkv-round2-20261125/audit.json). Raw host logs and custom QKV traces are archived in `raw-evidence.tar.gz`; extract it into the result directory before offline parsing. Generic Ascend driver logs and compiled executables are omitted; model weights are retained. Machine-specific root paths use named placeholders, with original/published SHA-256 recorded.
