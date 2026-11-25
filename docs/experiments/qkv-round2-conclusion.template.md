# QKV round two — one-page conclusion

{timing}

Thirty formal samples plus six retained warm-ups per policy. Same 910B2 (physical 4 / ACL 0), B16, 40 layers, seq=3500, seed=1234; one binary, six interleaved periods. Positive changes mean slower. All policies pass output and complete forty-layer K/V golden; all 37 loaded compute kernels match round one. No exclusive device lock is held.

1. **The main original cost is centralized pending/S0 waiting.** Attention-off barely changes short-control wall time; centralized service with Shared publication remains slow. Original/improved sampled pending means are {pending} µs, with different coverage. Controls do not provide an additive causal decomposition. Pop/snapshot/score/publication and S0 dispatch gaps are retained.
2. **Ordinary MIX and feedback work, with a coverage bottleneck.** The sync-coexistence test actually assigns {mix} ordinary MIX tasks. A 1,024-task short chain assigns only {chain} eligible tasks, largely falling back on missing/stale summaries. Qwen's forty MIX tasks remain synchronous attention, so MIX support adds no direct Qwen coverage. Improved admission reduces central service when immediate capacity is absent, at the cost of more Shared fallback.
3. **Better learned-QKV prediction does not establish a net gain over linear scheduling.** Composite MAE on 12,361 actual heldout actions is {mae} µs. AICPU batch-eight/16-candidate shadow p95 is QKV/linear {shadow} µs. Learned dispatch passes its gates; learned versus improved median changes by {trained_delta}. Linear dispatch is not formally benchmarked. Non-tie replay ranking agreement is equal for the two models, showing no attention advantage.

{coverage}

Decision means start after dequeue; pop is sampled separately. Start labels are register/promotion/doorbell proxies, FIN is AICPU-observed, and actual AICore start is unavailable. Each complete replay has 11,166 tasks/23,605 edges, no dangling endpoints, and recomputes releases and successor readiness after an intervention. Real supervision and replay labels remain separate. Failed and superseded attempts are archived.

**One next check: S0 service gaps and delayed summary refresh.** Hold weights fixed and explain pending waits plus short-MIX fallbacks. This round ends with code and data delivery.

[Detailed report](REPORT.en.md) · [Commands](REPRODUCE.md) · [Samples](measure/summary.json) · [Feedback](coverage.json) · [Models](models/) · [Implementation diff](runtime-and-tests.patch)
