# QKV round two — detailed report

| Policy | n | Median ms | p95 ms | Median vs Shared | p95 vs Shared |
|---|---:|---:|---:|---:|---:|
| Shared | 30 | 36.830900 | 37.007384 | +0.00% | +0.00% |
| Original QKV | 30 | 40.004970 | 40.195601 | +8.62% | +8.62% |
| Improved QKV | 30 | 37.668440 | 37.791733 | +2.27% | +2.12% |
| Learned QKV | 30 | 37.615719 | 37.809745 | +2.13% | +2.17% |


Thirty formal samples plus six retained warm-ups per policy; same 910B2 card (physical 4 / ACL 0), B16, 40 layers, seq=3500, seed=1234. Six interleaved periods use one binary. Positive percentages mean slower. All four policies pass output and complete 40-layer K/V golden (repository rtol=.05, atol=.1); all 37 loaded compute kernels match each other and round one. No exclusive device-lock tool is available.

**Primary first-version cost: centralized pending/S0 service waiting.** Short controls retain their first round and use five subsequent samples: Shared 36.84874; original 39.46816; attention-off 39.45798; centralized Shared publication 38.94626 ms. Removing attention barely changes wall time; retaining centralized service while publishing Shared remains expensive. This supports centralized routing as the main cause; short controls do not provide an additive causal decomposition. Final sampled pending mean: original 49.04 µs (n=292), improved 9.94 µs (n=32); admission coverage differs. S0 stage means in µs: original pop=0.391, snapshot=1.378, score=3.494, publish=0.215; improved pop=0.158, snapshot=1.706, score=1.174, publish=0.192. Dispatch gaps, DUMMY CPU pressure and early-slot observations are in diagnostics-summary.json.

**Ordinary MIX and feedback: real assignment, limited short-chain coverage.** The sync-coexistence scene assigns 20/23 ordinary MIX tasks. The 1,024-task short chain assigns 0/1008 eligible tasks, with initialization/staleness fallbacks retained. Qwen's forty MIX tasks are synchronous attention and retain their protocol, so ordinary MIX adds no direct Qwen coverage. Timestamp/version checks distinguish missing, stale and inconsistent summaries. Improved admission bypasses central service without immediate compatible capacity, trading assignment coverage for Shared fallback.

| Policy | Eligible | Assigned | Coverage | Multi-task batches | Fallback | Decision mean µs (batches) |
|---|---:|---:|---:|---:|---:|---:|
| Original QKV | 298830 | 288134 | 96.42% | 40281 | 10696 | 6.202 (43950) |
| Improved QKV | 298831 | 33553 | 11.23% | 9627 | 265278 | 3.900 (13615) |
| Learned QKV | 298830 | 33750 | 11.29% | 9585 | 265080 | 5.169 (13222) |


**Learned QKV: prediction benefit does not establish scheduling benefit over linear.** Actual-action train/validation/test counts are 24,170/11,843/12,361, split by complete runs and sequence length. Composite holdout MAE improves from 7.616 to 6.526 µs (-14.31%), with neither target worse than linear. Actual AICPU batch-eight/16-candidate shadow p95 is 4.655 µs versus linear 1.637 µs (n=62), below the 50 µs gate. Learned versus improved fixed-QKV formal median changes by -0.14%. Linear dispatch was not formally benchmarked, so an end-to-end advantage over a linear scheduler is unproven.

Complete replay has 11,166 tasks/23,605 edges per run and no dangling endpoints. Heldout seq=3500 makespan errors: +1.40%, -0.34%, +1.47%. Of 48 counterfactuals, 14 are non-ties; linear/attention ranking agreement is 42.86%/42.86%. No attention ranking advantage is established. Per-task start-proxy error remains substantial. Replay labels remain separate from real supervision. Earlier incomplete replay is archived; actual-action labels retain their sampling limitation rather than inventing missing records.

**One next investigation: S0 service gaps and delayed summary refresh.** It connects centralized waits with missing/stale MIX summaries. Hold weights fixed and test that causal chain. No further tuning is performed here.

Raw samples: measure/*/block*/result.json (all_ms includes warm-up); coverage.json; models/ (weights, normalization, errors and failures); REPRODUCE.md; runtime-and-tests.patch. Decision means start after dequeue; pop is reported separately. Diagnostic logging never enters formal samples. AICore actual start is unavailable; runtime labels span register/promotion/doorbell proxies to AICPU-observed FIN.
