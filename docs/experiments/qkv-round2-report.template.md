# QKV round two — detailed report

{p0}

Thirty formal samples plus six retained warm-ups per policy; same 910B2 card (physical 4 / ACL 0), B16, 40 layers, seq=3500, seed=1234. Six interleaved periods use one binary. Positive percentages mean slower. All four policies pass output and complete 40-layer K/V golden (repository rtol=.05, atol=.1); all 37 loaded compute kernels match each other and round one. No exclusive device-lock tool is available.

**Primary first-version cost: centralized pending/S0 service waiting.** Short controls retain their first round and use five subsequent samples: {p1}. Removing attention barely changes wall time; retaining centralized service while publishing Shared remains expensive. This supports centralized routing as the main cause; short controls do not provide an additive causal decomposition. Final sampled pending mean: original {p2} µs (n={p3}), improved {p4} µs (n={p5}); admission coverage differs. S0 stage means in µs: original {p6}; improved {p7}. Dispatch gaps, DUMMY CPU pressure and early-slot observations are in diagnostics-summary.json.

**Ordinary MIX and feedback: real assignment, limited short-chain coverage.** The sync-coexistence scene assigns {p8}/{p9} ordinary MIX tasks. The 1,024-task short chain assigns {p10}/{p11} eligible tasks, with initialization/staleness fallbacks retained. Qwen's forty MIX tasks are synchronous attention and retain their protocol, so ordinary MIX adds no direct Qwen coverage. Timestamp/version checks distinguish missing, stale and inconsistent summaries. Improved admission bypasses central service without immediate compatible capacity, trading assignment coverage for Shared fallback.

{p12}

**Learned QKV: prediction benefit does not establish scheduling benefit over linear.** Actual-action train/validation/test counts are 24,170/11,843/12,361, split by complete runs and sequence length. Composite holdout MAE improves from {p13} to {p14} µs ({p15}%), with neither target worse than linear. Actual AICPU batch-eight/16-candidate shadow p95 is {p16} µs versus linear {p17} µs (n={p18}), below the 50 µs gate. {p19} Linear dispatch was not formally benchmarked, so an end-to-end advantage over a linear scheduler is unproven.

Complete replay has 11,166 tasks/23,605 edges per run and no dangling endpoints. Heldout seq=3500 makespan errors: {p20}. Of {p21} counterfactuals, {p22} are non-ties; linear/attention ranking agreement is {p23}/{p24}. No attention ranking advantage is established. Per-task start-proxy error remains substantial. Replay labels remain separate from real supervision. Earlier incomplete replay is archived; actual-action labels retain their sampling limitation rather than inventing missing records.

**One next investigation: S0 service gaps and delayed summary refresh.** It connects centralized waits with missing/stale MIX summaries. Hold weights fixed and test that causal chain. No further tuning is performed here.

Raw samples: measure/*/block*/result.json (all_ms includes warm-up); coverage.json; models/ (weights, normalization, errors and failures); REPRODUCE.md; runtime-and-tests.patch. Decision means start after dequeue; pop is reported separately. Diagnostic logging never enters formal samples. AICore actual start is unavailable; runtime labels span register/promotion/doorbell proxies to AICPU-observed FIN.
