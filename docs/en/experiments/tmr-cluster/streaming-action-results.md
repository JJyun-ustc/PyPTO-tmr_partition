# Streaming TMR: visibility, calibration and infrastructure cost

2026-11-27. **This is not an active A–D scheduling-benefit comparison. C/D resource calendars and scoring are not connected to active NPU overrides; the empty-admission build takes the default path.** The measurements establish A/B cost, factual prefix visibility, diamond correctness and calibration limits, not analytical or learned benefit.

| Policy | n | median ms | p95 ms | vs Shared median |
|---|---:|---:|---:|---:|
| A · Shared | 60 | 36.650350 | 36.730563 | +0.00% |
| B · infrastructure | 60 | 38.206820 | 38.349504 | +4.25% |
| C guard · default only | 60 | 38.197500 | 38.305580 | +4.22% |
| D guard · default only | 60 | 38.194300 | 38.376780 | +4.21% |

| seq_len | A median / p95 ms | B median / p95 ms | B vs A median |
|---|---:|---:|---:|
| 2250 | 33.175060 / 33.275701 | 34.757010 / 34.875016 | +4.77% |
| 5000 | 40.525130 / 40.621897 | 42.119290 / 42.250695 | +3.93% |

Target: batch 16, 40 layers, seq_len 3500, seed 1234; sealed heldouts 2250/5000. Each arm has 12 periods × (1 retained warm-up + 5 ordinary samples). All 720 ordinary samples and 144 warm-ups are retained. B's median paired-period change is +4.23%, bootstrap 95% interval [+4.06%, +4.40%] over 12 independent periods. p95 uses linear quantiles. Golden, pilot and diagnostic measurements are excluded. Differences among default-only C/D arms are not model effects.

One Ascend910B2, physical card 3 / ACL 0; one binary, five S plus one O, actual AICPU IDs 2–6 for S and 7 for O, 24 AIC/48 AIV. DFX=1, scheduler profiling=0 at build; detailed events disabled for ordinary measurements. All timing arms and real-kernel placement calibration load identical fingerprints for 37 compute kernels. Starting commit: `8ebbff77377175bbbdc8d8f7252897f3eb73e6f8`. Device locking was unavailable. This cannot re-estimate the historical 7.11% regression's causal decomposition measured on card 4.

- Correctness: all 12 per-arm output/full 40-layer K/V golden checks pass, with output/current-token K/V rows poisoned before every arm. Native tests 55/55, CPU prefix tests 24/24, four onboard diamonds and six coexistence regression groups pass. Eligible counts match successful block claims on every run; this is not a complete execution-ledger proof for every resource combination.
- Coverage: target B processes 9962 eligible tasks per run through new claiming infrastructure, 597720 successful block claims across its 60 ordinary runs. Alternative comparisons, overrides and affinity-queue publications are zero. Infrastructure coverage is not model coverage. No active placement benefit is observed or evaluated.
- Final-build visibility: among 9962 ordinary single-block Shared queue observations, 52 (0.522%) have a usable join within the fanin-4 budget; 8920 have a registered join exceeding that budget. Reinterpreting only captured successor indices/epochs at budget 128 yields 8972 (90.06%). This diagnostic denominator is not actual three-head comparison opportunities; ranking-change rate is unmeasured.
- Calibration: 6048 same-address/drained custom reuse tasks; 504 sparse CPU-proxy and 144 AICore-profiled real Qwen placement runs. All 23,040 selected AICore records satisfy causal timestamp inequalities. Every profiled launch matches the independent baseline's 10,604 compute task IDs and 19,208 block invocations. CPU publication proxy minus AICore interval median/p95 is 4.72/13.86 µs; pending-promotion proxy 0.02/8.295 µs, kept separate. Physical-core-matched contrasts change direction across six periods; occupancy matching retains only 43 tile pairs with mixed signs. No reliable locality class is admitted.
- Cost/models: every-32nd queue-pop invocation median/p95 0.040/0.780 µs, n=2098; includes empty and batch calls, not neural inference latency. B also enables metadata registration, so the regression cannot be attributed to queue operations alone. Factual span calibration heldout MAE/p95 absolute error 1.790/5.640 µs does not calibrate two-action gains. Independent a0/a1 gain interventions are missing. New E/F models were not trained, QKV shadow inference was not sampled, and net advantage over linear is unestablished. Earlier weights predict a different target.

Next priority: bounded frontier summaries for already-visible high-fanin joins, while containing metadata hot-path cost. The fanin-4 budget currently hides most usable information. Establish candidate/gain measurements and independent action calibration before active C/D or learned residual promotion.

[Mechanism](streaming-action-mechanism.md) · [中文](../../../zh/experiments/tmr-cluster/streaming-action-results.md) · [Reproduce](streaming-action-reproduce.md) · [Raw samples](../../../../official-results/streaming-action-20261127/final-formal-3500/samples.json) · [Freeze](../../../../official-results/streaming-action-20261127/formal-freeze.json) · [Diagnostics](../../../../official-results/streaming-action-20261127/final-diagnose-3500/diagnostic-summary.json) · [Calibration](../../../../official-results/streaming-action-20261127/qwen-core-calibration/summary.json) · [Audit](../../../../official-results/streaming-action-20261127/audit.json) · [Implementation diff](../../../../official-results/streaming-action-20261127/runtime-and-tests.patch). Failures and superseded attempts remain. Simulation scene compilation is blocked by absent g++-15; CPU/native and real-device evidence are separate. Published with these explicit implementation and validation limits.

[Complete raw archive](../../../../official-results/streaming-action-20261127/raw-evidence.tar.gz) includes logs, AICore records, failed and superseded attempts. Only machine paths are normalized and generated Python caches omitted. [Original/public file hashes](../../../../official-results/streaming-action-20261127/publication-file-hashes.json) record every transformation. Local-only status fields in original audits describe collection time; see the [publication manifest](../../../../official-results/streaming-action-20261127/publication-manifest.json) for packaging status.
