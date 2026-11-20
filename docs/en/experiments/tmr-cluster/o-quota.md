# v14: O block quotas versus v8 and Shared

v14 assigns immutable per-side block ranges in O. S no longer corrects placement at first ready,
WAIT completion does not accumulate consumer votes, and cross-side pickup is removed.
Qwen improves over v8 but remains slower than Shared. Bgemm and Paged Attention regress against v8.
The policy and prices were frozen before measurement; no best-candidate substitution or tuning was used.
See [the algorithm](algorithm-v14.md).

## Device-wall results

Median milliseconds, 30 samples per cell. Positive percentages mean slower.

| Case | Shared | v8 (ms) | v14 | v14 vs v8 | v14 vs Shared |
| ---- | -----: | ------: | --: | --------: | ------------: |
| Bgemm | 1.04637 | 1.00804 | 1.15184 | +14.27% | +10.08% |
| Scale | 4.85766 | 3.86222 | 5.67701 | +46.99% | +16.87% |
| Uniform | 2.45813 | 2.20300 | 2.93216 | +33.10% | +19.28% |
| Skew | 3.28223 | 2.90045 | 3.87212 | +33.50% | +17.97% |
| Qwen | 36.10933 | 46.00712 | 40.56889 | -11.82% | +12.35% |

| Case | Shared p95 | v8 p95 | v14 p95 |
| ---- | ---------: | -----: | ------: |
| Bgemm | 1.19994 | 1.18068 | 1.33322 |
| Scale | 4.90197 | 3.94833 | 5.75317 |
| Uniform | 2.51731 | 2.29943 | 3.00060 |
| Skew | 3.33492 | 2.98684 | 3.95045 |
| Qwen | 36.50633 | 46.60215 | 41.20776 |

Shared is upstream with only the maximum AICPU thread count changed from four to six.
All 591 v8 source files match published snapshot `7ac2fb88cf978435d8c06ccfd701bbcc036f8d5d`,
including the locally untracked partition header. Independent runtime builds were used.
Loaded compute-kernel FNV hashes match across versions.
Qwen is 40 layers, batch 16, sequence 3500, maximum sequence 5500, no LM head, CANN fused attention.
Scale is the larger Bgemm workload; Uniform/Skew are Paged Attention workloads.

## Protocol and limitations

Ascend910B2 physical card 6, logical device 0; O on CPU7, S on CPU2-6.
All six version permutations, five rounds per process, rotated case order, all first rounds retained:
450 ordinary samples, separate 15 model golden runs and 15 level-4 diagnostic runs.
Torch/OMP/MKL/OpenBLAS and compilation were limited to two threads/workers.
No task-submit or exclusive device lock was available.

A connection interruption terminated the process. The 255 validated samples were retained;
the interrupted v14/Qwen/block3 invocation was archived and rerun in full before continuing.
Source and binary fingerprints match at restart and after every invocation.
The experiment spans a session/container restart and may include environmental drift.
All six batch medians, raw samples, and the restart record are preserved.

## Independent diagnostics

One instrumented run per version/case; milliseconds. O score is inside O submit.
O submit includes resource-reclamation/wait time, not just scoring CPU work.
Level-4 Qwen collection substantially expands execution (Shared O submit is about 1.20 seconds
versus ordinary device wall about 36 ms); these captures validate events and mechanisms,
not representative proportions of ordinary execution.
S feedback and S phases sum CPU time over five threads and are not device critical-path durations.
These values include tracing overhead and cannot be subtracted from ordinary wall time.

| Case | Shared O submit | v8 O submit | v14 O submit | v14 O score | v14 S feedback sum | v14 split tasks |
| ---- | --------------: | ----------: | -----------: | ----------: | -----------------: | --------------: |
| Bgemm | 1.21746 | 1.36792 | 1.49376 | 0.12116 | 0.02510 | 0 |
| Scale | 4.92962 | 5.24786 | 6.61672 | 0.49508 | 0.14184 | 0 |
| Uniform | 3.00582 | 3.42208 | 3.78800 | 0.27220 | 0.08482 | 0 |
| Skew | 3.94152 | 4.54986 | 5.02972 | 0.35930 | 0.10546 | 0 |
| Qwen | 1199.52908 | 1201.83624 | 1201.46142 | 1.36668 | 30.19438 | 422 |

| Case | Shared complete | Shared dispatch | Shared resolve | v8 complete | v8 dispatch | v8 resolve | v14 complete | v14 dispatch | v14 resolve |
| ---- | --------------: | --------------: | -------------: | ----------: | ----------: | ---------: | -----------: | -----------: | ----------: |
| Bgemm | 1.12352 | 1.68534 | 0.01118 | 1.11640 | 1.61152 | 0.02608 | 1.19712 | 1.73250 | 0.02194 |
| Scale | 3.91256 | 6.48170 | 0.08996 | 3.31314 | 5.92670 | 0.06452 | 4.58008 | 7.67160 | 0.14768 |
| Uniform | 2.24520 | 3.34176 | 0.07306 | 2.21756 | 3.79796 | 0.34102 | 2.35034 | 3.37334 | 0.09328 |
| Skew | 2.93032 | 4.46792 | 0.08522 | 2.91438 | 4.91066 | 0.42656 | 3.15746 | 4.64142 | 0.10828 |
| Qwen | 25.59220 | 25.15490 | 7.48764 | 20.73002 | 26.00382 | 8.56960 | 25.14532 | 25.80296 | 9.05042 |

Capture completeness flags remain in diagnostics.json. Incomplete streams describe only captured work.
The legacy cross_pickup trace slot measures an empty segment in v14, not a pickup operation.
All S affinity_votes call counts are zero. Per-side assigned, ready, claimed, executed,
and finished_units/6 agree. AIV and MIX hardware regressions each exercise five actual splits.

O still reads creation-time predecessor distributions and scores each task. Ready publication
may require two queue entries. Local range claims and FIN bookkeeping remain, as do periodic
leader summaries. The 64-iteration/task intervals apply to summary publication/reads, not local
per-event bookkeeping. Single-writer counters avoid a new shared fetch-add but are not free.
All four microbenchmarks have zero split tasks: their single-block tasks incur scoring and
feedback costs without a multi-block split benefit; dual-queue publication cannot explain
those regressions. Qwen actually splits 422 tasks.
Equivalent-block prices do not model kernel duration or MIX interference. Stale fixed quotas
cannot be repaired by an idle side. These are plausible contributors, not isolated causal claims:
no single-mechanism ablation was performed. v14 is a mechanism experiment, not a general replacement.

## Validation and reproduction

184 CPU cases across eight targets; 15 board regression cases; 16 terminal ledger groups;
DFX-off build and runtime pre-commit checks pass. All 15 model golden runs pass, including Qwen output
and every layer's KV cache. Full-project scenes/simulation and O-position controls were not run.

Use runtime/examples/a2a3/tensormap_and_ringbuffer/benchmark_o_quota/run_study.py with independent
--shared, --v8, --v14 checkouts and --output, running phases preflight, measure, diagnostics in order.
Then run analyze_study.py on the output directory. Use --resume after interruption.
Published raw files over 500 KiB are gzip-compressed. Large compressed files are split into numbered parts.
Reassemble with `cat <file>.gz.part* > <file>.gz`, then decompress before reanalysis.
No binaries, virtual environments, caches, or generic driver logs are published.

[summary](../../../../official-results/v14-o-quota-20261120/summary.json),
[samples](../../../../official-results/v14-o-quota-20261120/samples.json),
[CSV](../../../../official-results/v14-o-quota-20261120/results.csv),
[batch medians](../../../../official-results/v14-o-quota-20261120/batch-results.csv),
[diagnostics](../../../../official-results/v14-o-quota-20261120/diagnostics.json).
