# 流式 TMR：出版复现入口

本页从 `runtime` 子模块根目录执行。当前源码快照已包含补丁；**不要再次应用补丁**。独立使用补丁时才从起点提交应用。先把出版包解压到该运行时的 `results/`，获得原目录结构和离线解析工具。

```bash
# From the publication checkout:
git submodule update --init --recursive runtime
mkdir -p runtime/results
tar -xzf official-results/streaming-action-20261127/raw-evidence.tar.gz -C runtime/results
cd runtime
```

## Commands / 命令

This experiment starts from runtime commit `8ebbff77377175bbbdc8d8f7252897f3eb73e6f8`, available as the publication repository's `runtime` submodule revision on `runtime-qkv-ready-batch`. The patch applies at that base runtime root; the pinned publication snapshot already includes it. Do not apply it at the publication root. The patch includes new example scripts and the analytical CPU/native reference implementation.

本轮起点是第二轮运行时提交，不是出版仓库的 main。不要在出版目录里直接构建。保持计算 kernel 不变；从起点提交复现时先应用补丁，使用固定源码快照则不重复应用。随后同时构建 host 和 AICPU。没有修改 kernel 权重或使用训练版策略。

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --no-build-isolation --no-deps -e . --config-settings=build.targets=_task_interface
# Load the installed CANN environment when not already loaded; this host used:
source /usr/local/Ascend/cann-9.0.0/set_env.sh
python -c 'import multiprocessing,runpy,sys; multiprocessing.cpu_count=lambda:2;sys.argv=["simpler_setup/build_runtimes.py","--platforms","a2a3"];runpy.run_path(sys.argv[0],run_name="__main__")'
cmake -B tests/ut/cpp/build -S tests/ut/cpp
cmake --build tests/ut/cpp/build --target test_a2a3_tmr_scheduler_state --parallel 2
tests/ut/cpp/build/test_a2a3_tmr_scheduler_state
python -m unittest discover -s examples/a2a3/tensormap_and_ringbuffer/benchmark_streaming_action -p test_reference.py
```

Board commands below use physical card 3 / ACL logical device 0 on this host. The scripts call the existing preflight and refuse a mismatching card. Recheck your own mapping. `task-submit` is absent here, so these runs have no device exclusivity lock. Use the repository's device queue wrapper when available. All run output directories must be new, to prevent overwriting raw evidence.

```bash
TASK_STREAM_TOOLS=examples/a2a3/tensormap_and_ringbuffer/benchmark_streaming_action
TASK_STREAM_OUT=results/streaming-reproduction
python "$TASK_STREAM_TOOLS/run_scene.py" "$TASK_STREAM_TOOLS/test_diamond.py" --output "$TASK_STREAM_OUT/diamond"
python "$TASK_STREAM_TOOLS/check_diamond.py" "$TASK_STREAM_OUT/diamond"
python examples/a2a3/tensormap_and_ringbuffer/benchmark_qkv/run_regressions.py --output "$TASK_STREAM_OUT/regressions" --mode 7 --trace 0 --card 3
python "$TASK_STREAM_TOOLS/qwen_pair_study.py" collect --output "$TASK_STREAM_OUT/qwen-pair"
python "$TASK_STREAM_TOOLS/qwen_pair_analyze.py" "$TASK_STREAM_OUT/qwen-pair"
# expected_compute_counts.json is supplied with this result; it was derived from
# the complete original Qwen Shared trace, not inferred from profiling output.
python "$TASK_STREAM_TOOLS/qwen_pair_study.py" core --output "$TASK_STREAM_OUT/qwen-core" --expected-counts results/streaming-action-20261127/qwen-core-calibration/expected_compute_counts.json
python "$TASK_STREAM_TOOLS/qwen_pair_analyze.py" "$TASK_STREAM_OUT/qwen-core"
python "$TASK_STREAM_TOOLS/calibrate_spans.py" "$TASK_STREAM_OUT/qwen-core"
```

The two calibration collections differ in instrumentation. `collect` retains ordinary host device_wall with sparse CPU trace; `core` explicitly enables AICore profiling and stores no ordinary timing value when the framework suppresses it. Both are diagnostics, excluded from formal samples. Kernel 33→34 is controlled only for its first tile per layer, with unchanged input addresses and consumer kernel.

```bash
for TASK_STREAM_SEQ in 3500 2250 5000; do
  python "$TASK_STREAM_TOOLS/run_interleaved.py" golden --seq-len "$TASK_STREAM_SEQ" --output "$TASK_STREAM_OUT/golden-$TASK_STREAM_SEQ"
done
python "$TASK_STREAM_TOOLS/run_interleaved.py" pilot --output "$TASK_STREAM_OUT/pilot"
python "$TASK_STREAM_TOOLS/run_interleaved.py" diagnose --output "$TASK_STREAM_OUT/diagnose"
python "$TASK_STREAM_TOOLS/analyze_diagnostics.py" "$TASK_STREAM_OUT/diagnose"
# Review/freeze admission and sample count BEFORE the following commands.
for TASK_STREAM_SEQ in 3500 2250 5000; do
  python results/streaming-action-20261127/recover_formal_parser.py collect --seq-len "$TASK_STREAM_SEQ" --output "$TASK_STREAM_OUT/formal-$TASK_STREAM_SEQ"
done
python "$TASK_STREAM_TOOLS/summarize.py" "$TASK_STREAM_OUT/formal-3500" "$TASK_STREAM_OUT/formal-2250" "$TASK_STREAM_OUT/formal-5000"
```

Each formal invocation keeps one worker, one fixture and the same 37 resident compute kernels while switching modes. Per policy: 12 periods × (1 warm-up + 5 ordinary samples), with rotated/reversed order. Warm-ups, first runs and all ordinary samples are retained. No detailed event trace or per-run AICore profiling is enabled for these invocations.

`SIMPLER_TMR_QKV_BATCH`: 0 Shared; 7 B infrastructure/default ordering; 8 C guarded default; 9 D guarded default. **Modes 8/9 have no active analytical override in this build.** Every locality class remains neutral. They do not evaluate analytical or learned benefit. Historical modes 1–6 retain their prior meanings. `SIMPLER_TMR_QKV_TRACE=3` is the new sparse, first-tile Qwen calibration diagnostic; 0 is required for ordinary timing.

Qwen golden poisons `out` and all 40 layers' current-token K/V output rows before each policy, then compares the full output and the entire 40-layer K/V buffers against the original PyTorch fixture. Historical context rows remain unchanged. The later validation harness first checks exact equality per chunk, then applies the original allclose predicate to non-identical chunks; every element is still checked, including NaNs. This avoids redundant float comparisons over unchanged context without weakening the predicate. The target 3500 validation used the original predicate directly. Tolerances are the existing repository values (rtol 0.05, atol 0.1).

`expected_counts.py FULL_TRACE_FOLDER OUTPUT_JSON` independently derives the expected task/block map (10,604 compute task IDs and 19,208 invocations for the target). `visibility_budget.py FULL_TRACE_FOLDER` reproduces the post-hoc fanin-budget sensitivity. It uses only the successor indices captured at the observation and registration epochs no newer than that observation. The full trace is an explanation artifact, never an online policy input.

Simulation: the runtime simulation binaries were built, but the focused scene cannot compile because `g++-15` is missing. The failed attempt is retained. Native C++ tests, CPU prefix simulation, and actual Ascend results are reported separately. Do not treat the simulation failure as an Ascend access failure.

Build cache evidence: DFX=1 and scheduler profiling=0 for onboard host/AICPU/AICore. `build-config/` includes caches for other built architectures as provenance; only a2a3/onboard is used in this result. `formal-freeze.json` is the authoritative final runtime/harness fingerprint. Earlier `phase1-frozen-source.json` and `harness-freeze.json` are phase snapshots, not the formal build declaration.

Formal host C++ logging may be buffered past a Python sample boundary. The wrapper preserves the unchanged frozen collector, then associates each native `device_wall` by invocation ID, verifying all 288 complete `chip.run` invocation IDs, all 288 labels/end markers, actual mode sequence, unique timing IDs, eligible claims, runtime fingerprints and 37 loaded kernels. A positional-parser failure with worker exit 0 does not invalidate complete uniquely identified samples. Original logs and the first parser failure remain unchanged. See `parser-recovery.json` in each formal folder. No run was dropped or rerun to replace a timing sample.
