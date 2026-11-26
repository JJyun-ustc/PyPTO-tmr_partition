# QKV round two — reproduction / 第二轮复现

Runtime starting commit: `e245d805eddb5db982c99e86a52c56a081d502b4`.
Branch: `experiment/qkv-ready-batch`; published runtime branch: `runtime-qkv-ready-batch`.
Physical card 4 maps to ACL device 0, Ascend910B2. The device queue tool is absent;
`hardware.json` records the lack of an exclusive lock. Recheck isolation before running.

Batch 16, 40 layers, seq_len 3500, max_seq 5500, seed 1234, six AICPU threads:
five schedulers and one orchestrator. The observed per-side capacities are
AIC/MIX **10/14 groups**, AIV **20/28 cores**. All policies load the same 37
compute-kernel fingerprints, including the first-round artifacts.
`host-topology.json`, `build-config/`, and phase `frozen.json` record the configuration.

正式计时每进程六轮，首轮保留为 warm-up，后五轮为正式样本；六个时段交错运行，
每策略 30 个样本。详细诊断与正式样本分离。完整任务/依赖日志的冷路径输出会显著
增大诊断 device_wall，不能把它当性能样本。golden 使用仓库容差：rtol=0.05、atol=0.1，
完整比较输出和全部 40 层 K/V cache。

Run from the runtime repository root with the repository's CANN environment loaded.
For a frozen reproduction, copy the published `models/` into the new result root;
keep the checked-in generated weights unchanged. The collection/training workflow below
creates a new independently measured experiment and may produce different weights.

```bash
set -e
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
source .claude/skills/testing/load-env.sh
python -m pip install --no-build-isolation --no-deps -e . --config-settings=build.targets=_task_interface
PYTHONPATH=python:. python -c 'import multiprocessing,os,runpy; multiprocessing.cpu_count=lambda:int(os.environ["PYPTO_BUILD_JOBS"]); runpy.run_module("simpler_setup.build_runtimes",run_name="__main__")' --platforms a2a3 --profiling-dfx 1
cmake -B tests/ut/cpp/build -S tests/ut/cpp
cmake --build tests/ut/cpp/build --target test_a2a3_tmr_scheduler_state test_a2a3_tmr_cluster_partitioner test_a2a3_tmr_task_state test_a2a3_tmr_wiring --parallel "$PYPTO_BUILD_JOBS"
ctest --test-dir tests/ut/cpp/build -R '^test_a2a3_tmr_(scheduler_state|cluster_partitioner|task_state|wiring)$' --output-on-failure
python -m unittest discover -s examples/a2a3/tensormap_and_ringbuffer/benchmark_qkv -p test_round2_offline.py

TASK_QKV_ROOT=results/qkv-round2-reproduction
TASK_QKV_TOOLS=examples/a2a3/tensormap_and_ringbuffer/benchmark_qkv
python "$TASK_QKV_TOOLS/round2_study.py" --output "$TASK_QKV_ROOT" --phase probe
python "$TASK_QKV_TOOLS/round2_study.py" --output "$TASK_QKV_ROOT" --phase calibration
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python "$TASK_QKV_TOOLS/round2_train.py" --data "$TASK_QKV_ROOT/calibration" --output "$TASK_QKV_ROOT/models"
python "$TASK_QKV_TOOLS/round2_shadow.py" "$TASK_QKV_ROOT/models"
```

For newly trained weights, export them **before** the final build and board gates.
The optional fixture update computes the test expectation independently in PyTorch.
Do not regenerate weights or modify runtime sources during formal timing.

```bash
python "$TASK_QKV_TOOLS/round2_export.py" "$TASK_QKV_ROOT/models" src/a2a3/runtime/tensormap_and_ringbuffer/runtime/qkv_model_weights.h --update-test-fixture
clang-format -i src/a2a3/runtime/tensormap_and_ringbuffer/runtime/qkv_model_weights.h tests/ut/cpp/a2a3/runtime/tensormap_and_ringbuffer/test_scheduler_state.cpp
PYTHONPATH=python:. python -c 'import multiprocessing,os,runpy; multiprocessing.cpu_count=lambda:int(os.environ["PYPTO_BUILD_JOBS"]); runpy.run_module("simpler_setup.build_runtimes",run_name="__main__")' --platforms a2a3 --profiling-dfx 1
cmake --build tests/ut/cpp/build --target test_a2a3_tmr_scheduler_state --parallel "$PYPTO_BUILD_JOBS"
ctest --test-dir tests/ut/cpp/build -R '^test_a2a3_tmr_scheduler_state$' --output-on-failure
python "$TASK_QKV_TOOLS/round2_study.py" --output "$TASK_QKV_ROOT" --phase shadow
python "$TASK_QKV_TOOLS/round2_gate.py" "$TASK_QKV_ROOT"
python "$TASK_QKV_TOOLS/round2_study.py" --output "$TASK_QKV_ROOT" --phase replay-complete
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python "$TASK_QKV_TOOLS/round2_replay.py" --data "$TASK_QKV_ROOT/replay-complete" --models "$TASK_QKV_ROOT/models"
python "$TASK_QKV_TOOLS/run_regressions.py" --output "$TASK_QKV_ROOT/regressions-final" --mode 3 --trace 2
python "$TASK_QKV_TOOLS/round2_study.py" --output "$TASK_QKV_ROOT" --phase golden
python "$TASK_QKV_TOOLS/round2_study.py" --output "$TASK_QKV_ROOT" --phase measure
python "$TASK_QKV_TOOLS/round2_analyze.py" "$TASK_QKV_ROOT"
```

Mode 4 is calibrated only for this Qwen workload at seq_len=3500. Unknown kernel/
launch specifications fall back to the fixed formula. A failed quality or AICPU
budget gate excludes mode 4 from the formal comparison; no other training trial
or parameter sweep is added. Full replay is an offline diagnostic, never real-action
supervision. Its global service scale is fitted on training runs only.

`SIMPLER_TMR_QKV_BATCH`: 0 Shared, 1 original QKV, 2 attention-off control,
3 improved QKV, 4 frozen learned QKV, 6 centralized-Shared-publication control.
`SIMPLER_TMR_QKV_TRACE`: 0 off (formal), 1 sampled, 2 full DAG.
Fixed cap 8, pending queue 64, decision budget 50 µs, maximum summary age 200 µs.

真实训练标签只使用实际执行侧；512/1750 的完整运行作为训练分区，2625 验证，3500
留出，不随机拆分重复层。第一批追踪漏掉快速路径，因此只将其中可靠的实际动作行用于
两模型共同的监督数据，旧回放归入 `models/legacy-partial-replay/`，不得作为最终排序证据。
`replay-complete/` 是重新采集的完整 DAG：每次 11,166 个任务、23,605 条依赖，提交数与
记录数相等，所有端点存在；两次训练长度运行校准回放，2625 验证，三个 3500 运行留出。
后继 ready 和资源释放随分配重新计算，sync cohort 使用不同物理组，O 内联 allocation
不占用 S。初版实际监督数据仍保留其采样覆盖限制，未将缺失记录补造为真实标签。

Start markers are explicit proxies: register publication, observed pending promotion,
or doorbell permission. FIN-only fallback is censored; inline O allocation has its
own marker. AICore actual start is unavailable. Creation times are fixed replay
anchors; subsequent dependency readiness and physical resource calendars are recomputed.
Recorded later side choices are held fixed for each single-action counterfactual.
Fast direct-ready routes remain on the original Shared protocol; their ready timestamps
are not used as measured normal-route wait labels.

`failed-layout/`, `smoke/`, `golden-pretrained/`, `golden-before-dag-fix/`,
`diagnostics-before-dag-fix/`, and archived replay attempts retain unsuccessful or
superseded diagnostics. Successful golden results from earlier builds are not formal
timing samples. Do not reuse artifacts left by a failed incremental runtime build.
