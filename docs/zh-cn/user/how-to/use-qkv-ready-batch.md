# 固定 QKV 自然 ready 批量调度机制与复现

[English](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/docs/en/experiments/tmr-cluster/algorithm-qkv-ready-batch.md) · [实验报告](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/docs/zh/experiments/tmr-cluster/qkv-ready-batch.md)

## 实现边界与开关

以可复现 Shared `d8f997fab` 为起点，复用 `7ac2fb88c` 的分侧队列、任务侧别 ABI、范围领取和六线程拓扑。
Qwen orchestration 使用普通 Shared 提交，不设置 Auto；旧在线放置评分不参与本轮。
`SIMPLER_TMR_QKV_BATCH=0`（默认）选择 Shared，`=1` 选择 QKV，使用同一运行时二进制和计算 kernel。
固定五个 S、一个 O：侧 0 为 CPU2/3 上的 S0/S1，侧 1 为 CPU4/5/6 上的 S2/S3/S4，O 在 CPU7。

只接管普通 ready 路径的纯 AIC/AIV 整任务；谓词、严格绑侧、Auto scope、sync-start、DUMMY、MIX、
已经进入提前派发或已有 block 领取的任务保持原路径。任务 block 数超过任一侧对应静态容量即走 Shared。
第一版为固定公式的 CPU 实现，不训练，不卸载到 NPU，不跨侧领取，不按策略拆分 block。
同侧多个 S 仍可用原来的范围 CAS 分担一个任务的 block。

## 自然批次、唯一发布与推进

依赖就绪与提前派发检查完成后，`READY_CLAIMED` 的唯一获胜者才可进入 64 项有界 MPMC 待分配队列。
提前派发 helper 成功执行 NONE→DISPATCHED 的 CAS，才证明没有提前派发 stager 占有任务；
单独在调用前读取状态存在竞态。成功入队即把唯一最终发布责任交给 S0；队列满则立即发布 Shared。
重复 ready 通知被生命周期 CAS 拒绝，后续范围重入队不重新进入分配器。

S0 每次非阻塞取当前已经发布的连续前缀，最多 8 项；只有一项立即处理，不等凑批。
一轮决策后回原调度循环。服务点位于低负载检查和 sync-drain 的 `continue` 之前。
分配器只发布队列，不领取硬件、不加入或绕过同步 rendezvous。待分配任务保留原生命周期引用，不能提前回收。

同步 drain 的 barrier 只等待参与线程确认和 staging；容量不足会推进 attempt 并返回调度轮询，容量足够则
完成 staging 并重新开门。普通 ready 同步任务已经完成所有前驱；early sync 候选要求前驱 block 已全部发布，
因此未分配的 pending 前驱不能成为该 early drain 的先决条件。原队列容量与生命周期约束继续适用。
任务最终只发布到一个队列；发布到侧队列后不迁移，实时资源检查和 `claim_block_range()` 继续由原派发路径负责。

## 唯一固定 QKV 公式

每侧每纯资源类别保留三个永久 K/V 条目：排队积压、running/pending 占用、空闲容量压力。
共 12 个固定条目；每个候选行的掩码仅保留对应侧与资源类别的三个条目，无需扫描队列或在途任务。

```text
Q = [side_sign, resource_sign, blocks/capacity, age, 1, 1], d=6
K_j = [side_sign, resource_sign, occupancy_j, type_j/2,
       is_queue_j, is_running_j]
V_j = [occupancy_j, is_running_j * occupancy_j]
S_j = dot(Q,K_j)/sqrt(6) + (is_running_j ? 0.25 : 0)
Z = stable masked_softmax(S) * V
attention_pressure = 0.5 * Z[0] + 0.5 * Z[1]
score = (queued_blocks + batch_reserved + blocks)/capacity + attention_pressure
```

类型编号 queue=0、running=1、idle=2。age 为进入 pending 后等待时间除以 1 ms，并截断至 1。
三种 occupancy 分别为 `queued_blocks/capacity`、`occupied_slots/(2*capacity)`、`1-idle_cores/capacity`。
总积压另保留在基础项，避免 softmax 平均抹掉规模；注意力项与基础项均无量纲。
V 中的待释放压力近似执行槽占用比例，不是可靠的剩余执行时长。

同批读取同一个近似快照，按 ready 顺序逐个决策；每次选定后立即增加该侧、该资源类别的
`batch_reserved`，下一任务感知这次分配。K/V 与快照在批内不变。
精确同分时先选批内预留更少的侧，再选侧 0。不兼容侧在评分前排除。
摘要不可用、无合法侧、超过固定 50 us 预算均回退 Shared；评分前后均检查预算，超时结果不能获得发布权，
必需的回退发布本身不属于评分预算。

## 反馈账本与资源摘要

S0 单写累计 assigned block；各 ready 发布者与各 S 的成功范围领取维护各自单写、单调的原子账本。
不为每次 FIN 增加全局 RMW。已领取量领先被采样发布量时，对估计队列差值截断为零；真实派发仍保持精确。
共享普通纯资源积压按静态容量比例分摊到两侧，侧队列使用实际 block 账本。

每个 S 每 64 次循环从自己的 CoreTracker 发布每资源一个打包原子字：capacity、idle、pending_free。

```text
occupied_slots = 2*capacity - 2*idle - pending_free
```

IDLE core 有两个空执行槽，不出现在 PENDING 可领取掩码中。
MIX、同步、严格任务和提前派发使用的硬件资源均影响这些掩码。
O 从原有 round-robin 分配契约计算静态容量，不能读取尚未完成握手初始化的 CoreTracker。
动态摘要初始不可用；每个原子字只读一次，是轻量异步估计，不是全局同步的硬件瞬时快照。

## 确定性证据与统计口径

`QkvFormula.KvStateOverridesLowerBaseLoadAndBatchChangesNextChoice` 设置两侧容量均为 8。
空闲且无积压时两个单 block 任务依次选 0、1，验证批内预留影响后续任务。
侧 1 积压为 1 时，基础负载项更偏好侧 0；把侧 0 K/V 改为满占用且无空闲后，QKV 改选侧 1，
恢复其状态后又选侧 0，证明注意力项确实能逆转基础项的选择。

`QKV_SUMMARY` 保留候选任务数、两侧实际分配数、assigned/claimed block、pending 剩余及决策 tick。
候选数在静态宽度检查前累计，减去 oversize 回退才是符合宽度限制的 eligible 数。
`QKV_BATCHES` 保留大小 1..8 的自然分布，以及 full、snapshot、no-side、budget、oversize 五类回退。
决策时间包含摘要读取、QKV 计算和最终队列发布，不含 pending pop 与报告开销。
正常计时保留总量、最大值和 log2(ns) 直方图；只有 swimlane=4 的独立诊断输出逐批明细。
最多保留 4096 个逐批精确样本，总批数与直方图不截断。

## 从发布仓库构建、验证和计时

以下从主仓库的固定 runtime 子模块开始；所有编译都在新 checkout 内完成。
先安装 CANN 工具链并加载其环境，创建本地 venv，安装 CPU torch、项目构建依赖。
若主仓库机器资源配置可用，先读取 `.claude/skills/testing/load-env.sh`；否则显式使用下面的两工作线程限制。

```bash
git clone https://github.com/JJyun-ustc/PyPTO-tmr_partition.git
cd PyPTO-tmr_partition
git submodule update --init runtime
cd runtime
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install nanobind==2.15.0 scikit-build-core==1.1.2 cmake==4.4.2 ninja==1.13.2
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install cloudpickle==3.1.3 PyYAML==6.0.3 pytest==9.2.0
export PYPTO_BUILD_JOBS=2 PYPTO_TEST_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL="$PYPTO_BUILD_JOBS" MAKEFLAGS="-j$PYPTO_BUILD_JOBS"
pip install --no-build-isolation -e . --config-settings=build.targets=_task_interface
PYTHONPATH=python:. python -c \
  'import multiprocessing,os,runpy; multiprocessing.cpu_count=lambda:int(os.environ["PYPTO_BUILD_JOBS"]); runpy.run_module("simpler_setup.build_runtimes",run_name="__main__")' \
  --platforms a2a3 --profiling-dfx 1
cmake -S tests/ut/cpp -B tests/ut/cpp/build
cmake --build tests/ut/cpp/build --parallel "$PYPTO_BUILD_JOBS" \
  --target test_a2a3_tmr_scheduler_state test_a2a3_tmr_cluster_partitioner \
  test_a2a3_tmr_task_state test_a2a3_tmr_wiring
ctest --test-dir tests/ut/cpp/build --output-on-failure \
  -R '^test_a2a3_tmr_(scheduler_state|cluster_partitioner|task_state|wiring)$'
python examples/a2a3/tensormap_and_ringbuffer/benchmark_qkv/run_study.py \
  --output results/qkv-rerun --card 4 --device 0 --phase validate
python examples/a2a3/tensormap_and_ringbuffer/benchmark_qkv/run_regressions.py \
  --output results/qkv-regressions-rerun --card 4 --device 0
python examples/a2a3/tensormap_and_ringbuffer/benchmark_qkv/run_study.py \
  --output results/qkv-rerun --card 4 --device 0 --phase measure
python examples/a2a3/tensormap_and_ringbuffer/benchmark_qkv/analyze_results.py results/qkv-rerun
```

本轮容器物理卡 4 映射 ACL 逻辑设备 0。脚本查询指定物理卡，并用 CANN SoC ini 验证 Ascend910B；
原 shell 预检查假定卡 0 和旧版 board 格式，不适用于此映射。其他环境须替换两个编号。
有 `task-submit` 时用其持有整个实验的设备锁；本轮该工具缺失，未持锁，查询记录保留。

固定参数为 batch 16、40 层、seq_len 3500、seed 1234，单批上限 8、pending 容量 64、预算 50 us。
验证阶段分别跑两模式完整输出/KV golden，再独立采集诊断；已有目录与已冻结源码/二进制变更会被拒绝。
正式阶段六个时段交替 Shared→QKV、QKV→Shared；每个新进程一轮固定 warm-up 后五轮正式样本。
保留全部 72 轮及首轮，统计只用 60 个正式样本。37 个实际加载 kernel 指纹必须逐次一致。
p95 按 `(n-1)*0.95` 线性插值。正式使用 `info` 冷路径汇总、swimlane=0；`timing` 级别会屏蔽汇总。
所有失败采集尝试整组保留，诊断不混入正式时长，不根据结果调参。

主仓库中的[全部证据](https://github.com/JJyun-ustc/PyPTO-tmr_partition/tree/main/official-results/qkv-ready-batch-20261125)经过明确记录的路径脱敏与 gzip 压缩，
用 `gzip -dc <file>.gz` 阅读；发布清单给出原始及发布 SHA-256，压缩前后所有数值样本一致。
