# 固定 QKV 自然 ready 批量调度：Shared 对照报告

[English](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/docs/en/experiments/tmr-cluster/qkv-ready-batch.md) · [机制与复现](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/docs/zh/experiments/tmr-cluster/algorithm-qkv-ready-batch.md)

日期：2026-11-25（UTC）。本轮完成实现、构建、CPU 验证、板上 golden、独立诊断和正式计时；结果为性能回退，停在本轮代码与数据交付。

| Qwen device_wall | Shared | QKV 原型 | 相对变化 |
| --- | ---: | ---: | ---: |
| 正式样本数 | 30 | 30 | — |
| 中位数 | 36.697130 ms | 39.307130 ms | **+7.11%** |
| p95 | 36.829344 ms | 39.462639 ms | **+7.15%** |

六个时段均为 QKV 较慢，各时段中位数相对回退 7.06%、7.27%、7.56%、7.23%、6.70%、6.70%。这是本轮固定策略的观察结果，不据此调整参数或补做消融。

**一致性与正确性。** 实验 worktree 分支 `experiment/qkv-ready-batch`（发布分支 `runtime-qkv-ready-batch`） 从 Shared `d8f997fab` 起步，复用 `7ac2fb88c` 的分侧队列、范围领取与六线程拓扑。两种模式使用同一套本地构建的 runtime 二进制，以 `SIMPLER_TMR_QKV_BATCH=0/1` 切换。Ascend 910B2，物理卡 4 / ACL 逻辑设备 0；每轮均记录相同 CPU 池 `[2,3,4,5,6,7]`，S0/S1 位于侧 0，S2/S3/S4 位于侧 1，O 位于 CPU 7。主机 torch、OMP、MKL、OpenBLAS 均为 2 线程。

负载固定为仓库 Qwen3-14B decode fixture：batch 16、40 层、seq_len 3500、MAX_SEQ 5500、seed 1234；沿用层 0 权重重复到各层的参考设置，输出为 hidden→hidden。两种模式的输出及全部 40 层 K/V cache golden 均通过，容差沿用仓库 `rtol=0.05, atol=0.1`。所有正式进程实际加载的 **37 个 kernel 代码指纹全部一致**，orchestration hash 同为 `0xd0f1f953f424f7e4`。540 个 runtime 源码/二进制指纹在冻结前后相同。构建采用 C++17、`-O3 -g`，DFX=1，其余 profiling=0；完整 flags 与工具链记录见原始数据。

**覆盖率、自然批次和回退（仅 30 个 QKV 正式轮次）。**

- 符合条件的任务 **298,830**，原型实际分配 **288,522（96.55%）**；侧 0/1 分别为 111,343 / 177,179 个。
- 自然批次 **44,793**；多任务批 **40,325**，其中满 8 任务批 **31,414**。批大小 1..8 的完整分布为 `[4468,3520,1210,1303,908,1314,656,31414]`。
- Shared 回退 **10,308**：待分配队列满 10,152，摘要不可用 156；无合法侧、预算、超容量均为 0。超容量路径另在多 block 板上回归中实际走到；无合法侧在 CPU 用例覆盖。golden 中另观察到 1 次预算回退，原始日志保留。
- 每轮分配/领取 block 账本相等、pending 最终为空；实际领取路径断言执行侧等于任务队列侧。43 项 CPU 调度器测试通过，其中 8 项为 QKV 用例，另有任务状态、wiring 和复用基础设施测试通过。板上同步共存、普通多 block、DUMMY、提前派发 4 组 golden 通过。同步共存中的 6 个单任务批全部分侧发布，证明低负载与同步任务共存时能够推进。

**决策成本。** 正式轮次共记录 44,793 批，平均 **5.283 µs/批**，最大 **15.40 µs**；每轮累计决策时间平均 **7.889 ms**。计时范围包含摘要读取、QKV 计算与最终队列发布，不含 pending pop。独立诊断保留 1,487 个逐批样本：中位数 **6.060 µs**、p95 **7.634 µs**，这些诊断样本不进入正式 device_wall。running/pending 的 V 是执行槽占用近似量，不是剩余执行时长预测。

**采样与保留。** 六个时段按 Shared→QKV、QKV→Shared 交替先后；每个新进程 1 次固定 warm-up 加 5 个正式样本。保留全部 72 个普通轮次（60 正式 + 12 warm-up），首个正式样本未删除；p95 按 `(n-1)*0.95` 线性插值。正式采集使用 `info` 级别冷路径汇总，swimlane=0，不输出逐批详细诊断。第一次尝试的 `timing` 日志级别屏蔽了汇总，导致采集校验失败；该次两模式共 12 个时长全部保留在 `attempts/measurement-timing-loglevel/`，整组排除后用统一日志配置重新采集。逻辑设备编号错误及修正初始化竞态前的 Shared 验证记录也保留。本容器没有 `task-submit`，本轮未持设备锁；设备查询与前后占用记录随数据交付。

**下一轮最值得检验的一项：S0 集中决策的服务成本与 pending 排队延迟。** 本轮覆盖率高、多任务批充足，且每轮约 7.9 ms 决策工作伴随约 3.4% 的队列满回退，优先检查它如何占用 S0 和影响关键路径。现有数据支持提出这个假设，尚未证明它是 7.11% 回退的原因；本轮不继续实验。

发布遵循既有约定：不上传二进制、venv、缓存和通用 Ascend 驱动日志；全部时长、失败采集尝试、golden/CPU/板上回归结果、QKV 诊断与指纹保留。日志、patch 及超过 500 KiB 的文件 gzip 压缩，用 `gzip -dc <file>.gz` 阅读。本机路径替换为命名占位符，发布清单同时保留原文件与发布文件的 SHA-256。原始本地完整压缩包未修改。测量后仅补标准版权注释和文档；[变更记录](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/official-results/qkv-ready-batch-20261125/post-measurement-changes.json) 区分发布版和原始测量 patch，未改变调度公式、参数或计算 kernel。累计决策时间是 CPU 工作量，不能直接与 device_wall 相加或相减。

交付入口：

- [完整机制、所有权与同步推进证明、构建和复现命令](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/docs/zh/experiments/tmr-cluster/algorithm-qkv-ready-batch.md)
- [全部实验数据](https://github.com/JJyun-ustc/PyPTO-tmr_partition/tree/main/official-results/qkv-ready-batch-20261125)
- [原始完整实现 patch](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/official-results/qkv-ready-batch-20261125/implementation.patch.gz)、[核心 patch](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/official-results/qkv-ready-batch-20261125/runtime-and-tests.patch.gz)
- [正式样本与统计](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/official-results/qkv-ready-batch-20261125/summary.json)、[反馈汇总](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/official-results/qkv-ready-batch-20261125/feedback.json)、[一致性审计](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/official-results/qkv-ready-batch-20261125/audit.json)
- [发布文件与校验清单](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/main/official-results/qkv-ready-batch-20261125/publication-manifest.json)
