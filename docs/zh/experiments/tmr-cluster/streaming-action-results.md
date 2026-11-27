# 流式 TMR：可见性、标定与基础设施成本实测

日期：2026-11-27。**本报告不是主动 A–D 调度收益对照：C/D 板上评分与资源日历未接入主动改派，仅运行空准入表下的默认路径。** 本轮得到 A/B 成本、真实前缀可见性、菱形正确性和标定结果；不能据此判断主动 C/D 或训练型 QKV 的收益。

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

目标配置：B16、40 层、seq_len=3500、seed=1234；2250/5000 为冻结后留出负载。每组 12 个时段，每时段 1 次预热和 5 次正式样本；共 720 次正式样本及 144 次预热全部保留。B 相对 A 的时段配对变化中位数为 +4.23%，12 时段 bootstrap 95% 区间 [+4.06%, +4.40%]。p95 使用线性分位数。诊断、golden 和 pilot 均不混入正式样本；不能把 C/D 占位组之间的波动解释为模型收益。

同一 Ascend910B2（物理卡 3 / ACL 0）、同一二进制、5 S + 1 O、AICPU cpu_id 2–7；S 为 2–6，O 为 7，计算资源为 24 AIC/48 AIV。DFX 编译为 1、调度 profiling 为 0，正式运行关闭详细事件。所有正式组与 Qwen 配对标定的 37 个实际加载计算 kernel 指纹相同。起点提交 `8ebbff77377175bbbdc8d8f7252897f3eb73e6f8`。未取得设备独占锁（工具缺失），不能用此轮直接重估旧卡 4 上初版慢 7.11% 的因果分解。

- **正确性**：三种序列长度 × 四组，共 12 次逐策略输出及全部 40 层 K/V cache golden 通过，每次先 NaN 污染输出和当前 token 的 K/V 行。55 项 native 测试、24 项 CPU 前缀测试、四个板上菱形及六组同步/MIX/multi-block/DUMMY/提前派发共存回归通过。每轮符合条件任务数与实际 block 领取数一致；这不是完整执行账本对所有资源组合的证明。
- **实际覆盖**：目标负载 B 每轮 9962 个符合条件任务走新增领取基础设施，60 轮共 597720 次成功 block 领取。替代候选比较、改派和亲和队列发布均为 0；不是“模型覆盖 100%”。所有任务维持默认领取，没有测到主动选侧或迁移收益。
- **最终构建的流式可见性**：Shared 诊断的 9962 次普通单 block 队列观察中，预算内可用汇合 52 次（0.522%）；完整登记但入度超预算 8920 次。只用当时已记录索引和 epoch，事后将入度预算从 4 放宽到 128，可用 8972 次（90.06%）。这是独立诊断与预算敏感性，分母不是实际三队列候选比较机会，也没有测排序改变率。
- **标定与时间口径**：同地址、逐条件 drain 的重用微基准保留 6048 个任务；实际 Qwen 33→34 完成 504 次 CPU 代理诊断及 144 次 AICore 补充诊断。后者 23,040 个选定任务均满足因果时间不等式，每次全量核对 10,604 个计算任务 ID、19,208 个 block 调用。CPU 发布代理相对 AICore 区间的差值中位数/p95 为 4.72/13.86 µs，pending 提升代理为 0.02/8.295 µs，两者分开。实际物理核匹配后两侧六时段差异均未稳定；进一步占用匹配仅留 43 对 tile，符号仍变化，不能当作可靠局部性先验。
- **开销与训练**：B 每 32 次采样的队头领取调用耗时中位数/p95 为 0.040/0.780 µs，n=2098，包含空调用和批领取，不是模型推理预算样本。B 同时增加元数据登记，端到端回退不能全部归因于队列。跨度表独立留出 MAE/p95 绝对误差为 1.790/5.640 µs；该标签不等于两动作净收益。尚无独立 a0/a1 净收益配对标定；本轮未训练 E/F、没有 AICPU QKV 影子预算样本，也未证明 QKV 相对线性有净收益。旧轮次权重的目标不同，不直接移植为动作残差。

下一轮优先检查**如何在控制元数据热路径开销的同时，为高入度可见汇合维护有界前沿摘要**。当前 4 入度预算人为屏蔽了大量已经可见的信息；基础设施已有成本，需先使候选与收益可测，再补独立动作标定和主动 C/D，随后才谈学习残差。

[机制](streaming-action-mechanism.md) · [English](../../../en/experiments/tmr-cluster/streaming-action-results.md) · [复现](streaming-action-reproduce.md) · [计时原始样本](../../../../official-results/streaming-action-20261127/final-formal-3500/samples.json) · [冻结配置](../../../../official-results/streaming-action-20261127/formal-freeze.json) · [诊断汇总](../../../../official-results/streaming-action-20261127/final-diagnose-3500/diagnostic-summary.json) · [标定](../../../../official-results/streaming-action-20261127/qwen-core-calibration/summary.json) · [审计](../../../../official-results/streaming-action-20261127/audit.json) · [实现 diff](../../../../official-results/streaming-action-20261127/runtime-and-tests.patch)。原始日志、失败与过渡版本保留；模拟 scene 因缺 g++-15 未运行，CPU/native 与实卡证据单独报告。本次出版保留未完成范围，不将默认路径占位组描述为主动策略。

完整证据在 [raw-evidence.tar.gz](../../../../official-results/streaming-action-20261127/raw-evidence.tar.gz)，包含日志、AICore records、失败与过渡版本。仅归一化机器路径并排除生成的 Python 缓存；原始与出版 SHA-256 对照见 [publication-file-hashes.json](../../../../official-results/streaming-action-20261127/publication-file-hashes.json)。原审计中“未推送”字段为采集时状态，出版信息见 [publication-manifest.json](../../../../official-results/streaming-action-20261127/publication-manifest.json)。
