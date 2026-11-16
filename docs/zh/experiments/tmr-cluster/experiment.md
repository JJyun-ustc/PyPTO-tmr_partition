# v7：降低 O 控制成本的分阶段实验

本页为最新 v7 结果。此前 v4–v6 的探索和原始结论保留在 [完整实验历史](https://github.com/JJyun-ustc/PyPTO-tmr_partition/blob/runtime-snapshot/docs/investigations/2026-11-aicpu-cluster-bgemm.md)。

本轮以当前 v6 源码为新鲜基线，围绕 O 的逐节点控制成本分阶段实现四项改动。
完整状态机、压力口径及阈值见
[当前机制说明](algorithm.md)。

## 实现与实验分阶段

| 阶段目录 | 累积引入的行为 |
| --- | --- |
| `baseline` | 当前 v6：逐节点完整评分，8 节点快照，最终侧在生成时固定，无跨侧领取 |
| `groups` | 依赖前驱识别小组，最多 8 节点、32 次放置的继承范围、64 槽 O 本地表 |
| `deferred` | 生成只记偏好，ready 时按最后释放前驱实际侧及队列拥塞轻量最终绑定 |
| `summary` | 64 次放置才刷新 O 快照；S 每 64 循环发布压缩资源字；单写者分片账本；普通轮关闭逐节点决策计时 |
| `steal` | 独立单 block 自动队列，每 32 循环可检查，远端 6/8 条阈值，一次一个，尝试后 128 循环冷却 |
| `final` | 相同四项组合，补上来源时间戳缺失的回归修复、格式整理及本地记账接口清理后重新测量 |

这些是累积阶段，不是四因素独立消融。`summary` 同时改变反馈布局、刷新频率、
本地预留归并和普通轮计时开销，不能把它的收益单独归因于某一个字段压缩。
固定 256 次放置校准和原评分权重保留；新反馈时序会改变采样及拟合结果。

任务 payload 的自动元数据保持 40 字节，tensor/scalar 偏移不变；组号与 scope 在其中
压缩表示。新增自动队列使用 DeviceArena 布局，host、AICPU 与 AICore 三端从当前
worktree 一起重建。严格/不可领取队列与可领取自动队列分开，不从混合队列弹出后再筛选。

`preferred`、`queued`、`actual` 三种侧别分别记录。生成预留由 O 累加，首次 ready
同时撤销偏好预留并按最终队列入账；派发永远扣原队列，即便执行 S 位于另一侧。
重入队不重复入账，完成/CONSUMED 不重复扣账。每个 S 以及 O 各有自己的摘要行，
不存在多个 S 高频修改同一个全侧计数器。摘要读中变化时沿用旧行，不阻塞调度。

## 设备与测量口径

本环境此次枚举为 **physical/logical device 0**，910B2，PCIe `0000:C1:00.0`。
这是本轮重新采集的结果，不与此前 physical card 1/4 的历史样本合并；枚举变化本身
也不能证明物理硬件是否更换。预检前后设备健康均为 OK，预检未见运行中的 NPU 进程。
`task-submit` 仍不存在，因此计时未持有设备独占锁。

仓库架构脚本预期 `Chip Name=Ascend910` 与独立 `NPU Name=B2`，当前驱动直接输出
`Chip Name=910B2` 而没有 `NPU Name`，脚本不能解析。已保留脚本失败输出，并通过
成功的 `npu-smi info -t board -i 0 -c 0` 与 CANN
`Ascend910B2.ini` 的 `Short_SoC_version=Ascend910B` 核实。
所有阶段保持六线程，允许 CPU 为 2–7，O 固定在 CPU 7，未做 O 位置对照。

每阶段先单独跑四个黄金校验，再对四个配对用例各跑两批反序的 15 轮。
计时关闭 swimlane、使用 `device_wall`，保留全部 30 个样本，包括各进程首轮。
每阶段 Shared/Automatic 共用一套二进制，并核验前后 SHA256 相同。
所有阶段的 AIC/AIV 计算代码 FNV 哈希均分别为
`0x212c407e5c39dda2` / `0x4c0e49fc6b5f2f9f`。
另外每阶段独立执行带 swimlane level 4 的黄金诊断；诊断轮不进入计时统计。

初次基线采集在后续 orchestration 编译时遇到正在修改的头文件，编译失败；该批保存在
`baseline/invalid-build-overlap`，完全排除。随后恢复基线源码、保持每个阶段源码不变直到
整个测试完成，重新取得完整基线。各阶段源码、日志、capture 副本均独立保存。

## 各阶段结果

下表为全部 30 轮的中位数，单位 ms；括号为 Automatic 相对同阶段 Shared 的变化。

| 阶段 | Case0 Shared | Case0 Automatic | Scale Shared | Scale Automatic |
| --- | ---: | ---: | ---: | ---: |
| v6 基线 | 1.0443 | 1.1994（+14.85%） | 4.9412 | 5.7833（+17.04%） |
| 仅加组复用 | 1.0460 | 1.1792（+12.74%） | 4.8845 | 5.4011（+10.58%） |
| 再加延迟绑定 | 1.0528 | 1.2247（+16.33%） | 4.9221 | 5.4856（+11.45%） |
| 再加低频摘要 | 1.0169 | 0.9836（−3.28%） | 4.8099 | 3.9837（−17.18%） |
| 再加受限领取 | 1.0518 | 1.0161（−3.39%） | 4.9652 | 4.0872（−17.68%） |
| 最终代码复核 | 1.0504 | 1.0125（−3.60%） | 4.8201 | 3.9146（−18.79%） |

最终 p95：Case0 Shared/Automatic 为 **1.1602 / 1.1340 ms**，
Scale 为 **4.8575 / 3.9705 ms**。
相对本轮 v6 的 Automatic 中位数，最终降低 **15.58% / 32.31%**。

观察与限制：

- 组复用对 Scale 有改善；单独叠加延迟绑定没有显示收益。
- 明显变化出现在第三阶段的组合，O 与 S 的共享反馈访问显著下降。
- 领取前后相对 Shared 的优势相近，不能证明领取本身带来额外加速。
  `steal` 到 `final` 的小幅差别也不能单独归因于来源修复：构建顺序、设备漂移和拟合样本都变化。
- Case0 的优势较小；顺序构建且设备无独占锁，不能据此宣称所有短图都有稳定收益。
- 本图生成仍由 O 串行执行；没有缓冲等待小组凑齐，也没有把完整评分搬到 S 或并行评分。

## O 与 S 的控制成本

以下来自独立的、包含计时器开销的单次诊断，单位 ms。不同线程重叠；O 细分阶段属于
submit 的一部分；S observe、ready-binding 可嵌在其他阶段中，**不能相加还原 wall time**。

| 指标 | Case0 v6 | Case0 最终 | Scale v6 | Scale 最终 |
| --- | ---: | ---: | ---: | ---: |
| O 全部 `orch_submit` 区间之和 | 1.5920 | 1.3700 | 6.9959 | 5.2279 |
| O 放置与 wire/ready 细分阶段之和 | 0.4812 | 0.3021 | 2.3435 | 1.1031 |
| 全部 S 四类派发遍历区间之和 | 3.4501 | 3.1300 | 17.0965 | 13.1123 |
| 全部 S 资源反馈区间之和 | 0.1229 | 0.0105 | 0.5878 | 0.0431 |
| 全部 S 新增 ready-binding 区间之和 | 未单列 | 0.1234 | 未单列 | 0.4025 |
| 全部 S 领取检查区间之和 | 无 | 0.0167 | 无 | 0.0822 |

四类派发包括 sync、strict、auto、shared；循环总次数也会随整体执行时间变化，不能将
累计区间差异解释成单次派发的纯成本差。最大单线程派发遍历累计为 Case0
0.9149→0.7936 ms、Scale 4.5399→3.2313 ms。

最终 S 反馈最大单线程区间之和为 Case0 **0.0036 ms**、Scale **0.0149 ms**；
ready-binding 对应最大单线程为 **0.0354 / 0.1678 ms**。
ready-binding 数值包含 ready stamping、账本、采样条件和计时器，不能解释成纯选择规则耗时。
领取检查在该两次诊断中没有触发实际借用，表中仍记录了检查成本。

## 状态、侧别和领取证据

最终诊断的入口完整评分/继承次数为 Case0 **500/500**，Scale **2048/2048**。
快照从 v6 的 **250/1024** 降到 **16/64**；实际只对组入口使用的 shape 刷新。
仍为 256 次冷放置，且校准恰好一次。

最终诊断的偏好侧分配/实际执行侧任务数分别为：

| 用例 | 偏好侧 0/1 | 实际侧 0/1 |
| --- | --- | --- |
| Case0 | 128 / 872 | 127 / 873 |
| Scale | 73 / 4023 | 105 / 3991 |

偏好不再等于执行位置。当前结果也显著偏向 O 所在的 cluster 1，不能将性能改善解释成
两侧按 2:3 完美均衡；低频摘要与阈值提供的是调整能力，并非强制均衡保证。

最终普通 Case0 黄金轮中，五个 S 各成功跨侧领取一个任务，共 **5 个 block**；
该轮正确性通过。独立的高开销诊断轮则没有触发领取，因此没有拿零领取的诊断轮
推断普通计时轮的领取次数。对带版本 4 账本的黄金轮/诊断轮检查：
`preferred == released`、`queued == taken`、各 shape 总 preferred/queued/actual 相等。
允许每侧 queued 与 actual 不等；不能沿用 v6 的“每侧 assigned == started”检查。

## 回归与可复现材料

8 个相关 C++ 目标、共 **190 个测试**通过。新增测试覆盖组大小/年龄/scope/槽复用、
版本竞争沿用旧摘要、偏好侧与队列侧及执行侧分离、队列侧扣账、领取阈值、严格/多 block
不可领取以及校准结束后仍保留实际来源。另修复了老 TaskState 合成 fixture 缺失 task
描述导致 ready 路由空指针的问题，显式设置 Shared 哨兵 -1。

来源时间戳缺失用例先复现失败，再修复：采样之前启动的共享生产者可能给出 0 时间戳，
这不代表 O 发起 ready。`origin-before-fix.log` 保留失败证据；修复后全部目标通过。

板上额外 **15 个黄金回归用例**通过，覆盖自动 AIV 多 block 4/16/24/48/96、
MIX 多 block、自动 sync-start 共享回退、严格跨侧向量依赖、早派发生产者，以及自动/严格
DUMMY、长链与 dense fanout/fanin。源码已放到
`tests/st/a2a3/tensormap_and_ringbuffer/cluster_partition_study/`。
仿真运行库构建通过；仿真场景 kernel 编译因环境缺少 `g++-15` 失败，未声称仿真回归通过。
没有运行整个 runtime 的全量系统测试。

材料位于 `../official-results/v7-control-20261116/`：

- `summary.json`：六个阶段的 30 轮样本、中位数/p95、成本及状态账本。
- 每阶段 `source/`、`run/`、`diagnostics/`、`captures/`：源码、原始 host/device 日志、独立图与时间线。
- `implementation.patch`：相对本轮 v6 基线的改动；`verified-unit.log` 与 `regressions/`：验证记录。
- `plan.json`：分阶段阈值；架构失败输出、板卡查询与 SHA256 一起保留。

在 runtime 自身的项目虚拟环境中，按本机测试资源上限构建并测量当前版本：

```bash
source .venv/bin/activate
source ../.claude/skills/testing/load-env.sh
python examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm/run_control_stage.py \
  ../official-results/<new-study-directory> --jobs "$PYPTO_BUILD_JOBS"
```

输出目录必须全新，避免覆盖证据；当前驱动示例默认查询 physical 0。
仅重新测已有二进制可运行 `reproduce_cluster_study.py --physical-card <id> --output <new-dir>`。
比较本次所有阶段用 `analyze_control_study.py ../official-results/v7-control-20261116`。
