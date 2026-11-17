# AICPU cluster 放置：Paged Attention 配对测试

日期：2026-11-17。算法保持 v8 三项评分和 ready 延迟绑定，禁用跨侧领取；
本次只扩展 benchmark，没有针对新负载调整权重。
当前机制见 [算法说明](../user/how-to/use-aicpu-cluster-placement.md)，
BGEMM 的权重消融和历史结果见 [BGEMM 调查](2026-11-aicpu-cluster-bgemm.md)。

## 为什么补充这个 benchmark

BGEMM 对 O 的生成和调度吞吐有价值，但其模式主要是独立矩阵任务加小范围累加。
此前诊断中完整成熟评分没有遇到未完成 WAIT，因此不能凭它标定跨侧边惩罚。
仓库还有 Paged Attention、滑动窗口依赖，以及 Qwen/DeepSeek 解码用例。
本次采用已有 Paged Attention 的四个算子和原始依赖构建，补充不同的依赖结构。

Paged Attention 用于大模型解码：query 依次读取分页 KV cache，计算各分片的
注意力，再用在线 softmax 合并结果。每个 KV 分片产生四个单 block 任务：

```text
QK（AIC 矩阵乘）→ SF（AIV softmax）→ PV（AIC 矩阵乘）
                        └──────────────────┐
上一个分片的 UP ────────────────────────→ UP（AIV 在线合并）
```

UP 同时读取本分片的 softmax 统计量和 PV 输出，并更新跨分片累加状态。
因此存在矩阵/向量资源交替、分片流水、多输入汇合以及同一输出的顺序归约。
其中 SF→UP 可能被传递依赖缩减覆盖；不能把图上的每条逻辑边当成独立 WAIT。
这仍是规则的算子图，不代表已经验证任意动态 DAG 或完整模型性能。

## 配置与公平性

- Uniform：16 个请求，每个上下文 4096，512 个 KV 分片，2048 个计算任务。
- Skew：32 个请求，上下文长度按 128、512、2048、8192 重复 8 次，
  共 680 个 KV 分片、2720 个计算任务，长短请求的后续工作量差异明显。
- 两组均为 16 个 query heads、1 个 KV head、head_dim=128、block_size=128、
  BF16 输入、FP32 输出；固定随机种子 9102，同一组两策略输入相同。
- 沿用原始算子源码和 golden，容差 RTOL=ATOL=1e-3。
- 两策略使用同一套运行库、同一 orchestration 和算子机器码；通过额外 scalar
  参数选择是否对四类计算任务调用 `set_aicpu_cluster_auto()`。
- 两策略都是六 AICPU 线程、两个 AICPU cluster。cluster 0 为 S0/S1（CPU 2/3），
  cluster 1 为 S2/S3/S4（CPU 4/5/6）和 O（CPU 7），O 位置不变。
- Shared 的普通任务进入按 AIC/AIV/MIX 分类的全局共享队列，由两侧 S 共同领取；
  自动策略分侧入队。Shared 不是单 cluster，也不是只有一侧 S 工作。
- 每用例先做一次 golden，再以 A 正序、B 反序两批各测 15 轮。
  计时关闭 swimlane/dep-gen，保留全部轮次，包括各进程首轮；不只取最好的一轮。
  另做四个启用 level-4 swimlane 和依赖捕获的 golden，诊断不混入计时。

## 普通计时结果

单位 ms，主指标为设备端整体 `device_wall`。

| 负载 | Shared 中位数 | Auto 中位数 | Auto 相对 Shared | Shared p95 | Auto p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | 2.42840 | 2.15587 | -11.22% | 2.46762 | 2.22286 |
| Skew | 3.25837 | 2.88967 | -11.32% | 3.30064 | 3.38219 |

Skew 的 Auto p95 比 Shared 高约 2.47%，虽然中位数改善，尾部没有同步改善。
不能只报告平均或中位数而忽略这一点。

两批反序结果和全部 120 个普通计时样本保留在 `summary.json` 中。
这些是同卡、同二进制的局部结果，不是对所有注意力规模或模型的普遍加速保证。

## 独立诊断结果与限制

四个额外诊断 golden 均通过，算子机器码 hash 完全一致，capture 无丢失或截断。
两组自动运行的 8 行分资源类别账本满足 preferred=released、queued=taken=actual，
总量守恒；5 个 S 的跨侧 checks、attempts、remote_blocks 全为零。
本次运行库三份二进制 hash 与 v8 所选 queue-future 阶段完全相同。

Uniform 有 512 次完整偏好决策、1536 次继承；Skew 为 680/2040，均为 75% 继承。
依赖捕获分别记录 512/680 个具有多个 WAIT 前驱的节点，能够补充 BGEMM 的结构覆盖。
但捕获的逻辑 WAIT 不等于完整评分时仍 pending 的前驱，传递缩减及组继承仍会避开
部分评分；本次没有重新增加逐项评分探针，不能宣称已验证每个评分项或得到通用权重。
本轮 AIC/AIV 的 queue_ns 两侧均保持 2000ns 先验，仅部分交接系数更新。

以下为启用诊断的单轮控制成本，单位 ms：

| 用例 | O submit | O 放置及 wire 分段合计 | S observe 各线程合计 | S ready_binding 各线程合计 |
| --- | ---: | ---: | ---: | ---: |
| UniformShared | 3.030 | 0.444 | 0.439 | 0.000 |
| UniformAuto | 2.970 | 0.528 | 0.277 | 0.345 |
| SkewShared | 4.071 | 0.613 | 0.595 | 0.000 |
| SkewAuto | 4.575 | 0.970 | 0.392 | 0.627 |

O 分段计时包含在 submit 内，不与 submit 再相加；S 项跨线程重叠，部分还嵌套于
派发路径，不能相加当成设备关键路径。计时器开销和跟踪开销均在这些诊断数值内。
Skew 的自动策略 O submit 比 Shared 高约 12.4%，所以整体中位数改善不等于 O 成本
必然降低；这里更可能同时涉及依赖交接和调度行为，不能从这四轮证明单一因果。

## 复现与材料

从 runtime 目录，在已经安装并构建 a2a3/onboard/TMR 的项目环境内运行：

```bash
source ../.claude/skills/testing/load-env.sh
python examples/a2a3/tensormap_and_ringbuffer/benchmark_paged_attention/run_cluster_study.py \
  ../official-results/new-paged-attention/paired
python examples/a2a3/tensormap_and_ringbuffer/benchmark_paged_attention/analyze_cluster_study.py \
  ../official-results/new-paged-attention/paired
```

脚本固定查询物理卡 0、使用逻辑卡 0；换机器必须先核实映射并按实际情况修改。
存在 task-submit 时逐次通过它运行；本机没有该工具，未持有独占锁。
板卡查询和 CANN 配置确认是 910B2/Ascend910B；架构预检脚本因驱动未输出单独
NPU Name 字段而返回失败，原始输出及显式核验结果均保留，没有假定预检成功。

原始材料位于 PyPTO 主仓库的 `official-results/v9-paged-attention-20261117/paired/`：
`summary.json`、`invocations.json`、逐轮 host 日志、debug/device 日志和 `captures/`。
主指标不包含编译、Python 输入生成或 CPU golden 的时间；它也不是整网解码吞吐。
