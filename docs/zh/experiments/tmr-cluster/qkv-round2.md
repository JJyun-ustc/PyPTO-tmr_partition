# QKV 第二轮：一页结论

[English](../../../en/experiments/tmr-cluster/qkv-round2.md) · [Mechanism](algorithm-qkv-round2.md)

| Policy | n | Median ms | p95 ms | Median vs Shared | p95 vs Shared |
|---|---:|---:|---:|---:|---:|
| Shared | 30 | 36.830900 | 37.007384 | +0.00% | +0.00% |
| Original QKV | 30 | 40.004970 | 40.195601 | +8.62% | +8.62% |
| Improved QKV | 30 | 37.668440 | 37.791733 | +2.27% | +2.12% |
| Learned QKV | 30 | 37.615719 | 37.809745 | +2.13% | +2.17% |


每策略 30 个正式样本，另保留 6 个 warm-up；同一 910B2（物理 4 / ACL 0）、B16、40 层、seq=3500、seed=1234，同一二进制、六时段交错。正值表示变慢。四策略输出和全部 40 层 K/V golden 通过，37 个实际加载 kernel 与初版一致。未持设备独占锁。

1. **初版主要开销在集中 pending / S0 等待。** 关闭注意力的短对照几乎不改变端到端时间；保留集中服务、最终发往 Shared 仍明显变慢。原/改进版独立诊断的 pending 均值为 49.04/9.94 µs，覆盖不同，不能作可加的精确因果分解。pop、快照、评分、发布和 S0 派发间隙数据均保留。
2. **普通 MIX 和反馈已接入，但覆盖瓶颈仍在。** 同步共存用例真正选侧 20/23 个普通 MIX；1,024 任务短链仅 0/1008 个符合条件任务被选侧，大量因摘要未初始化或过旧回退。Qwen 的 40 个 MIX 都是同步 attention，继续同步协议，没有因此直接扩大 Qwen 覆盖。改进版减少缺少即时容量时的集中入队，收益伴随更多 Shared 回退。
3. **学习型 QKV 的误差改善未证明相对线性调度有净收益。** 留出 12,361 个真实动作上，综合 MAE 7.616 → 6.526 (-14.31%) µs；8 任务/16 候选 AICPU 影子 p95 为 QKV/线性 4.655/1.637 µs，模型通过门槛后实际派发。训练版相对改进版中位数变化 -0.14%。本轮没有线性策略板上派发对照；回放的非同分排序一致率两者相同，未显示注意力优势。

| Policy | Eligible | Assigned | Coverage | Multi-task batches | Fallback | Decision mean µs (batches) |
|---|---:|---:|---:|---:|---:|---:|
| Original QKV | 298830 | 288134 | 96.42% | 40281 | 10696 | 6.202 (43950) |
| Improved QKV | 298831 | 33553 | 11.23% | 9627 | 265278 | 3.900 (13615) |
| Learned QKV | 298830 | 33750 | 11.29% | 9585 | 265080 | 5.169 (13222) |


决策均值从出队后计时，pop 另作采样。启动仅有寄存器发布/提升/doorbell 代理，FIN 为 AICPU 观测，未声称取得实际 AICore 开始时刻。完整回放逐次为 11,166 任务、23,605 边，无悬空端点，资源释放与后继 ready 随动作重算；真实监督与反事实标签分开。早期追踪遗漏、失败与过渡版本全部归档。

**下一轮只优先检查 S0 服务间隙与摘要刷新滞后。** 保持权重不变，解释 pending 等待和短 MIX 链的大量回退。本轮停止在代码与数据交付。

[详细报告](../../../../official-results/qkv-round2-20261125/REPORT.zh-CN.md) · [复现命令](../../../../official-results/qkv-round2-20261125/REPRODUCE.md) · [完整样本](../../../../official-results/qkv-round2-20261125/measure/summary.json) · [反馈与回退](../../../../official-results/qkv-round2-20261125/coverage.json) · [训练权重和误差](../../../../official-results/qkv-round2-20261125/models/) · [实现 diff](../../../../official-results/qkv-round2-20261125/runtime-and-tests.patch)


[Publication manifest](../../../../official-results/qkv-round2-20261125/publication-manifest.json) · [Audit](../../../../official-results/qkv-round2-20261125/audit.json). Raw host logs and custom QKV traces are archived in `raw-evidence.tar.gz`; extract it into the result directory before offline parsing. Generic Ascend driver logs and compiled executables are omitted; model weights are retained. Machine-specific root paths use named placeholders, with original/published SHA-256 recorded.
