# v14：O 配额分发与 v8 / Shared 对比

v14 已实现 O 统一评分和双侧固定 block 配额，删除 S 首次 ready 改侧、WAIT 票数累计和跨侧领取。
Qwen decode 相比 v8 改善，但未超过 Shared；Bgemm 和 Paged Attention 相比 v8 回退。
本轮固定公式和参数，不根据中途成绩调参，不替换为最好候选。
算法见 [algorithm-v14.md](algorithm-v14.md)。

## 普通计时

单位 ms，device_wall 中位数；正百分比表示更慢。每格 30 个样本。

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

Shared 为原始共享队列运行时，仅将最大 AICPU 线程数从 4 改为 6；v8 为已发布快照。
v8 的全部 591 个源码文件均与发布提交 `7ac2fb88cf978435d8c06ccfd701bbcc036f8d5d` 相同，
包含在旧工作树中未被 git 跟踪的分区头文件，另有完整补充指纹证明。
v14 在独立工作树独立构建；三个版本实际装载的计算 kernel FNV 哈希逐项一致。
Qwen 是 40 层、batch 16、seq_len 3500、max_seq_len 5500，无 LM head，使用 CANN fused attention。
Scale 为 Bgemm 扩展负载，Uniform 和 Skew 为 Paged Attention 的均匀与偏斜负载。

## 协议与边界

同一张 Ascend910B2，物理卡 6 / 逻辑 0，O=CPU7，S=CPU2–6，两个 AICPU cluster。
六种版本顺序全部执行，每进程五轮，每组轮换负载顺序；全部首轮保留，总计 450 个普通样本。
独立执行 15 次模型 golden 和 15 次 level-4 诊断，均不混入普通计时。
CPU/OMP/MKL/OpenBLAS 均限制两线程，编译两 worker。
本容器没有 task-submit，未持独占设备锁；结果不能当作隔离条件下的稳定收益保证。

连接中断时进程退出，保留已验证的 255 个样本，归档被中断的 v14/Qwen/block3，整次重跑该进程后续跑。
没有删除任何已完成的性能样本。恢复时和每次运行结束时检查源码、二进制指纹；均保持不变。
实验跨越一次会话/容器重启，可能有环境漂移，因此同时保存全部六组中位数和重启记录。
普通计时只能比较整体效果；没有针对单一机制的消融，不能把所有差值归因于某一行代码。

## 耗时在哪里

以下是各版本各负载一次独立诊断，单位 ms，包含插桩开销，不能与普通计时直接相减。
O submit 是全部任务提交区间累计，包含提交期间的资源回收/等待，并不等于纯评分 CPU 时间。
O score 是其中的评分子区间，不能重复相加。
尤其 Qwen 的 level-4 采集会显著放大时长（例如 Shared O submit 约 1.20 秒，而普通 device_wall 约 36 ms），
这里的诊断主要用于验证机制和事件，不能用其比例解释普通计时的瓶颈构成。
S feedback 是五个 S 线程的 CPU 时间总和，不是设备关键路径时长。

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

S complete/dispatch/resolve 同样是五线程累计时间，可能并行，不能相加当作 device_wall。
原始采集记录的完整性标记保存在 diagnostics.json；发生丢记录时只解释已采集区间。

删除三条路径后仍有下列成本：O 每任务读取创建期前驱分布、计算固定候选配额并登记预留；
首次 ready 需要按非零配额发布一到两个队列条目；S 本侧领取更新范围游标与累计账本；
完成时更新单写者 finished 计数，每侧 leader 每 64 个循环汇总所有资源类别，O 每 64 次创建读取摘要。
“64 次更新”约束摘要传输与读取频率，不代表任务创建、领取、FIN 本地记账也只每 64 次执行。
普通 S 不再查询两侧摘要，也没有每条 WAIT 边读取前驱/给消费者原子加票的路径。
创建期亲和保留在 O，并不意味着前驱分布读取完全消失。

诊断中的 `cross_pickup` 名称保留为兼容 trace 槽；v14 对应空计时段，调用次数不是跨侧领取次数。
`affinity_votes` 的 S 调用数为零；两侧每个资源类别 assigned=ready=claimed=executed=finished_units/6。
实际拆分任务数见上表和原始日志；板上多 block AIV/MIX 回归也分别实际触发 5 次拆分。

四个微基准诊断的拆分数均为 0：这些负载是单 block 任务，只承担 O 评分及反馈成本，
没有多 block 拆分收益，不能将其回退解释为双队列发布的开销。
Qwen 诊断实际拆分 422 个任务，确认新的多 block 路径被真实负载使用。

这套等价 block 评分没有 kernel 时长模型，不能准确表达不同 kernel 的计算时间和 MIX 与纯 AIC/AIV 的竞争。
反馈可能滞后，固定配额又无法由空闲侧协助收尾，因此简化控制路径不自动带来更好的整体负载均衡。
当前证据支持保留 v14 作为机制验证版本，不支持全面替代 v8 或 Shared。
后续优化可先对 O 候选评分、双队列发布和周期汇总分别做消融，再考虑用轻量在线时间标定替代等 block 假设；
这些均未混入本轮 v14，也尚未证明有效。

## 验证与复现

184 个 CPU 用例、8 个测试目标，15 个板上回归用例，16 组配额终态核对通过；DFX 关闭构建和运行时提交检查通过。
15 次模型 golden 全通过，Qwen 输出与全部 40 层 KV cache 均检查。
未运行完整项目的所有场景/仿真，也未做 O 位置对照。

从发布仓库递归拉取子模块后，分别准备 Shared、v8、v14 独立构建目录，运行：

```bash
python runtime/examples/a2a3/tensormap_and_ringbuffer/benchmark_o_quota/run_study.py \
  --shared <shared-checkout> --v8 <v8-checkout> --v14 <v14-checkout> \
  --output <study-output> --phase preflight
# 同样的参数分别运行 --phase measure 与 --phase diagnostics
python runtime/examples/a2a3/tensormap_and_ringbuffer/benchmark_o_quota/analyze_study.py <study-output>
```

中断续跑使用 `--resume`，保留完成记录，归档不完整调用。正式矩阵不接受筛选负载或版本来替代全矩阵。
公开结果中大于 500 KiB 的原始文件使用 gzip 保存；压缩后仍较大的文件拆为 `.part001` 等分片。
先按文件名顺序用 `cat <file>.gz.part* > <file>.gz` 合并，再解压 `.gz` 后重新分析。
构建产物、venv、缓存和通用 driver plog/security/run 日志不发布。

[summary](../../../../official-results/v14-o-quota-20261120/summary.json),
[samples](../../../../official-results/v14-o-quota-20261120/samples.json),
[CSV](../../../../official-results/v14-o-quota-20261120/results.csv),
[batch medians](../../../../official-results/v14-o-quota-20261120/batch-results.csv),
[diagnostics](../../../../official-results/v14-o-quota-20261120/diagnostics.json).
