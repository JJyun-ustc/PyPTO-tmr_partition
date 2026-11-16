# AICPU cluster 二分放置实验说明

## 研究问题与系统边界

研究对象是 910B2 上 TMR 动态任务图的 AICPU cluster 放置。O 生成任务，
决定初始调度侧；S 派发计算、处理完成事件并释放后继依赖。
交接涉及共享描述符、依赖计数和 ready 队列控制信息，不表示复制 tensor 数据。
决策发生在每个节点 WAIT 缩减后、依赖连接和 ready 发布前；后继可能尚未生成，
不等待整个 scope，不求解完整图的全局最优切分。

六线程布局：cluster 0 的 S0/S1 位于 CPU 2/3；cluster 1 的 S2/S3/S4 位于
CPU 4/5/6，唯一 O 位于 CPU 7。O 的位置保持固定，没有进行位置对照。
24 个 AI Core 组按 S 轮转分配为 5/5/5/5/4 组，所以 AIC 容量为 10/14，
AIV 容量为 20/28。AI Core 组与 AICPU cluster 是不同概念。

## block、压力与队列

block 是任务的一份逻辑执行实例，一个任务可以有多个 block；它不等于 kernel
内部循环次数。当前 BGEMM 的每次 AIC/AIV 提交都是单 block。

```text
blocked[c] = max(0, assigned[c] - ready_published[c])
ready_auto[c] = max(0, ready_published[c] - dispatched[c])
depth[c] = 本侧定向队列项数 + 共享队列项数 * 本侧S数 / 总S数
B[c] = max(ready_auto[c], depth[c])
```

blocked 是尚未 ready 的自动 block；ready_auto 是 ready 后尚未派发的 block。
队列项数对多 block 任务只是近似。工作量除以 capacity 表示大致的资源波数，
不表示执行时间或资源占用率。成熟评分的队列项按 S 数归一化。

默认不设 hint 使用共享队列；严格指定 0/1 和自动放置任务使用对应侧的定向队列。
最新版不保留独立自动队列，也没有跨侧领取或迁移计数。多 block 可以在所选侧
分批派发。自动多 block sync-start 走共享同步路径，严格绑定的多 block sync-start
被拒绝。DUMMY 及谓词跳过任务使用相应 DUMMY 队列。

## 最新算法：固定校准和快照批次

1. 每次执行重置参数，前 256 次自动放置使用冷启动规则并采样。
2. 第 257 次决策前读取一次累计样本，拟合一次参数，然后冻结。
3. 不再使用逐 shape 成熟门槛、滑动窗口、EWMA、反复轮询和更新时间间隔。
4. 每个 queue 系数至少 32 个样本，交接等渠道至少 16 个；不足或快照竞争时
   保留先验，但不延长校准阶段。校准后停止普通延迟采样，继续资源反馈与记账。
5. 每 8 次自动放置构成一批。每个 shape 首次出现时读取快照，批内复用。
6. 每个新分配任务立即在 O 本地增加 blocked 或 ready 预留；S 的并发进展在
   下一次刷新时体现。快照更统一但可能更旧，不是精确的全局原子状态。

当前 submit 会立即返回供后续节点使用的输出句柄；O 仍顺序生成与评分，
S 的执行与之重叠。O 不等待前一个 kernel 结束。本次没有缓存整批任务后并行评分，
没有借用 S 进行规划，也没有引入 scope 刷新屏障。

冷启动：

```text
cold_score[c] = 1024 * (blocked[c] + B[c] + 新任务block数) / capacity[c]
```

容量未知时用 S 数；若前驱亲和侧得分不超过最小值 256，可优先亲和侧。
冷启动分数是归一化平衡量，不是 ns。

后续评分按 AIC/AIV/MIX/DUMMY 分别使用自己的参数：

```text
Q[c] = B[c] / S_count[c]
R[c] = 2*(capacity[c]-free[c])/capacity[c] + (capacity[c]-idle[c])/capacity[c]
C[v,c] = publish_ns[O所在侧][c]                  若无未完成WAIT前驱
       = max(适用的 dependency_ns[前驱侧][c])   否则
X[v,c] = 已知远端未完成WAIT数，减去C已代表的一次远端交接（若适用）
F[v,c] = min(blocked[c] + 当前待等待任务block数, capacity[c]) / capacity[c]
score[v,c] = C[v,c] + 0.5*queue_ns[c]*Q[c] + 0.25*queue_ns[c]*R[c]
             + 128 ns*X[v,c] + 0.25*queue_ns[c]*F[v,c]
```

未知位置按两个来源的最大交接系数估计，不增加未知边的结构惩罚。
没有未完成 WAIT 的当前任务不加入 F；容量未知时 F 使用 S 数兜底。
free 包含 IDLE 与可派发 PENDING，idle 只是真正空闲；blocked 不会减少它们。
未来项每侧最多 queue_ns/4，双方均饱和后不能继续分辨更深的积压。
所有权重都是策略先验，模型没有 kernel 时长和前驱剩余时间预测。

先验 queue=2000 ns；publish/dependency 两个矩阵的同侧为 250 ns、跨侧为 750 ns。
有效样本为 ready 发布到首次资源准备的观测；安静交接要求队列为空、存在 idle
证据、真正使用 IDLE 槽，pickup 等待不超过 5 us，准备不超过 20 us。
queue 观测按 S/(S+深度) 归一化，并近似扣除安静 pickup 基线。
采样不覆盖 O 整个提交过程，不能解释为纯硬件交接时延。

## BGEMM 工作负载与计时

每组计算 C[group] = sum_k A[group,k] @ B[group,k]，AIC 做矩阵乘法，AIV 累加。
组之间独立，同组累加有依赖。

| 用例 | 分组 | grid_k | tile | kernel内部循环 | AIC+AIV节点 |
| --- | ---: | ---: | --- | ---: | ---: |
| Case0 | 250 | 2 | 128×128 | 4 | 1000 |
| Scale | 512 | 4 | 64×64 | 1 | 4096 |

Shared/Automatic 使用相同 kernel、确定性输入和六线程拓扑。每轮恢复输出，
重新生成任务图并重置模型；计算结果与学习参数不跨轮继承，但缓存、温度和
环境状态可能有影响。因此使用全部有效轮次的中位数和 p95，不挑最快一轮。
正确性与诊断单独运行；计时各 30 轮，两批各 15 轮，第二批反序，关闭 swimlane。

## 最近结果：2026-11-16

本次逻辑设备 0 对应物理卡 4，910B2 健康状态前后均为 OK。原版和新版均在
当前卡重新测量；没有与此前物理卡 1 的数据混用。没有 task-submit，未持独占锁。

| 规模／版本 | Shared中位数／p95 ms | Automatic中位数／p95 ms | 自动相对共享 |
| --- | ---: | ---: | ---: |
| Case0／原v5 | 1.048／1.142 | 1.419／1.497 | +35.42% |
| Case0／简化版 | 1.047／1.139 | 1.197／1.323 | +14.32% |
| Scale／原v5 | 4.963／4.998 | 6.629／6.686 | +33.58% |
| Scale／简化版 | 4.913／4.950 | 5.809／5.867 | +18.23% |

自动策略自身中位数下降 15.66% / 12.37%，两批方向一致，但仍慢于共享队列。
这是一组组合修改，未做逐项消融，不能分别归因于冻结权重、取消领取或批次化。

诊断中 Case0 分配/派发为 327:673，Scale 为 1437:2659；每侧分配等于派发。
两者均为 256 次冷启动，之后分别 744/3840 次评分；校准一次，快照 250/1024 次。
诊断运行不用于计时，也不能据此宣称切边或负载已经最优。

验证通过 17 项控制器、32 项调度状态和 45 项依赖连接测试。新旧各四个 BGEMM
正确性检查通过；额外通过四线程 Bgemm64、自动 4/16/24/48/96-block AIV、
sync-start 回退和自动 DUMMY 链。新增复现脚本只做语法和参数检查，发布副本没有
重新编译或重新跑 NPU；算法代码与被测工作区逐文件校验一致。

## 历史证据及局限

- v4：四项评分，Case0/Scale 自动中位数 1.421/6.605 ms，仍慢于 Shared。
- v5：加入有界未来项后为 1.419/6.633 ms，未见加速。与 v6 的当前卡对照是
  重新运行的 v5，不是直接引用历史卡上的这组数字。
- 更早版本探索过累计学习、队列空层跳过、计数器缓存行隔离等；英文原始调查保留
  在 runtime/docs/investigations/2026-11-aicpu-cluster-bgemm.md。
- 早期离线前缀规划排除了规划时间，不能作为集成在线算法的端到端收益。
- 未纳入不用于结论的 O 位置对照残留，也未复制通用驱动安全日志。
- 参数冻结后不能适应后续任务分布变化；未来项会饱和；不同 kernel 计算量没有区分。

## 代码和证据入口

核心决策器位于 runtime/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/
aicpu_cluster_partition.h；O 接入为同目录 orchestrator.cpp；ready 路由为
scheduler/scheduler.h，S 采样和派发为 scheduler/scheduler_dispatch.cpp。
这些文件在 runtime-snapshot 分支，也会随 main 的递归子模块克隆获得。

main 的 official-results/ 保留原始计时、诊断、汇总、实现补丁和版本快照。
v6-simplified-20261116/summary.json 含每批逐轮值与校验信息；源码来源 SHA 见
同目录文档下 source-provenance.json。历史绝对路径只是采集时路径，不是复现安装路径。
