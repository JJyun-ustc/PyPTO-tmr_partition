# a2a3 AICPU cluster 自动放置：有界分组与延迟绑定

本页描述当前 v7 实现。cluster 始终指 **AICPU cluster**，不是 AI Core 的计算资源组。
910B2 六线程实验拓扑为：cluster 0 的 S0/S1 使用 CPU 2/3，cluster 1 的
S2/S3/S4 使用 CPU 4/5/6，O 固定使用 CPU 7。默认线程数仍为四线程。

```cpp
CoreTaskArgs args;
args.set_aicpu_cluster_auto();
rt_submit_aiv_task(kernel_id, args);
```

不设置提示使用 Shared 队列；`set_aicpu_cluster(0/1)` 严格绑定。
自动多 block sync-start 继续使用共享同步路径，严格指定侧的多 block sync-start 不支持。
自动放置不搬运 tensor 数据，也不预测 kernel 时长或前驱剩余时间。

## 执行流程与三种侧别

```text
O 生成节点、缩减 WAIT 依赖
  → 寻找可复用的小组；入口完整评分，组内继承偏好
  → 写入 preferred_cluster，预留 block 工作量
  → 连接依赖
  → 已 ready：O 轻量确定队列侧
    未 ready：最后一个依赖释放者 S 轻量确定队列侧
  → S 优先取本侧工作，满足条件时直接领取另一侧自动任务
  → 首次准备派发时记录实际侧；后继可以看到这个侧
```

| 字段 | 含义与写入时机 |
| --- | --- |
| `placement.preferred_cluster` | O 生成时的偏好，之后不改；等待阶段按它记未来预留 |
| `task.target_aicpu_cluster` | 自动任务未 ready 时为 -1；首次 ready 发布确定最终队列侧，此后不改 |
| `placement.dispatch_cluster` | 首次领取并准备派发的实际 AICPU cluster，初始 -1；跨侧领取后可与队列侧不同 |
| `placement.release_cluster` | 使本节点 ready 的实际前驱侧，或 O 发布的来源侧；用于交接采样与 ready 亲和 |
| `placement.ready_writer` | 实际执行 ready 发布代码的线程行，不等同于前驱执行侧；O 使用专属行 5 |

其中 -1 的“未绑定自动任务”和普通 Shared 任务通过 `placement.scope != 0` 区分。
所有自动任务禁用普通 early-resolve；sync-start 的自动回退不进入这个自动状态机。
任务描述在依赖连接前完整初始化；ready 路由的唯一领取者完成绑定、记账再发布队列。

多 block 自动任务的所有块仍在同侧，实际侧记录其首次准备侧。单 block 跨侧领取
只有一个执行侧。Shared 前驱如果在反馈启用前已被取走，其实际侧可能未知；算法保留未知，
不把偏好伪装为实测执行位置。

## 1. 在线识别有界小组

O 使用缩减后的 WAIT 前驱列表识别小组，不要求完整 scope 或完整图已生成：

1. 前驱携带有效组号，且组属于当前 scope，才可继承。
2. 每组最多 **8 个逻辑任务节点**，包括入口；最多跨越 **32 次自动放置**。
3. O 的本地表只有 **64 个槽**，组号校验防止槽复用后串组。表被替换只会使后继重新完整评分。
4. 找到第一个可用前驱后，取其实际领取侧；尚未领取时取其偏好侧。
5. 本次后续 fanin 回调不再读取生产者的状态或位置；仍会遍历既有 fanin 容器。
   没有可用组时扫描待完成 WAIT 来源并完整评分，建立新组。

组成员计数只由 O 修改，S 不写这个表。一个大 scope 会产生很多独立小组。
组只复用偏好；组内各节点在 ready 时仍可改队列侧，领取时仍有有限补救机会。
当前 BGEMM 主要形成 GEMM→ADD 的二节点组，没有按 `grid_k` 或 tile 参数硬编码分组。

已经创建的后继可能保留旧偏好，但这不等于最终绑定：生产者跨侧执行后，后继 ready
时使用真正释放依赖的生产者实际侧。生产者派发后新生成的后继则直接继承实际侧。
多前驱任务只保证 ready 阶段跟随最后释放者，不能代表所有前驱的位置。

## 2. ready 路径只做轻量绑定

设候选侧为 `c`，另一侧为 `o`，同类队列摘要深度为 `D`，S 数量为 `n`：

```text
c = 已知的最后释放前驱实际侧；否则取生成偏好
若候选侧没有 S，改用有 S 的另一侧
若 D[c] >= 4，且 D[c]/n[c] > D[o]/n[o] + 2，则改用 o
否则保留 c
```

实现使用整数交叉乘法，不读取每个 S 的资源、不重新扫描 fanin、不执行完整评分。
已 ready 的 O 节点以偏好为起点。来源未知时不制造 S→S 亲和。
采样时间戳缺失不会把 S 发布改记为 O 发布；来源语义独立于是否仍在采样。

队列摘要是滞后的提示。同一个长 fanout 内的一批任务可能仍集中选择同侧；这里没有
对整个 fanout 做联合分配，后续通过摘要刷新和受限领取调整。

## 3. 单写者摘要、O 本地预留与压力口径

**block** 是任务的一次逻辑执行实例，一个多 block 任务包含多个这样的实例。
容量是兼容的 AIC/AIV 核数，MIX 是兼容计算组数，DUMMY 为 S 数量。
block backlog / capacity 估计的是工作波数，并不是执行时间。

每个发布线程、每种资源类别各占一个 **64 字节摘要行**：

- 一个资源字：capacity/free/idle 各占 8 bit，S 每 **64 个调度循环**发布一次。
- `released[偏好侧]`：已经撤销的生成偏好预留，累计 block 数。
- `queued[最终队列侧]`：首次 ready 发布量，累计 block 数。
- `taken[原队列侧]`：已经派发的 block 数，跨侧领取仍扣原队列。
- 一个版本字：使同一行的撤销与入队变更保持一致。

各线程只写自己的行；计数使用单写者 load/store。版本字的 RMW 也只由本线程写。
S 开始处理 ready 任务到实际发布 block 的短准备窗口仍可能计为 ready；不等任务完成
或 CONSUMED 才扣减。多 block 重入队不再次计 ready，只扣本次派发的范围。
谓词失败时，在 DUMMY 路径按原资源类别一次扣除所有逻辑 block。

每侧最低编号的 S 另外单独发布该侧队列提示：严格/不可迁移队列深度 + 自动队列深度 +
按 S 数量分摊的 Shared 深度。高位记录这个发布者是否有空闲资源，供冷启动交接采样参考；
它不是全侧精确空闲位图。队列条目与 block 在多 block 时单位不完全相同，深度只是兜底近似。

O **每 64 次自动放置定义一个刷新周期**，每种 shape 在完整评分首次使用时才读取摘要。
O 对每行非阻塞读一次版本；遇到正在更新或读中变化，沿用该行上次一致结果，不旋转等待。
容量资源字和队列提示仍为近似快照，不保证所有线程同时一致。

刷新时，对所有发布线程求和：

```text
future[c] = max(0, O_assigned_preference[c] - sum(released[c]))
ready_ledger[c] = max(0, sum(queued[c]) - sum(taken[c]))
B[c] = max(ready_ledger[c], side_queue_depth[c])
```

`future` 即评分结构中的 `blocked`。两次刷新之间，O 为每个新节点只增加本地
`future[偏好侧] += block数`，包括马上会发布的节点。这是“尚未被快照吸收的本地预留”，
不保证这时任务仍在等前驱。刷新时从累计量**重建**，不再叠加旧本地增量。

| 状态事件 | 未来预留 | ready 压力 | 实际侧计数 |
| --- | --- | --- | --- |
| O 分配偏好 | assigned 加一次 | 无 | 无 |
| 首次 ready | released 按偏好加一次 | queued 按最终队列加一次 | 无 |
| 同侧或跨侧派发 k 个 block | 无 | taken 按原队列加 k | started 按实际 S 所在侧加 k |
| 多 block 重新入队 | 无 | 不重新加 queued | 无 |
| 完成/CONSUMED | 不重复扣 | 不重复扣 | 不重复计 |

因此跨侧领取不需要迁入/迁出账本：任务直接从原队列派发，没有重新入另一侧队列。
`B` 对账本与深度取最大，不能将它们相加。读者仍可能因状态滞后暂时高估或低估压力；
这些量是调度提示，不是资源锁，也不是精确的全局瞬时排队长度。

## 4. 受限跨侧领取

当前有三类普通计算队列：全局 Shared、每侧严格/不可迁移队列、每侧可领取自动队列。
另有 DUMMY 和 sync-start 专用路径。只有自动、单 block、普通计算任务进入可领取队列；
严格指定、多 block、DUMMY、失败谓词和 sync-start 不进入。

每个 S 在处理完本侧与 Shared 后，最多每 **32 次正常派发循环**检查一次：

- 本 S 有该类别的 IDLE 派发机会。
- 本侧同类严格、自动、Shared、sync 队列都为空。
- 对侧可领取队列至少有 `2 × 对侧S数 + 2` 个条目：当前为 6 或 8。
- 一次最多领取 **一个**任务，直接派发；不转入本侧队列。
- 发起一次领取尝试后，本 S 冷却 **128 次正常派发循环**，即使 pop 因竞争失败也冷却。

队列 pop 与 block claim 沿用原有原子协议。阈值、低频检查和不重新入本侧队列限制来回争抢。
实际领取侧发布给后继；交接采样按真实来源侧与接收 S 侧归类，不按原偏好归类。

## 保留的入口评分与固定校准

只在小组入口执行原 v6 的评分，继承节点也计入放置次数。
前 256 次自动放置使用冷评分，257 前读取一次 S 的样本并冻结系数；没有滚动学习和成熟门槛。
每个 shape 独立拟合，排队通道至少 32 个样本，交接通道至少 16 个；不足保留先验。
普通采样在冻结后停止，摘要和压力记账继续。每轮 runtime 执行重置这些状态。

```text
cold[c] = 1024 × (future[c] + B[c] + 新任务block数) / capacity[c]
```

容量未知时用 S 数量；前驱亲和侧在最低分 +256 以内可以优先。
固定校准后仍为：

```text
Q[c] = B[c] / S_count[c]
R[c] = 2*(capacity[c]-free[c])/capacity[c] + (capacity[c]-idle[c])/capacity[c]
F[c] = min(future[c] + (待完成WAIT存在时的新block数), capacity[c]) / capacity[c]
score[c] = C[c] + 0.5*queue_ns[c]*Q[c] + 0.25*queue_ns[c]*R[c]
           + 128ns*extra_remote[c] + 0.25*queue_ns[c]*F[c]
```

`C` 无待完成 WAIT 时取 O→S 发布系数；否则取适用 S→S 系数最大值。
前驱实际领取侧已知时使用实际侧，否则自动前驱使用偏好作估计；未知侧覆盖两个来源。
`extra_remote` 只补计没有被最大交接项代表的已知远端 WAIT。
queue 初始先验为 2000 ns，发布和依赖矩阵先验为同侧 250/跨侧 750 ns。
0.5、0.25、128 ns 均为策略权重，不是实测硬件单边时延。
静默接收基线的扣除仍是近似分解；摘要更稀疏也会改变采样条件，系数不能解释成纯通信常数。

## 验证与实验入口

实现位于 `runtime/aicpu_cluster_partition.h`、`runtime/orchestrator.cpp`、
`runtime/scheduler/scheduler.h` 与 `scheduler_dispatch.cpp`，路径均相对于
`src/a2a3/runtime/tensormap_and_ringbuffer/`。

[实验记录](../../investigations/2026-11-aicpu-cluster-bgemm.md) 保留 Shared、v6、
仅分组、加入延迟绑定、加入摘要以及加入跨侧领取的阶段结果。
每阶段同一套二进制先做独立 golden，再各策略两批反序、共 30 轮计时，保留全部样本。
O 提交与 S 阶段成本从独立诊断轮取得；跨线程重叠，observe 和 ready 绑定也可能嵌套，不能相加还原总时间。

可用 `examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm/reproduce_cluster_study.py`
重新测当前源码；`run_control_stage.py` 负责构建、保存源码及独立诊断。
所有实验保持 O 的位置固定。本轮结果不代表全图最优切分，也不证明其他动态图或重 kernel 场景同样受益。
