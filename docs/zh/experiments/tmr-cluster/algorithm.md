# AICPU cluster 放置：v11 ready 亲和版本

本分支 `runtime-ready-affinity` 是独立的 v11 实现。旧 v8 保留在 `runtime-snapshot`，
不通过运行时开关加载旧模型。公共 API 与线程拓扑沿用，决策器、任务元数据及反馈重新实现。
这里的 cluster 指 AICPU cluster，不是 AI Core cluster。

## API 与任务生命周期

- 默认 Shared：按资源类别进入全局队列，由两侧 S 竞争领取。
- `set_aicpu_cluster(0/1)`：严格绑定该侧，不能跨侧领取。
- `set_aicpu_cluster_auto()`：O 产生依赖亲和偏好；第一次 ready 决定队列侧。
- 自动 sync-start 仍使用共享回退；DUMMY/失败谓词走专用完成路径。

```text
O 生成任务 → 汇总 WAIT 前驱亲和 → 尚未 ready：等待，不预留负载
                              → 已 ready：调用统一 ready 选择
最后释放 WAIT 的 S → 调用同一 ready 选择 → 分侧入队
本侧 S → 原子认领 block 范围 → 派发 → 原有完成与后继通知
本侧资源空闲且普通队列全空 → leader 低频尝试领取对侧自动单 block
```

六线程拓扑：侧0 S0/S1 使用 CPU2/3；侧1 S2/S3/S4 使用 CPU4/5/6，O 使用 CPU7。
同侧 S 共同领取分侧队列，不是每个 S 独占队列。O 不对 blocked 任务读取调度负载。
默认 active 线程数保持原配置；这里的六线程由研究 benchmark 显式指定。

## O 只汇总亲和

`AffinityVote` 遍历已缩减的 WAIT 前驱：优先实际执行侧；严格任务可使用其保证的目标侧；
其他自动前驱尚未派发时，只允许沿当前 scope 继承有限跳数的偏好。

- 全部已知、同侧且不含推断：强偏好。
- 已知部分同侧，但包含未知或推断：弱偏好。
- 两侧冲突、全部未知、没有 WAIT：中性。

多个前驱可以形成同侧共识，不要求只有一个 WAIT。推断偏好最多传播7条依赖边，
不能跨 scope；实际执行侧不受此推断限制。不维护组表、成员数、组 block 预算或全图分组。
O 不学习 queue_ns/交接价格，不维护 W/R/I，不进行未来容量预留。

## 第一次 ready 决定队列侧

每个 AIC/AIV/MIX/DUMMY 类别分别估计：

```text
R[c] = max(累计首次入队block - 累计已认领block, 0)
P[c] = 256 × (R[c] + 新任务block数) / K[c]
```

K 是该侧兼容计算容量，不是 S 数；容量提示未建立时以 S 数暂代。
两侧压力相差超过64，即四分之一波，选较轻侧。压力接近时依次：强亲和、
仅一侧有可派发资源的提示、弱亲和、最后通知前驱侧、较低压力。同压取侧0。
O 直接发布时没有最后前驱侧。当前资源提示包含 IDLE 与可预派发 PENDING 槽，
不是“所有资源都真正空闲”。参数是预声明候选值，不是实测 kernel 时间。

偏好、队列侧、实际侧分别存储。最后通知侧不会覆盖强亲和；未知或弱亲和也可能被
压力覆盖。`ReadyTaskState` 占16字节，旧40字节学习/分组元数据不再存在。

每次前驱完成处理开始一个 ready 批；每类别第一次使用或最多16个 ready任务后
重新读取提示。O 的直接 ready 也最多16个复用同一视图。不等待凑批、不延迟发布。
同线程首次入队与认领增量立即修正缓存，其他写入者的变化等待低频提示。

## 反馈与多 block 账本

每个线程/类别拥有独立64字节计数行：enqueued、claimed、executed、resources。
计数按侧分开，单写者更新，不使用争写的全局 fetch_add。

| 事件 | 操作 |
| --- | --- |
| 首次 ready | 按最终队列侧 enqueued += 全部逻辑block数 |
| 成功原子认领范围 | 按原队列侧 claimed += 本次认领数 |
| 剩余 block 重入队 | 不再次增加 enqueued |
| 派发 | 按实际侧 executed += 本次发布数，仅作守恒诊断 |
| 完成 | 不更新放置专用 retired；仍执行原有正确性计数与依赖处理 |

每侧最低编号 S 每64次调度循环汇总工作、容量、可派发资源及每写入者的吸收水位。
reader 单次尝试读取一致版本，失败保留先前有效视图，不自旋。
本地有效 R = 摘要 R + 自身未吸收 enqueued - 自身未吸收 claimed，并截断为非负。
这是近似提示；认领的正确性依赖原子 block 范围与真实资源跟踪器。

多 block task 整体进入一侧，同侧多个 S 分段认领、派发。一个队列项可代表许多 block，
不以队列项数代替 block 压力。认领后的准备、排队执行和长 kernel 剩余时间不计入 R；
低频资源提示只能部分补偿，模型不预测执行时间。

## 受限跨侧领取

本侧严格/多 block、自动单 block、Shared 普通计算队列均空时，侧 leader 才探测远端；
这是比“本侧没有某一种合适任务”更保守的限制。它还必须有该类别 IDLE 资源，
远端自动队列深度至少 `2 × 远端S数 + 2`。每32次满足本侧空队列的检查机会探测，
发起领取尝试后延后128次机会再探测，一次最多领取一个任务。

只访问普通 Auto 单 block队列，不领取严格、多 block、DUMMY、sync-start。
成功领取直接派发，不重新入另一侧队列。claimed 扣原队列侧，executed 和实际侧记
接收侧；后继优先读取实际侧。因而终态按队列侧 enqueued=claimed，但跨侧领取时
不要求逐侧 claimed=executed，只要求总数相等。

`READY_REMOTE_PICKUP` 是源码级常量；实验关闭/开启分别构建并保存指纹，没有新增环境开关。
跨侧领取不能补救多 block整体误放，也没有改变 kernel 的执行位置映射或搬运 tensor。

## 实验与边界

比较必须使用独立冻结官方 Shared、同卡重测 v8、v11无领取、v11有领取。
每组先 golden，再15轮正序及15轮反序，统计所有30轮的中位数/p95。
独立插桩诊断观察 O submit、ready 绑定、S工作和领取次数，不与普通 device_wall 相加。

实验入口：`examples/a2a3/tensormap_and_ringbuffer/benchmark_ready_affinity/`。
实验记录见 [v11报告](ready-affinity.md)。
版本发布不代表已证明普遍加速；异构 kernel 时长、并发发布滞后、整任务多 block绑定
仍是明确限制。
