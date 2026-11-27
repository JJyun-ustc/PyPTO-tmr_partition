# 流式 TMR：空槽触发的动作调度实验方案

**执行状态（2026-11-27）**：[已有实测结果](streaming-action-results.md)。仅完成可见性、标定与 A/B 基础设施成本；主动 C/D、独立两动作收益标定和新残差训练尚未完成。以下仍为目标规格，不代表全部已实现。

**状态：设计提案，2026-11-27。尚未实现或测量。**

[English](../../../en/experiments/tmr-cluster/streaming-action-plan.md) · [已完成的第二轮结果](qkv-round2.md)

本方案更新下一轮实验设计，不覆盖历史结果。代码参考版本：runtime `8ebbff77377175bbbdc8d8f7252897f3eb73e6f8`；发布仓库基线 `5cb3b7920f2a35d8d5846118f6939d12eb232c92`。
第一份主要交付是 A–D 对照和菱形图的两种提交时序。只有分析式策略的可信收益能够覆盖成本后，才继续训练。

## 1. 实验依据与决策点变化

第二轮 Shared／原 QKV／改进 QKV／训练版的中位数为 **36.831／40.005／37.668／37.616 ms**。原版最主要的已观察开销是集中 pending/S0 等待；改进版和训练版仍慢于 Shared，训练版没有证明稳定净收益。独立用例中的普通 MIX 覆盖率不能代表 Qwen 同步 attention 的覆盖率。

下一轮从集中分配改为：**S 确实可以派发时，做一次有界动作比较**。首阶段只处理普通、尚未领取过 block 的单 block AIC/AIV。多 block、MIX、sync-start、严格绑侧、谓词特殊路径、DUMMY 和已被 stager 接管的任务保留原协议，但它们的真实资源占用仍进入日历。不引入在线训练或整图扫描。

AICPU 的侧别是调度域，不能直接等同于计算核缓存局部性域。同侧偏好必须来自实际生产者／消费者物理组和输入重用的配对标定；位置未知、区域未知或收益方向不稳定时，提示为中性。

## 2. 在线信息契约

决策时只能读取已原子发布、且在本次观察版本中可见的元数据及已登记依赖。仅对事后整图按提交时间过滤，不足以证明在线可见性。

| 输入 | 可用条件 | 用途 |
|---|---|---|
| 任务资源、kernel、launch 规格、ready 年龄 | 任务已发布 | 运行跨度与等待年龄 |
| 前驱位置及 tensor 区域 | 依赖已登记且实际位置已知 | 经标定的局部性 |
| 本侧空槽、对侧摘要、running/pending/已承诺工作 | 快照可读，含时间戳和版本 | 最早可执行时间 |
| 三个队列的队头 | 已发布，代次可校验 | 默认与替代任务 |
| 后继及全部相关入边 | 提交和依赖登记完整 | 可见汇合估计 |

后继状态分为 `UNSEEN`（未知）、`REGISTERING`（登记中）、`USABLE`（可用于估计）。`USABLE` 仅代表元数据完整，不代表已经 ready。没有明确的提交范围／fanout 封口协议时，空后继索引意味着未来后继未知。看到一个登记完整的后继，也不代表全部未来后继已确定。实际执行位置未知时，局部性中性处理。

特征不得包含最终 fanout、未来图变体或尚未提交的后继。每次决策保存观察到的发布／登记 epoch。完整图仅用于事后解释和明确标注的信息参考实验。拿到整图的启发式策略并非严格性能上界；小图精确搜索也只在其时间模型假设下给出界。

## 3. 队列、默认动作与所有权

每类纯资源提供三个逻辑队列：无偏好、偏好 0、偏好 1。偏好是软提示，不授予独占所有权。每个任务只发布到其中一个队列。无偏好队列可以沿用现有 Shared 队列并容纳不符合新策略条件的普通任务；偏好队列只容纳符合条件的任务。如果默认队头不符合条件，就执行其原路径，不为了制造比较而跳过它。

B–F 的默认规则统一为：**三个队列的合法队头中，领取 ready 最早的任务**；同分按队列编号，再按任务代次／ID。它不等于未修改 Shared A 的原领取顺序。B–F 共用分类、提示、队列、窥视和领取路径，B 选择时忽略亲和提示。因此 B–A 测的是包括元数据和分类在内的整个结构改动，不是单独的队列指令成本。

偏好队列满时，在发布所有权转移前回退无偏好队列。无偏好队列满时，必须保留原有容量、背压和推进协议；在 Gate 0 证明，不能丢任务、无限自旋或重复发布。O 的零入边、前驱均已完成等直接 ready 路径，必须和 S 路径使用同一发布所有权凭证，不能重复执行生命周期 CAS。

### 安全观察是前置条件

当前 Vyukov 队列是在取得 dequeue ticket **之后**才读取非原子任务载荷。先读载荷再检查两遍 sequence，在槽位回收时仍可能构成 C++ 数据竞争。先 pop 再 push 也不能当安全窥视。

必须先实现不可变、原子发布的队头元数据，证明读取期生命周期，校验代次，并支持按精确 ticket 消费。队列位置所有权与原有任务／block 所有权仍然分开。推测评分期间不得解引用可能已回收的任务载荷。

以下为拟议接口，当前代码尚不存在：

```cpp
struct HeadToken {
  uint64_t queue_ticket, task_generation, metadata_epoch;
  TaskId task_id;
};
struct VisibleHead { HeadToken token; ImmutableTaskFeatures features; };
// ReadGuard pins metadata lifetime; validate head and task generation on commit.
std::optional<VisibleHead> observe_head(ReadGuard&);
OwnedQueueItem consume_exact(const HeadToken&);  // may fail without consuming
// Then use the existing task/block claim protocol before prepare/publish.
```

ticket 改变就使旧估计失效。只做有界次数的默认领取重试，或下个调度循环再试，不自旋。出队后若 drain、sync 或槽位所有权改变，必须通过已证明安全的持有任务继续执行／重入队协议维持唯一所有者。启用 C 前，需一起验证队列存储、读取保护和任务复用。

## 4. 空槽触发流程

策略只输出所选任务、预计净收益和理由；比较细节保存在诊断记录中。

可用槽必须是原普通 `DispatchPhase::IDLE` 路径确实允许立即派发的槽，遵守 sync、drain、staging、prepared/pending 的优先级。仅 pending 邮箱为空不够。两个动作都针对**同一个物理槽**，首版不同时优化计算组放置。

初始限制：每次原普通 IDLE 调度 pass 最多比较一次，其余槽按 B 填充。保留原跨任务 prepare、批量寄存器发布与屏障。记录被预算限制的机会数，pilot 后冻结该限制。新策略不使用 S0 集中待分配队列。

```text
on_original_normal_idle_pass(side, physical_slot):
    heads = observe_three_legal_heads_with_lifetime_guards()
    U = oldest_legal_head(heads)                    # B default
    if U absent: return_to_original_loop()
    T = oldest_other_eligible_head(heads, U)       # fixed bounded rule
    if U ineligible or T absent or U/T preference_age_expired:
        try_original_claim_and_dispatch(U); return
    snapshot = capture_visible_versioned_state()
    if snapshot unreliable or budget exceeded:
        try_original_claim_and_dispatch(U); return
    a0 = simulate(U_now, T_later, default_B_continuation, snapshot)
    a1 = simulate(T_now, U_later, default_B_continuation, snapshot)
    chosen = T if gain(a0,a1) > optimism_bound + extra_cost + delta else U
    consume_exact_then_original_claim(chosen)     # no slot/head reservation by peek
    prepare_with_original_protocol_or_retry_next_pass()
```

只有一个合法队头时，无论对侧摘要年龄或亲和提示如何，直接领取。初始 `10 µs` 到期规则表示 ready 年龄超过该值后禁止模型继续改变默认动作，不是 ready 到启动必小于 10 µs 的保证。不得为了等亲和侧而让可用槽空转。分别记录无替代项、到期、摘要过旧／缺失、竞争失败和超预算回退。分别采样观察／出队、快照／日历、评分、提交／发布及 S 派发间隙；完整路径总成本必须包含未成功改派的比较。

```mermaid
flowchart LR
  O[已发布任务与已登记依赖] --> Q[唯一逻辑队列]
  Q --> I[S 有合法空槽]
  I --> B[默认 U 与替代 T]
  B --> C[同一快照模拟两个动作]
  C --> G[收益超过误差和成本门槛]
  G --> A[校验 ticket 并沿用原子领取]
  A --> P[原 prepare 与批量发布]
```

## 5. 有界资源日历与评价目标

根据 running、pending、prepared 和提前派发承诺增量维护日历，不扫描整个队列。以 `(任务代次, block)` 去重，同一 Shared 工作只能放入一次，不能复制进两侧。遵守实际兼容计算组，包括混合占用和同步批次。观察到槽空闲只是预测，不是资源预留。

每次最多观察三个队头、比较一个替代动作、使用一个可见汇合。pilot 初始上限为索引两个后继、join 入度不超过四。溢出或登记不完整时，本次停用汇合前瞻；不能截断前驱后把遗漏部分当作完成。日历规模受硬件资源上限约束，模型最多使用八个事件。

对已运行时间 `e`，使用条件剩余时间：

\[
\hat r(e)=Q_{50}\{d_i-e\mid d_i>e\}.
\]

历史跨度为 `6、8、10、15、17 µs`、已运行 `11 µs` 时，估计剩余 `5 µs`。没有存活样本时，用另行标定的粗粒度／尾部回退或放弃比较，不能通过 `median(d)-e` 推出剩余为零。观察到 FIN 后以真实事件为准。

\[
\hat F_v(a)=\max(t,\hat H_v(a),\hat R_v(a),\hat D_v(a))+\hat P_v(a).
\]

`H` 表示兼容槽可用时间；已 ready 的任务在 `R` 中使用真实 ready 时间。若单列 `D`，它必须是包含服务时长的最早**派发完成／发布时刻**；服务已含在跨度或日历代理口径中时，不再单列，避免重复计费。`P` 在明确的启动代理口径下包含已标定的放置和并发影响。

对登记完整、以完成事件释放依赖的可见汇合：

\[
\hat R_J(a)=\max_{p\in pred(J)}[\hat F_p(a)+\hat E^{remaining}_{p,J}(a)].
\]

已完成依赖使用实际释放事件，只计算未来尚未发生的边开销。RETAIN 与 WAIT 语义不同，应使用真实释放门控；不支持的门控停用汇合估计。发生在 kernel 内的局部性差异计入 `P`，不能再收一次边成本。

两个动作固定使用同一终点集合。没有可用汇合时，纳入 T、U 和可见受影响竞争任务。只有依赖语义证明支配关系时，才允许用 J 的 ready 终点替代前驱终点；保留其他竞争任务。对终点取最大值得到 `M(a)`，不求和。在相同默认后续规则下同时模拟两个任务及有界竞争任务。

| 可见前缀 | 默认动作终点 | 替代动作终点 | 预计收益 |
|---|---:|---:|---:|
| T:18→14、U:10→16 µs；无可用 J | 18 | 16 | 2 µs |
| J 可见，另一前驱在 20 µs 到达 | 20 | 20 | 0 µs |
| J 可见，另一前驱在 12 µs 到达 | 18 | 16 | 2 µs |

最后一行 J 自身提前 4 µs，但 U 的竞争将净局部收益压到 2 µs。之后才提交的后继，不能反过来改变本次决策的终点定义或训练标签。

## 6. 误差门槛与独立配对标定

\[
\hat G=\hat M(a_0)-\hat M(a_1),\qquad
\text{override iff }\hat G>Q_{90}[(\hat G-G_{obs})_+\mid b]+C_{extra}+\delta.
\]

pilot 初始 `delta=0.5 µs`。正式计时前冻结它、到期时间、摘要年龄上限和全部参数。`C_extra` 仅计动作特有且尚未建模的成本；共同推理成本由整轮计时检验，不能重复扣除。经验高估分位数不等于实际收益具有 90% 保证。

按摘要年龄、并发程度、汇合可见性分档。标定使用独立于模型拟合和调参的配对时段。每个保留档至少需要 30 个独立配对块，否则逐级合并到粗档；粗档也不足时走默认。块内相关重复不能当作独立样本，可保守地采用块内最大高估等统计量。初始六个时段不足以支持细档尾部分位数。

| 标定任务 | 匹配对照 | 必需产物 |
|---|---|---|
| 控制交接 | 轻链，同侧／跨侧；两个实验臂均预留消费者槽 | FIN 观察→边释放、ready 发布→领取、领取→派发的配对差 |
| 数据局部性 | 同消费者 kernel；4 KiB、64 KiB、1 MiB 重用及低重用对照 | 按物理组的跨度差和置信度；不稳定则中性 |
| 跨度与剩余时间 | kernel、规格、资源掩码、并发档 | p50/p90、条件剩余表、样本数、回退层级 |
| 动作收益 | 匹配的预备状态，强制 a0 或 a1 一次，之后统一按 B | T/U 完成、可见 J ready、整图结束、配对局部收益 |

每类微基准先做六个配对时段，每条件每时段一次预热、二十次重复。固定 kernel、输入和物理映射，交换方向和先后顺序。观察结果前预注册可观测预备状态的匹配／剔除规则。状态匹配不代表缓存状态逐位相同。基准中的预留槽只是实验装置，不是在线窥视时可以预留资源。

不能独立改变 CPU 侧和物理计算组时，报告组合放置效应，不强行归因为硬件传输。资源等待不能充当控制交接标签。不可复现状态下的 Qwen 只能提供实际时间线，不能拼出虚假的配对因果标签；回放标签单独保存。

## 7. 事件、时钟与流式回放

拟议诊断格式：

```text
run_id, task_id, task_generation, pred_id, edge_kind, tensor_region
submit_ts, edge_registered_ts, edge_released_ts, ready_ts, publish_ts
claim_ts, dispatch_ts, start_proxy_ts, start_proxy_kind, fin_seen_ts
publisher_s, claimer_s, producer_side, execution_side, physical_group
resource_shape, slot_state, summary_ts, summary_version
decision_ts, observed_submit_epoch, observed_registration_epoch
visible_successor_state, head_tokens, endpoint_ids, a0, a1
predicted_gain, optimism_bin, decision_reason, decision_cost
```

跨线程比较前验证时钟偏移、漂移和单位。保留启动代理分类：空闲寄存器发布、观察到 pending 晋升、doorbell、仅 FIN 删失，不能自动混为真实启动。仅 FIN 的样本不能监督运行跨度。记录采样概率和缺失情况，覆盖 O 快速路径及任务生命周期复用。

回放只暴露当时前缀和登记 epoch 可见的事件。改变动作后重新计算资源释放和后继 ready，不能继续把记录中的未来 ready 当作固定真值。如果 O 提交受执行进度影响，固定提交轨迹的回放仅估计该轨迹下收益。自适应策略完整 rollout 与单动作反事实须分别验证。

第二轮轨迹／回放是历史诊断，不能证明新方案的可见性契约或微秒级动作收益精度。旧整图和旧残差权重不能当作新配对标签。整图结束时间误差较小，也可能同时存在很大的逐任务时间误差。

## 8. 执行门槛与实验矩阵

### Gate 0：协议和可观测性

先实现时钟、发布／登记可见性、安全队头元数据和领取、有界容量、drain 推进，再做模型。覆盖队列绕回／回收、两个 S 竞争队头、槽失效、重复 ready、队满回退、单候选、不合格队头、退出和同步／drain 共存。断言唯一发布所有者、唯一 block 领取、恰好一次派发及退出时队列清空。保留 TaskPayload ABI／tensor offset 和原屏障协议。

### Gate 1：流式 TMR 实际能看到什么

先跑未改动 Shared 的诊断。只读前缀的影子队列可估计三队列机会，但不改变 Shared 领取；明确标注“影子机会”，B 中重新测真实比例。分别报告后继已提交、登记完整、可用汇合、动作排序改变、摘要缺失／过旧、单候选的分母和比例。

必过用例：

1. 相同可见前缀、不同未来图：未来提交前决策、特征和理由完全一致。
2. 同一菱形图，J 在 T 领取前／后提交，兄弟分支快／慢：只有提前可见且登记完整的 J 能影响前瞻。
3. J 部分登记、后继索引为空且未封口：区分登记中／未知与已证明不存在。

不能把读未来的策略冒充在线策略。事后整图信息参考仍用同一时间模型；若额外给真实未来时长，须另标注。

### Gate 2：先完成分析式 A–D

| 组别 | 行为 | 解释 |
|---|---|---|
| A Shared | 原公共领取 | 实用基线 |
| B 结构对照 | 三队列、相同提示／窥视／领取，选最早合法队头 | 结构成本 |
| C 两动作 | B 加日历、前驱局部性、被挤工作和收益门槛 | 流式局部信息价值 |
| D 可见汇合 | C 加完整可见的有界汇合 | 图信息的独立增益 |

先跑等 block 异耗时、菱形两种提交时序、fanout→join，再跑 decode。加入短链、普通 MIX 干扰和同步 attention 共存，但 MIX 自身仍不参与新策略选任务。检查依赖合法、恰好一次派发、兼容计算组匹配、退出清空及输出正确。

目标 decode 沿用 Qwen3-14B、batch 16、40 层、seq_len 3500。建议留出长度 2250、5000，先做内存／输入／golden 可运行性检查，然后在调参前冻结。另留出完整图形家族或规模。每个实际派发策略均需通过输出和全部 40 层 KV golden。后续模型结构和阈值不得再利用留出结果调参。

若可见汇合很少，依据开发 pilot 冻结 C-local 分支，D 保留为独立图信息消融。若 C、D 均无可信的覆盖成本后的价值，交付负面的 A–D 结果，停止升级模型。

### Gate 3：残差学习，再决定是否真实派发

学习**两动作收益差的残差** `G_observed - G_analytical`。E 用线性模型，F 用最多八事件、单头、8 维投影的 QKV。Q 编码动作、目标侧、最早开工差和被挤队头；K/V 编码真实生产者、可见兄弟前驱、兼容槽释放和竞争任务。未知后继有 mask／缺失标志。依赖 max、兼容性和领取协议仍显式执行。

D／C-local、E、F 使用相同输入和派发路径。板上配对数据与经标定回放反事实分开保存和报告。真实动作耗时仅监督执行过的动作；反事实收益差必须来自配对或明确标注的仿真。按整轮和图形家族／规模切分，同一配对两臂不得跨集合；归一化只拟合训练集，并保留独立标定块和封存测试集。

报告留出集上**选错动作导致的实际时间损失**均值／p95、选择率、标定和失败案例，不能只报残差 MAE。终点定义固定于决策时刻。回放要重算下游事件，使用其排序前须验证动作收益尺度上的误差。

真实派发要求 F 的留出动作损失优于 E 和分析式，且尾部没有不可接受的回退。AICPU 影子测量覆盖真实多样快照、冷态和缺失 mask。初始预算为新增**完整触发比较路径 p95 ≤1 µs**，拆报推理、窥视和校验。不能把旧十六评分批次 p95 除以十六当作单次触发测量。pilot 冻结验收阈值，未通过就保留影子模式。

通过门槛后，**E 和 F 都实际派发**，与同轮 A、选定分析式对照比较。第二轮未正式派发线性模型，因此不能证明 F 相对 E 的净收益。若后继可见率低，以前驱／资源模型为主，汇合条目作为独立消融。

## 9. 正式计时与报告

从十二个交错时段起步，每策略／负载每时段保留一次预热、五次正式样本，即 **60 个正式样本＋12 个预热样本**。独立 pilot 后、正式运行前冻结样本量，禁止可选停止或中途改权重／阈值。A–D 以及后续 A／分析式／E／F 均为四组：十二时段采用四次循环位置轮换，依次正向／反向／正向。如果保留全部六组，采用六次正向和六次反向轮换；这保证位置平衡，不宣称完整的一阶携带效应平衡。

固定板卡、CPU 核映射、物理组映射、kernel 指纹、输入、编译选项、元数据模式、标定表和模型哈希。优先同二进制在静止／重置状态切换模式，保持相同进程／输入／缓存重置和预热条件；否则记录固定的进程设计。冻结调度元数据重置语义，避免策略间状态泄漏。详细诊断不混入正式计时。

保存所有原始样本和预热，按时段配对统计并按块重采样给出区间：

- device_wall 中位数、p95、相对 Shared 差及不确定性，包括 p95 区间和失败。
- 按时钟／代理类型统计 pending／ready→claim、claim→dispatch 分布，以及 S 派发间隙。
- 汇合机会，D 相对 C 推进的汇合，兼容槽空闲时间及跨侧改变影响到的任务。
- 比较次数、改派、默认／回退原因、类型覆盖率、每次和累计开销；同时计入 O 元数据成本。
- 留出动作损失、实际 E 与 F 的 device_wall 净差和分歧案例。

预测局部终点改善不代表整轮收益。回退、模型被拒绝和无收益结果均照实交付。

## 10. 具体改动位置与验收交付

下表针对 runtime `8ebbff773`，行号为该版本附近位置。新名称／接口均为拟议设计，不是当前可运行命令或已完成功能。

| 现有位置 | 拟议改动 |
|---|---|
| `runtime/scheduler/scheduler.h`（约 74–302、786、1349） | 安全队头观察／精确消费证明，三队列，统一各 ready 路径的发布所有权 |
| `runtime/orchestrator.cpp`（约 1318–1360、`wire_fanin_task`） | 提交／边登记 epoch，后继元数据完整标记，覆盖直接 ready 快速分支 |
| `runtime/scheduler/scheduler_dispatch.cpp`（约 468–611、1326） | 在普通 IDLE 加分支，保留原领取与批量发布；新策略不经 S0 批分配 |
| `runtime/scheduler/scheduler_completion.cpp`、`scheduler_context.h` | 增量放置／日历事件、启动代理类型、真实依赖释放语义 |
| `runtime/shared/runtime_init.cpp`、`runtime/qkv_trace.h` | 队列容量、arena／生命周期／重置、版本摘要和新事件格式 |
| `tests/ut/cpp/a2a3/runtime/tensormap_and_ringbuffer/test_scheduler_state.cpp`、`test_wiring.cpp` | 生命周期竞争、前缀可见性、时钟／日历确定性用例 |
| `tests/st/a2a3/tensormap_and_ringbuffer/` | 菱形提交时序、短链、fanout/join、混合干扰、golden 和 drain |

runtime 路径均位于 `src/a2a3/runtime/tensormap_and_ringbuffer/` 下；`scheduler_context.h` 位于其 `runtime/scheduler/` 目录。拟新增 `runtime/streaming_metadata.h` 和 `runtime/streaming_action.h`。元数据放在 TaskPayload 外，保留现有 tensor ABI。

修改前（现有流程的概念性描述）：

```cpp
// Eligible ready work can enter the centralized QKV allocation path.
// Normal dispatch takes a ready batch, claims block ranges, prepares, then publishes.
```

修改后（拟议控制边界）：

```cpp
if (policy == Shared) return original_shared_dispatch();
// B-F share the same queues, hints and normal-IDLE ownership protocol.
if (!normal_idle_slot || sync_or_drain_precedes) return original_protocol();
auto heads = observe_heads_with_read_guard();
auto choice = policy == Structure ? default_head(heads)
                                  : bounded_two_action_choice(heads, snapshot);
// Conditional consume and original unique block claim; preserve batched barriers.
return prepare_claimed_choice_in_original_batch(choice);
```

上述伪代码不能用来绕过真实同步／drain 检查，也不代表当前队列已有安全 peek。实施顺序为：安全队列前提及测试 → 时钟／可见性诊断 → 日历与标定 → B/C/D → 最后 E/F。

拟议结果目录为 `official-results/streaming-action-<collection-date>/`，只有产生新数据后才创建。拟议工具目录 `examples/a2a3/tensormap_and_ringbuffer/benchmark_streaming_action/` 包括事件格式／校验器、配对标定、前缀回放、切分清单、线性／QKV 训练、AICPU 影子测试、冻结策略清单和交错运行器。工具实现并验证后再发布可执行 CLI 命令。

验收包须有：实现 diff 和准确构建／测试命令；卡／拓扑／kernel 指纹；事件格式和时钟验证；协议测试及两种菱形时序；原始配对和标定表；前缀回放测试及限制；若进入训练则提供切分、权重和归一化；全部正式／预热样本及配对统计；一页 A–D 结论，以及通过门槛后 E/F 的实测净收益。板卡不可用时，CPU／协议／回放证据单列，板上标定、golden、影子预算和 device_wall 对照明确记为未完成。
