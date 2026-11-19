# v11：独立重写的 ready 亲和放置实验

日期：2026-11-19。此版本在独立 worktree、`runtime-ready-affinity` 分支实现。
v8 所在 `runtime-snapshot` 不修改；官方 Shared 使用另一独立、冻结运行库。
实验结果目录为 main 分支 `official-results/v11-ready-affinity-20261119/`。
发布研究版本不代表已经证明整体更快。

## 实现与版本边界

公共 Auto/严格/Shared API、线程拓扑、依赖处理、原子 block 范围认领及计算 kernel 保留。
删除旧 `aicpu_cluster_partition.h`，新 `ready_affinity.h` 独立实现亲和、ready 压力及单写者反馈。
不存在旧模型/新模型运行时切换，删除价格学习、校准、组表、W预留、I/retired负载计数。
任务放置元数据由40字节缩为16字节。详细机制见
[算法与记账](../user/how-to/use-aicpu-cluster-placement.md)。

```text
O：汇总 WAIT 前驱实际侧/有限跳数偏好，不读取调度负载
  → blocked：只保留强/弱/中性偏好
  → ready：O 或处理最后依赖的 S 调用统一选择
  → R/K 明显失衡选轻侧，否则亲和/可派发提示择侧
  → 首次入队记全部 block；成功认领多少扣多少；重入队不重复记账
  → 派发记录实际侧；完成不增加放置专用计数
  → 空队列侧 leader 有 IDLE 资源时，低频尝试领取远端自动单 block
```

多 WAIT 前驱允许同侧共识。尚未执行的前驱只能产生弱偏好；偏好传播最多7条边，
不能跨scope。实际侧可覆盖旧偏好。不是把多个父组并成一个大组。
普通多 block task 整体绑定一侧，不能跨侧拆分或领取。
共享 sync-start 等特殊任务仍沿用既有路径；单个“首次派发侧”不能表达共享多 block
任务完整的跨侧执行分布，普通计算全 Auto 的本轮负载不以混用作为设计要求。

ready 近似压力为 `256*(未认领block+新任务block)/兼容容量`，差超过64才覆盖亲和。
摘要每64次S循环发布，ready每批/每类别最多16个任务复用一次视图，立即修正本线程
未吸收的发布/认领净增量。容量未知时以该侧 S 数暂代，不是永久把 S 数当计算容量。

跨侧领取比“本侧缺一种工作”更保守：本侧各普通计算队列全空、远端单 block队列
深度至少 `2*远端S数+2`，只由 leader探测。有资格的空队列检查机会每32次探测，
发起领取后延后128次。扣原队列侧 claimed，实际侧另记，不能要求逐侧 claimed=executed。

## 对照与测量协议

官方 runtime 固定 `d8f997fabd1b6988b69a94feef61ab669dda0eea`，`src` 唯一差异为
线程上限4→6；官方队列、依赖处理和 early-resolve 不变。benchmark是研究用例，
官方 orchestration 去掉 Auto 提示，不能把自动运行库里的 Shared 分支冒充官方。

本轮物理卡5、逻辑设备0，910B2、PCIe `0000:41:00.0`；O固定CPU7，S使用CPU2–6。
与v10卡3不同，所有本轮对照重新测量，不跨卡比较绝对时间。预检默认卡0查询失败，
已明确查询卡5，并核对CANN的 `Ascend910B2.ini` / `Short_SoC_version=Ascend910B`。
`task-submit` 不存在，未持有独占设备锁。

顺序：官方初测 → 本卡v8 → v11关闭领取 → v11开启领取 → 官方复核。
两个v11阶段只改变源码常量 `READY_REMOTE_PICKUP`，分别构建和保存指纹。
所有常量在普通计时前声明，没有依照中途成绩调整权重或只选最快轮次。

每阶段八个用例：BGEMM 1000/4096节点、PA等长2048/长短混合2720节点、
Multi16、MultiSkew、短前驱Fanout、延迟前驱Burst。参数和计算kernel沿用v10。
Burst 是人工控制压力测试：生产者固定有界标量循环，不代表真实模型计算。

每用例先1次golden，再15轮正序、15轮反序，保留进程首轮，报告30轮中位数/p95。
另做独立level4时间线/依赖捕获，不与普通计时混合。O/S分段存在嵌套和并行，不能相加。
各阶段普通计算机器码hash应一致；官方运行库前后指纹必须保持不变。

## 全部结果

单位：普通 device_wall 毫秒。第一张为中位数，第二张为p95，均保留全部30轮。

### 中位数

|阶段|Bgemm|Scale|Uniform|Skew|Multi16|MultiSkew|Fanout|Burst|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|official-before|1.0341|4.8328|2.4649|3.2625|1.2152|4.8472|4.8462|14.0921|
|v8|0.9904|3.8404|2.1965|2.8784|1.2827|5.0247|4.4633|14.7075|
|ready-no-pickup|0.9832|5.1061|2.6646|3.4878|1.1647|4.6928|4.5204|13.7732|
|ready-pickup|0.9840|5.1357|2.6628|3.5130|1.1681|4.6181|4.5475|13.7389|
|official-after|1.0334|4.8404|2.4578|3.2542|1.2200|4.9481|4.8895|14.0967|

### p95

|阶段|Bgemm|Scale|Uniform|Skew|Multi16|MultiSkew|Fanout|Burst|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|official-before|1.1219|4.8879|2.5182|3.2880|1.3879|5.0203|5.0571|14.2726|
|v8|1.1101|3.8989|2.2688|2.9533|1.4439|5.3636|4.6746|15.6310|
|ready-no-pickup|1.1050|5.1489|2.7051|3.5338|1.3572|6.0832|4.8730|13.9233|
|ready-pickup|1.1121|5.1868|2.6913|3.6010|1.3543|5.9850|5.1149|13.8635|
|official-after|1.1193|4.8665|2.5060|3.2947|1.3722|5.1117|5.0719|14.2590|


官方初测/复核同一组二进制。MultiSkew中位数前后漂移约2.08%，其他用例最大约0.89%；
主相对值使用末尾复核，不挑选更有利的一批。未持有独占锁，小幅差异不作确定收益。

## 诊断、验证与判断

v11开启领取相对v8：BGEMM约−0.65%、Scale约+33.73%、PA等长约+21.23%、
PA混合约+22.05%、Multi16约−8.93%、MultiSkew约−8.09%、Fanout约+1.89%、
Burst约−6.59%。相对官方复核，Scale/两组PA仍慢约6.10%/8.34%/7.95%。
因此这次发布是独立研究版本，不能替代v8作为“已验证更优”的通用策略。

单次插桩诊断（累计毫秒；线程间重叠，不可与普通wall相加）：

| 指标 | v8 | v11无领取 | v11有领取 |
| --- | ---: | ---: | ---: |
| Scale O submit | 5.232 | 5.935 | 5.905 |
| Scale O策略区间 | 0.495 | 0.193 | 0.194 |
| Scale O连接/ready发布区间 | 0.589 | 1.088 | 1.069 |
| Scale S ready绑定累计 | 0.407 | 0.791 | 0.788 |
| PA Skew O submit | 4.112 | 4.674 | 4.693 |
| PA Skew S ready绑定累计 | 0.496 | 0.909 | 0.906 |

O策略区间是v8五个策略阶段累计与v11 affinity区间的比较，插桩次数不同，
不能当作无插桩的严格节省值。它说明简化的局部目标有所体现，但O整体提交没有变快，
连接/发布与S ready路径变贵。捕获WAIT边的跨侧数，Scale为37→570，PA Skew为466→1708；
这些是依赖捕获边，可能含多个参数边，不能解释成精确交接次数或独立因果证明。
当前证据与“更激进分散增加控制路径交接”的解释相符，但没有单因素消融来证明唯一原因。

Burst的384个后继全部在生产者结束前提交，生产者约9.076ms。
v8实际26945个block全部在侧1；v11有领取为11200/15745，约42%/58%。
诊断中的生产者结束后计算尾段从7.935ms降至5.828ms；这是插桩区间，不是普通wall分解。
去掉O未来预留、在ready时分散发布确实缓解了这组人工突发，但不推出真实模型普遍加速。

领取开启相对关闭，八组中位数变化均小于约1.6%。八次独立诊断合计只领取9个block：
Skew 1、MultiSkew 2、Fanout 2、Burst 4，其余为0。因此不能把总体改善主要归于跨侧领取，
也没有足够证据认定其控制成本已被收益稳定覆盖。保留开启版本用于发布所描述的完整方案，
关闭版本源码和结果同时归档，参数没有在计时中回调。

验证：40组阶段/用例，共1200个普通样本、40次golden、40次独立诊断。
最终184项C++测试、15个板上回归通过；64行benchmark账本、12行额外回归账本满足
逐队列侧enqueued=claimed且全局claimed=executed。Skew诊断真实发生跨侧领取，
AIC队列侧541/819，实际540/820，正好体现扣原队列、实际侧另记。
所有阶段计算机器码hash一致；时间线检查CPU2–7角色一致；官方源码/二进制冻结再次核验。
关闭/开启领取两种DFX-off配置均编译通过。仿真场景未运行，配置的g++-15不可用。

旧v8全部591个已跟踪src文件与原发布提交7ac2fb8逐文件一致。
新策略源码和实验构建归档逐字节一致，开启/关闭源码唯一差异为领取常量。


完成路径不维护retired，不能把未认领block压力解释为精确剩余计算工作。
低频可派发资源提示不能预测长kernel结束；异构kernel的block也不等重。
其他发布者的变化仍有滞后，多block整体误放没有事后跨侧补救。
这些限制与观察到的性能需一起解释，不能只凭均衡率或切边数评价收益。

## 复现与归档

从main递归克隆得到新运行库；v8请另建指向 `runtime-snapshot` 的独立checkout。
官方请从固定upstream runtime提交另建worktree，只应用 `official-runtime.patch` 的线程上限修改。
三个checkout都必须各自构建，不能复制其他checkout的.so。
研究用例的Shared变体还需移除官方API不存在的Auto提示，精确源码已归档。

激活各自的Python3.10虚拟环境并安装项目依赖、CANN/PTOAS后，
按仓库testing流程限制编译并行度为2。在新运行库目录执行：

```bash
export PYPTO_BUILD_JOBS=2 PYPTO_TEST_JOBS=2
export PYTHONPATH="$PWD/python:$PWD:$PYTHONPATH"
STUDY=/path/to/new-results
HARNESS=examples/a2a3/tensormap_and_ringbuffer/benchmark_ready_affinity
python "$HARNESS/prepare_variant.py" --pickup disabled --archive "$STUDY/ready-no-pickup-artifacts"
python "$HARNESS/run_study.py" official-before --shared --runtime /path/to/frozen-official \
  --physical-card DEVICE --output "$STUDY/official-before"
python "$HARNESS/run_matrix.py" --study "$STUDY" --v8 /path/to/v8-runtime \
  --official /path/to/frozen-official --physical-card DEVICE
python "$HARNESS/validate_runtime.py" --study "$STUDY"
python "$HARNESS/analyze_study.py" "$STUDY"
python "$HARNESS/build_report.py" "$STUDY"
```

DEVICE为物理卡号；场景逻辑设备使用0，换机需核对可见设备映射。
脚本仍包含本机依赖fallback路径，换环境应改为自己的依赖路径/完整本地venv。
输出目录拒绝覆盖；先完成官方初测和关闭领取的源码构建，再运行matrix。
运行库、bindings及PTO-ISA均在各自worktree构建；本机PTO-ISA以同pin的本地git源码
克隆准备，不复制其他checkout编译产物。

发布保留源码、补丁、运行命令、全部普通样本、诊断捕获、测试日志及SHA256清单。
Git中不包含虚拟环境、编译产物、通用驱动plog/security/run重复日志或凭据；
本地二进制留档，Git保留其指纹。标注maybe_unused以兼容DFX关闭的声明，在计时配置
重建后SHA256完全相同，验证文件为 `annotation-verification.json`。
