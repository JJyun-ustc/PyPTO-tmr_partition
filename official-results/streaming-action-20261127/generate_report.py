import hashlib,json,re,subprocess
from pathlib import Path
import sys
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'examples/a2a3/tensormap_and_ringbuffer/benchmark_streaming_action'))
from analyze_diagnostics import main as diag_main
sys.argv=['analyze_diagnostics.py',str(OUT/'final-diagnose-3500')]
diag_main()
summary={s:json.loads((OUT/f'final-formal-{s}/summary.json').read_text()) for s in (3500,2250,5000)}
diag=json.loads((OUT/'final-diagnose-3500/diagnostic-summary.json').read_text())
kernels=json.loads((OUT/'final-formal-3500/kernel-hashes.json').read_text())
freeze=json.loads((OUT/'formal-freeze.json').read_text())
audit={'formal_runs':0,'ordinary_runs':0,'warmups':0,'golden_checks':[],'loaded_compute_kernels':len(kernels),'same_kernels':True,'same_source_and_binaries':True,'actual_overrides':0,'claim_count_matches':True,'CPP_tests':55,'CPU_reference_tests':24,'onboard_regression_groups':6,'onboard_diamond_cases':4,'active_C_D':False,'new_gain_models_trained':False,'publication':'local only; no push','format_check':'one scheduler.h comment-alignment violation retained while runtime source was frozen'}
for seq in summary:
    folder=OUT/f'final-formal-{seq}'
    m=json.loads((folder/'manifest.json').read_text())
    assert m['fingerprints']==freeze['source_and_binaries']
    assert json.loads((folder/'kernel-hashes.json').read_text())==kernels
    rows=json.loads((folder/'samples.json').read_text())
    assert len(rows)==288
    for r in rows:
        assert r['device_wall_ms'] is not None
        c=r['counters']
        if c['mode']:
            assert c['eligible']==c['eligible_claimed'] and c['dropped']==0
        audit['formal_runs']+=1
        audit['ordinary_runs']+=not r['warmup']
        audit['warmups']+=r['warmup']
        audit['actual_overrides']+=c['overrides']
    g=OUT/f'final-golden-{seq}'
    assert json.loads((g/'kernel-hashes.json').read_text())==kernels
    checked=[json.loads(line.split(' ',1)[1]) for line in (g/'host.log').read_text().splitlines() if line.startswith('STREAM_GOLDEN_PASS ')]
    assert {v['mode'] for v in checked}=={'shared','structure','C_guarded','D_guarded'} and len(checked)==4
    assert all(v['poisoned'] and v['layers']==40 for v in checked)
    audit['golden_checks']+= [{'seq_len':seq,**v} for v in checked]
for folder in (OUT/'qwen-core-calibration').glob('period*'):
    hashes=dict(re.findall(r'func_id=(\d+).*?code_fnv1a64=(0x[0-9a-f]+)',(folder/'host.log').read_text()))
    assert hashes==kernels
(OUT/'audit.json').write_text(json.dumps(audit,indent=2))
labels={'shared':'A · Shared','structure':'B · infrastructure','C_guarded':'C guard · default only','D_guarded':'D guard · default only'}
rows=[]
for name,v in summary[3500]['policies'].items():
    rows.append(f"| {labels[name]} | 60 | {v['median_ms']:.6f} | {v['p95_ms']:.6f} | {v['median_relative_shared_percent']:+.2f}% |")
table='| Policy | n | median ms | p95 ms | vs Shared median |\n|---|---:|---:|---:|---:|\n'+'\n'.join(rows)
held=[]
for seq in (2250,5000):
    a,b=[summary[seq]['policies'][x] for x in ('shared','structure')]
    held.append(f"| {seq} | {a['median_ms']:.6f} / {a['p95_ms']:.6f} | {b['median_ms']:.6f} / {b['p95_ms']:.6f} | {b['median_relative_shared_percent']:+.2f}% |")
heldtable='| seq_len | A median / p95 ms | B median / p95 ms | B vs A median |\n|---|---:|---:|---:|\n'+'\n'.join(held)
b=summary[3500]['policies']['structure']
ci=b['paired_period_bootstrap_95_percent']
v=diag['shared_visibility'];bud=diag['shared_visibility_budget_sensitivity']
large=next(x for x in bud['budgets'] if x['max_fanin']==128)
cost=diag['policies']['structure']['queue_pop_cost_including_empty_and_batch_calls']
common=f'''{table}

{heldtable}
'''
zh=f'''# 流式 TMR：可见性、标定与基础设施成本实测

日期：2026-11-27。**本报告不是主动 A–D 调度收益对照：C/D 板上评分与资源日历未接入主动改派，仅运行空准入表下的默认路径。** 本轮得到 A/B 成本、真实前缀可见性、菱形正确性和标定结果；不能据此判断主动 C/D 或训练型 QKV 的收益。

{common}

目标配置：B16、40 层、seq_len=3500、seed=1234；2250/5000 为冻结后留出负载。每组 12 个时段，每时段 1 次预热和 5 次正式样本；共 720 次正式样本及 144 次预热全部保留。B 相对 A 的时段配对变化中位数为 {b['paired_period_median_percent']:+.2f}%，12 时段 bootstrap 95% 区间 [{ci[0]:+.2f}%, {ci[1]:+.2f}%]。p95 使用线性分位数。诊断、golden 和 pilot 均不混入正式样本；不能把 C/D 占位组之间的波动解释为模型收益。

同一 Ascend910B2（物理卡 3 / ACL 0）、同一二进制、5 S + 1 O、AICPU cpu_id 2–7；S 为 2–6，O 为 7，计算资源为 24 AIC/48 AIV。DFX 编译为 1、调度 profiling 为 0，正式运行关闭详细事件。所有正式组与 Qwen 配对标定的 37 个实际加载计算 kernel 指纹相同。起点提交 `8ebbff77377175bbbdc8d8f7252897f3eb73e6f8`。未取得设备独占锁（工具缺失），不能用此轮直接重估旧卡 4 上初版慢 7.11% 的因果分解。

- **正确性**：三种序列长度 × 四组，共 12 次逐策略输出及全部 40 层 K/V cache golden 通过，每次先 NaN 污染输出和当前 token 的 K/V 行。55 项 native 测试、24 项 CPU 前缀测试、四个板上菱形及六组同步/MIX/multi-block/DUMMY/提前派发共存回归通过。每轮符合条件任务数与实际 block 领取数一致；这不是完整执行账本对所有资源组合的证明。
- **实际覆盖**：目标负载 B 每轮 {b['eligible_tasks_total']//60} 个符合条件任务走新增领取基础设施，60 轮共 {b['actual_eligible_block_claims_total']} 次成功 block 领取。替代候选比较、改派和亲和队列发布均为 0；不是“模型覆盖 100%”。所有任务维持默认领取，没有测到主动选侧或迁移收益。
- **最终构建的流式可见性**：Shared 诊断的 {v['observed_normal_single_block_claims']} 次普通单 block 队列观察中，预算内可用汇合 {v['usable_bounded_join']} 次（{100*v['usable_fraction']:.3f}%）；完整登记但入度超预算 {v['complete_fanin_over_budget']} 次。只用当时已记录索引和 epoch，事后将入度预算从 4 放宽到 128，可用 {large['usable']} 次（{100*large['fraction']:.2f}%）。这是独立诊断与预算敏感性，分母不是实际三队列候选比较机会，也没有测排序改变率。
- **标定与时间口径**：同地址、逐条件 drain 的重用微基准保留 6048 个任务；实际 Qwen 33→34 完成 504 次 CPU 代理诊断及 144 次 AICore 补充诊断。后者 23,040 个选定任务均满足因果时间不等式，每次全量核对 10,604 个计算任务 ID、19,208 个 block 调用。CPU 发布代理相对 AICore 区间的差值中位数/p95 为 4.72/13.86 µs，pending 提升代理为 0.02/8.295 µs，两者分开。实际物理核匹配后两侧六时段差异均未稳定；进一步占用匹配仅留 43 对 tile，符号仍变化，不能当作可靠局部性先验。
- **开销与训练**：B 每 32 次采样的队头领取调用耗时中位数/p95 为 {cost['median_us']:.3f}/{cost['p95_us']:.3f} µs，n={cost['n']}，包含空调用和批领取，不是模型推理预算样本。B 同时增加元数据登记，端到端回退不能全部归因于队列。跨度表独立留出 MAE/p95 绝对误差为 1.790/5.640 µs；该标签不等于两动作净收益。尚无独立 a0/a1 净收益配对标定；本轮未训练 E/F、没有 AICPU QKV 影子预算样本，也未证明 QKV 相对线性有净收益。旧轮次权重的目标不同，不直接移植为动作残差。

下一轮优先检查**如何在控制元数据热路径开销的同时，为高入度可见汇合维护有界前沿摘要**。当前 4 入度预算人为屏蔽了大量已经可见的信息；基础设施已有成本，需先使候选与收益可测，再补独立动作标定和主动 C/D，随后才谈学习残差。

[机制](MECHANISM.zh-CN.md) · [English](REPORT.en.md) · [复现](REPRODUCE.md) · [计时原始样本](final-formal-3500/samples.json) · [冻结配置](formal-freeze.json) · [诊断汇总](final-diagnose-3500/diagnostic-summary.json) · [标定](qwen-core-calibration/summary.json) · [审计](audit.json) · [实现 diff](runtime-and-tests.patch)。原始日志、失败与过渡版本保留；模拟 scene 因缺 g++-15 未运行，CPU/native 与实卡证据单独报告。当前仅本地交付，未推送。
'''
en=f'''# Streaming TMR: visibility, calibration and infrastructure cost

2026-11-27. **This is not an active A–D scheduling-benefit comparison. C/D resource calendars and scoring are not connected to active NPU overrides; the empty-admission build takes the default path.** The measurements establish A/B cost, factual prefix visibility, diamond correctness and calibration limits, not analytical or learned benefit.

{common}

Target: batch 16, 40 layers, seq_len 3500, seed 1234; sealed heldouts 2250/5000. Each arm has 12 periods × (1 retained warm-up + 5 ordinary samples). All 720 ordinary samples and 144 warm-ups are retained. B's median paired-period change is {b['paired_period_median_percent']:+.2f}%, bootstrap 95% interval [{ci[0]:+.2f}%, {ci[1]:+.2f}%] over 12 independent periods. p95 uses linear quantiles. Golden, pilot and diagnostic measurements are excluded. Differences among default-only C/D arms are not model effects.

One Ascend910B2, physical card 3 / ACL 0; one binary, five S plus one O, actual AICPU IDs 2–6 for S and 7 for O, 24 AIC/48 AIV. DFX=1, scheduler profiling=0 at build; detailed events disabled for ordinary measurements. All timing arms and real-kernel placement calibration load identical fingerprints for 37 compute kernels. Starting commit: `8ebbff77377175bbbdc8d8f7252897f3eb73e6f8`. Device locking was unavailable. This cannot re-estimate the historical 7.11% regression's causal decomposition measured on card 4.

- Correctness: all 12 per-arm output/full 40-layer K/V golden checks pass, with output/current-token K/V rows poisoned before every arm. Native tests 55/55, CPU prefix tests 24/24, four onboard diamonds and six coexistence regression groups pass. Eligible counts match successful block claims on every run; this is not a complete execution-ledger proof for every resource combination.
- Coverage: target B processes {b['eligible_tasks_total']//60} eligible tasks per run through new claiming infrastructure, {b['actual_eligible_block_claims_total']} successful block claims across its 60 ordinary runs. Alternative comparisons, overrides and affinity-queue publications are zero. Infrastructure coverage is not model coverage. No active placement benefit is observed or evaluated.
- Final-build visibility: among {v['observed_normal_single_block_claims']} ordinary single-block Shared queue observations, {v['usable_bounded_join']} ({100*v['usable_fraction']:.3f}%) have a usable join within the fanin-4 budget; {v['complete_fanin_over_budget']} have a registered join exceeding that budget. Reinterpreting only captured successor indices/epochs at budget 128 yields {large['usable']} ({100*large['fraction']:.2f}%). This diagnostic denominator is not actual three-head comparison opportunities; ranking-change rate is unmeasured.
- Calibration: 6048 same-address/drained custom reuse tasks; 504 sparse CPU-proxy and 144 AICore-profiled real Qwen placement runs. All 23,040 selected AICore records satisfy causal timestamp inequalities. Every profiled launch matches the independent baseline's 10,604 compute task IDs and 19,208 block invocations. CPU publication proxy minus AICore interval median/p95 is 4.72/13.86 µs; pending-promotion proxy 0.02/8.295 µs, kept separate. Physical-core-matched contrasts change direction across six periods; occupancy matching retains only 43 tile pairs with mixed signs. No reliable locality class is admitted.
- Cost/models: every-32nd queue-pop invocation median/p95 {cost['median_us']:.3f}/{cost['p95_us']:.3f} µs, n={cost['n']}; includes empty and batch calls, not neural inference latency. B also enables metadata registration, so the regression cannot be attributed to queue operations alone. Factual span calibration heldout MAE/p95 absolute error 1.790/5.640 µs does not calibrate two-action gains. Independent a0/a1 gain interventions are missing. New E/F models were not trained, QKV shadow inference was not sampled, and net advantage over linear is unestablished. Earlier weights predict a different target.

Next priority: bounded frontier summaries for already-visible high-fanin joins, while containing metadata hot-path cost. The fanin-4 budget currently hides most usable information. Establish candidate/gain measurements and independent action calibration before active C/D or learned residual promotion.

[Mechanism](MECHANISM.en.md) · [中文](REPORT.zh-CN.md) · [Reproduce](REPRODUCE.md) · [Raw samples](final-formal-3500/samples.json) · [Freeze](formal-freeze.json) · [Diagnostics](final-diagnose-3500/diagnostic-summary.json) · [Calibration](qwen-core-calibration/summary.json) · [Audit](audit.json) · [Implementation diff](runtime-and-tests.patch). Failures and superseded attempts remain. Simulation scene compilation is blocked by absent g++-15; CPU/native and real-device evidence are separate. Local delivery only; not pushed.
'''
(OUT/'REPORT.zh-CN.md').write_text(zh)
(OUT/'REPORT.en.md').write_text(en)
print('AUDIT',json.dumps({k:v for k,v in audit.items() if k!='golden_checks'}))
print(table)
print(heldtable)
