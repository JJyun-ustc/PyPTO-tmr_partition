#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Export every interleaved result and independent control-path diagnostic without selecting a winning policy."""

import argparse
import csv
import json
from pathlib import Path

LABELS = ("Bgemm", "Scale", "Uniform", "Skew")
STAGES = ("v8", "v12", "v13", "v13-no-pickup")


def phase_ns(diagnostic, name, include_o=False):
    return sum(
        rows.get(name, {}).get("ns", 0)
        for owner, rows in diagnostic["stages"].items()
        if owner.startswith("S") and (include_o or owner != "S5")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--document", type=Path, required=True)
    args = parser.parse_args()
    root = args.study.resolve()
    report = json.loads((root / "summary.json").read_text())
    stages = report["stages"]
    assert tuple(stages) == STAGES
    for stage, row in stages.items():
        for label, timing in row["timing"]["cases"].items():
            assert timing["n"] == 32
            for baseline in ("v8", "v12", "v13-no-pickup"):
                ref = stages[baseline]["timing"]["cases"][label]
                timing["vs_" + baseline + "_pct"] = 100 * (timing["median_ms"] / ref["median_ms"] - 1)
    report.update(
        ordinary_samples=512,
        passed_golden=16,
        independent_diagnostics=16,
        parameters=json.loads((root / "frozen-parameters.json").read_text()),
    )
    for name in ("unit-validation", "source-preservation", "code-review", "architecture"):
        p = root / (name + ".json")
        if p.exists():
            report[name] = json.loads(p.read_text())
    p = root / "final-validation/results.json"
    if p.exists():
        report["hardware_validation"] = json.loads(p.read_text())
    (root / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (root / "results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("version", "case", "n", "median_ms", "p95_ms", "block_medians_ms"))
        for stage in STAGES:
            for label in LABELS:
                row = stages[stage]["timing"]["cases"][label]
                writer.writerow(
                    (stage, label, row["n"], row["median_ms"], row["p95_ms"], json.dumps(row["batch_median_ms"]))
                )
    lines = [
        "# v13：O预测预留、S轻量修正与受限领取",
        "",
        "本版按请求将完整压力评分移回O，移除逐FIN压力更新，保留任务级实际分布亲和和受限跨侧领取。",
        "不实现范围级亲和、kernel时长预测或价格学习。所有配置在正式实验前固定，保留全部结果，不选最好版本。",
        "",
        "## 实现与固定实验口径",
        "",
        "O每次Auto创建完整评分；两侧leader每N=64循环仅汇总本侧S，O每M=64创建读取两侧摘要。",
        "每个writer同时记录对两个队列侧的变化，按writer归属汇总，不按队列侧重复汇总。O行独立于S汇总。",
        "首次ready以单写者版本事务同时R[p]+=u、E[q]+=u；领取C[原队列]+=x，重入队和完成不重复记账。",
        "O使用A_current−R_O_current−ΣR_S_snapshot预测预留，以及O当前E−C和S快照E−C队列量。",
        "差值保持有符号，评分才截断；摘要替换旧累计快照，本线程净增量用吸收水位校正。",
        "O所在侧摘要独立附带O行副本供S查看，O评分忽略副本并使用当前O值，防止重复。",
        "",
        "固定策略价格：本地250、跨侧750、队列2000且权重1/2、未来预留上限500且最多一波兼容容量。",
        "这些是ns尺度的策略价格，不是本轮实测时延。当前等待任务只进未来候选项，立即ready任务只进队列候选项；A只增加一次。",
        "S默认接受预测，实际完成票优势严格超过1/4或创建票优势严格超过1/2才允许在无明显拥塞时改侧。",
        "S ready不刷新摘要、不扫描fanin/writer、不完整评分；逐FIN模型账移除，但每条WAIT仍累计实际票。",
        "前驱按两侧实际block分布归一化投票，不记录范围、不映射消费者读取区域，仍须整任务完成WAIT。",
        "",
        "本侧正常派发后，leader每32次兼容IDLE空缺机会探测；对侧提示有Auto工作即可尝试，不要求一波。",
        "实际尝试后冷却128次，每次最多min(8,兼容IDLE容量)个block；失败返回。严格、sync、DUMMY、失败谓词排除，MIX检查使用资源完整兼容。",
        "",
        "物理卡0/逻辑0，910B2，PCIe 0000:C1:00.0；S0/S1在CPU2/3，S2/S3/S4在CPU4/5/6，O在CPU7。",
        "task-submit不可用，未持独占设备锁；board查询与CANN配置独立核对a2a3，未做O位置对照。",
        "v8、v12、新版v13、仅源码常量关闭领取的v13-no-pickup均独立worktree构建，同卡重测。",
        "",
        "采用4组Williams版本顺序，每组每版本8轮；每版本在每个顺序位置出现一次，每个有向相邻版本对出现一次。",
        "用例顺序逐组轮转。每项32轮，全部进程首轮与全部样本计入，共512个普通device_wall样本。",
        "另外16次golden和16次独立诊断；预检查不计入正式统计。计时为设备一次运行墙钟，不含编译、golden或数据传输。",
        "每个版本每用例4次进程启动，其首轮波动也计入p95，不能与之前两进程30轮协议的p95直接混比。",
        "",
        "## 完整性能结果",
        "",
        "中位数 / p95，单位ms。负数变化代表更快。",
        "",
        "|版本|BGEMM Case0|BGEMM Scale|PA Uniform|PA Skew|",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        rows = stages[stage]["timing"]["cases"]
        lines.append(
            "|" + stage + "|" + "|".join(f"{rows[c]['median_ms']:.5f} / {rows[c]['p95_ms']:.5f}" for c in LABELS) + "|"
        )
    lines += ["", "|用例|v13相对v8|v13相对v12|v13开启领取相对关闭|", "|---|---:|---:|---:|"]
    for label in LABELS:
        r = stages["v13"]["timing"]["cases"][label]
        lines.append(f"|{label}|{r['vs_v8_pct']:+.2f}%|{r['vs_v12_pct']:+.2f}%|{r['vs_v13-no-pickup_pct']:+.2f}%|")
    lines += [
        "",
        "本轮未保住v8普通负载性能：四项中位数均更慢。相对v12只有Skew中位数改善，其他三项回退。",
        "领取开启相对关闭的变化约为−0.68%、−0.01%、+1.43%、+0.73%，没有证明稳定净收益覆盖其成本。",
        "以上是固定候选的一次完整实验，不删除异常轮、不根据中途成绩调整参数。",
        "同一交错组的各版本中位数保留在summary.json与CSV，不能把32个进程内样本视为32次完全独立环境重复。",
        "",
        "## 独立O/S诊断",
        "",
        "|版本/用例|O提交ms|O完整评分ms|S ready累计ms|S反馈累计ms|WAIT票辅助累计ms|远端block|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        for label in LABELS:
            d = stages[stage]["diagnostics"][label]
            score = d["stages"].get("O", {}).get("score", {}).get("ns", 0) / 1e6
            votes = phase_ns(d, "affinity_votes", True) / 1e6
            remote = sum(row["remote"] for row in d["ready_feedback"])
            lines.append(
                f"|{stage}/{label}|{d['O_submit_ms']:.5f}|{score:.5f}|{d['S_ready_binding_sum_ms']:.5f}|"
                f"{phase_ns(d, 'resource_feedback') / 1e6:.5f}|{votes:.5f}|{remote}|"
            )
    lines += [
        "",
        "Scale诊断中v12→v13：S ready累计0.56692→0.51582ms，但S反馈0.33328→0.77136ms，O提交5.94228→6.36470ms。",
        "本版每N循环刷新所有类别的S缓存，低频并不保证共享读取总成本下降；还需区分调度循环次数与每次刷新成本。",
        "O完整评分累计0.26798ms和WAIT辅助0.08070ms是定位线索，不能单独解释全部普通耗时退化。",
        "v8/v12没有新增的O score/WAIT辅助计时区间，表中0表示未单独采集，不表示没有这些成本。",
        "WAIT辅助区间包含读取、任务级分布归一化、原子累计和计时开销，不是纯原子指令延迟。",
        "S5表示专用O账本writer；WAIT辅助总和包含O在wiring时处理已完成前驱的成本。",
        "各S累计区间并行、完成/resolve/ready有嵌套，不能相加还原普通device_wall；细粒度诊断本身也会扰动执行。",
        "",
        "|版本/用例|S完成区间累计ms|S领取阶段包络ms|探测|实际尝试|远端block|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        for label in LABELS:
            d = stages[stage]["diagnostics"][label]
            probes = sum(row["probes"] for row in d["ready_feedback"])
            attempts = sum(row["attempts"] for row in d["ready_feedback"])
            remote = sum(row["remote"] for row in d["ready_feedback"])
            lines.append(
                f"|{stage}/{label}|{d['S_phase_sum_ms']['complete']:.5f}|"
                f"{phase_ns(d, 'cross_pickup') / 1e6:.5f}|{probes}|{attempts}|{remote}|"
            )
    lines += [
        "",
        "关闭领取版仍保留阶段计时壳，包络非零不表示执行了领取检查；是否实际检查以probes/attempts为准。",
        "v13 Scale/Uniform/Skew单次诊断分别101/49/70次尝试，仅领取9/3/4个单block任务。",
        "存在不少无效尝试；提示陈旧与并发领取是可能原因，尚未逐次分类。",
        "开启/关闭领取使用相同评分与账本，唯一源码差异见pickup-only.patch；比较的是检查开销和调度收益的净效果。",
        "不同版本的完成区间还受依赖拓扑、并发度和插桩影响，不把一次诊断当作已隔离的因果证明。",
        "",
        "## 正确性、守恒与解释边界",
        "",
        "全部性能配置golden通过；每次调用前后检查运行库指纹，计算kernel代码哈希在全部对照一致。",
        "v13终态每侧A=R、E=C，全局ΣC=Σ实际派发；跨侧时不要求每侧C=实际派发，队列侧和执行侧有不同语义。",
        "CPU用例覆盖R/E事务、并发读取、writer归属、O独立账、摘要替换、负数差值、阈值及范围唯一领取。",
        "194个CPU用例、15个板上正确性用例及DFX关闭构建通过；证据在final-validation和unit-validation.json。",
        "板上多block回归未触发远端分支，范围唯一性由CPU并发用例覆盖；不能据此声称端到端覆盖或收益。仿真因缺g++-15未运行。",
        "",
        "即使完成路径不再逐FIN记账，WAIT边票累计仍有额外成本，已独立报告。没有范围亲和或时长预测，",
        "任务级block数无法区分不等长kernel，MIX与纯AIC/AIV共享资源也未完整建模。",
        "有界预留饱和后无法表达更多未来压力；领取后队列压力即减少，O不预测已领取长任务的剩余执行时间。",
        "异步摘要有滞后，提示不是硬资源预留，真实派发依赖原资源跟踪与原子认领。",
        "小幅收益须结合四组中位数、p95与无独占锁条件判断，不宣称普遍加速。",
        "",
        "## 复现与原始证据",
        "",
        "运行库根目录：examples/a2a3/tensormap_and_ringbuffer/benchmark_orchestrator_pressure。",
        "prepare_study.py冻结源码、二进制指纹和参数；run_interleaved.py接收四个独立worktree路径；",
        "analyze_study.py验证拓扑/守恒/计算代码并生成摘要；build_report.py生成本报告。",
        "每个worktree独立构建，不复制其他worktree的编译产物；构建/测试并行度2，普通实验的OMP/MKL/OpenBLAS线程数统一2。",
        "summary.json保存全部逐轮样本与诊断，results.csv保存统计，invocations.json保存实际版本顺序与命令。",
        "各版本保留host日志、研究设备日志、swimlane/deps捕获与精确候选源码；发布排除二进制、缓存、venv及泛化驱动日志。",
        "",
    ]
    args.document.parent.mkdir(parents=True, exist_ok=True)
    args.document.write_text("\n".join(lines))
    (root / "tables.md").write_text(
        "\n".join(lines[lines.index("## 完整性能结果") : lines.index("## 正确性、守恒与解释边界")]).rstrip() + "\n"
    )


if __name__ == "__main__":
    main()
