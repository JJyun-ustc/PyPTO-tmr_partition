#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Export complete block-pressure study tables and Chinese experimental findings."""

import argparse
import csv
import json
from pathlib import Path

LABELS = ("Bgemm", "Scale", "Uniform", "Skew")
STAGES = ("official-before", "v8", "v11", "v12", "official-after")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--document", type=Path, required=True)
    args = parser.parse_args()
    root = args.study.resolve()
    report = json.loads((root / "summary.json").read_text())
    stages = report["stages"]
    assert tuple(stages) == STAGES
    assert all(row["timing"]["cases"][label]["n"] == 30 for row in stages.values() for label in LABELS)
    report["ordinary_samples"] = 600
    report["golden_invocations"] = report["diagnostic_invocations"] = 20
    report["kernel_time_model"] = False
    report["preflight_failed_invocations"] = 1
    report["parameters"] = {
        "wave_scale": 256,
        "remote_price_waves": 0.5,
        "publication_price_waves": 0.125,
        "switch_margin_waves": 0.0625,
        "weak_confidence": 0.25,
        "ready_refresh_budget": 16,
        "feedback_loop_interval": 64,
        "cache_age_check_us": 20,
        "pickup_probe_loops": 32,
        "pickup_cooldown_loops": 128,
        "pickup_range_limit": 8,
    }
    for name in ("unit-validation", "source-preservation", "architecture", "code-review"):
        path = root / (name + ".json")
        if path.exists():
            report[name] = json.loads(path.read_text())
    path = root / "final-validation/results.json"
    if path.exists():
        report["hardware_validation"] = json.loads(path.read_text())
    drift = {}
    for label in LABELS:
        before = stages["official-before"]["timing"]["cases"][label]["median_ms"]
        after = stages["official-after"]["timing"]["cases"][label]["median_ms"]
        drift[label] = 100 * (after / before - 1)
        current = stages["v12"]["timing"]["cases"][label]
        for baseline in ("v8", "v11"):
            ref = stages[baseline]["timing"]["cases"][label]
            current["vs_" + baseline + "_pct"] = 100 * (current["median_ms"] / ref["median_ms"] - 1)
    report["official_drift_pct"] = drift
    (root / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (root / "results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("stage", "case", "n", "median_ms", "p95_ms", "batch_a_median_ms", "batch_b_median_ms"))
        for stage in STAGES:
            for label in LABELS:
                row = stages[stage]["timing"]["cases"][label]
                writer.writerow(
                    (
                        stage,
                        label,
                        row["n"],
                        row["median_ms"],
                        row["p95_ms"],
                        row["batch_median_ms"]["a"],
                        row["batch_median_ms"]["b"],
                    )
                )
    lines = [
        "# v12：亲和代价、未完成压力与跨事件缓存的组合实验",
        "",
        "本轮按用户要求将修改一次组合，仅保留BGEMM与Paged Attention性能负载；不实现预计block执行时间。",
        "官方Shared、v8、v11均为独立对照，不根据本轮性能结果调整参数。正数表示更慢，负数表示更快。",
        "",
        "## 实现与实验口径",
        "",
        "- v12保留O生成亲和、首次ready最终绑定；加入多数亲和票和前驱完成后的真实分布更新。",
        "- 评分统一以波数表达负载与交接策略价格，删除四分之一波直接覆盖亲和；没有时长字段、时长采样或学习。",
        "- Q统计排队block，I统计已领取未FIN的block等价量；同侧领取不使总压力下降，跨侧领取转移到实际侧。",
        "- 缓存不因前驱事件换批失效；16次ready决策预算与低频年龄检查，单写者账本加摘要吸收水位。",
        "- 普通Auto多block可按范围远端领取，每次最多8个且受本S空闲资源限制；严格、sync-start、DUMMY排除。",
        "- 官方Shared是冻结上游独立worktree，src唯一差异是线程上限4→6，调度与依赖代码保持官方原样。",
        "",
        "每组先golden，再15轮正序与15轮反序，全部30轮计入中位数和p95；不选最好一轮。",
        "5阶段×4用例，共600个普通样本、20次golden与20次独立插桩诊断。",
        "反序指同一版本内的用例顺序反转；版本按Shared→v8→v11→v12→Shared固定顺序，未做版本交错随机化。",
        "device_wall是设备一次运行从开始到结束的墙钟时间，包含O/S与计算及其重叠，不包括编译、golden或H2D/D2H。",
        "O提交等指标来自另外的诊断运行，不能与普通device_wall相加。",
        "",
        "本容器当前物理卡0/逻辑0，Ascend910B2，PCIe 0000:C1:00.0；与v11历史报告的卡5不同，所有对照本轮重测。",
        "S0/S1在CPU2/3，S2/S3/S4在CPU4/5/6，O在CPU7；不做O位置对照。",
        "task-submit不可用，未持有独占设备锁。默认硬件检查不能解析该驱动的Chip Name=910B2且无NPU Name格式，",
        "已用成功board查询及CANN的Ascend910B2.ini/Short_SoC_version=Ascend910B独立核对架构。",
        "",
        "## 普通计时结果",
        "",
        "中位数 / p95，单位ms。",
        "",
        "|策略|BGEMM Case0|BGEMM Scale|PA Uniform|PA Skew|",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        values = stages[stage]["timing"]["cases"]
        lines.append(
            "|"
            + stage
            + "|"
            + "|".join(f"{values[c]['median_ms']:.5f} / {values[c]['p95_ms']:.5f}" for c in LABELS)
            + "|"
        )
    lines += [
        "",
        "|用例|v12相对v11中位数|v12相对v8中位数|v12相对末次官方Shared|官方首尾漂移|",
        "|---|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        row = stages["v12"]["timing"]["cases"][label]
        lines.append(
            f"|{label}|{row['vs_v11_pct']:+.2f}%|{row['vs_v8_pct']:+.2f}%|{row['vs_frozen_shared_pct']:+.2f}%|{drift[label]:+.2f}%|"
        )
    lines += [
        "",
        "小幅差异需结合首尾漂移、两批中位数及无独占锁条件判断，不能据此宣称稳定加速。",
        "",
        "## 单次插桩诊断",
        "",
        "|版本/用例|O提交ms|S ready绑定累计ms|S摘要刷新次数|S ready决策数|捕获跨侧WAIT边|远端领取block|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in ("v8", "v11", "v12"):
        for label in LABELS:
            d = stages[stage]["diagnostics"][label]
            refreshes = sum(row["refreshes"] for row in d["ready_feedback"] if row["writer"] != 5)
            decisions = sum(
                rows.get("ready_binding", {}).get("calls", 0)
                for owner, rows in d["stages"].items()
                if owner.startswith("S") and owner != "S5"
            )
            remote = sum(row["remote"] for row in d["ready_feedback"])
            lines.append(
                f"|{stage}/{label}|{d['O_submit_ms']:.5f}|{d['S_ready_binding_sum_ms']:.5f}|{refreshes}|{decisions}|{d['cross_side_captured_wait_edges']}|{remote}|"
            )
    lines += [
        "",
        "v12在本次Scale/Uniform/Skew诊断中减少摘要刷新及S ready绑定时间，但Skew普通中位数与p95仍变差。",
        "这说明局部控制路径改善不能直接推导整体完成时间改善；本轮仍未追平v8。",
        "Skew诊断中v12的AIC队列侧分布为298/1062，AIV为367/993，比v11更偏向侧1；",
        "未建模kernel时长，不能仅凭block数证明真实计算工作是否平衡。固定亲和价格与状态滞后是后续排查候选，尚非已隔离的原因。",
        "四个v12独立性能诊断的远端领取量均为0，不能用本轮性能说明跨侧领取收益；多block的机制验证另看正确性回归。",
        "各线程区间有重叠；诊断开启细粒度计时并改变开销，只能用于定位线索，不能据此分解普通总时间。",
        "WAIT边统计仅使用两端都只在一侧执行的捕获记录，可能包含参数依赖重复项，不等于精确交接事件数。",
        "",
        "## 验证与解释边界",
        "",
        "所有性能用例均通过golden；各阶段前后运行库指纹不变，计算kernel代码哈希在所有对照一致。",
        "v12终态按原队列侧enqueued=claimed，按实际侧accepted=executed=completed_units/6；跨侧时不要求逐侧claimed=executed。",
        "191个C++用例覆盖缓存跨事件复用、水位吸收、未完成压力、MIX分数完成与并发范围唯一领取；DFX关闭构建通过。",
        "15个板上正确性回归全部通过，另列于final-validation/results.json，不将其作为新增性能benchmark。",
        "这些多block硬件回归未观察到远端领取，跨侧多block范围派发尚无板上触发覆盖；范围唯一领取由并发CPU用例验证。",
        "v11首次启动被Python绑定提交标记检查拦截，尚未接触设备；在自身worktree重建绑定后重新完整运行，失败记录保留在preflight。",
        "仿真场景未运行：配置的g++-15缺失。",
        "",
        "本轮是组合版本对比，不能把差异归因于某个单项改动。前驱实际分布累计增加每条WAIT边一次原子操作，",
        "完成账本也增加本地更新，任务放置元数据从16增至40字节；减少摘要读取不保证总控制成本下降。交接波数价格尚未标定。",
        "同类block仍按等价工作建模，MIX与纯AIC/AIV的资源竞争、其他线程变化滞后、blocked未来释放尚未完整建模。",
        "BGEMM/PA此处主要是单block任务，不能用这轮性能结果证明多block领取的收益；多block依靠额外正确性回归及并发范围测试验证。",
        "",
        "## 复现与证据",
        "",
        "入口为examples/a2a3/tensormap_and_ringbuffer/benchmark_ready_pressure。先在各自worktree构建运行库和绑定，",
        "官方Shared执行run_study.py official-before，再用run_matrix.py传入v8/v11/官方worktree路径顺序完成其余阶段。",
        "每个策略须使用自身build/lib中的运行库，不复制其他worktree编译产物。构建与测试并行度均为2。",
        "",
        "summary.json保存全部逐轮样本、诊断摘要与验证；results.csv保存中位数/p95及分批结果；",
        "各阶段保留golden、计时日志、设备研究日志和swimlane/deps捕获；v12-artifacts保存精确源码和本地二进制清单。",
        "Git发布源码、指纹与原始研究结果，排除二进制、venv、缓存及泛化驱动日志。旧v8/v11分支保留不改。",
        "",
    ]
    args.document.parent.mkdir(parents=True, exist_ok=True)
    args.document.write_text("\n".join(lines))
    (root / "tables.md").write_text(
        "\n".join(lines[lines.index("## 普通计时结果") : lines.index("## 验证与解释边界")]) + "\n"
    )


if __name__ == "__main__":
    main()
