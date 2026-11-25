#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Render the frozen experiment's bilingual one-page conclusions and sample plot."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from round2_analyze import counters

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
LABELS = {"shared": "Shared", "original": "Original QKV", "improved": "Improved QKV", "trained": "Learned QKV"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    root = args.root
    result = json.loads((root / "measure/summary.json").read_text())
    coverage = json.loads((root / "coverage.json").read_text())
    diag = json.loads((root / "diagnostics-summary.json").read_text())
    metrics = json.loads((root / "models/metrics.json").read_text())["test"]
    shadow = json.loads((root / "models/aicpu-shadow.json").read_text())
    replay = json.loads((root / "models/replay-validation.json").read_text())
    ranking = json.loads((root / "models/replay-ranking.json").read_text())
    mix = counters(root / "regressions-final/qkv_sync_coexist")[0]
    chain = counters(root / "regressions-final/qkv_mix_chain")[0]
    for mode in result:
        g = json.loads((root / "golden" / mode / "golden/result.json").read_text())
        assert g["passed"] and g["golden"]
        assert len(result[mode]["all_ms"]) == 30
    tables = "| Policy | n | Median ms | p95 ms | Median vs Shared | p95 vs Shared |\n|---|---:|---:|---:|---:|---:|\n"
    for mode, r in result.items():
        tables += (
            "| "
            f"{LABELS[mode]}"
            " | "
            f"{r['n']}"
            " | "
            f"{r['median_ms']:.6f}"
            " | "
            f"{r['p95_ms']:.6f}"
            " | "
            f"{r['vs_shared_percent']['median_ms']:+.2f}"
            "% | "
            f"{r['vs_shared_percent']['p95_ms']:+.2f}"
            "% |\n"
        )
    ctable = (
        "| Policy | Eligible | Assigned | Coverage | Multi-task batches | Fallback | "
        "Decision mean µs (batches) |\n|---|---:|---:|---:|---:|---:|---:|\n"
    )
    for mode, d in coverage.items():
        r = d["formal"]
        ctable += (
            "| "
            f"{LABELS[mode]}"
            " | "
            f"{r['eligible']}"
            " | "
            f"{r['assigned']}"
            " | "
            f"{r['coverage_percent']:.2f}"
            "% | "
            f"{r['multi_batches']}"
            " | "
            f"{sum(r['fallback'].values())}"
            " | "
            f"{r['mean_decision_us']:.3f}"
            " ("
            f"{r['batches']}"
            ") |\n"
        )
    train_delta = 100 * (metrics["attention"]["composite_mae_us"] / metrics["linear"]["composite_mae_us"] - 1)
    orig = diag["diagnostics/original/sample"]
    improved = diag["diagnostics/improved/sample"]
    wait0 = orig["timeline"]["ready_to_pop_us"]
    wait1 = improved["timeline"]["ready_to_pop_us"]

    def stages(d):
        return ", ".join((f"{k}={v.get('mean', 0):.3f}" for k, v in d["phase_us"].items()))

    held = [r for r in replay if "3500" in r["run_id"]]
    errors = ", ".join(f"{r['relative_error_percent']:+.2f}%" for r in held)
    rank = ranking["attention"]
    linear_rank = ranking["linear"]
    cf_count = rank["n"]
    nt = rank["non_tie_n"]
    extra = (
        100 * (result["trained"]["median_ms"] / result["improved"]["median_ms"] - 1) if "trained" in result else None
    )
    modeltable = "| Model | Wait MAE µs | Runtime-proxy MAE µs | Composite MAE µs |\n|---|---:|---:|---:|\n"
    for m in ("base", "linear", "attention"):
        r = metrics[m]
        modeltable += f"| {m} | {r['mae_wait_us']:.3f} | {r['mae_run_us']:.3f} | {r['composite_mae_us']:.3f} |\n"
    shared_short = json.loads((root / "probe/shared/short/result.json").read_text())["all_ms"][1:]
    original_short = json.loads((root / "probe/original/short/result.json").read_text())["all_ms"][1:]
    base_short = json.loads((root / "probe/base_only/short/result.json").read_text())["all_ms"][1:]
    central_short = json.loads((root / "probe-control/central_shared/short/result.json").read_text())["all_ms"][1:]
    controls = (
        "Shared "
        f"{statistics.median(shared_short):.5f}"
        "; original "
        f"{statistics.median(original_short):.5f}"
        "; attention-off "
        f"{statistics.median(base_short):.5f}"
        "; centralized Shared publication "
        f"{statistics.median(central_short):.5f}"
        " ms"
    )
    template = Path(__file__).resolve().parents[4] / "docs/zh-cn/experiments/qkv-round2-report.template.md"
    zh = template.read_text().format_map(
        {
            "p0": format(tables, ""),
            "p1": format(controls, ""),
            "p2": format(wait0["mean"], ".2f"),
            "p3": format(wait0["n"], ""),
            "p4": format(wait1["mean"], ".2f"),
            "p5": format(wait1["n"], ""),
            "p6": format(stages(orig), ""),
            "p7": format(stages(improved), ""),
            "p8": format(mix["shape_assigned"][2], ""),
            "p9": format(mix["shape_eligible"][2], ""),
            "p10": format(chain["shape_assigned"][2], ""),
            "p11": format(chain["shape_eligible"][2], ""),
            "p12": format(
                dict(
                    zip(
                        [
                            "full",
                            "snapshot",
                            "no_side",
                            "budget",
                            "oversize",
                            "uninitialized",
                            "stale",
                            "busy",
                            "version",
                        ],
                        chain["fallback"],
                    )
                ),
                "",
            ),
            "p13": format(ctable, ""),
            "p14": format(metrics["linear"]["composite_mae_us"], ".3f"),
            "p15": format(metrics["attention"]["composite_mae_us"], ".3f"),
            "p16": format(train_delta, "+.2f"),
            "p17": format(shadow["attention_us"]["p95"], ".3f"),
            "p18": format(shadow["linear_us"]["p95"], ".3f"),
            "p19": format(shadow["attention_us"]["n"], ""),
            "p20": format(format(extra, "+.2f") + "%" if extra is not None else "N/A", ""),
            "p21": format(errors, ""),
            "p22": format(cf_count, ""),
            "p23": format(nt, ""),
            "p24": format(linear_rank["non_tie_agreement"], ".2%"),
            "p25": format(rank["non_tie_agreement"], ".2%"),
        }
    )
    template = Path(__file__).resolve().parents[4] / "docs/experiments/qkv-round2-report.template.md"
    en = template.read_text().format_map(
        {
            "p0": format(tables, ""),
            "p1": format(controls, ""),
            "p2": format(wait0["mean"], ".2f"),
            "p3": format(wait0["n"], ""),
            "p4": format(wait1["mean"], ".2f"),
            "p5": format(wait1["n"], ""),
            "p6": format(stages(orig), ""),
            "p7": format(stages(improved), ""),
            "p8": format(mix["shape_assigned"][2], ""),
            "p9": format(mix["shape_eligible"][2], ""),
            "p10": format(chain["shape_assigned"][2], ""),
            "p11": format(chain["shape_eligible"][2], ""),
            "p12": format(ctable, ""),
            "p13": format(metrics["linear"]["composite_mae_us"], ".3f"),
            "p14": format(metrics["attention"]["composite_mae_us"], ".3f"),
            "p15": format(train_delta, "+.2f"),
            "p16": format(shadow["attention_us"]["p95"], ".3f"),
            "p17": format(shadow["linear_us"]["p95"], ".3f"),
            "p18": format(shadow["attention_us"]["n"], ""),
            "p19": format(
                "Learned versus improved fixed-QKV formal median changes by " + format(extra, "+.2f") + "%."
                if extra is not None
                else "Learned dispatch was excluded.",
                "",
            ),
            "p20": format(errors, ""),
            "p21": format(cf_count, ""),
            "p22": format(nt, ""),
            "p23": format(linear_rank["non_tie_agreement"], ".2%"),
            "p24": format(rank["non_tie_agreement"], ".2%"),
        }
    )
    (root / "REPORT.zh-CN.md").write_text(zh)
    (root / "REPORT.en.md").write_text(en)
    (root / "model-comparison.md").write_text(modeltable)
    summary_values = {
        "timing": tables,
        "coverage": ctable,
        "pending": f"{wait0['mean']:.2f}/{wait1['mean']:.2f}",
        "mix": f"{mix['shape_assigned'][2]}/{mix['shape_eligible'][2]}",
        "chain": f"{chain['shape_assigned'][2]}/{chain['shape_eligible'][2]}",
        "mae": (
            f"{metrics['linear']['composite_mae_us']:.3f}"
            " → "
            f"{metrics['attention']['composite_mae_us']:.3f}"
            " ("
            f"{train_delta:+.2f}"
            "%)"
        ),
        "shadow": f"{shadow['attention_us']['p95']:.3f}/{shadow['linear_us']['p95']:.3f}",
        "trained_delta": format(extra, "+.2f") + "%" if extra is not None else "N/A",
    }
    repository = Path(__file__).resolve().parents[4]
    for language, folder in (("zh-CN", "docs/zh-cn/experiments"), ("en", "docs/experiments")):
        template = repository / folder / "qkv-round2-conclusion.template.md"
        (root / f"CONCLUSION.{language}.md").write_text(template.read_text().format_map(summary_values))
    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        modes = list(result)
        colors = ["#687c95", "#d27660", "#3b9487", "#8c71b6"]
        rng = np.random.default_rng(1234)
        for i, mode in enumerate(modes):
            vals = result[mode]["all_ms"]
            ax.scatter(i + rng.uniform(-0.1, 0.1, len(vals)), vals, s=18, alpha=0.7, color=colors[i])
            ax.hlines(statistics.median(vals), i - 0.22, i + 0.22, color="black", linewidth=2)
        ax.set_xticks(range(len(modes)), [LABELS[m] for m in modes])
        ax.set_ylabel("Qwen device_wall (ms)")
        ax.set_title("30 retained formal samples per policy; black line = median")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(root / "samples.png", dpi=180)
        fig.savefig(root / "samples.pdf")
        plt.close(fig)
    print(tables)
    print(ctable)


if __name__ == "__main__":
    main()
