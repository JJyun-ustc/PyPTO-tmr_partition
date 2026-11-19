# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Export all medians, p95, controls, and diagnostic costs without selecting rounds."""

import argparse
import csv
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    root = args.study.resolve()
    summary = json.loads((root / "summary.json").read_text())
    stages = summary["stages"]
    v8 = stages["v8"]["timing"]["cases"]
    official = stages["official-after"]["timing"]["cases"]
    table = ["|阶段|" + "|".join(official) + "|", "|---|" + "---:|" * len(official)]
    p95 = table.copy()
    rows = []
    for name, stage in stages.items():
        cases = stage["timing"]["cases"]
        table.append("|" + name + "|" + "|".join(f"{cases[c]['median_ms']:.4f}" for c in official) + "|")
        p95.append("|" + name + "|" + "|".join(f"{cases[c]['p95_ms']:.4f}" for c in official) + "|")
        for case, timing in cases.items():
            diag = stage["diagnostics"][case]
            rows.append(
                {
                    "stage": name,
                    "case": case,
                    "n": timing["n"],
                    "median_ms": timing["median_ms"],
                    "p95_ms": timing["p95_ms"],
                    "vs_v8_pct": 100 * (timing["median_ms"] / v8[case]["median_ms"] - 1),
                    "vs_official_pct": 100 * (timing["median_ms"] / official[case]["median_ms"] - 1),
                    "O_submit_ms": diag["O_submit_ms"],
                    "S_ready_binding_sum_ms": diag["S_ready_binding_sum_ms"],
                    "O_ready_binding_ms": diag["O_ready_binding_ms"],
                    "remote_blocks": sum(r["remote"] for r in diag["ready_feedback"]),
                }
            )
    with (root / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    drift = {
        case: 100 * (official[case]["median_ms"] / stages["official-before"]["timing"]["cases"][case]["median_ms"] - 1)
        for case in official
    }
    summary["official_drift_pct"] = drift
    summary["normal_sample_count"] = sum(r["n"] for r in rows)
    validation = summary.get("validation", {})
    validation.update(json.loads((root / "final-validation/results.json").read_text()))
    assert len(validation["scenes"]) == 6 and all(code == 0 for code in validation["scenes"].values())
    ledgers = []
    for log in (root / "final-validation/device-logs/debug").rglob("device-*.log"):
        for line in log.read_text().splitlines():
            if "AICPU_READY_LEDGER " not in line:
                continue
            values = {
                key: [int(v) for v in value.split(",")]
                for key, value in re.findall(r"(queued|claimed|executed)=([0-9,]+)", line)
            }
            assert values["queued"] == values["claimed"]
            assert sum(values["claimed"]) == sum(values["executed"])
            ledgers.append(values)
    validation["regression_ledger_rows"] = len(ledgers)
    validation["benchmark_ledger_rows"] = sum(
        len(d["ready_ledger"]) for v in stages.values() for d in v["diagnostics"].values()
    )
    validation["golden_scene_cases"] = sum(
        len(re.findall(r"::[^\n]* PASSED", (root / f"final-validation/{name}.log").read_text()))
        for name in validation["scenes"]
    )
    validation["DFX_off"] = json.loads((root / "dfx-off.json").read_text())
    summary["validation"] = validation
    summary["source_boundary"] = (
        "v8 worktree unchanged; new runtime-ready-affinity branch; official runtime src delta maxthreads 4->6 only"
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "tables.md").write_text("\n".join(table) + "\n\n" + "\n".join(p95) + "\n")
    print("\n".join(table))
    print(
        "normal_samples", summary["normal_sample_count"], "max_official_drift_pct", max(abs(x) for x in drift.values())
    )


if __name__ == "__main__":
    main()
