"""Summarize no-pickup and score comparisons without pooling diagnostic timings."""

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "plan.json").read_text())
    report = {
        "plan": plan,
        "qualification": (
            "Sequential source builds, fresh same-binary Shared controls, all 30 samples per case retained. "
            "No exclusive device lock. Small weight differences are local observations, not global optima. "
            "Held-out Heavy/Deep cases share the BGEMM family and do not establish general-DAG performance."
        ),
        "variants": {},
    }
    hashes = set()
    for stage in plan["stages"] + ["final"]:
        directory = root / stage
        if not (directory / "run/summary.json").exists():
            continue
        timing = json.loads((directory / "run/summary.json").read_text())
        assert (directory / "run/performance/binaries-before.sha256").read_bytes() == (
            directory / "run/performance/binaries-after.sha256"
        ).read_bytes()
        compute = set()
        for log in (directory / "run/performance").glob("*.log"):
            compute.update(re.findall(r"func_id=(\d+)\b[^\n]*code_fnv1a64=(0x[0-9a-f]+)", log.read_text()))
        assert len(compute) == 2, stage
        hashes.add(tuple(sorted(compute)))
        costs_path = directory / "control-costs.json"
        with costs_path.open("w") as output:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("analyze_control_costs.py")),
                    str(directory / "diagnostics"),
                ],
                stdout=output,
                check=True,
            )
        ratios = {
            family: timing["cases"][family + "Adaptive"]["median_ms"] / timing["cases"][family + "Shared"]["median_ms"]
            for family in ("Case0", "Scale")
        }
        row = {
            "timing": timing,
            "paired_ratios": ratios,
            "geometric_mean_ratio": math.sqrt(math.prod(ratios.values())),
            "diagnostics": json.loads(costs_path.read_text()),
        }
        if (directory / "holdout/summary.json").exists():
            row["holdout"] = json.loads((directory / "holdout/summary.json").read_text())
        report["variants"][stage] = row
        print(stage, ratios)
    assert len(hashes) == 1
    report["compute_code_hashes"] = list(next(iter(hashes)))
    if (root / "features.json").exists():
        report["feature_probe"] = json.loads((root / "features.json").read_text())
    if (root / "selection.json").exists():
        report["selection"] = json.loads((root / "selection.json").read_text())
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
