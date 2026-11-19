"""Summarize completed control-path variants; position experiments are excluded."""

import argparse
import json
import re
import statistics
from pathlib import Path


def quantile(values, p):
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = int(index)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (index - lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = {
        "qualification": (
            "Each variant has its own same-binary controls. Variants were built sequentially; "
            "small between-variant gains are not proof of causality."
        ),
        "variants": {},
    }
    for variant in ("before", "empty-tiers", "batched", "isolated"):
        performance = {}
        for case in ("ScaleShared", "ScaleAdaptive", "Case0Shared", "Case0Adaptive"):
            values = []
            for batch in ("a", "b"):
                path = args.root / variant / "performance" / f"{case}-{batch}.log"
                content = path.read_text()
                assert f"Cluster{case} PASSED" in content
                sample = [
                    int(ns) / 1e6
                    for ns in re.findall(r"name=chip\.run\.runner_run\.device_wall ts=\d+ dur=(\d+)\b", content)
                ]
                assert len(sample) == 15, (path, len(sample))
                values.extend(sample)
            performance[case] = {
                "n": len(values),
                "median_ms": statistics.median(values),
                "p95_ms": quantile(values, 0.95),
            }
        for family in ("Scale", "Case0"):
            performance[family + "_overhead_pct"] = (
                performance[family + "Adaptive"]["median_ms"] / performance[family + "Shared"]["median_ms"] - 1
            ) * 100
        report["variants"][variant] = performance
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
