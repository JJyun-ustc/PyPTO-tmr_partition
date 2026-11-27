#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Summarize frozen blocked measurements; the independent unit is a period."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    samples = json.loads((folder / "samples.json").read_text())
    grouped = defaultdict(list)
    by_period = defaultdict(dict)
    for row in samples:
        if row["device_wall_ms"] is None:
            raise ValueError("diagnostic/profile samples cannot be ordinary timing samples")
        if not row["warmup"]:
            grouped[row["mode"]].append(row["device_wall_ms"])
    for mode in grouped:
        for period in range(12):
            values = [
                r["device_wall_ms"] for r in samples if not r["warmup"] and r["mode"] == mode and r["period"] == period
            ]
            if len(values) != 5:
                raise ValueError("expected five measurements per policy per period")
            by_period[mode][period] = float(np.median(values))
    if any(len(v) != 60 for v in grouped.values()):
        raise ValueError("expected 60 ordinary samples per policy")
    baseline = float(np.median(grouped["shared"]))
    summary = {
        "seq_len": manifest["seq_len"],
        "formal_samples": 60,
        "warmups_per_policy": 12,
        "periods": 12,
        "p95_method": "numpy linear quantile",
        "bootstrap": "10000 resamples of 12 paired period medians, seed 1234",
        "C_D_active": manifest["C_D_active"],
        "policies": {},
    }
    for mode, values in grouped.items():
        contrasts = np.array([100 * (by_period[mode][p] / by_period["shared"][p] - 1) for p in range(12)])
        rng = np.random.default_rng(1234)
        draws = np.median(contrasts[rng.integers(0, 12, (10000, 12))], axis=1)
        counters = [r["counters"] for r in samples if r["mode"] == mode and not r["warmup"]]
        summary["policies"][mode] = {
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.quantile(values, 0.95)),
            "median_relative_shared_percent": 100 * (float(np.median(values)) / baseline - 1),
            "period_relative_shared_percent": contrasts.tolist(),
            "paired_period_median_percent": float(np.median(contrasts)),
            "paired_period_bootstrap_95_percent": np.quantile(draws, [0.025, 0.975]).tolist(),
            "eligible_tasks_total": sum(c["eligible"] for c in counters),
            "actual_eligible_block_claims_total": sum(c["eligible_claimed"] for c in counters),
            "alternative_comparisons_total": sum(c["multi"] for c in counters),
            "overrides_total": sum(c["overrides"] for c in counters),
            "claim_races_total": sum(c["race"] for c in counters),
        }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    args = parser.parse_args()
    for folder in args.folders:
        print(json.dumps(summarize(folder), indent=2))


if __name__ == "__main__":
    main()
