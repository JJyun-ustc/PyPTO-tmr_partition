#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Aggregate retained QKV scheduler feedback; keep diagnostic timing separate."""
import argparse
import json
from pathlib import Path
import re
import statistics

from run_study import percentile


def records(dest):
    candidates = list((dest / "ascend/debug/device-0").glob("*.log"))
    if not candidates:
        candidates = list((dest / "ascend").rglob("*.log"))
    text = "\n".join(p.read_text(errors="replace") for p in candidates)
    pattern = (r"QKV_SUMMARY eligible=(\d+) assigned_tasks=(\d+),(\d+) batches=(\d+) ticks=(\d+) "
               r"max_ticks=(\d+) freq=(\d+) assigned_blocks=(\d+),(\d+) claimed_blocks=(\d+),(\d+) pending=(\d+)")
    rows = []
    for m in re.finditer(pattern, text):
        row = dict(zip(("candidates", "tasks0", "tasks1", "batches", "ticks", "max_ticks", "freq",
                        "blocks0", "blocks1", "claimed0", "claimed1", "pending"), map(int, m.groups())))
        assert row["blocks0"] == row["claimed0"] and row["blocks1"] == row["claimed1"] and row["pending"] == 0
        rows.append(row)
    distributions = re.findall(r"QKV_BATCHES sizes1to8=([\d,]+) fallback_full_snapshot_noside_budget_oversize=([\d,]+)", text)
    assert len(distributions) == len(rows), dest
    for row, (sizes, fallback) in zip(rows, distributions):
        row["sizes"] = list(map(int, sizes.split(",")))
        row["fallback"] = list(map(int, fallback.split(",")))
        row["eligible"] = row["candidates"] - row["fallback"][4]
        row["assigned"] = row["tasks0"] + row["tasks1"]
        assert row["candidates"] == row["assigned"] + sum(row["fallback"]), row
        assert sum((i + 1) * n for i, n in enumerate(row["sizes"])) == row["assigned"] + sum(row["fallback"][1:4]), row
    exact = [int(v) for v in re.findall(r"QKV_DECISION batch=\d+ ticks=(\d+)", text)]
    return rows, exact


def aggregate(rows):
    result = {key: sum(r[key] for r in rows) for key in ("candidates", "eligible", "assigned", "tasks0", "tasks1", "batches", "ticks")}
    result["runs"] = len(rows)
    result["natural_batch_histogram"] = [sum(r["sizes"][i] for r in rows) for i in range(8)]
    result["multi_task_batches"] = sum(result["natural_batch_histogram"][1:])
    result["fallback_full_snapshot_noside_budget_oversize"] = [sum(r["fallback"][i] for r in rows) for i in range(5)]
    result["mean_decision_us"] = sum(r["ticks"] / r["freq"] * 1e6 for r in rows) / result["batches"]
    result["max_decision_us"] = max(r["max_ticks"] / r["freq"] * 1e6 for r in rows)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    out = args.output
    diagnostic, exact = records(out / "qkv/diagnostic")
    result = {"diagnostic": aggregate(diagnostic)}
    assert len(diagnostic) == 1 and len(exact) == diagnostic[0]["batches"]
    exact_us = [v / diagnostic[0]["freq"] * 1e6 for v in exact]
    result["diagnostic_exact_decision_us"] = {"n": len(exact_us), "median": statistics.median(exact_us),
                                               "p95": percentile(exact_us, .95), "all": exact_us}
    formal, warmup = [], []
    for b in range(6):
        rows, exact = records(out / f"qkv/block{b}")
        assert len(rows) == 6 and not exact
        warmup += rows[:1]
        formal += rows[1:]
    result["formal"] = aggregate(formal)
    result["warmup"] = aggregate(warmup)
    (out / "feedback.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "diagnostic_exact_decision_us"}, indent=2))


if __name__ == "__main__":
    main()
