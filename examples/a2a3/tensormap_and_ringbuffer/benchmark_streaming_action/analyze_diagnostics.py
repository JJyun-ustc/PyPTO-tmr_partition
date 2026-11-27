#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Audit the final build's full traces and sampled queue-observation costs."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from analyze import visibility_summary
from qwen_pair_analyze import launches
from visibility_budget import calculate


def distribution(values):
    return {
        "n": len(values),
        "median_us": float(np.median(values)) if values else None,
        "p95_us": float(np.quantile(values, 0.95)) if values else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    runs = launches(args.folder)
    labels = json.loads((args.folder / "manifest.json").read_text())["schedule"]
    if len(runs) != len(labels):
        raise ValueError("incomplete diagnostic launches")
    records = []
    for path in (args.folder / "ascend").rglob("*.log"):
        for line in path.read_text(errors="replace").splitlines():
            if any(tag in line for tag in ("STREAM_RUN ", "STREAM_THREAD ", "STREAM_COST ")):
                stamp = re.search(r"\d{4}-\d\d-\d\d-\d\d:\d\d:\d\d\.\d+\.\d+", line)
                if stamp:
                    records.append((stamp[0], line))
    costs, threads, mode = {}, {}, None
    for _, line in sorted(records):
        if "STREAM_RUN " in line:
            mode = int(re.search(r"mode=(\d+)", line)[1])
            costs[mode], threads[mode] = [], []
        elif "STREAM_COST " in line:
            costs[mode].append(int(re.search(r"ticks=(\d+)", line)[1]) / 50)
        else:
            threads[mode].append({k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line.split("STREAM_THREAD ", 1)[1])})
    summary = {"scope": "diagnostic only; all device_wall excluded from ordinary measurements", "policies": {}}
    for label, run in zip(labels, runs):
        rows = run["tasks"]
        if len(rows) != run["meta"]["tasks"] or run["meta"]["dropped"] or len({r["task"] for r in rows}) != len(rows):
            raise ValueError("task record audit")
        for row in rows:
            row["visibility"] = run["visible"][row["task"]]
        eligible = [r for r in rows if r["shape"] < 2 and r["blocks"] == 1 and r["ready"] and r["visibility"]["claim"]]
        mode = {"shared": 0, "structure": 7, "C_guarded": 8, "D_guarded": 9}[label["mode"]]
        summary["policies"][label["mode"]] = {
            "tasks": len(rows),
            "edges": run["meta"]["edges"],
            "ready_to_claim": distribution(
                [
                    (r["visibility"]["claim"] - r["ready"]) / 50
                    for r in eligible
                    if r["visibility"]["claim"] >= r["ready"]
                ]
            ),
            "claim_to_dispatch": distribution(
                [
                    (r["dispatch"] - r["visibility"]["claim"]) / 50
                    for r in eligible
                    if r["dispatch"] >= r["visibility"]["claim"]
                ]
            ),
            "queue_pop_cost_including_empty_and_batch_calls": distribution(costs[mode]),
            "queue_pop_cost_sampling": "every 32nd invocation, at most 2048 records per S; not model inference latency",
            "thread_counters": threads[mode],
        }
        if mode == 0:
            summary["shared_visibility"] = visibility_summary(rows)
            summary["shared_visibility_budget_sensitivity"] = calculate(rows)
    (args.folder / "diagnostic-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
