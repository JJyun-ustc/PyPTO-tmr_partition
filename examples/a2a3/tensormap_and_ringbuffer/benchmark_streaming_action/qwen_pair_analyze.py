#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Matched real-kernel spans; factual placement contrasts are not action-gain labels."""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from analyze import NAMES


def launches(folder):
    lines = []
    for path in sorted((folder / "ascend").rglob("*.log")):
        for line in path.read_text(errors="replace").splitlines():
            if not any(tag in line for tag in ("QKV_R2_META ", "QKV_R2_TASKS ", "STREAM_VIS ")):
                continue
            stamp = re.search(r"\d{4}-\d\d-\d\d-\d\d:\d\d:\d\d\.\d+\.\d+", line)
            if stamp:
                lines.append((stamp[0], line))
    result, current = [], None
    for _, line in sorted(lines, key=lambda x: x[0]):
        if "QKV_R2_META " in line:
            if current is not None:
                result.append(current)
            meta = dict((k, int(v)) for k, v in re.findall(r"(level|tasks|edges|dropped)=(\d+)", line))
            current = {"meta": meta, "tasks": [], "visible": {}}
        elif "QKV_R2_TASKS " in line:
            for raw in line.split("QKV_R2_TASKS ", 1)[1].split(";")[:-1]:
                cols = raw.split(",")
                row = dict(zip(NAMES, map(int, cols[:25])))
                row["local"] = list(map(float, cols[25:30]))
                row["global"] = list(map(float, cols[30:35]))
                current["tasks"].append(row)
        else:
            fields = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", line.split("STREAM_VIS ", 1)[1])}
            current["visible"][fields["task"]] = fields
    if current is not None:
        result.append(current)
    return result


def parse_period(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    runs = launches(folder)
    if len(runs) != len(manifest["schedule"]):
        raise ValueError(f"missing launches: {folder}: {len(runs)}")
    expected_path = folder.parent / "expected_compute_counts.json"
    expected_counts = json.loads(expected_path.read_text()) if expected_path.exists() else None
    labelled = []
    for label, run in zip(manifest["schedule"], runs):
        if run["meta"] != {"level": 3, "tasks": 160, "edges": 0, "dropped": 0}:
            raise ValueError(run["meta"])
        core_records = None
        if manifest.get("profile_root"):
            profile = folder / "core_records" / f"run{label['ordinal']:04d}/chip_swimlane_records.json"
            data = json.loads(profile.read_text())
            if data["metadata"]["clock_freq_hz"] != 50000000:
                raise ValueError("AICore clock frequency")
            core_records = defaultdict(list)
            for entry in data["aicore_tasks"]:
                if len(entry) != 7 or entry[4] < entry[3]:
                    raise ValueError("AICore record schema")
                core_records[entry[1]].append(entry)
            if expected_counts is None or {str(k): len(v) for k, v in core_records.items()} != expected_counts:
                raise ValueError("AICore task/block counts differ from complete baseline trace")
        counts = Counter()
        for task in run["tasks"]:
            kernel = next(task[k] for k in ("k0", "k1", "k2") if task[k] >= 0)
            layer = counts[kernel]
            counts[kernel] += 1
            info = run["visible"][task["task"]]
            if kernel in (33, 34):
                side = label["producer_side"] if kernel == 33 else label["consumer_side"]
                if task["actual_side"] != side:
                    raise ValueError("controlled side was not honored")
            if task["blocks"] != 1:
                raise ValueError("non-single-block sample")
            task.update(label)
            task.update(kernel=kernel, layer=layer, physical_group=info["physical_group"], claim=info["claim"])
            task["span_us"] = (
                (task["fin"] - task["start_proxy"]) / 50.0
                if task["proxy_kind"] in (1, 2) and task["fin"] > task["start_proxy"]
                else None
            )
            task["cpu_proxy_kind"] = task["proxy_kind"]
            task["cpu_span_us"] = task["span_us"]
            if core_records is not None:
                entries = core_records[task["task"]]
                if len(entries) != 1:
                    raise ValueError("selected single-block kernel has duplicate records")
                entry = entries[0]
                task["aicore_start"] = entry[3]
                task["aicore_end"] = entry[4]
                task["aicore_core"] = entry[0]
                task["span_us"] = (entry[4] - entry[3]) / 50.0
                task["proxy_kind"] = 7
            labelled.append(task)
        if counts != {31: 40, 32: 40, 33: 40, 34: 40}:
            raise ValueError(counts)
    (folder / "labelled.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labelled))
    return labelled


def contrasts(rows, strict):
    paired = defaultdict(dict)
    for row in rows:
        if row["kernel"] == 34 and not row["warmup"] and row["span_us"] is not None:
            paired[row["period"], row["repeat"], row["consumer_side"], row["layer"]][row["cross"]] = row
    pairs, missing = [], Counter()
    for identity, arms in paired.items():
        if len(arms) != 2:
            missing["missing_idle_proxy_arm"] += 1
            continue
        same, cross = arms[0], arms[1]
        if (
            same["task"] != cross["task"]
            or (same["physical_group"], same["scheduler"])
            != (
                cross["physical_group"],
                cross["scheduler"],
            )
            or same.get("aicore_core", same["local"][4]) != cross.get("aicore_core", cross["local"][4])
        ):
            missing["different_consumer_group"] += 1
            continue
        if same["proxy_kind"] != cross["proxy_kind"]:
            missing["different_start_proxy_kind"] += 1
            continue
        if strict and (
            same["local"][:3] != cross["local"][:3]
            or [int(x) // 4 for x in same["global"][:4]] != [int(x) // 4 for x in cross["global"][:4]]
        ):
            missing["different_occupancy"] += 1
            continue
        pairs.append(
            {
                "period": identity[0],
                "repeat": identity[1],
                "consumer_side": identity[2],
                "layer": identity[3],
                "proxy_kind": same["proxy_kind"],
                "same_us": same["span_us"],
                "cross_us": cross["span_us"],
                "difference_us": cross["span_us"] - same["span_us"],
            }
        )
    return pairs, dict(missing)


def summarize(pairs):
    summary = []
    for side in (0, 1):
        for proxy in sorted({r["proxy_kind"] for r in pairs}):
            periods = []
            for period in range(6):
                by_run = defaultdict(list)
                for row in pairs:
                    if row["consumer_side"] == side and row["period"] == period and row["proxy_kind"] == proxy:
                        by_run[row["repeat"]].append(row["difference_us"])
                values = [statistics.median(v) for v in by_run.values()]
                periods.append(
                    {
                        "period": period,
                        "matched_run_pairs": len(values),
                        "median_difference_us": statistics.median(values) if values else None,
                    }
                )
            values = [p["median_difference_us"] for p in periods if p["median_difference_us"] is not None]
            summary.append(
                {
                    "consumer_side": side,
                    "proxy_kind": proxy,
                    "periods": periods,
                    "matched_periods": len(values),
                    "median_difference_us": statistics.median(values) if values else None,
                    "stable_above_half_us": len(values) == 6
                    and (all(v > 0.5 for v in values) or all(v < -0.5 for v in values)),
                }
            )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    rows = []
    for folder in sorted(args.folder.glob("period*")):
        if not (folder / "samples.json").exists():
            raise ValueError(f"collection incomplete: {folder}")
        rows.extend(parse_period(folder))
    summary = {
        "scope": "33->34 first tile; placement contrasts, not two-action gain labels",
        "timing": "AICore post-ACK to post-kernel/pre-FIN dispatcher interval"
        if any(r["proxy_kind"] == 7 for r in rows)
        else "CPU publication/pending-promotion to FIN-observed proxies; families kept separate",
        "start_time_alignment": "absolute CPU/AICore timestamps not assumed interchangeable",
    }
    core = [r for r in rows if r["proxy_kind"] == 7]
    if core:
        summary["clock_audit"] = {
            "selected_records": len(core),
            "dispatch_after_aicore_start": sum(r["dispatch"] > r["aicore_start"] for r in core),
            "fin_seen_before_aicore_end": sum(r["fin"] < r["aicore_end"] for r in core),
            "cpu_minus_aicore_interval_us": {},
        }
        for kind in sorted({r["cpu_proxy_kind"] for r in core}):
            selected = [r for r in core if r["cpu_proxy_kind"] == kind]
            differences = [(r["fin"] - r["start_proxy"] - r["aicore_end"] + r["aicore_start"]) / 50 for r in selected]
            summary["clock_audit"]["cpu_minus_aicore_interval_us"][str(kind)] = {
                "n": len(selected),
                "median": float(np.median(differences)),
                "p95": float(np.quantile(differences, 0.95)),
                "valid_cpu_duration_label": kind in (1, 2),
            }
    for name, strict in (("physical_group", False), ("physical_and_occupancy", True)):
        pairs, excluded = contrasts(rows, strict)
        (args.folder / f"{name}-pairs.json").write_text(json.dumps(pairs, indent=2))
        summary[name] = {"pairs": len(pairs), "excluded": excluded, "by_side": summarize(pairs)}
    (args.folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
