#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Audit prefix visibility and paired placement spans without inventing actual starts."""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

NAMES = (
    "task created ready pending_pop publish dispatch start_proxy first_fin fin completed "
    "k0 k1 k2 blocks shape mask sync selected actual_side proxy_kind scheduler fanout "
    "dummy_begin dummy_end summary_age"
).split()


def read_trace(folder):
    logs = sorted((folder / "ascend").rglob("*.log"))
    text = "\n".join(p.read_text(errors="replace") for p in logs)
    rows = []
    for match in re.finditer(r"QKV_R2_TASKS ([^\n]+)", text):
        for record in match[1].split(";")[:-1]:
            fields = record.split(",")
            if len(fields) != 35:
                raise ValueError("unknown task schema")
            rows.append(dict(zip(NAMES, map(int, fields[:25]))))
    visible = {}
    for line in text.splitlines():
        if "STREAM_VIS " in line:
            values = {key: int(value) for key, value in re.findall(r"(\w+)=(-?\d+)", line.split("STREAM_VIS ", 1)[1])}
            visible[values["task"]] = values
    if len({r["task"] for r in rows}) != len(rows):
        raise ValueError("multiple launches or duplicate task records")
    if rows and len(visible) != len(rows):
        raise ValueError("incomplete visibility schema")
    meta = re.findall(r"QKV_R2_META level=(\d+) tasks=(\d+) edges=(\d+) dropped=(\d+)", text)
    if len(meta) != 1 or int(meta[0][1]) != len(rows) or int(meta[0][3]):
        raise ValueError("truncated or multiple-launch trace")
    host = (folder / "host.log").read_text()
    freq = set(map(int, re.findall(r"dev_cnt_hz=(\d+)", host)))
    if freq != {50000000}:
        raise ValueError(f"unverified tick frequency: {freq}")
    for r in rows:
        r.update({"visibility": visible[r["task"]]})
    return rows


def visibility_summary(rows):
    observed = [r for r in rows if r["visibility"]["observed"]]
    count = Counter(r["visibility"]["state"] for r in observed)
    return {
        "tasks": len(rows),
        "observed_normal_single_block_claims": len(observed),
        "denominator": "actual Shared batch-pop observations; not three-queue comparison opportunities",
        "unknown": count[0],
        "registering": count[1],
        "usable_bounded_join": count[2],
        "index_overflow": count[3],
        "complete_fanin_over_budget": count[4],
        "usable_fraction": count[2] / len(observed) if observed else None,
        "ranking_change": "not measured by this observational hook",
    }


def calibration(rows):
    tasks = [r for r in rows if r["shape"] == 1]
    if len(tasks) != 6048:
        raise ValueError(f"expected 6048 calibration tasks, got {len(tasks)}")
    labelled, samples = [], defaultdict(list)
    cursor = 0
    for period in range(6):
        for size in (4096, 65536, 1048576):
            for reuse in (1, 8):
                for producer_side in (0, 1):
                    for arm in (0, 1):
                        cross = 1 - arm if period % 2 else arm
                        for repeat in range(21):
                            for consumer in (0, 1):
                                task = dict(tasks[cursor])
                                cursor += 1
                                expected = 1 - producer_side if consumer and cross else producer_side
                                if task["actual_side"] != expected:
                                    raise ValueError("strict placement violated")
                                task.update(
                                    period=period,
                                    bytes=size,
                                    reuse=reuse,
                                    cross=cross,
                                    repeat=repeat,
                                    consumer=consumer,
                                    warmup=repeat == 0,
                                )
                                task["span_us"] = (
                                    (task["fin"] - task["start_proxy"]) / 50.0
                                    if task["proxy_kind"] == 1 and task["fin"] > task["start_proxy"]
                                    else None
                                )
                                labelled.append(task)
                                if consumer and repeat and task["span_us"] is not None:
                                    group = task["visibility"]["physical_group"]
                                    samples[period, size, reuse, expected, group, cross].append(task["span_us"])
    comparisons = paired_periods(samples)
    return labelled, comparisons, summarize_pairs(comparisons)


def paired_periods(samples):
    comparisons = []
    for period in range(6):
        for size in (4096, 65536, 1048576):
            for reuse in (1, 8):
                for side in (0, 1):
                    group_diffs = []
                    groups = sorted({k[4] for k in samples if k[:4] == (period, size, reuse, side)})
                    for group in groups:
                        same = samples[period, size, reuse, side, group, 0]
                        cross = samples[period, size, reuse, side, group, 1]
                        if len(same) >= 2 and len(cross) >= 2:
                            diff = statistics.median(cross) - statistics.median(same)
                            group_diffs.append(diff)
                    comparisons.append(
                        {
                            "period": period,
                            "bytes": size,
                            "reuse": reuse,
                            "consumer_side": side,
                            "matched_groups": len(group_diffs),
                            "cross_minus_same_us": statistics.median(group_diffs) if group_diffs else None,
                        }
                    )
    return comparisons


def summarize_pairs(comparisons):
    summary = []
    for size in (4096, 65536, 1048576):
        for reuse in (1, 8):
            for side in (0, 1):
                values = [
                    r["cross_minus_same_us"]
                    for r in comparisons
                    if r["bytes"] == size
                    and r["reuse"] == reuse
                    and r["consumer_side"] == side
                    and r["cross_minus_same_us"] is not None
                ]
                summary.append(
                    {
                        "bytes": size,
                        "reuse": reuse,
                        "consumer_side": side,
                        "paired_periods": len(values),
                        "period_differences_us": values,
                        "median_difference_us": statistics.median(values) if values else None,
                        "consistent_nonzero_direction": len(values) == 6
                        and (all(x > 0 for x in values) or all(x < 0 for x in values)),
                        "scope": "combined producer-side/physical-group placement; custom reuse kernel only",
                    }
                )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("calibration", "visibility"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    rows = read_trace(args.folder)
    if args.kind == "calibration":
        labelled, comparisons, summary = calibration(rows)
        (args.folder / "labelled-tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labelled))
        (args.folder / "paired-periods.json").write_text(json.dumps(comparisons, indent=2))
    else:
        summary = visibility_summary(rows)
    (args.folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
