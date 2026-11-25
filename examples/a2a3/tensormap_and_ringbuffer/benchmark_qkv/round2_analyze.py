#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Aggregate formal counters and diagnostic timelines with explicit denominators."""

import argparse
import json
import re
from pathlib import Path

from round2_data import distribution, parse, timeline_stats

SUMMARY = (
    r"QKV_SUMMARY eligible=(\d+) assigned_tasks=(\d+),(\d+) batches=(\d+) ticks=(\d+) "
    r"max_ticks=(\d+) freq=(\d+) assigned_blocks=(\d+),(\d+) claimed_blocks=(\d+),(\d+) pending=(\d+)"
)
NAMES = [
    "eligible",
    "side0",
    "side1",
    "batches",
    "ticks",
    "max_ticks",
    "freq",
    "blocks0",
    "blocks1",
    "claimed0",
    "claimed1",
    "pending",
]
FALLBACKS = [
    "full",
    "snapshot",
    "no_side",
    "budget",
    "oversize",
    "uninitialized",
    "stale",
    "busy",
    "version",
]


def counters(folder):
    text = (folder / "feedback.txt").read_text()
    rows = [dict(zip(NAMES, map(int, m))) for m in re.findall(SUMMARY, text)]
    batches = re.findall(
        r"QKV_BATCHES sizes1to8=([\d,]+) fallback_full_snapshot_noside_budget_oversize=([\d,]+)",
        text,
    )
    coverage = re.findall(
        (
            "QKV_R2_COVERAGE shape_eligible=([\\d,]+) assigned=([\\d,]+) uninitialized=(\\d+) "
            "stale=(\\d+) busy=(\\d+) version=(\\d+)"
        ),
        text,
    )
    assert len(rows) == len(batches)
    for i, (r, (sizes, fb)) in enumerate(zip(rows, batches)):
        r["sizes"] = list(map(int, sizes.split(",")))
        r["fallback"] = list(map(int, fb.split(",")))
        if coverage:
            cov = coverage[i]
            r["shape_eligible"] = list(map(int, cov[0].split(",")))
            r["shape_assigned"] = list(map(int, cov[1].split(",")))
            r["fallback"] += list(map(int, cov[2:]))
        else:
            r["shape_eligible"] = [0] * 3
            r["shape_assigned"] = [0] * 3
            r["fallback"] += [0] * 4
        r["assigned"] = r["side0"] + r["side1"]
        assert r["eligible"] == r["assigned"] + sum(r["fallback"]), (folder, r)
        assert r["blocks0"] == r["claimed0"] and r["blocks1"] == r["claimed1"] and r["pending"] == 0, (folder, r)
        assert sum(r["sizes"]) == r["batches"]
    return rows


def aggregate(rows):
    out = {
        k: sum(r[k] for r in rows)
        for k in [
            "eligible",
            "assigned",
            "batches",
            "blocks0",
            "blocks1",
            "claimed0",
            "claimed1",
        ]
    }
    out["runs"] = len(rows)
    out["coverage_percent"] = 100 * out["assigned"] / max(1, out["eligible"])
    out["fallback"] = {k: sum(r["fallback"][i] for r in rows) for i, k in enumerate(FALLBACKS)}
    out["natural_batch_sizes"] = [sum(r["sizes"][i] for r in rows) for i in range(8)]
    out["multi_batches"] = sum(out["natural_batch_sizes"][1:])
    out["shape_eligible"] = [sum(r["shape_eligible"][i] for r in rows) for i in range(3)]
    out["shape_assigned"] = [sum(r["shape_assigned"][i] for r in rows) for i in range(3)]
    out["mean_decision_us"] = sum(r["ticks"] / r["freq"] * 1e6 for r in rows) / max(1, out["batches"])
    out["max_decision_us"] = max([r["max_ticks"] / r["freq"] * 1e6 for r in rows] + [0])
    return out


def diagnostics(folder):
    d = parse(folder)
    ts = d["tasks"]
    dummy = [t for t in ts if t["shape"] == 3 and t["proxy_kind"] == 5]
    text = (folder / "feedback.txt").read_text()
    gaps = [(int(b), int(n)) for b, n in re.findall(r"QKV_R2_GAP bin=(\d+) count=(\d+)", text)]
    total = sum(n for b, n in gaps)
    cum = 0
    p95 = None
    for b, n in gaps:
        cum += n
        if p95 is None and cum >= total * 0.95:
            p95 = [2**b / 1000, 2 ** (b + 1) / 1000]
    # Bin is log2(nanoseconds), unlike META gap_max which is raw ticks.
    phase = {
        name: distribution([r[i] / 50 for r in d["phases"]])
        for i, name in enumerate(("pop", "snapshot", "score", "publish"))
    }
    start = min(t["created"] for t in ts)
    end = max(max(t["fin"], t["dummy_end"]) for t in ts)
    span = (end - start) / 50
    return {
        "tasks": len(ts),
        "meta": d["meta"],
        "timeline": timeline_stats(ts),
        "by_shape": {str(s): timeline_stats([t for t in ts if t["shape"] == s]) for s in range(4)},
        "phase_us": phase,
        "s0_dispatch_gap_p95_us_bin": p95,
        "s0_dispatch_gap_samples": total,
        "dummy_fanout": distribution([t["fanout"] for t in dummy]),
        "dummy_s_pressure": {
            str(s): {
                "inline_us": sum((t["dummy_end"] - t["dummy_begin"]) / 50 for t in dummy if t.get("scheduler") == s),
                "fraction_of_span": sum(
                    (t["dummy_end"] - t["dummy_begin"]) / 50 for t in dummy if t.get("scheduler") == s
                )
                / span,
            }
            for s in range(5)
        },
        "early_slots": [
            list(map(int, m))
            for m in re.findall(
                (
                    "QKV_R2_EARLY thread=(\\d+) staged=(\\d+) census=(\\d+) running_max=(\\d+) "
                    "pending_max=(\\d+) slot_sum=(\\d+)"
                ),
                text,
            )
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    root = args.root
    result = {}
    for mode in ["original", "improved", "trained"]:
        if not (root / "measure" / mode).exists():
            continue
        formal = []
        warm = []
        for i in range(6):
            rows = counters(root / "measure" / mode / f"block{i}")
            assert len(rows) == 6
            warm += rows[:1]
            formal += rows[1:]
        result[mode] = {"formal": aggregate(formal), "warmup": aggregate(warm)}
    (root / "coverage.json").write_text(json.dumps(result, indent=2))
    diag = {}
    for p in (root / "probe").glob("*/sample/result.json"):
        diag[str(p.parent.relative_to(root))] = diagnostics(p.parent)
    for p in (root / "diagnostics").glob("*/sample/result.json"):
        diag[str(p.parent.relative_to(root))] = diagnostics(p.parent)
    (root / "diagnostics-summary.json").write_text(json.dumps(diag, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
