#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check quota conservation and summarize independent diagnostic runs."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from simpler_setup.tools.swimlane_converter import read_perf_data


def fields(line: str) -> dict[str, Any]:
    return {
        k: [int(n) for n in v.split(",")] if "," in v else int(v) for k, v in re.findall(r"([a-z_]+)=([0-9,]+)", line)
    }


def diagnostic(dest, quota):
    logs = list((dest / "device-logs/debug/device-0").glob("*.log"))
    log = max(logs, key=lambda p: p.stat().st_size).read_text()
    result: dict[str, Any] = {
        "topology": {
            int(t): {"cpu": int(c), "role": role}
            for c, t, role in re.findall(r"cpu=(\d+) exec_idx=(\d+) ACTIVE\((sched|orch)\)", log)
        }
    }
    result["trace_ms"] = {}
    for owner, stage, calls, ns in re.findall(
        r"AICPU_PARTITION_TRACE owner=(\w+) stage=(\w+) calls=(\d+) ns=(\d+)", log
    ):
        result["trace_ms"].setdefault(owner, {})[stage] = {"calls": int(calls), "ms": int(ns) / 1e6}
    if quota:
        ready = [fields(line) for line in log.splitlines() if "AICPU_READY_LEDGER" in line]
        reserved = [fields(line) for line in log.splitlines() if "AICPU_RESERVATION_LEDGER" in line]
        finished = [fields(line) for line in log.splitlines() if "AICPU_QUOTA_FINISHED" in line]
        assert len(ready) == len(reserved) == len(finished) == 4, dest
        for r, a, f in zip(ready, reserved, finished):
            assert r["shape"] == a["shape"] == f["shape"], dest
            assert r["queued"] == r["claimed"] == r["executed"] == a["assigned"] == a["revoked"], (dest, r, a)
            assert [6 * n for n in r["executed"]] == f["units"], (dest, r, f)
        result["quota_ledger"] = ready
        result["conservation_passed"] = True
        match = re.search(r"AICPU_QUOTA_SPLITS tasks=(\d+)", log)
        assert match is not None, dest
        result["split_tasks"] = int(match[1])
        for owner, stages in result["trace_ms"].items():
            if owner.startswith("S"):
                assert stages.get("affinity_votes", {}).get("calls", 0) == 0
    captures = list(dest.rglob("chip_swimlane_records.json"))
    if captures:
        perf = read_perf_data(captures[0])
        submits = [
            row for phases in perf["aicpu_orchestrator_phases"] for row in phases if row["phase"] == "orch_submit"
        ]
        result["O_submit_ms"] = sum(row["end_time_us"] - row["start_time_us"] for row in submits) / 1000
        result["S_phase_sum_ms"] = {
            phase: sum(
                row["end_time_us"] - row["start_time_us"]
                for rows in perf["scheduler_records"]
                for row in rows
                if row["phase"] == phase
            )
            / 1000
            for phase in ("complete", "dispatch", "resolve")
        }
        result["capture_streams"] = [row["capture"] for row in perf["scheduler_streams"]]
        mapping = perf["core_to_thread"]
        sides = {}
        for row in perf["tasks"]:
            side = 0 if mapping[row["core_id"]] < 2 else 1
            sides.setdefault(str(row["task_id"]), set()).add(side)
        result["captured_both_side_tasks"] = sum(len(s) == 2 for s in sides.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    root = args.study.resolve()
    summary = json.loads((root / "summary.json").read_text())
    diagnostics = {s: {} for s in summary["stages"]}
    with (root / "results.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "stage", "n", "median_ms", "p95_ms", "vs_v8_pct", "vs_shared_pct"])
        for stage, cases in summary["stages"].items():
            for case, stats in cases.items():
                v8 = summary["stages"]["v8"][case]["median_ms"]
                shared = summary["stages"]["shared"][case]["median_ms"]
                writer.writerow(
                    [
                        case,
                        stage,
                        stats["n"],
                        stats["median_ms"],
                        stats["p95_ms"],
                        100 * (stats["median_ms"] / v8 - 1),
                        100 * (stats["median_ms"] / shared - 1),
                    ]
                )
                diagnostics[stage][case] = diagnostic(root / stage / case / "diagnostic", stage == "v14")
    (root / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print("Independent diagnostics and v14 conservation passed")


if __name__ == "__main__":
    main()
