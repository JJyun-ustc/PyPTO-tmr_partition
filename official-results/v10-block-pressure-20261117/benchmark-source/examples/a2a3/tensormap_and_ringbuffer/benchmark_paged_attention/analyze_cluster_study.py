#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Summarize diagnostic DAGs and control work separately from ordinary timing."""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

from simpler_setup.tools.swimlane_converter import read_perf_data

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.with_name("benchmark_bgemm")))
from analyze_control_study import fields  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    reports = {}
    compute_hashes = set()
    for host in sorted(root.glob("*-diagnostic.log")):
        case = host.stem.removesuffix("-diagnostic")
        text = host.read_text()
        assert f"TestClusterPagedAttention::{case} PASSED" in text
        pids = re.findall(r"simpler_inner_[0-9a-f]+_(\d+)\.json", text)
        assert pids, host
        devices = list((root / "device-logs").rglob(f"device-{pids[0]}_*.log"))
        device = max(devices, key=lambda p: p.read_text().count("AICPU_PARTITION_TRACE"))
        lines = device.read_text().splitlines()
        stages = {}
        for line in lines:
            match = re.search(r"AICPU_PARTITION_TRACE owner=(\w+) stage=(\w+) calls=(\d+) ns=(\d+)", line)
            if match:
                owner, stage, calls, ns = match.groups()
                stages.setdefault(owner, {})[stage] = {"calls": int(calls), "ns": int(ns)}
        report = {
            "device_log": str(device.relative_to(root)),
            "stages": stages,
            "O_placement_and_wire_ns": sum(v["ns"] for v in stages.get("O", {}).values()),
        }
        finals = [fields(line) for line in lines if "AICPU_ADAPTIVE_FINAL" in line]
        ledgers = [fields(line) for line in lines if "AICPU_PLACEMENT_LEDGER" in line]
        groups = [fields(line) for line in lines if "AICPU_PLACEMENT_GROUP" in line]
        pickups = [fields(line) for line in lines if "AICPU_PLACEMENT_PICKUP" in line]
        if case.endswith("Auto"):
            expected = 2048 if case.startswith("Uniform") else 2720
            assert len(finals) == 1 and sum(finals[0]["decisions"]) == sum(finals[0]["started"]) == expected
            assert len(ledgers) == 4
            for row in ledgers:
                assert row["preferred"] == row["released"]
                assert row["queued"] == row["taken"] == row["actual"]
                assert sum(row["preferred"]) == sum(row["queued"])
            assert len(pickups) == 5 and all(
                row["checks"] == row["attempts"] == row["remote_blocks"] == 0 for row in pickups
            )
        else:
            assert not finals and not ledgers
        report.update(final=finals, ledgers=ledgers, groups=groups, pickups=pickups)
        report["model"] = [fields(line) for line in lines if "AICPU_ADAPTIVE_MODEL" in line]
        capture = root / "captures" / case
        perf = read_perf_data(capture / "chip_swimlane_records.json")
        assert all(
            not row["capture"]["dropped"] and not row["capture"]["truncated"] for row in perf["scheduler_streams"]
        )
        report["compute_tasks"] = len(perf["tasks"])
        report["kernel_counts"] = dict(collections.Counter(row["func_id"] for row in perf["tasks"]))
        report["O_submit_ns"] = (
            sum(
                row["end_time_us"] - row["start_time_us"]
                for rows in perf["aicpu_orchestrator_phases"]
                for row in rows
                if row["phase"] == "orch_submit"
            )
            * 1000
        )
        report["S_observe_sum_ns"] = sum(
            rows.get("observe", {}).get("ns", 0) for owner, rows in stages.items() if owner.startswith("S")
        )
        report["S_ready_binding_sum_ns"] = sum(
            rows.get("ready_binding", {}).get("ns", 0) for owner, rows in stages.items() if owner.startswith("S")
        )
        deps = json.loads((capture / "deps.json").read_text())
        incoming = collections.defaultdict(set)
        for edge in deps["edges"]:
            if "wait" in edge["flags"]:
                incoming[edge["succ"]].add(edge["pred"])
        report["recorded_wait_edges_unique"] = sum(map(len, incoming.values()))
        report["recorded_multiple_wait_predecessor_nodes"] = sum(len(preds) >= 2 for preds in incoming.values())
        report["wait_fanin_histogram"] = dict(collections.Counter(len(preds) for preds in incoming.values()))
        report["wait_graph_qualification"] = (
            "Edges retained by dependency capture; not proof every predecessor was pending at full placement scoring."
        )
        codes = tuple(re.findall(r"func_id=(\d+).*?code_fnv1a64=(0x[0-9a-f]+)", text))
        assert len(codes) == 4, host
        compute_hashes.add(codes)
        reports[case] = report
    assert len(reports) == 4 and len(compute_hashes) == 1
    (root / "diagnostic-summary.json").write_text(
        json.dumps(
            {
                "qualification": (
                    "Separate instrumented golden runs, not ordinary timing samples. "
                    "O stage timers are exclusive; S timers overlap/nest across threads."
                ),
                "compute_code_hashes_match": True,
                "cases": reports,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                case: {
                    k: v
                    for k, v in row.items()
                    if k
                    in (
                        "compute_tasks",
                        "recorded_multiple_wait_predecessor_nodes",
                        "groups",
                        "O_submit_ns",
                        "O_placement_and_wire_ns",
                        "S_observe_sum_ns",
                        "S_ready_binding_sum_ns",
                    )
                }
                for case, row in reports.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
