#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate stage ledgers and summarize ordinary timings and separate diagnostics."""

import argparse
import json
import re
from pathlib import Path

from simpler_setup.tools.swimlane_converter import read_perf_data

STAGES = ("v8", "v12", "v13", "v13-no-pickup")


def fields(line):
    return {
        key: [int(v) for v in value.split(",")] if "," in value else int(value)
        for key, value in re.findall(r"([a-z_][a-z_0-9]*)=([0-9,]+)", line)
    }


def diagnostic(root, host):
    label = host.stem.removesuffix("-diagnostic")
    text = host.read_text()
    pids = re.findall(r"simpler_inner_[0-9a-f]+_(\d+)\.json", text)
    assert pids, host
    candidates = list((root / "device-logs").rglob(f"device-{pids[0]}_*.log"))
    path = max(
        candidates,
        key=lambda p: (
            p.read_text().count("AICPU_READY_LEDGER") + p.read_text().count("AICPU_PLACEMENT_LEDGER"),
            p.stat().st_size,
        ),
    )
    lines = path.read_text().splitlines()
    topology = {
        int(execution): {"cpu": int(cpu), "role": role}
        for cpu, execution, role in re.findall(r"cpu=(\d+) exec_idx=(\d+) ACTIVE\((sched|orch)\)", path.read_text())
    }
    assert topology == {i: {"cpu": i + 2, "role": "orch" if i == 5 else "sched"} for i in range(6)}, (host, topology)
    result = {"device_log": str(path.relative_to(root)), "topology": topology, "stages": {}}
    for line in lines:
        match = re.search(r"AICPU_PARTITION_TRACE owner=(\w+) stage=(\w+) calls=(\d+) ns=(\d+)", line)
        if match:
            owner, name, calls, ns = match.groups()
            result["stages"].setdefault(owner, {})[name] = {"calls": int(calls), "ns": int(ns)}
    tags = {
        "ready_final": "AICPU_READY_AFFINITY",
        "ready_ledger": "AICPU_READY_LEDGER",
        "ready_feedback": "AICPU_READY_FEEDBACK",
        "completion_ledger": "AICPU_READY_COMPLETION",
        "reservation_ledger": "AICPU_RESERVATION_LEDGER",
        "final": "AICPU_ADAPTIVE_FINAL",
        "groups": "AICPU_PLACEMENT_GROUP",
        "placement": "AICPU_PLACEMENT_LEDGER",
        "blocks": "AICPU_BLOCK_LEDGER",
        "prices": "AICPU_ADAPTIVE_MODEL",
        "price_updates": "AICPU_ADAPTIVE_UPDATES",
    }
    for key, tag in tags.items():
        result[key] = [fields(line) for line in lines if tag in line]
    for row in result["placement"]:
        assert row["preferred"] == row["released"], (host, row)
        assert row["queued"] == row["taken"] == row["actual"], (host, row)
        assert sum(row["preferred"]) == sum(row["queued"]), (host, row)
    for row in result["ready_ledger"]:
        assert row["queued"] == row["claimed"], (host, row)
        assert sum(row["claimed"]) == sum(row["executed"]), (host, row)
    for row in result["blocks"]:
        assert row["queued"] == row["claimed"] == row["published"] == row["retired"], (host, row)
    for row in result["reservation_ledger"]:
        assert row["assigned"] == row["revoked"], (host, row)
    for row in result["completion_ledger"]:
        assert [6 * v for v in row["accepted"]] == row["completed_units"], (host, row)
        execution = next(x for x in result["ready_ledger"] if x["shape"] == row["shape"])
        assert row["accepted"] == execution["executed"], (host, row)
    capture = root / "captures" / label
    perf = read_perf_data(capture / "chip_swimlane_records.json")
    assert all(not row["capture"]["dropped"] and not row["capture"]["truncated"] for row in perf["scheduler_streams"])
    deps = json.loads((capture / "deps.json").read_text())
    result["tasks"] = len(deps["tasks"])
    result["logical_blocks"] = sum(t["block_num"] for t in deps["tasks"])
    result["captured_execution_records"] = len(perf["tasks"])
    submits = [row for phases in perf["aicpu_orchestrator_phases"] for row in phases if row["phase"] == "orch_submit"]
    result["O_submit_ms"] = sum(row["end_time_us"] - row["start_time_us"] for row in submits) / 1000
    result["O_placement_and_wire_ms"] = sum(row["ns"] for row in result["stages"].get("O", {}).values()) / 1e6
    result["S_ready_binding_sum_ms"] = (
        sum(
            rows.get("ready_binding", {}).get("ns", 0)
            for owner, rows in result["stages"].items()
            if owner.startswith("S") and owner != "S5"
        )
        / 1e6
    )
    o_ready = result["stages"].get("S5", {}).get("ready_binding")
    result["O_ready_binding_ms"] = o_ready["ns"] / 1e6 if o_ready is not None else None
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
    execution_sides = {}
    mapping = perf["core_to_thread"]
    for row in perf["tasks"]:
        thread = mapping[row["core_id"]]
        execution_sides.setdefault(str(row["task_id"]), set()).add(0 if thread < 2 else 1)
    wait_edges = [edge for edge in deps["edges"] if "wait" in edge.get("flags", [])]
    eligible = [
        edge
        for edge in wait_edges
        if len(execution_sides.get(str(edge["pred"]), set())) == 1
        and len(execution_sides.get(str(edge["succ"]), set())) == 1
    ]
    result["captured_wait_edges"] = len(wait_edges)
    result["single_side_wait_edges"] = len(eligible)
    result["cross_side_captured_wait_edges"] = sum(
        execution_sides[str(edge["pred"])] != execution_sides[str(edge["succ"])] for edge in eligible
    )
    result["mixed_execution_tasks"] = sum(len(sides) > 1 for sides in execution_sides.values())
    result["compute_hashes"] = dict(re.findall(r"func_id=(\d+).*?code_fnv1a64=(0x[0-9a-f]+)", text))
    if label == "Burst":
        producer = str(deps["tasks"][0]["task_id"])
        root_records = [row for row in perf["tasks"] if str(row["task_id"]) == producer]
        root_end = max(row["end_time_us"] for row in root_records)
        child_submits = [row for row in submits if str(row["task_id"]) != producer]
        result["burst"] = {
            "producer_execution_us": sum(row["duration_us"] for row in root_records),
            "producer_end_us": root_end,
            "last_child_submit_end_us": max(row["end_time_us"] for row in child_submits),
            "children_submitted_before_producer_finished": sum(row["end_time_us"] < root_end for row in child_submits),
            "post_producer_compute_tail_us": max(row["end_time_us"] for row in perf["tasks"]) - root_end,
        }
        assert result["burst"]["children_submitted_before_producer_finished"] >= 300, (host, result["burst"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report = {
        "qualification": (
            "v8 and v12 are immutable independent controls; version orders interleaved. "
            "All controls remeasured on the card recorded by each stage; do not compare absolute times across cards. "
            "Diagnostics are separate instrumented runs "
            "and not summed into ordinary device_wall."
        ),
        "stages": {},
    }
    hashes = {}
    for stage in STAGES:
        directory = root / stage
        if not (directory / "summary.json").exists():
            continue
        timing = json.loads((directory / "summary.json").read_text())
        rows = {
            host.stem.removesuffix("-diagnostic"): diagnostic(directory, host)
            for host in directory.glob("*-diagnostic.log")
        }
        for label, row in rows.items():
            if label in hashes:
                assert row["compute_hashes"] == hashes[label], (stage, label, row["compute_hashes"], hashes[label])
            else:
                hashes[label] = row["compute_hashes"]
        report["stages"][stage] = {"timing": timing, "diagnostics": rows}
        (root / stage / "diagnostic-summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    baseline = report["stages"]["v8"]["timing"]["cases"]
    for stage, rows in report["stages"].items():
        for label, timing in rows["timing"]["cases"].items():
            timing["vs_v8_pct"] = 100 * (timing["median_ms"] / baseline[label]["median_ms"] - 1)
    report["compute_code_hashes_match"] = True
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("| stage | " + " | ".join(baseline) + " |")
    for stage, rows in report["stages"].items():
        values = rows["timing"]["cases"]
        print(
            "| "
            + stage
            + " | "
            + " | ".join(f"{values[label]['median_ms']:.4f}" if label in values else "-" for label in baseline)
            + " |"
        )


if __name__ == "__main__":
    main()
