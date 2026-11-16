"""Read cold diagnostic stage totals; these are not end-to-end performance samples."""

import argparse
import json
import re
from pathlib import Path

from simpler_setup.tools.swimlane_converter import read_perf_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    reports = {}
    for host in sorted(args.root.glob("Cluster*.log")):
        text = host.read_text()
        if " PASSED" not in text:
            raise ValueError(f"case did not pass: {host}")
        pids = re.findall(r"simpler_inner_[0-9a-f]+_(\d+)\.json", text)
        if not pids:
            continue
        paths = list((args.root / "device-logs").rglob(f"device-{pids[0]}_*.log"))
        device = max(paths, key=lambda p: p.read_text().count("AICPU_PARTITION_TRACE"))
        content = device.read_text()
        stages = {}
        for owner, stage, calls, ns in re.findall(
            r"AICPU_PARTITION_TRACE owner=(\w+) stage=(\w+) calls=(\d+) ns=(\d+)", content
        ):
            stages.setdefault(owner, {})[stage] = {"calls": int(calls), "ns": int(ns)}
        aggregated = {}
        for owner, rows in stages.items():
            if not owner.startswith("S"):
                continue
            for name, row in rows.items():
                total = aggregated.setdefault(name, {"sum_ns": 0, "max_thread_ns": 0, "calls": 0})
                total["sum_ns"] += row["ns"]
                total["max_thread_ns"] = max(total["max_thread_ns"], row["ns"])
                total["calls"] += row["calls"]
        report = {
            "device_log": str(device),
            "stages": stages,
            "scheduler": aggregated,
            "qualification": (
                "O stages are exclusive. S observe is nested in dispatch; threads overlap. Timer overhead is included."
            ),
        }
        report["O_total_ns"] = sum(row["ns"] for row in stages.get("O", {}).values())
        capture = re.search(r"at (\S+/deps\.json)", text)
        if capture:
            path = Path(capture[1]).with_name("chip_swimlane_records.json")
            perf = read_perf_data(path)
            tasks = perf["tasks"]
            assert all(
                not row["capture"]["dropped"] and not row["capture"]["truncated"] for row in perf["scheduler_streams"]
            )
            submit = [
                row for rows in perf["aicpu_orchestrator_phases"] for row in rows if row["phase"] == "orch_submit"
            ]
            report["capture"] = str(path)
            report["tasks"] = len(tasks)
            report["orch_submit_ns"] = sum(row["end_time_us"] - row["start_time_us"] for row in submit) * 1000
            report["kernel_total_us"] = {
                kind: sum(task["duration_us"] for task in tasks if task["core_type"] == kind) for kind in ("aic", "aiv")
            }
        reports[host.stem] = report
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
