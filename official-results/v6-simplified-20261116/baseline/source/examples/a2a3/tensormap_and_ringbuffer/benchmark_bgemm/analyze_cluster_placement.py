"""Summarize paired six-thread BGEMM device-wall runs from STRACE logs."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


DEVICE_WALL = re.compile(r"name=chip\.run\.runner_run\.device_wall ts=\d+ dur=(\d+)\b")
CASES = ("ScaleShared", "ScaleAdaptive", "Case0Shared", "Case0Adaptive")


def learning_summary(root):
    reports = {}
    for family, expected in (("Scale", 4096), ("Case0", 1000)):
        host = root / f"Cluster{family}Adaptive-smoke.log"
        if not host.exists():
            continue
        host_text = host.read_text()
        if f"TestBenchmarkBgemm::Cluster{family}Adaptive PASSED" not in host_text:
            raise ValueError(f"golden check failed in {host}")
        matches = re.findall(r"simpler_inner_[0-9a-f]+_(\d+)\.json", host_text)
        if not matches:
            raise ValueError(f"worker PID missing in {host}")
        paths = list((root / "device-logs").rglob(f"device-{matches[0]}_*.log"))
        device = max(paths, key=lambda p: p.read_text().count("AICPU_ADAPTIVE_FINAL"))
        lines = device.read_text().splitlines()

        def fields(line):
            result = {}
            for key, value in re.findall(r"([a-z_][a-z_0-9]*)=([0-9,]+)", line):
                result[key] = [int(v) for v in value.split(",")] if "," in value else int(value)
            return result

        finals = [fields(line) for line in lines if "AICPU_ADAPTIVE_FINAL" in line]
        if len(finals) != 1 or finals[0].get("version") != 2:
            raise ValueError(f"expected one v2 terminal report in {device}")
        final = finals[0]
        assert sum(final["decisions"]) == sum(final["started"]) == expected
        assert final["cold"] + final["learned"] == expected and final["cold"] >= 256 and final["learned"] > 0
        assert final["polls"] <= expected // 64
        for c in (0, 1):
            assert final["decisions"][c] + final["moves"][c] - final["moves"][1-c] == final["started"][c]
        updates = [fields(line) for line in lines if "AICPU_ADAPTIVE_UPDATES" in line]
        active = [row for row in updates if row["shape"] < 2]
        assert sum(row["o1"] for row in active) > 0
        assert sum(row["s1"] if row["cluster"] == 0 else row["s0"] for row in active) > 0
        assert max(row[key] for row in updates for key in ("queue", "o0", "o1", "s0", "s1")) <= (expected - 1) // 256
        reports[family] = {"final": final, "channel_updates": updates, "device_log": str(device),
                           "models": [fields(line) for line in lines if "AICPU_ADAPTIVE_MODEL" in line]}
    return reports


def percentile(values, fraction):
    ordered = sorted(values)
    point = (len(ordered) - 1) * fraction
    low = int(point)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (point - low)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--outputs", type=Path, help="Runtime outputs directory with separate swimlane and dep-gen runs")
    args = parser.parse_args()
    results = args.results
    before = (results / "binaries-before.sha256").read_text()
    after = (results / "binaries-after.sha256").read_text()
    if before != after:
        raise ValueError("runtime binaries changed during the measurement suite")
    for line in before.splitlines():
        expected, path = line.split(maxsplit=1)
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"runtime binary changed after the suite: {path}")

    records = {}
    for case in CASES:
        durations = []
        for batch in ("a", "b"):
            log = (results / f"{case}-{batch}.log").read_text()
            if f"TestBenchmarkBgemm::Cluster{case} PASSED" not in log:
                raise ValueError(f"correctness result missing in {case}-{batch}.log")
            sample = [int(ns) / 1e6 for ns in DEVICE_WALL.findall(log)]
            if len(sample) != 15:
                raise ValueError(f"expected 15 device-wall samples in {case}-{batch}.log; found {len(sample)}")
            durations.extend(sample)
        records[case] = {
            "n": len(durations),
            "median_ms": statistics.median(durations),
            "p95_ms": percentile(durations, .95),
            "min_ms": min(durations),
            "max_ms": max(durations),
        }
    for family in ("Scale", "Case0"):
        shared = records[f"{family}Shared"]["median_ms"]
        adaptive = records[f"{family}Adaptive"]["median_ms"]
        records[family] = {"adaptive_vs_shared_median_pct": (adaptive / shared - 1) * 100}
    learning = learning_summary(results.parent)
    if learning:
        records["learning"] = learning
    if args.outputs:
        from simpler_setup.tools.swimlane_converter import read_perf_data

        diagnostics = {}
        for family in ("Scale", "Case0"):
            dependency_paths = list(args.outputs.glob(f"TestBenchmarkBgemm_Cluster{family}Shared_*/deps.json"))
            dependency_path = max(dependency_paths, key=lambda path: path.stat().st_mtime)
            graph = json.loads(dependency_path.read_text())
            edges = {(int(edge["pred"]), int(edge["succ"])) for edge in graph["edges"] if "wait" in edge["flags"]}
            expected_tasks = 4096 if family == "Scale" else 1000
            if len(graph["tasks"]) != expected_tasks:
                raise ValueError(f"unexpected task count in {dependency_path}")
            for mode in ("Shared", "Adaptive"):
                case = family + mode
                paths = list(args.outputs.glob(f"TestBenchmarkBgemm_Cluster{case}_*/chip_swimlane_records.json"))
                path = max(paths, key=lambda item: item.stat().st_mtime)
                perf = read_perf_data(path)
                tasks = perf["tasks"]
                if len(tasks) != expected_tasks or len({task["task_id"] for task in tasks}) != expected_tasks:
                    raise ValueError(f"missing or repeated tasks in {path}")
                if any(stream["capture"]["dropped"] or stream["capture"]["truncated"]
                       for stream in perf["scheduler_streams"]):
                    raise ValueError(f"incomplete scheduler capture in {path}")
                thread = perf["core_to_thread"]
                task_cluster = {task["task_id"]: int(thread[task["core_id"]] >= 2) for task in tasks}
                if any(pred not in task_cluster or succ not in task_cluster for pred, succ in edges):
                    raise ValueError(f"dependency tasks missing in {path}")
                waits = [row["end_time_us"] - row["start_time_us"] for rows in perf["scheduler_records"]
                         for row in rows if row["phase"] == "ready_wait"]
                if len(waits) != expected_tasks:
                    raise ValueError(f"ready-wait capture incomplete in {path}")
                diagnostics[case] = {
                    "capture": str(path), "dependency_graph": str(dependency_path),
                    "tasks": len(tasks), "wait_edges": len(edges),
                    "cross_wait_edges": sum(task_cluster[pred] != task_cluster[succ] for pred, succ in edges),
                    "cluster_tasks": [sum(cluster == c for cluster in task_cluster.values()) for c in (0, 1)],
                    "ready_wait_mean_us": statistics.mean(waits),
                    "ready_wait_median_us": statistics.median(waits),
                    "stream_drops": 0,
                }
        records["diagnostics"] = diagnostics
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
