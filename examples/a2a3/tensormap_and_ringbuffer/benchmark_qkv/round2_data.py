#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Parse low-overhead task timelines and full-DAG diagnostic records."""

import argparse
import gzip
import json
import re
import statistics
from pathlib import Path

from run_study import percentile

NAMES = [
    "task",
    "created",
    "ready",
    "pop",
    "publish",
    "dispatch",
    "start_proxy",
    "first_fin",
    "fin",
    "completed",
    "k0",
    "k1",
    "k2",
    "blocks",
    "shape",
    "mask",
    "sync",
    "selected",
    "actual_side",
    "proxy_kind",
    "fanout",
    "dummy_begin",
    "dummy_end",
    "summary_age",
]


def parse(folder):
    result = json.loads((folder / "result.json").read_text())
    text = (folder / "feedback.txt").read_text()
    tasks = []
    edges = []
    for line in text.splitlines():
        if "QKV_R2_TASKS " in line:
            values = line.split("QKV_R2_TASKS ", 1)[1].split('"')[0]
            for row in values.split(";")[:-1]:
                if not row:
                    continue
                cols = row.split(",")
                if len(cols) not in (34, 35):
                    continue
                names = NAMES if len(cols) == 34 else NAMES[:20] + ["scheduler"] + NAMES[20:]
                offset = len(names)
                task = dict(zip(names, map(int, cols[:offset])))
                task["features"] = [
                    list(map(float, cols[offset : offset + 5])),
                    list(map(float, cols[offset + 5 : offset + 10])),
                ]
                task.update(
                    {
                        "run_id": result["run_id"],
                        "mode": result["mode"],
                        "seq_len": result["seq_len"],
                        "seed": result["seed"],
                    }
                )
                task["kernel"] = next((task[k] for k in ("k0", "k1", "k2") if task[k] >= 0), -1)
                tasks.append(task)
        elif "QKV_R2_EDGES " in line:
            values = line.split("QKV_R2_EDGES ", 1)[1].split('"')[0]
            edges.extend([list(map(int, pair.split(","))) for pair in values.split(";") if pair])
    phase = [list(map(int, row)) for row in re.findall(r"QKV_R2_PHASE (\d+),(\d+),(\d+),(\d+)", text)]
    meta = re.findall(
        (
            "QKV_R2_META level=(\\d+) tasks=(\\d+) edges=(\\d+) dropped=(\\d+) "
            "phase_samples=(\\d+) early=([\\d,]+) gap_max=(\\d+)"
        ),
        text,
    )
    assert meta and all(int(x[3]) == 0 for x in meta), (folder, meta)
    if result["rounds"] == 1 and int(meta[0][0]) == 2:
        assert len(tasks) == int(meta[0][1]) and len(edges) == int(meta[0][2])
    ids = {t["task"] for t in tasks}
    dangling = sorted({a for a, b in edges if a not in ids} | {b for a, b in edges if b not in ids})
    audit = re.findall(r"QKV_R2_AUDIT submitted=(\d+) traced=(\d+)", text)
    if int(meta[0][0]) == 2 and audit:
        assert all(int(a) == int(b) for a, b in audit), (folder, audit)
        assert not dangling, (folder, "dangling DAG endpoints", dangling)
    return {
        "integrity": {
            "submitted_traced_audit": audit,
            "dangling_endpoint_count": len(dangling),
            "dangling_ids": dangling,
        },
        "run": result,
        "tasks": tasks,
        "edges": edges,
        "phases": phase,
        "meta": meta,
    }


def distribution(values):
    return (
        {
            "n": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }
        if values
        else {"n": 0}
    )


def timeline_stats(tasks):
    pairs = {
        "ready_to_pop_us": ("ready", "pop"),
        "pop_to_publish_us": ("pop", "publish"),
        "queue_to_dispatch_us": ("publish", "dispatch"),
        "ready_to_dispatch_us": ("ready", "dispatch"),
        "ready_to_start_proxy_us": ("ready", "start_proxy"),
        "dispatch_to_start_proxy_us": ("dispatch", "start_proxy"),
        "runtime_proxy_us": ("start_proxy", "fin"),
        "dummy_processing_us": ("dummy_begin", "dummy_end"),
    }
    return {
        name: distribution([(t[end] - t[start]) / 50 for t in tasks if t[start] > 0 and t[end] >= t[start]])
        for name, (start, end) in pairs.items()
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    args = ap.parse_args()
    output = {}
    runs = []
    for result in sorted(args.folder.rglob("result.json")):
        folder = result.parent
        if not (folder / "feedback.txt").exists() or not json.loads(result.read_text())["trace"]:
            continue
        data = parse(folder)
        runs.append(data)
        path = folder / "tasks.jsonl.gz"
        with gzip.open(path, "wt") as f:
            for task in data["tasks"]:
                f.write(json.dumps(task) + "\n")
        (folder / "dag.json").write_text(json.dumps({"run": data["run"], "edges": data["edges"]}))
        phase = {
            name: distribution([row[i] / 50 for row in data["phases"]])
            for i, name in enumerate(("pop", "snapshot", "score", "publish"))
        }
        output[str(folder.relative_to(args.folder))] = {
            "timeline": timeline_stats(data["tasks"]),
            "phase_us": phase,
            "meta": data["meta"],
            "parsed_tasks": len(data["tasks"]),
            "by_shape": {
                str(shape): timeline_stats([t for t in data["tasks"] if t["shape"] == shape]) for shape in range(4)
            },
        }
    (args.folder / "timeline-summary.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
