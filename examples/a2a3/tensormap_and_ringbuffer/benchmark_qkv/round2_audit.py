#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Audit frozen artifacts, loaded kernels, actual side placement and all samples."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import numpy as np
from round2_analyze import counters
from round2_data import parse
from round2_study import digest, fingerprint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    root = args.root
    frozen = json.loads((root / "measure/frozen.json").read_text())
    assert frozen["fingerprints"] == fingerprint()
    assert json.loads((root / "golden/frozen.json").read_text())["fingerprints"] == frozen["fingerprints"]
    for name, value in frozen["calibration_and_model_hashes"].items():
        assert digest(root / name) == value
    kernels = json.loads((root / "measure/kernel-hashes.json").read_text())
    assert len(kernels) == 37
    for phase in ("golden", "diagnostics", "replay-complete", "shadow"):
        assert json.loads((root / phase / "kernel-hashes.json").read_text()) == kernels
    first = root.parent / "qkv-ready-batch-20261125/kernel-hashes.json"
    assert json.loads(first.read_text()) == kernels
    sample_rows = []
    placement = []
    model_attempts = []
    order = []
    periods = []
    bodies = {}
    for block in range(6):
        for mode in ("shared", "original", "improved", "trained"):
            folder = root / "measure" / mode / f"block{block}"
            run = json.loads((folder / "result.json").read_text())
            assert run["passed"] and run["trace"] == 0
            assert len(run["all_ms"]) == 6 and run["formal_ms"] == run["all_ms"][1:]
            text = (folder / "feedback.txt").read_text()
            assert not any(tag in text for tag in ("QKV_R2_TASKS", "QKV_R2_META", "QKV_DECISION", "QKV_R2_SHADOW"))
            if mode != "shared":
                assert len(counters(folder)) == 6
            if mode == "trained":
                calls = re.findall(r"QKV_R2_LEARNED attempted=(\d+) missing_calibration=(\d+)", text)
                assert len(calls) == 6 and all(int(b) == 0 for a, b in calls)
                model_attempts.extend(int(a) for a, b in calls[1:])
            host = (folder / "host.log").read_text()
            body = [
                int(x) / 1e6
                for x in re.findall(r"name=chip\.run\.runner_run\.device_wall\.sched ts=\d+ dur=(\d+)", host)
            ]
            assert len(body) == 6
            bodies.setdefault(mode, []).extend(body[1:])
            order.append(
                {"block": block, "mode": mode, "completion_mtime_ns": (folder / "result.json").stat().st_mtime_ns}
            )
            sample_rows.extend(
                {
                    "block": block,
                    "mode": mode,
                    "round": i,
                    "warmup": i == 0,
                    "device_wall_ms": v,
                    "run_id": run["run_id"],
                }
                for i, v in enumerate(run["all_ms"])
            )
        baseline = statistics.median(
            json.loads((root / f"measure/shared/block{block}/result.json").read_text())["formal_ms"]
        )
        improved = statistics.median(
            json.loads((root / f"measure/improved/block{block}/result.json").read_text())["formal_ms"]
        )
        learned = statistics.median(
            json.loads((root / f"measure/trained/block{block}/result.json").read_text())["formal_ms"]
        )
        periods.append(
            {
                "block": block,
                "shared_ms": baseline,
                "improved_ms": improved,
                "trained_ms": learned,
                "trained_vs_improved_percent": 100 * (learned / improved - 1),
            }
        )
    for mode in ("shared", "original", "improved", "trained"):
        golden = json.loads((root / f"golden/{mode}/golden/result.json").read_text())
        assert golden["passed"] and golden["golden"]
    full = []
    for p in sorted((root / "replay-complete").rglob("result.json")):
        d = parse(p.parent)
        assert d["integrity"]["dangling_endpoint_count"] == 0 and d["integrity"]["submitted_traced_audit"]
        assert len({t["task"] for t in d["tasks"]}) == len(d["tasks"])
        bound = [t for t in d["tasks"] if t["selected"] >= 0]
        assert all(t["selected"] == t["actual_side"] for t in bound)
        placement.append(
            {"run_id": d["run"]["run_id"], "bound_tasks": len(bound), "all_observed_on_selected_side": True}
        )
        full.append(
            {
                "run_id": d["run"]["run_id"],
                "tasks": len(d["tasks"]),
                "edges": len(d["edges"]),
                "integrity": d["integrity"],
            }
        )
    assert len(full) == 6
    order.sort(key=lambda r: r["completion_mtime_ns"])
    (root / "collection-order.json").write_text(json.dumps(order, indent=2))
    with (root / "samples.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    delta = np.array([r["trained_vs_improved_percent"] for r in periods])
    rng = np.random.default_rng(1234)
    boot = np.median(delta[rng.integers(0, 6, (10000, 6))], axis=1)
    result = {
        "formal_n": 120,
        "warmup_n": 24,
        "policies": 4,
        "goldens": 4,
        "same_loaded_kernel_ids": 37,
        "first_round_kernel_match": True,
        "source_and_runtime_frozen": True,
        "calibration_weights_gate_frozen": True,
        "periods": periods,
        "exploratory_period_bootstrap": {
            "unit": "whole period, six periods",
            "seed": 1234,
            "resamples": 10000,
            "statistic": "median of within-period median percent changes, trained versus improved",
            "ci95_percent": np.percentile(boot, [2.5, 97.5]).tolist(),
        },
        "formal_scheduler_body_median_ms": {k: statistics.median(v) for k, v in bodies.items()},
        "learned_attempts_formal": sum(model_attempts),
        "learned_missing_calibration_formal": 0,
        "complete_dags": full,
        "placement": placement,
    }
    (root / "audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("complete_dags", "placement", "periods")}, indent=2))


if __name__ == "__main__":
    main()
