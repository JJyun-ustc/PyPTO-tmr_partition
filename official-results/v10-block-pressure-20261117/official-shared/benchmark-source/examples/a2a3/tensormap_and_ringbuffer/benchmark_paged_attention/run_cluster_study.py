#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare Shared and automatic paged attention with paired inputs and binaries."""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.with_name("benchmark_bgemm")))
from reproduce_cluster_study import RUNTIME_ROOT, binary_hashes, percentile, verify_hardware  # noqa: E402

CASES = ("UniformShared", "UniformAuto", "SkewShared", "SkewAuto")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    hardware = verify_hardware(0, output)
    queue = shutil.which("task-submit")
    logs = output / "device-logs"
    logs.mkdir()
    os.environ["ASCEND_PROCESS_LOG_PATH"] = str(logs)
    before = binary_hashes()
    samples = {case: {} for case in CASES}
    invocation = []
    for batch, cases, rounds in (
        ("golden", CASES, 1),
        ("a", CASES, 15),
        ("b", tuple(reversed(CASES)), 15),
        ("diagnostic", CASES, 1),
    ):
        for case in cases:
            command = [
                sys.executable,
                str(HERE / "test_cluster_paged_attention.py"),
                "-p",
                "a2a3",
                "-d",
                "0",
                "--case",
                case,
                "--manual",
                "include",
                "--rounds",
                str(rounds),
                "--log-level",
                "info" if batch == "diagnostic" else "timing",
            ]
            if batch in ("a", "b"):
                command.append("--skip-golden")
            if batch == "diagnostic":
                command.extend(["--enable-chip-swimlane", "4", "--enable-dep-gen"])
            if queue:
                command = [
                    queue,
                    "--timeout",
                    "1800",
                    "--max-time",
                    "1800",
                    "--device",
                    "0",
                    "--run",
                    shlex.join(command),
                ]
            path = output / f"{case}-{batch}.log"
            with path.open("w") as log:
                subprocess.run(command, cwd=RUNTIME_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            text = path.read_text()
            assert f"TestClusterPagedAttention::{case} PASSED" in text, path
            if batch in ("a", "b"):
                values = [
                    int(v) / 1e6
                    for v in re.findall(r"name=chip\.run\.runner_run\.device_wall ts=\d+ dur=(\d+)\b", text)
                ]
                assert len(values) == rounds, (path, len(values))
                samples[case][batch] = values
            if batch == "diagnostic":
                candidates = [
                    p
                    for p in (RUNTIME_ROOT / "outputs").glob(f"TestClusterPagedAttention_{case}_*")
                    if (p / "chip_swimlane_records.json").exists()
                    and 0 <= path.stat().st_mtime - (p / "chip_swimlane_records.json").stat().st_mtime < 180
                ]
                assert candidates, path
                capture = max(candidates, key=lambda p: p.stat().st_mtime)
                target = output / "captures" / case
                target.mkdir(parents=True)
                for name in ("chip_swimlane_records.json", "deps.json"):
                    source = capture / name
                    if source.exists():
                        shutil.copyfile(source, target / name)
            invocation.append({"case": case, "batch": batch, "command": command, "passed": True})
            (output / "invocations.json").write_text(json.dumps(invocation, indent=2) + "\n")
            print(case, batch, "passed", flush=True)
    assert binary_hashes() == before
    report = {
        "hardware": hardware,
        "exclusive_lock": bool(queue),
        "binary_hashes": before,
        "algorithm": "v8 queue-future; no cross-cluster pickup",
        "cases": {},
    }
    for case, batches in samples.items():
        values = batches["a"] + batches["b"]
        report["cases"][case] = {
            "n": len(values),
            "median_ms": statistics.median(values),
            "p95_ms": percentile(values, 0.95),
            "samples_ms": batches,
            "batch_median_ms": {batch: statistics.median(v) for batch, v in batches.items()},
        }
    for family in ("Uniform", "Skew"):
        c = report["cases"]
        report[family] = {
            "automatic_vs_shared_pct": 100 * (c[family + "Auto"]["median_ms"] / c[family + "Shared"]["median_ms"] - 1)
        }
    sources = [
        HERE / "test_cluster_paged_attention.py",
        HERE / "kernels/orchestration/paged_attention_orch.cpp",
        RUNTIME_ROOT / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/aicpu_cluster_partition.h",
    ]
    report["source_sha256"] = {
        str(p.relative_to(RUNTIME_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    report["postflight"] = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True).stdout
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({f: report[f] for f in ("Uniform", "Skew")}, indent=2))


if __name__ == "__main__":
    main()
