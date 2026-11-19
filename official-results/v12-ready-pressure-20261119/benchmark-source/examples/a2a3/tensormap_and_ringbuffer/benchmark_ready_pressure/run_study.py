#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Measure combined block-pressure Auto stages against a separate, frozen official Shared worktree."""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
from pathlib import Path

BINARIES = ("libhost_runtime.so", "libaicpu_kernel.so", "aicore_kernel.o")
SUITE = (
    ("Bgemm", "benchmark_bgemm/test_benchmark_bgemm.py", "TestBenchmarkBgemm", "ClusterCase0", "Adaptive"),
    ("Scale", "benchmark_bgemm/test_benchmark_bgemm.py", "TestBenchmarkBgemm", "ClusterScale", "Adaptive"),
    (
        "Uniform",
        "benchmark_paged_attention/test_cluster_paged_attention.py",
        "TestClusterPagedAttention",
        "Uniform",
        "Auto",
    ),
    ("Skew", "benchmark_paged_attention/test_cluster_paged_attention.py", "TestClusterPagedAttention", "Skew", "Auto"),
)


def hashes(root):
    return {
        name: hashlib.sha256((root / "build/lib/a2a3/onboard/tensormap_and_ringbuffer" / name).read_bytes()).hexdigest()
        for name in BINARIES
    }


def percentile(values, q):
    x = sorted(values)
    pos = (len(x) - 1) * q
    lo = int(pos)
    return x[lo] + (pos - lo) * (x[min(lo + 1, len(x) - 1)] - x[lo])


def main():  # noqa: PLR0912
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-card", type=int, required=True)
    parser.add_argument("--shared", action="store_true")
    args = parser.parse_args()
    official = args.shared
    root = args.runtime.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error("Result directory already exists")
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "device-logs"
    logs.mkdir()
    before = hashes(root)
    report = {
        "stage": args.stage,
        "physical_card": args.physical_card,
        "logical_device": 0,
        "exclusive_lock": bool(shutil.which("task-submit")),
        "worktree": str(root),
        "binary_hashes": before,
        "cases": {},
        "protocol": "golden + 15 forward + 15 reverse; diagnostics separate; torch threads=2 before worker forks",
    }
    for command in (("npu-smi", "info"), ("npu-smi", "info", "-t", "board", "-i", str(args.physical_card), "-c", "0")):
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        with (output / "hardware-preflight.log").open("a") as f:
            f.write(result.stdout + result.stderr)
    assert "910B2" in (output / "hardware-preflight.log").read_text()
    samples = {entry[0]: {} for entry in SUITE}
    invocations = []
    for batch, suite, rounds in (
        ("golden", SUITE, 1),
        ("a", SUITE, 15),
        ("b", tuple(reversed(SUITE)), 15),
        ("diagnostic", SUITE, 1),
    ):
        for label, script, cls, stem, auto_suffix in suite:
            case = stem + ("Shared" if official else auto_suffix)
            test = root / "examples/a2a3/tensormap_and_ringbuffer" / script
            wrapper = (
                "import multiprocessing,runpy,sys,torch; multiprocessing.cpu_count=lambda:2; "
                "torch.set_num_threads(2); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
            )
            cmd = [
                str(root / ".venv/bin/python"),
                "-c",
                wrapper,
                str(test),
                "-p",
                "a2a3",
                "-d",
                "0",
                "--manual",
                "include",
                "--case",
                case,
                "--rounds",
                str(rounds),
                "--log-level",
                "info" if batch == "diagnostic" else "timing",
            ]
            if batch != "golden":
                cmd.append("--skip-golden")
            if batch == "diagnostic":
                cmd.extend(["--enable-chip-swimlane", "4", "--enable-dep-gen"])
            queue = shutil.which("task-submit")
            if queue:
                cmd = [queue, "--timeout", "1800", "--max-time", "1800", "--device", "0", "--run", shlex.join(cmd)]
            env = os.environ.copy()
            env["PYTHONPATH"] = (
                str(root / "python")
                + ":"
                + str(root)
                + ":/data/pypto-work/pypto-home/venv/lib/python3.10/site-packages"
            )
            env["ASCEND_PROCESS_LOG_PATH"] = str(logs)
            host = output / f"{label}-{batch}.log"
            with host.open("w") as f:
                result = subprocess.run(
                    cmd, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT, check=False, timeout=600
                )
            invocations.append(
                {"label": label, "batch": batch, "case": case, "returncode": result.returncode, "command": cmd}
            )
            (output / "invocations.json").write_text(json.dumps(invocations, indent=2) + "\n")
            assert result.returncode == 0, host
            text = host.read_text()
            assert f"{cls}::{case} PASSED" in text, host
            if batch in ("a", "b"):
                values = [
                    int(v) / 1e6
                    for v in re.findall(r"name=chip\.run\.runner_run\.device_wall ts=\d+ dur=(\d+)\b", text)
                ]
                assert len(values) == rounds, (host, len(values))
                samples[label][batch] = values
            if batch == "diagnostic":
                choices = [
                    p
                    for p in (root / "outputs").glob(f"{cls}_{case}_*/chip_swimlane_records.json")
                    if 0 <= host.stat().st_mtime - p.stat().st_mtime < 120
                ]
                assert choices, host
                capture = max(choices, key=lambda p: p.stat().st_mtime)
                destination = output / "captures" / label
                destination.mkdir(parents=True)
                for name in ("chip_swimlane_records.json", "deps.json"):
                    p = capture.with_name(name)
                    if p.exists():
                        shutil.copyfile(p, destination / name)
            print(args.stage, label, batch, "passed", flush=True)
    assert hashes(root) == before
    for name, batches in samples.items():
        values = batches["a"] + batches["b"]
        report["cases"][name] = {
            "n": len(values),
            "median_ms": statistics.median(values),
            "p95_ms": percentile(values, 0.95),
            "samples_ms": batches,
            "batch_median_ms": {key: statistics.median(v) for key, v in batches.items()},
        }
    report["postflight"] = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True).stdout
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
