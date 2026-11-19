#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run golden checks and two reversed BGEMM timing batches on a 910B2."""

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parents[4]
CASES = ("Case0Shared", "Case0Adaptive", "ScaleShared", "ScaleAdaptive")
DEVICE_WALL = re.compile(r"name=chip\.run\.runner_run\.device_wall ts=\d+ dur=(\d+)\b")


def binary_hashes():
    directory = RUNTIME_ROOT / "build/lib/a2a3/onboard/tensormap_and_ringbuffer"
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("libhost_runtime.so", "libaicpu_kernel.so", "aicore_kernel.o")
    }


def verify_hardware(physical_card, output):
    if not shutil.which("npu-smi"):
        raise RuntimeError("npu-smi is required for this hardware-only study")
    commands = [
        ["npu-smi", "info"],
        ["npu-smi", "info", "-t", "board", "-i", str(physical_card), "-c", "0"],
    ]
    captures = []
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        captures.append(result.stdout + result.stderr)
    (output / "hardware-preflight.log").write_text("\n".join(captures))
    board = captures[1]
    full_name = re.search(r"Chip Name\s*:\s*(?:Ascend)?910B2\b", board)
    split_name = re.search(r"Chip Name\s*:\s*Ascend910\b", board) and re.search(r"NPU Name\s*:\s*B2\b", board)
    if not (full_name or split_name):
        raise RuntimeError("This study requires 910B2 silicon; inspect hardware-preflight.log")
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    config = cann / f"{platform.machine()}-linux/data/platform_config/Ascend910B2.ini"
    if not config.is_file() or "Short_SoC_version=Ascend910B" not in config.read_text():
        raise RuntimeError("Cannot confirm the 910B CANN platform configuration")
    precheck = subprocess.run(
        [str(RUNTIME_ROOT / ".claude/skills/onboard-arch-precheck/check.sh"), "a2a3"],
        capture_output=True,
        text=True,
        check=False,
    )
    (output / "architecture-precheck.log").write_text(precheck.stdout + precheck.stderr)
    return {
        "physical_card": physical_card,
        "architecture_precheck_returncode": precheck.returncode,
        "explicit_card_and_cann_check": "passed",
    }


def run_case(case, rounds, golden, output, device, queue):
    command = [
        sys.executable,
        str(Path(__file__).with_name("test_benchmark_bgemm.py")),
        "-p",
        "a2a3",
        "-d",
        str(device),
        "--case",
        f"Cluster{case}",
        "--manual",
        "include",
        "--rounds",
        str(rounds),
        "--log-level",
        "info" if golden else "timing",
    ]
    if not golden:
        command.append("--skip-golden")
    if queue:
        command = [
            queue,
            "--timeout",
            "1800",
            "--max-time",
            "1800",
            "--device",
            str(device),
            "--run",
            shlex.join(command),
        ]
    with output.open("w") as log:
        subprocess.run(command, cwd=RUNTIME_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    content = output.read_text()
    if f"TestBenchmarkBgemm::Cluster{case} PASSED" not in content:
        raise RuntimeError(f"Missing success marker: {output}")
    values = [int(ns) / 1e6 for ns in DEVICE_WALL.findall(content)]
    if not golden and len(values) != rounds:
        raise RuntimeError(f"Expected {rounds} device-wall markers in {output}, found {len(values)}")
    print(f"{case}: {'golden' if golden else str(rounds) + ' timing rounds'} passed", flush=True)
    return values


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Container logical device ID")
    parser.add_argument("--physical-card", type=int, required=True, help="Physical card for the board query")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; must not exist")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    hardware = verify_hardware(args.physical_card, output)
    queue = shutil.which("task-submit")
    if not queue:
        print("task-submit unavailable: results will be unlocked", file=sys.stderr, flush=True)
    logs = output / "device-logs"
    logs.mkdir()
    os.environ["ASCEND_PROCESS_LOG_PATH"] = str(logs)
    performance = output / "performance"
    performance.mkdir()
    before = binary_hashes()
    for case in CASES:
        run_case(case, 1, True, output / f"Cluster{case}-smoke.log", args.device, queue)
    samples = {case: {} for case in CASES}
    for batch, cases in (("a", CASES), ("b", tuple(reversed(CASES)))):
        for case in cases:
            samples[case][batch] = run_case(case, 15, False, performance / f"{case}-{batch}.log", args.device, queue)
    after = binary_hashes()
    (performance / "binaries-before.sha256").write_text(
        "".join(
            f"{digest}  {RUNTIME_ROOT / 'build/lib/a2a3/onboard/tensormap_and_ringbuffer' / name}\n"
            for name, digest in before.items()
        )
    )
    (performance / "binaries-after.sha256").write_text(
        "".join(
            f"{digest}  {RUNTIME_ROOT / 'build/lib/a2a3/onboard/tensormap_and_ringbuffer' / name}\n"
            for name, digest in after.items()
        )
    )
    if before != after:
        raise RuntimeError("Runtime binaries changed during the suite")
    report = {"hardware": hardware, "logical_device": args.device, "exclusive_lock": bool(queue), "cases": {}}
    for case, batches in samples.items():
        values = batches["a"] + batches["b"]
        report["cases"][case] = {
            "n": len(values),
            "median_ms": statistics.median(values),
            "p95_ms": percentile(values, 0.95),
            "samples_ms": batches,
            "batch_median_ms": {name: statistics.median(v) for name, v in batches.items()},
        }
    for family in ("Case0", "Scale"):
        cases = report["cases"]
        report[family] = {
            "automatic_vs_shared_pct": 100
            * (cases[family + "Adaptive"]["median_ms"] / cases[family + "Shared"]["median_ms"] - 1)
        }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({family: report[family] for family in ("Case0", "Scale")}, indent=2))


if __name__ == "__main__":
    main()
