#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Counterbalance four frozen runtime variants; retain every ordinary round and separate diagnostics."""

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

SUITE = (
    ("Bgemm", "benchmark_bgemm/test_benchmark_bgemm.py", "TestBenchmarkBgemm", "ClusterCase0Adaptive"),
    ("Scale", "benchmark_bgemm/test_benchmark_bgemm.py", "TestBenchmarkBgemm", "ClusterScaleAdaptive"),
    (
        "Uniform",
        "benchmark_paged_attention/test_cluster_paged_attention.py",
        "TestClusterPagedAttention",
        "UniformAuto",
    ),
    ("Skew", "benchmark_paged_attention/test_cluster_paged_attention.py", "TestClusterPagedAttention", "SkewAuto"),
)
STAGES = ("v8", "v12", "v13", "v13-no-pickup")
# Williams orders: every version occupies each position once and every ordered adjacent pair occurs once.
ORDERS = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))


def hashes(root):
    directory = root / "build/lib/a2a3/onboard/tensormap_and_ringbuffer"
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("libhost_runtime.so", "libaicpu_kernel.so", "aicore_kernel.o")
    }


def percentile(values, q):
    values = sorted(values)
    x = (len(values) - 1) * q
    lo = int(x)
    return values[lo] + (x - lo) * (values[min(lo + 1, len(values) - 1)] - values[lo])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v8", type=Path, required=True)
    parser.add_argument("--v12", type=Path, required=True)
    parser.add_argument("--v13", type=Path, required=True)
    parser.add_argument("--no-pickup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=True)
    roots = dict(zip(STAGES, (args.v8.resolve(), args.v12.resolve(), args.v13.resolve(), args.no_pickup.resolve())))
    assert not any((output / stage).exists() for stage in STAGES), "Refuse overwriting measurements"
    before = {stage: hashes(root) for stage, root in roots.items()}
    for stage in STAGES:
        (output / stage / "device-logs").mkdir(parents=True)
    invocations = []
    samples = {stage: {row[0]: {} for row in SUITE} for stage in STAGES}

    def invoke(stage, entry, batch, rounds, golden=False, diagnostic=False):
        label, script, cls, case = entry
        root = roots[stage]
        directory = output / stage
        wrapper = (
            "import multiprocessing,runpy,sys,torch; multiprocessing.cpu_count=lambda:2; "
            "torch.set_num_threads(2); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
        )
        cmd = [
            str(root / ".venv/bin/python"),
            "-c",
            wrapper,
            str(root / "examples/a2a3/tensormap_and_ringbuffer" / script),
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
            "info" if diagnostic else "timing",
        ]
        if not golden:
            cmd.append("--skip-golden")
        if diagnostic:
            cmd.extend(["--enable-chip-swimlane", "4", "--enable-dep-gen"])
        if shutil.which("task-submit"):
            cmd = ["task-submit", "--timeout", "1800", "--max-time", "1800", "--device", "0", "--run", shlex.join(cmd)]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{root / 'python'}:{root}:/data/pypto-work/pypto-home/venv/lib/python3.10/site-packages"
        env["ASCEND_PROCESS_LOG_PATH"] = str(directory / "device-logs")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[name] = "2"
        host = directory / f"{label}-{batch}.log"
        with host.open("w") as log:
            result = subprocess.run(
                cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=False
            )
        invocations.append(
            {
                "stage": stage,
                "case": label,
                "batch": batch,
                "rounds": rounds,
                "returncode": result.returncode,
                "command": cmd,
            }
        )
        (output / "invocations.json").write_text(json.dumps(invocations, indent=2) + "\n")
        assert result.returncode == 0, host
        text = host.read_text()
        assert f"{cls}::{case} PASSED" in text, host
        if batch.startswith("block"):
            values = [
                int(v) / 1e6 for v in re.findall(r"name=chip\.run\.runner_run\.device_wall ts=\d+ dur=(\d+)\b", text)
            ]
            assert len(values) == rounds, host
            samples[stage][label][batch] = values
        if diagnostic:
            captures = [
                p
                for p in (root / "outputs").glob(f"{cls}_{case}_*/chip_swimlane_records.json")
                if 0 <= host.stat().st_mtime - p.stat().st_mtime < 120
            ]
            assert captures, host
            capture = max(captures, key=lambda p: p.stat().st_mtime)
            dest = directory / "captures" / label
            dest.mkdir(parents=True)
            for name in ("chip_swimlane_records.json", "deps.json"):
                shutil.copyfile(capture.with_name(name), dest / name)
        assert hashes(root) == before[stage]
        print(stage, label, batch, "passed", flush=True)

    for entry in SUITE:
        for stage in STAGES:
            invoke(stage, entry, "golden", 1, golden=True)
    for block, order in enumerate(ORDERS):
        suite = SUITE[block:] + SUITE[:block]
        for entry in suite:
            for index in order:
                invoke(STAGES[index], entry, f"block{block}", 8)
    for entry in SUITE:
        for stage in reversed(STAGES):
            invoke(stage, entry, "diagnostic", 1, diagnostic=True)
    for stage in STAGES:
        cases = {}
        for label, batches in samples[stage].items():
            values = [v for batch in batches.values() for v in batch]
            assert len(values) == 32
            cases[label] = {
                "n": len(values),
                "median_ms": statistics.median(values),
                "p95_ms": percentile(values, 0.95),
                "samples_ms": batches,
                "batch_median_ms": {k: statistics.median(v) for k, v in batches.items()},
            }
        report = {
            "stage": stage,
            "worktree": str(roots[stage]),
            "physical_card": 0,
            "logical_device": 0,
            "binary_hashes": before[stage],
            "cases": cases,
            "protocol": (
                "4 balanced Williams blocks x 8 ordinary rounds = 32; "
                "first process rounds retained; diagnostics separate"
            ),
            "version_orders": [[STAGES[i] for i in order] for order in ORDERS],
            "exclusive_lock": bool(shutil.which("task-submit")),
        }
        (output / stage / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "timing-complete.json").write_text(
        json.dumps({"ordinary_samples": 512, "golden": 16, "diagnostics": 16}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
