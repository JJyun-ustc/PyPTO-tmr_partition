#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Interleaved, same-input placement calibration of Qwen kernels 33 -> 34."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from qwen_pair import generate_orchestration
from run_interleaved import ROOT, parse_host, read_device


def run_period(folder, source, period, smoke, core=False):
    rows = []
    for repeat in range(1 if smoke else (6 if core else 21)):
        for side in (0, 1):
            for arm in (0, 1):
                cross = 1 - arm if (repeat + period) % 2 else arm
                rows.append(
                    {
                        "mode": "shared",
                        "period": period,
                        "repeat": repeat,
                        "warmup": repeat == 0,
                        "consumer_side": side,
                        "producer_side": 1 - side if cross else side,
                        "cross": cross,
                        "ordinal": len(rows),
                    }
                )
    manifest = {
        "phase": "qwen_pair_calibration",
        "seq_len": 3500,
        "qwen_pair": True,
        "orchestration_source": str(source),
        "schedule": rows,
        "golden_ordinals": list(range(4)) if period == 0 and not smoke and not core else [],
    }
    if core:
        manifest["profile_root"] = str(folder / "core_records")
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "ascend").mkdir()
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    env = os.environ.copy()
    env.update(
        SIMPLER_TMR_QKV_TRACE="3",
        ASCEND_PROCESS_LOG_PATH=str(folder / "ascend"),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONPATH=f"{ROOT}/python:{ROOT}",
    )
    cmd = [sys.executable, str(Path(__file__).with_name("run_interleaved.py")), "_worker", str(path)]
    (folder / "command.json").write_text(json.dumps(cmd, indent=2))
    print("START Qwen-pair period", period, "runs", len(rows), flush=True)
    with (folder / "host.log").open("w") as log:
        run = subprocess.run(cmd, check=False, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    (folder / "exit.json").write_text(json.dumps({"returncode": run.returncode}))
    if run.returncode:
        raise RuntimeError(folder)
    samples = parse_host((folder / "host.log").read_text(), rows, require_timing=not core)
    actual = read_device(folder)
    if len(actual) != len(samples) or any(r["mode"] != 0 or r["tasks"] != 160 or r["dropped"] for r in actual):
        raise ValueError(f"incomplete sparse trace: {len(actual)} launches")
    (folder / "samples.json").write_text(json.dumps(samples, indent=2))
    print("PASS Qwen-pair period", period, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "collect", "core-smoke", "core"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-counts", type=Path)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    if args.phase.startswith("core"):
        if args.expected_counts is None:
            parser.error("core calibration requires --expected-counts from the full baseline trace")
        counts = json.loads(args.expected_counts.read_text())
        (folder / "expected_compute_counts.json").write_text(json.dumps(counts, indent=2))
    baseline_path = ROOT / "examples/a2a3/tensormap_and_ringbuffer/benchmark_qkv/run_study.py"
    spec = importlib.util.spec_from_file_location("qwen_pair_baseline", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    source = ROOT / "build_output/streaming-qwen-pair/decode_controlled.cpp"
    original = (
        ROOT
        / "examples/a2a3/tensormap_and_ringbuffer/benchmark_qwen_cluster/kernels/orchestration/decode_fwd_layers.cpp"
    )
    generate_orchestration(original, source)
    before = baseline.fingerprint()
    (folder / "frozen.json").write_text(
        json.dumps(
            {
                "hardware": baseline.preflight(3),
                "fingerprints": before,
                "original_orchestration_sha256": baseline.digest(original),
                "controlled_orchestration_sha256": baseline.digest(source),
                "record_scope": "first tile per layer; actual Qwen kernels and matched physical groups",
                "uncertainty": "placement under the actual decode background; not an isolated hardware-transfer label",
            },
            indent=2,
        )
    )
    for period in range(1 if args.phase in ("smoke", "core-smoke") else 6):
        if baseline.fingerprint() != before:
            raise ValueError("runtime changed")
        run_period(
            folder / f"period{period}",
            source,
            period,
            args.phase in ("smoke", "core-smoke"),
            core=args.phase.startswith("core"),
        )
    if baseline.fingerprint() != before:
        raise ValueError("runtime changed during collection")


if __name__ == "__main__":
    main()
