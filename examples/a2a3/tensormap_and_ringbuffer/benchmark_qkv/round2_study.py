#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Execute the second-round QKV investigation with immutable phase manifests."""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from run_study import ROOT, digest, fingerprint, percentile, preflight

MODES = {
    "shared": 0,
    "original": 1,
    "base_only": 2,
    "improved": 3,
    "trained": 4,
    "central_shared": 6,
}
WRAPPER = (
    "import multiprocessing,runpy,sys,torch; multiprocessing.cpu_count=lambda:2; "
    'torch.set_num_threads(2);sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")'
)


def invoke(out, mode, label, rounds=1, seq=3500, seed=1234, trace=0, golden=False):
    dest = out / mode / label
    if (dest / "result.json").exists():
        previous = json.loads((dest / "result.json").read_text())
        if previous["returncode"] == 0 and previous.get("passed", False):
            print("RESUME", mode, label, flush=True)
            return previous
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "ascend").mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "SIMPLER_TMR_QKV_BATCH": str(MODES[mode]),
            "SIMPLER_TMR_QKV_TRACE": str(trace),
            "PYTHONPATH": f"{ROOT}/python:{ROOT}",
            "ASCEND_PROCESS_LOG_PATH": str(dest / "ascend"),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
        }
    )
    script = ROOT / "examples/a2a3/tensormap_and_ringbuffer/benchmark_qwen_cluster/main.py"
    cmd = [
        sys.executable,
        "-c",
        WRAPPER,
        str(script),
        "-p",
        "a2a3",
        "-d",
        "0",
        "--seq-len",
        str(seq),
        "--seed",
        str(seed),
        "--compile-workers",
        "2",
        "--rounds",
        str(rounds),
        "--log-level",
        "info",
    ]
    if not golden:
        cmd.append("--skip-golden")
    print("START", mode, label, "trace", trace, flush=True)
    begin = time.time()
    with (dest / "host.log").open("w") as log:
        run = subprocess.run(cmd, check=False, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    text = (dest / "host.log").read_text()
    values = [int(x) / 1e6 for x in re.findall(r"name=chip\.run\.runner_run\.device_wall ts=\d+ dur=(\d+)\b", text)]
    row = {
        "mode": mode,
        "label": label,
        "rounds": rounds,
        "seq_len": seq,
        "seed": seed,
        "trace": trace,
        "golden": golden,
        "returncode": run.returncode,
        "passed": run.returncode == 0 and "[qwen] PASSED" in text,
        "elapsed_s": time.time() - begin,
        "cmd": cmd,
        "all_ms": values,
        "formal_ms": values[1:] if label.startswith("block") else [],
        "run_id": f"{mode}/{label}",
    }
    (dest / "result.json").write_text(json.dumps(row, indent=2))
    assert run.returncode == 0 and "[qwen] PASSED" in text, dest
    assert len(values) == rounds, (dest, values)
    codes = dict(re.findall(r"func_id=(\d+).*?code_fnv1a64=(0x[0-9a-f]+)", text))
    assert len(codes) == 37, (dest, len(codes))
    hashes = out / "kernel-hashes.json"
    if hashes.exists():
        assert json.loads(hashes.read_text()) == codes, dest
    else:
        hashes.write_text(json.dumps(codes, indent=2))
    logs = list((dest / "ascend/debug/device-0").glob("*.log"))
    device = "\n".join(p.read_text(errors="replace") for p in logs)
    (dest / "feedback.txt").write_text("\n".join(line for line in device.splitlines() if "QKV_" in line) + "\n")
    if trace:
        assert "QKV_R2_META" in device, dest
    print("PASS", mode, label, values, flush=True)
    return row


def measure_phase(phase, args, gate, before):
    modes = args.modes or (
        ["shared", "original", "improved", "trained"] if gate["deploy"] else ["shared", "original", "improved"]
    )
    for block in range(6):
        order = modes[(block // 2) % len(modes) :] + modes[: (block // 2) % len(modes)]
        if block % 2:
            order = order[::-1]
        for mode in order:
            assert fingerprint() == before
            invoke(phase, mode, f"block{block}", rounds=6)
    result = {}
    for mode in modes:
        values = []
        for block in range(6):
            values += json.loads((phase / mode / f"block{block}/result.json").read_text())["formal_ms"]
        assert len(values) == 30
        result[mode] = {
            "n": 30,
            "median_ms": statistics.median(values),
            "p95_ms": percentile(values, 0.95),
            "all_ms": values,
        }
    for mode in modes:
        result[mode]["vs_shared_percent"] = {
            key: 100 * (result[mode][key] / result["shared"][key] - 1) for key in ("median_ms", "p95_ms")
        }
    (phase / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def freeze_manifest(manifest, before, model_hashes, phase_name):
    if manifest.exists():
        assert json.loads(manifest.read_text())["fingerprints"] == before, "phase binary/source changed"
    else:
        manifest.write_text(
            json.dumps(
                {
                    "fingerprints": before,
                    "phase": phase_name,
                    "modes": MODES,
                    "calibration_and_model_hashes": model_hashes,
                    "warmup": "one retained first round per fresh process, excluded from formal",
                    "gate": (
                        "attention holdout composite MAE at least 2% below linear; neither target worse; "
                        "CPU batch-8 p95 <=50us"
                    ),
                },
                indent=2,
            )
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--phase",
        choices=(
            "probe",
            "calibration",
            "replay-complete",
            "golden",
            "measure",
            "shadow",
        ),
        required=True,
    )
    ap.add_argument("--modes", nargs="+", choices=MODES)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "hardware.json").write_text(json.dumps(preflight(4), indent=2))
    phase = out / args.phase
    phase.mkdir(exist_ok=True)
    before = fingerprint()
    manifest = phase / "frozen.json"
    model_hashes = {
        str(p.relative_to(out)): digest(p)
        for p in (
            out / "models/calibration.json",
            out / "models/weights.json",
            out / "models/gate.json",
        )
        if p.exists()
    }
    if args.phase == "measure":
        assert len(model_hashes) == 3, "complete offline training and gates before formal timing"
        gate = json.loads((out / "models/gate.json").read_text())
        assert not gate["deploy"] or gate.get("actual_aicpu_budget_pass"), (
            "measure actual AICPU shadow before trained dispatch"
        )
    freeze_manifest(manifest, before, model_hashes, args.phase)
    if args.phase == "probe":
        for mode in args.modes or ["shared", "original", "base_only"]:
            invoke(phase, mode, "sample", trace=1)
            invoke(phase, mode, "short", rounds=6)
    elif args.phase == "calibration":
        for seq in [512, 1750, 2625, 3500]:
            for repeat in range(3):
                mode = "improved" if repeat % 2 == 0 else "original"
                invoke(phase, mode, f"len{seq}-run{repeat}", seq=seq, trace=2)
    elif args.phase == "replay-complete":
        specs = [
            ("original", 512),
            ("original", 1750),
            ("improved", 2625),
            ("improved", 3500),
            ("original", 3500),
            ("trained", 3500),
        ]
        for mode, seq in specs:
            invoke(phase, mode, f"replay-len{seq}", seq=seq, trace=2)
    elif args.phase == "golden":
        modes = ["shared", "original", "improved"]
        if (out / "models/gate.json").exists() and json.loads((out / "models/gate.json").read_text())["deploy"]:
            assert json.loads((out / "models/gate.json").read_text()).get("actual_aicpu_budget_pass")
            modes.append("trained")
        for mode in args.modes or modes:
            invoke(phase, mode, "golden", golden=True)
    elif args.phase == "shadow":
        invoke(phase, "improved", "shadow", trace=2, rounds=1)
    else:
        measure_phase(phase, args, gate, before)

    assert fingerprint() == before
    if args.phase == "measure":
        assert all(digest(out / p) == v for p, v in model_hashes.items()), (
            "model/calibration changed during formal timing"
        )


if __name__ == "__main__":
    main()
