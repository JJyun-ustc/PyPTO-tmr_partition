#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Frozen QKV batch versus Shared: golden, diagnostics and interleaved timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint():
    paths = list((ROOT / "src").rglob("*.h")) + list((ROOT / "src").rglob("*.cpp"))
    paths += list((ROOT / "build/lib/a2a3/onboard/tensormap_and_ringbuffer").glob("*"))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths) if p.is_file()}


def preflight(device):
    # The repository shell precheck assumes card 0 and an older npu-smi schema.
    # Query the actual container-visible card and still verify CANN's SoC map.
    board = subprocess.check_output(["npu-smi", "info", "-t", "board", "-i", str(device), "-c", "0"], text=True)
    match = re.search(r"Chip Name\s*:\s*(910B\w+)", board)
    if not match:
        raise RuntimeError(f"Cannot identify A2 silicon: {board}")
    soc = "Ascend" + match[1]
    ini = Path(os.environ["ASCEND_HOME_PATH"]) / f"aarch64-linux/data/platform_config/{soc}.ini"
    assert "Short_SoC_version=Ascend910B" in ini.read_text(), ini
    return {"board": board, "soc": soc, "ini_sha256": digest(ini),
            "npu_smi": subprocess.check_output(["npu-smi", "info"], text=True),
            "isolation": "task-submit" if __import__("shutil").which("task-submit") else "unlocked; tool absent"}


def percentile(values, p):
    v = sorted(values)
    x = (len(v) - 1) * p
    i = int(x)
    return v[i] + (v[min(i + 1, len(v) - 1)] - v[i]) * (x - i)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--card", type=int, default=4)
    ap.add_argument("--phase", choices=("validate", "measure"), required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    hardware = preflight(args.card)
    hardware["acl_logical_device"] = args.device
    (out / "hardware.json").write_text(json.dumps(hardware, indent=2))
    # Caller owns the whole study's device lock when task-submit is available.
    print(hardware["isolation"], flush=True)
    manifest = out / "frozen.json"
    current = fingerprint()
    if manifest.exists():
        assert json.loads(manifest.read_text())["fingerprints"] == current, "Frozen sources/binaries changed"
    elif args.phase == "measure":
        raise RuntimeError("Validate before timing; frozen.json is absent")
    else:
        manifest.write_text(json.dumps({"fingerprints": current, "base": "d8f997fab",
            "config": {"batch": 16, "layers": 40, "seq_len": 3500, "seed": 1234,
                       "aicpu_threads": 6, "cpu_threads": 2, "max_batch": 8, "pending_capacity": 64,
                       "budget_ns": 50000, "pressure_weights": [0.5, 0.5]},
            "warmup": "one first round per fresh process; retained in raw, excluded from formal samples",
            "p95": "linear interpolation (n-1)*0.95", "frozen_at_unix": time.time()}, indent=2))
    invfile = out / "invocations.json"
    inv = json.loads(invfile.read_text()) if invfile.exists() else []

    def invoke(mode, label, rounds, golden=False, diagnostic=False):
        assert fingerprint() == current
        dest = out / mode / label
        dest.mkdir(parents=True, exist_ok=False)
        (dest / "ascend").mkdir()
        env = os.environ.copy()
        env.update({"SIMPLER_TMR_QKV_BATCH": "1" if mode == "qkv" else "0",
                    "PYTHONPATH": f"{ROOT}/python:{ROOT}",
                    "ASCEND_PROCESS_LOG_PATH": str(dest / "ascend"),
                    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
        wrapper = ('import multiprocessing,runpy,sys,torch; multiprocessing.cpu_count=lambda:2; '
                   'torch.set_num_threads(2); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")')
        cmd = [sys.executable, "-c", wrapper, str(ROOT / "examples/a2a3/tensormap_and_ringbuffer/benchmark_qwen_cluster/main.py"),
               "-p", "a2a3", "-d", str(args.device), "--seq-len", "3500", "--compile-workers", "2",
               "--rounds", str(rounds), "--log-level", "info"]
        if not golden:
            cmd += ["--skip-golden"]
        if diagnostic:
            cmd += ["--enable-chip-swimlane", "4"]
        print("START", mode, label, flush=True)
        start = time.time()
        with (dest / "host.log").open("w") as log:
            run = subprocess.run(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        host = (dest / "host.log").read_text()
        row = {"mode": mode, "label": label, "returncode": run.returncode, "cmd": cmd,
               "elapsed_s": time.time() - start, "golden": golden, "diagnostic": diagnostic}
        inv.append(row)
        invfile.write_text(json.dumps(inv, indent=2))
        assert run.returncode == 0 and "[qwen] PASSED" in host, dest
        codes = dict(re.findall(r"func_id=(\d+).*?code_fnv1a64=(0x[0-9a-f]+)", host))
        if len(codes) != 37:
            raise RuntimeError(f"Missing actual loaded kernel fingerprints in {dest}: {len(codes)}")
        hashfile = out / "kernel-hashes.json"
        if hashfile.exists():
            assert json.loads(hashfile.read_text()) == codes, dest
        else:
            hashfile.write_text(json.dumps(codes, indent=2))
        device_logs = "\n".join(p.read_text(errors="replace") for p in (dest / "ascend").rglob("*.log"))
        summary_lines = [line for line in device_logs.splitlines() if "QKV_" in line or "AICPU affinity" in line]
        (dest / "scheduler-summary.txt").write_text("\n".join(summary_lines) + "\n")
        if mode == "qkv":
            summaries = re.findall(r"QKV_SUMMARY eligible=(\d+) assigned_tasks=(\d+),(\d+) batches=(\d+).*?assigned_blocks=(\d+),(\d+) claimed_blocks=(\d+),(\d+) pending=(\d+)", device_logs)
            assert summaries and all(int(s[1]) + int(s[2]) > 0 and s[4:6] == s[6:8] and s[8] == "0" for s in summaries), dest
        values = [int(v) / 1e6 for v in re.findall(r"name=chip\.run\.runner_run\.device_wall ts=\d+ dur=(\d+)\b", host)]
        if label.startswith("block"):
            assert len(values) == rounds, (dest, values)
            (dest / "samples.json").write_text(json.dumps({"all_ms": values, "warmup_ms": values[0], "formal_ms": values[1:]}, indent=2))
        print("PASS", mode, label, values, flush=True)

    if args.phase == "validate":
        invoke("shared", "golden", 1, golden=True)
        invoke("qkv", "golden", 1, golden=True)
        invoke("qkv", "diagnostic", 1, diagnostic=True)
        (out / "validated.json").write_text(json.dumps({"golden": ["shared", "qkv"], "fingerprints": current}))
    else:
        assert json.loads((out / "validated.json").read_text())["fingerprints"] == current
        for b in range(6):
            for mode in (("shared", "qkv") if b % 2 == 0 else ("qkv", "shared")):
                invoke(mode, f"block{b}", 6)
        result = {}
        for mode in ("shared", "qkv"):
            vals = []
            for b in range(6):
                vals += json.loads((out / mode / f"block{b}/samples.json").read_text())["formal_ms"]
            assert len(vals) == 30
            result[mode] = {"n": len(vals), "median_ms": statistics.median(vals), "p95_ms": percentile(vals, .95), "all_ms": vals}
        result["change_percent"] = {k: 100 * (result["qkv"][k] / result["shared"][k] - 1) for k in ("median_ms", "p95_ms")}
        (out / "summary.json").write_text(json.dumps(result, indent=2))
        assert fingerprint() == current
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
