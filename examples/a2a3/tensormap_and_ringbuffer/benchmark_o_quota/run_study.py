#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reproduce the frozen O-quota, v8 and upstream Shared comparison."""

import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path

SUITE = (
    ("Bgemm", "benchmark_bgemm/test_benchmark_bgemm.py", "ClusterCase0Adaptive", "ClusterCase0Shared"),
    ("Scale", "benchmark_bgemm/test_benchmark_bgemm.py", "ClusterScaleAdaptive", "ClusterScaleShared"),
    ("Uniform", "benchmark_paged_attention/test_cluster_paged_attention.py", "UniformAuto", "UniformShared"),
    ("Skew", "benchmark_paged_attention/test_cluster_paged_attention.py", "SkewAuto", "SkewShared"),
    ("Qwen", "benchmark_qwen_cluster/main.py", None, None),
)
STAGES = ("shared", "v8", "v14")
ORDERS = list(itertools.permutations(STAGES))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fingerprint(root):
    files = subprocess.check_output(["git", "ls-files", "src"], cwd=root, text=True).splitlines()
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "src": {f: digest(root / f) for f in files},
        "binaries": {p.name: digest(p) for p in (root / "build/lib/a2a3/onboard/tensormap_and_ringbuffer").iterdir()},
    }


def percentile(v):
    v = sorted(v)
    pos = (len(v) - 1) * 0.95
    i = int(pos)
    return v[i] + (pos - i) * (v[min(i + 1, len(v) - 1)] - v[i])


def resume_invocation(dest, out, stage, label, name, inv, samples):
    completed = any(
        row["stage"] == stage and row["case"] == label and row["name"] == name and row["returncode"] == 0 for row in inv
    )
    if completed and (not name.startswith("block") or name in samples[stage].get(label, {})):
        print("RESUME", stage, label, name, flush=True)
        return True
    archive = out / "interrupted" / f"{stage}-{label}-{name}-{time.time_ns()}"
    archive.parent.mkdir(exist_ok=True)
    dest.rename(archive)
    return False


def collect_capture(root, dest, captures_before):
    captures = set((root / "outputs").glob("*/chip_swimlane_records.json")) - captures_before
    assert len(captures) == 1, (dest, captures)
    capture_dir = dest / "capture"
    capture_dir.mkdir()
    for capture in captures:
        for artifact in capture.parent.glob("*.json"):
            if artifact.name in (
                "chip_swimlane_records.json",
                "deps.json",
                "func_names.json",
            ) or artifact.name.startswith("name_map_"):
                shutil.copyfile(artifact, capture_dir / artifact.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--v8", type=Path, required=True)
    parser.add_argument("--v14", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "measure", "diagnostics"), required=True)
    parser.add_argument("--resume", action="store_true", help="Keep validated runs and archive interrupted runs")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--cases", nargs="+", choices=[row[0] for row in SUITE], default=[row[0] for row in SUITE])
    args = parser.parse_args()
    selected = [row for row in SUITE if row[0] in args.cases]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    roots = {s: getattr(args, s).resolve() for s in STAGES}
    manifest = out / "before.json"
    if manifest.exists():
        before = json.loads(manifest.read_text())
        for s, root in roots.items():
            assert fingerprint(root) == before[s], s
    else:
        before = {s: fingerprint(root) for s, root in roots.items()}
        manifest.write_text(json.dumps(before, indent=2) + "\n")
    invfile = out / "invocations.json"
    inv = json.loads(invfile.read_text()) if invfile.exists() else []
    samplefile = out / "samples.json"
    samples = json.loads(samplefile.read_text()) if samplefile.exists() else {s: {} for s in STAGES}

    def invoke(stage, entry, name, rounds, golden=False, diagnostic=False):
        label, script, auto, shared = entry
        root = roots[stage]
        dest = out / stage / label / name
        if args.resume and dest.exists():
            if resume_invocation(dest, out, stage, label, name, inv, samples):
                assert fingerprint(root) == before[stage], stage
                return
        assert not dest.exists(), dest
        dest.mkdir(parents=True)
        (dest / "device-logs").mkdir()
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
            "--rounds",
            str(rounds),
            "--log-level",
            "info" if diagnostic else "timing",
        ]
        if label == "Qwen":
            cmd += ["--seq-len", "3500", "--compile-workers", "2"]
        else:
            cmd += ["--manual", "include", "--case", shared if stage == "shared" else auto]
        if not golden:
            cmd += ["--skip-golden"]
        if diagnostic:
            cmd += ["--enable-chip-swimlane", "4", "--enable-dep-gen"]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{root}/python:{root}:/data/pypto-work/pypto-home/venv/lib/python3.10/site-packages"
        env["ASCEND_PROCESS_LOG_PATH"] = str(dest / "device-logs")
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[k] = "2"
        print("START", stage, label, name, flush=True)
        captures_before = set((root / "outputs").glob("*/chip_swimlane_records.json")) if diagnostic else set()
        with (dest / "host.log").open("w") as log:
            run = subprocess.run(
                cmd, check=False, cwd=dest, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800
            )
        inv.append({"stage": stage, "case": label, "name": name, "returncode": run.returncode, "cmd": cmd})
        invfile.write_text(json.dumps(inv, indent=2) + "\n")
        text = (dest / "host.log").read_text()
        assert run.returncode == 0, dest
        if diagnostic:
            collect_capture(root, dest, captures_before)
        if label == "Qwen":
            assert "[qwen] PASSED" in text, dest
        else:
            assert "PASSED" in text or "PASS" in text, dest
        codes = dict(re.findall(r"func_id=(\d+).*?code_fnv1a64=(0x[0-9a-f]+)", text))
        assert len(codes) == (37 if label == "Qwen" else 2 if label in ("Bgemm", "Scale") else 4), (dest, codes)
        hashfile = out / (label + "-kernel-hashes.json")
        if hashfile.exists():
            assert json.loads(hashfile.read_text()) == codes, dest
        else:
            hashfile.write_text(json.dumps(codes, indent=2) + "\n")
        if name.startswith("block"):
            vals = [
                int(v) / 1e6 for v in re.findall(r"name=chip\.run\.runner_run\.device_wall ts=\d+ dur=(\d+)\b", text)
            ]
            assert len(vals) == rounds, (dest, vals)
            samples[stage].setdefault(label, {})[name] = vals
            samplefile.write_text(json.dumps(samples, indent=2) + "\n")
            print("MEDIAN", stage, label, statistics.median(vals), flush=True)
        assert fingerprint(root) == before[stage], stage
        print("PASS", stage, label, name, flush=True)

    if args.phase == "preflight":
        for stage in args.stages:
            for entry in selected:
                invoke(stage, entry, "golden", 1, golden=True)
    elif args.phase == "measure":
        for i, order in enumerate(ORDERS):
            for stage in order:
                for entry in SUITE[i % len(SUITE) :] + SUITE[: i % len(SUITE)]:
                    invoke(stage, entry, f"block{i}", 5)
        summary = {}
        for stage in STAGES:
            summary[stage] = {}
            for label, *_ in SUITE:
                batches = samples[stage][label]
                vals = [v for batch in batches.values() for v in batch]
                assert len(vals) == 30
                summary[stage][label] = {
                    "n": len(vals),
                    "median_ms": statistics.median(vals),
                    "p95_ms": percentile(vals),
                    "batch_medians_ms": {k: statistics.median(v) for k, v in batches.items()},
                }
        report = {
            "stages": summary,
            "protocol": "six version permutations x five rounds; all first rounds retained",
            "parameters": {
                "future_weight": 0.5,
                "affinity_waves": 0.5,
                "split_cost_waves": 0.125,
                "O_refresh_creations": 64,
                "S_summary_loops": 64,
            },
            "exclusive_lock": False,
            "physical_card": 6,
            "logical_device": 0,
            "kernel_hashes_match": True,
            "source_and_binaries_preserved": True,
        }
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    else:
        for stage in args.stages:
            for entry in selected:
                invoke(stage, entry, "diagnostic", 1, diagnostic=True)


if __name__ == "__main__":
    main()
