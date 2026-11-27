#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Same-process policy switching, poison goldens, and frozen blocked measurements."""

import argparse
import importlib.util
import json
import multiprocessing
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import qwen_pair
import torch

from simpler_setup.goldens.qwen3_14b_decode import CACHE_ROWS, HEAD_DIM, N_LAYERS, NUM_KV_HEADS

ROOT = Path(__file__).resolve().parents[4]
MODES = {"shared": 0, "structure": 7, "C_guarded": 8, "D_guarded": 9}


def schedule(phase):
    modes = list(MODES)
    if phase in ("smoke", "golden", "diagnose"):
        order = modes + (["shared"] if phase == "smoke" else [])
        return [{"mode": mode, "period": -1, "warmup": False, "ordinal": i} for i, mode in enumerate(order)]
    result = []
    periods = 12 if phase == "formal" else 1
    for period in range(periods):
        order = modes[period % 4 :] + modes[: period % 4]
        if (period // 4) % 2:
            order = order[::-1]
        for mode in order:
            for sample in range(6):
                result.append({"mode": mode, "period": period, "warmup": sample == 0, "ordinal": len(result)})
    return result


def poison_golden(worker, child, golden):
    # Only current-token rows are outputs of this decode. All context rows retain
    # the original fixture. Poisoning catches skipped writes even across policies.
    poison = torch.full((NUM_KV_HEADS, HEAD_DIM), float("nan"), dtype=golden.k_cache.dtype)
    size = poison.numel() * poison.element_size()
    for layer in range(N_LAYERS):
        for token_slot in golden.slot_mapping.tolist():
            row = layer * CACHE_ROWS + token_slot * NUM_KV_HEADS
            offset = row * HEAD_DIM * poison.element_size()
            for name in ("k_cache", "v_cache"):
                worker.copy_to(child.buffers[name], poison, dst_offset=offset, nbytes=size)
    worker.copy_to(child.buffers["out"], torch.full_like(golden.out, float("nan")))


def worker_main(manifest):
    multiprocessing.cpu_count = lambda: 2
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location(
        "streaming_qwen_driver", ROOT / "examples/a2a3/tensormap_and_ringbuffer/benchmark_qwen_cluster/main.py"
    )
    driver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = driver
    spec.loader.exec_module(driver)
    driver.configure_logging("info")
    original_compare = driver._compare_tensor

    def compare_tensor(name, actual, expected):
        # Exact equality implies the original allclose predicate. Every element
        # is checked; non-identical chunks retain the original tolerances.
        a, b = actual.reshape(-1), expected.reshape(-1)
        for offset in range(0, a.numel(), 1 << 20):
            ac, bc = a[offset : offset + (1 << 20)], b[offset : offset + (1 << 20)]
            if not torch.equal(ac, bc):
                original_compare(f"{name}[{offset}:]", ac, bc)

    driver._compare_tensor = compare_tensor
    original_run = driver.Worker.run
    original_build = driver.ChildMemoryTaskArgs.build_args
    original_generate = driver._decode_generate_inputs
    state = {"index": 0, "verified": set()}
    rows = manifest["schedule"]
    golden_ordinals = set(manifest.get("golden_ordinals", range(len(rows)) if manifest["phase"] == "golden" else []))
    original_copy_compare = driver._copy_and_compare

    def copy_and_compare(worker, buffers, golden):
        if state["index"] == len(rows) and len(state["verified"]) == len(rows):
            return
        original_copy_compare(worker, buffers, golden)

    driver._copy_and_compare = copy_and_compare
    if manifest.get("qwen_pair"):
        qwen_pair.install(driver, state, manifest)

    def build_args(child, *args, **kwargs):
        state["child"] = child
        return original_build(child, *args, **kwargs)

    def generate(*args, **kwargs):
        state["golden"] = original_generate(*args, **kwargs)
        return state["golden"]

    def run(worker, *args, **kwargs):
        i = state["index"]
        row = rows[i]
        os.environ["SIMPLER_TMR_QKV_BATCH"] = str(MODES[row["mode"]])
        if manifest.get("profile_root"):
            cfg = args[2]
            cfg.enable_chip_swimlane = 1
            prefix = Path(manifest["profile_root"]) / f"run{i:04d}"
            prefix.mkdir(parents=True, exist_ok=True)
            cfg.output_prefix = str(prefix)
        if manifest.get("qwen_pair"):
            qwen_pair.update_control(worker, state["child"], row)
        if i in golden_ordinals:
            poison_golden(worker, state["child"], state["golden"])
        print("STREAM_SAMPLE_BEGIN " + json.dumps(row), flush=True)
        result = original_run(worker, *args, **kwargs)
        if i in golden_ordinals:
            driver._copy_and_compare(worker, state["child"].buffers, state["golden"])
            state["verified"].add(i)
            print("STREAM_GOLDEN_PASS " + json.dumps({"mode": row["mode"], "layers": 40, "poisoned": True}), flush=True)
        print("STREAM_SAMPLE_END " + str(i), flush=True)
        state["index"] += 1
        return result

    driver.ChildMemoryTaskArgs.build_args = build_args
    driver._decode_generate_inputs = generate
    driver.Worker.run = run
    try:
        rc = driver.run(
            [0],
            "a2a3",
            runtime="tensormap_and_ringbuffer",
            rounds=len(rows),
            skip_golden=not bool(golden_ordinals),
            seed=1234,
            seq_len=manifest["seq_len"],
            compile_workers=2,
        )
        if rc or state["index"] != len(rows):
            raise RuntimeError("incomplete schedule")
    finally:
        driver.Worker.run = original_run
        driver.ChildMemoryTaskArgs.build_args = original_build
        driver._decode_generate_inputs = original_generate
        driver._compare_tensor = original_compare
        driver._copy_and_compare = original_copy_compare


def parse_host(text, expected, require_timing=True):
    samples = []
    active = None
    for line in text.splitlines():
        if line.startswith("STREAM_SAMPLE_BEGIN "):
            if active is not None:
                raise ValueError("overlapping runs")
            active = json.loads(line.split(" ", 1)[1])
            active["values_ms"] = []
        match = re.search(r"name=chip\.run\.runner_run\.device_wall ts=\d+ dur=(\d+)\b", line)
        if match:
            if active is None:
                raise ValueError("unlabelled device timing")
            active["values_ms"].append(int(match[1]) / 1e6)
        if line.startswith("STREAM_SAMPLE_END "):
            if (
                active is None
                or active["ordinal"] != int(line.split()[1])
                or len(active["values_ms"]) > 1
                or (require_timing and len(active["values_ms"]) != 1)
            ):
                raise ValueError("sample/timing mismatch")
            values = active.pop("values_ms")
            active["device_wall_ms"] = values[0] if values else None
            samples.append(active)
            active = None
    if active is not None or len(samples) != len(expected):
        raise ValueError("truncated host samples")
    return samples


def read_device(folder):
    lines = []
    for path in (folder / "ascend").rglob("*.log"):
        for line in path.read_text(errors="replace").splitlines():
            if "STREAM_RUN " in line:
                stamp = re.search(r"\d{4}-\d\d-\d\d-\d\d:\d\d:\d\d\.\d+\.\d+", line)
                if stamp is None:
                    raise ValueError("device log clock format")
                fields = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", line.split("STREAM_RUN ", 1)[1])}
                lines.append((stamp[0], fields))
    return [fields for _, fields in sorted(lines, key=lambda x: x[0])]


def controller(args):
    path = ROOT / "examples/a2a3/tensormap_and_ringbuffer/benchmark_qkv/run_study.py"
    spec = importlib.util.spec_from_file_location("streaming_baseline_helpers", path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    fingerprint, preflight = baseline.fingerprint, baseline.preflight

    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "ascend").mkdir()
    manifest = {
        "phase": args.phase,
        "seq_len": args.seq_len,
        "schedule": schedule(args.phase),
        "modes": MODES,
        "admitted_locality_classes": [],
        "C_D_active": False,
        "reason": "No Qwen paired locality/action calibration admitted; neutral queues and default-only C/D",
        "hardware": preflight(3),
        "fingerprints": fingerprint(),
    }
    manifest_path = folder / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    env = os.environ.copy()
    env.update(
        SIMPLER_TMR_QKV_TRACE="2" if args.phase == "diagnose" else "0",
        ASCEND_PROCESS_LOG_PATH=str(folder / "ascend"),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONPATH=f"{ROOT}/python:{ROOT}",
    )
    cmd = [sys.executable, str(Path(__file__).resolve()), "_worker", str(manifest_path)]
    (folder / "command.json").write_text(json.dumps(cmd, indent=2))
    with (folder / "host.log").open("w") as log:
        result = subprocess.run(cmd, check=False, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    (folder / "exit.json").write_text(json.dumps({"returncode": result.returncode}))
    if result.returncode:
        raise RuntimeError(f"worker failed; see {folder / 'host.log'}")
    if fingerprint() != manifest["fingerprints"]:
        raise ValueError("runtime changed during collection")
    samples = parse_host((folder / "host.log").read_text(), manifest["schedule"])
    actual = read_device(folder)
    if [r["mode"] for r in actual] != [MODES[r["mode"]] for r in samples]:
        raise ValueError(f"device did not honor same-process mode switching: {Counter(r['mode'] for r in actual)}")
    for sample, counters in zip(samples, actual):
        sample["counters"] = counters
        if counters["mode"] and (counters["eligible"] != counters["eligible_claimed"] or counters["dropped"]):
            raise ValueError(f"eligible coverage or trace capacity failed: {counters}")
    hashes = dict(re.findall(r"func_id=(\d+).*?code_fnv1a64=(0x[0-9a-f]+)", (folder / "host.log").read_text()))
    if len(hashes) != 37:
        raise ValueError("missing loaded kernel fingerprints")
    (folder / "kernel-hashes.json").write_text(json.dumps(hashes, indent=2))
    (folder / "samples.json").write_text(json.dumps(samples, indent=2))
    print(
        json.dumps(
            {
                "phase": args.phase,
                "n": len(samples),
                "actual_modes": Counter(r["mode"] for r in actual),
                "device_wall_ms": [s["device_wall_ms"] for s in samples[:8]],
            }
        ),
        flush=True,
    )


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "_worker":
        worker_main(json.loads(Path(sys.argv[2]).read_text()))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "golden", "diagnose", "pilot", "formal"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=3500)
    controller(parser.parse_args())


if __name__ == "__main__":
    main()
