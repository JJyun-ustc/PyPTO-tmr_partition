#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Archive the exact independently built candidates and immutable control source digests before timing."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--no-pickup", type=Path, required=True)
    parser.add_argument("--v8", type=Path, required=True)
    parser.add_argument("--v12", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    study = args.study.resolve()
    study.mkdir(exist_ok=True)
    roots = {"v8": args.v8.resolve(), "v12": args.v12.resolve(), "v13": root, "v13-no-pickup": args.no_pickup.resolve()}
    preservation = {}
    for name, directory in roots.items():
        files = subprocess.check_output(["git", "ls-files", "src"], cwd=directory, text=True).splitlines()
        preservation[name] = {
            "HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip(),
            "src_sha256": {rel: hashlib.sha256((directory / rel).read_bytes()).hexdigest() for rel in files},
        }
    (study / "sources-before.json").write_text(json.dumps(preservation, indent=2) + "\n")
    for name in ("v13", "v13-no-pickup"):
        directory = roots[name]
        archive = study / (name + "-artifacts")
        archive.mkdir(exist_ok=False)
        for rel in ("src/a2a3/runtime/tensormap_and_ringbuffer", "tests/ut/cpp/a2a3/runtime"):
            shutil.copytree(
                directory / rel,
                archive / "source" / rel,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "compile_commands.json"),
            )
        shutil.copytree(directory / "build/lib/a2a3/onboard/tensormap_and_ringbuffer", archive / "binaries")
        manifest = {
            str(p.relative_to(archive)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in archive.rglob("*")
            if p.is_file()
        }
        (archive / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    patch = subprocess.check_output(["git", "diff", "--", "src"], cwd=roots["v13-no-pickup"])
    assert patch.count(b"-constexpr bool READY_REMOTE_PICKUP = true;") == 1
    assert patch.count(b"+constexpr bool READY_REMOTE_PICKUP = false;") == 1
    assert len([line for line in patch.splitlines() if line.startswith((b"+", b"-"))]) == 4
    (study / "pickup-only.patch").write_bytes(patch)
    (study / "implementation.patch").write_bytes(
        subprocess.check_output(["git", "diff", "16a8b00", "HEAD", "--", "src", "tests/ut/cpp"], cwd=root)
    )
    bench = Path("examples/a2a3/tensormap_and_ringbuffer")
    for name in ("benchmark_bgemm", "benchmark_paged_attention", "benchmark_orchestrator_pressure"):
        shutil.copytree(
            root / bench / name,
            study / "benchmark-source" / bench / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    params = {
        "N": 64,
        "M": 64,
        "local_price": 250,
        "remote_price": 750,
        "queue_price": 2000,
        "queue_weight_den": 2,
        "future_cap_price": 500,
        "future_cap": "one compatible capacity wave",
        "actual_advantage_den": 4,
        "inferred_advantage_den": 2,
        "probe_vacancies": 32,
        "cooldown_vacancies": 128,
        "max_remote_blocks": 8,
        "range_affinity": False,
        "duration_model": False,
        "price_learning": False,
    }
    (study / "frozen-parameters.json").write_text(json.dumps(params, indent=2) + "\n")
    print("sources, binaries and policy frozen", flush=True)


if __name__ == "__main__":
    main()
