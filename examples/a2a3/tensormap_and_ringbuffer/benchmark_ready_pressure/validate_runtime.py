#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run final Auto hardware regressions after the staged timing suite releases the card."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TARGETS = (
    "ready_affinity",
    "scheduler_state",
    "wiring",
    "ready_queue",
    "task_state",
    "shared_memory",
    "orchestrator_fanin",
    "aicore_completion_mailbox",
)
SCENES = (
    ("placement-dummy", "test_dummy_task.py"),
    ("placement-vector", "test_vector_example.py"),
    ("placement-early", "test_vector_example.py"),
    ("spmd_multiblock_aiv", "test_spmd_multiblock_aiv.py"),
    ("spmd_multiblock_mix", "test_spmd_multiblock_mix.py"),
    ("spmd_sync_start", "test_spmd_sync_start.py"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    study = args.study.resolve()
    deadline = time.monotonic() + 7200
    while not (study / "official-after/summary.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Timing suite did not finish; inspect stage runner logs")
        time.sleep(5)
    output = study / "final-validation"
    output.mkdir(exist_ok=False)
    logs = output / "device-logs"
    logs.mkdir()
    for p in (study / "v12-artifacts/source/src").rglob("*"):
        if p.is_file():
            actual = ROOT / p.relative_to(study / "v12-artifacts/source")
            assert p.read_bytes() == actual.read_bytes(), actual
    expected = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (study / "v12-artifacts/binaries").iterdir()
    }
    for name, value in expected.items():
        assert (
            hashlib.sha256((ROOT / "build/lib/a2a3/onboard/tensormap_and_ringbuffer" / name).read_bytes()).hexdigest()
            == value
        )
    with (output / "unit-build.log").open("w") as log:
        subprocess.run(
            [
                "cmake",
                "--build",
                "tests/ut/cpp/build",
                "--target",
                *["test_a2a3_tmr_" + t for t in TARGETS],
                "--parallel",
                os.environ.get("PYPTO_BUILD_JOBS", "2"),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    pattern = "test_a2a3_tmr_(" + "|".join(TARGETS) + ")$"
    with (output / "unit.log").open("w") as log:
        subprocess.run(
            [
                "ctest",
                "--test-dir",
                "tests/ut/cpp/build",
                "-R",
                pattern,
                "--parallel",
                os.environ.get("PYPTO_TEST_JOBS", "2"),
                "--output-on-failure",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    counts = {}
    for name in TARGETS:
        text = subprocess.check_output(
            [str(ROOT / "tests/ut/cpp/build" / ("test_a2a3_tmr_" + name)), "--gtest_list_tests"], text=True
        )
        counts[name] = sum(line.startswith("  ") and not line.lstrip().startswith("#") for line in text.splitlines())
    report = {"binary_hashes": expected, "unit_counts": counts, "scenes": {}}
    for directory, filename in SCENES:
        path = ROOT / "tests/st/a2a3/tensormap_and_ringbuffer/cluster_partition_study" / directory / filename
        wrapper = (
            "import multiprocessing,runpy,sys,torch; multiprocessing.cpu_count=lambda:2; "
            "torch.set_num_threads(2); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
        )
        command = [
            str(ROOT / ".venv/bin/python"),
            "-c",
            wrapper,
            str(path),
            "-p",
            "a2a3",
            "-d",
            "0",
            "--manual",
            "include",
            "--log-level",
            "info",
            "--enable-chip-swimlane",
            "4",
        ]
        if shutil.which("task-submit"):
            command = [
                "task-submit",
                "--timeout",
                "1800",
                "--max-time",
                "1800",
                "--device",
                "0",
                "--run",
                shlex.join(command),
            ]
        env = os.environ.copy()
        env["ASCEND_PROCESS_LOG_PATH"] = str(logs)
        env["PYTHONPATH"] = (
            str(ROOT / "python") + ":" + str(ROOT) + ":/data/pypto-work/pypto-home/venv/lib/python3.10/site-packages"
        )
        with (output / f"{directory}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        report["scenes"][directory] = result.returncode
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        assert result.returncode == 0, directory
        print(directory, "passed", flush=True)
    print("final validation passed", sum(counts.values()), "C++ cases", flush=True)


if __name__ == "__main__":
    main()
