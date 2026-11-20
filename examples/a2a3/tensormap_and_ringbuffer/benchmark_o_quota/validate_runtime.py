#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate v14 split ranges and the preserved strict/sync dispatch paths."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-regressions", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    scenes = {
        "quota-aiv": root / "tests/st/a2a3/tensormap_and_ringbuffer/o_quota_multiblock_aiv/test_spmd_multiblock_aiv.py",
        "quota-mix": root / "tests/st/a2a3/tensormap_and_ringbuffer/o_quota_multiblock_mix/test_spmd_multiblock_mix.py",
        "dummy": args.legacy_regressions / "placement-dummy/test_dummy_task.py",
        "strict": args.legacy_regressions / "placement-vector/test_vector_example.py",
        "early": args.legacy_regressions / "placement-early/test_vector_example.py",
        "sync": args.legacy_regressions / "runtime-regressions/spmd_sync_start/test_spmd_sync_start.py",
    }
    wrapper = (
        "import multiprocessing,runpy,sys,torch; multiprocessing.cpu_count=lambda:2; "
        "torch.set_num_threads(2); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    results = {}
    for name, script in scenes.items():
        dest = out / name
        dest.mkdir(exist_ok=False)
        (dest / "device-logs").mkdir()
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{root}/python:{root}:/data/pypto-work/pypto-home/venv/lib/python3.10/site-packages"
        env["ASCEND_PROCESS_LOG_PATH"] = str(dest / "device-logs")
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[k] = "2"
        cmd = [
            str(root / ".venv/bin/python"),
            "-c",
            wrapper,
            str(script),
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
        print("START", name, flush=True)
        with (dest / "host.log").open("w") as log:
            run = subprocess.run(
                cmd, cwd=dest, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=False
            )
        text = (dest / "host.log").read_text()
        assert run.returncode == 0 and "PASSED" in text, dest
        splits = []
        for p in (dest / "device-logs/debug/device-0").glob("*.log"):
            splits += [int(v) for v in re.findall(r"AICPU_QUOTA_SPLITS tasks=(\d+)", p.read_text())]
        if name.startswith("quota-"):
            assert splits and max(splits) > 0, (name, splits)
        results[name] = {
            "returncode": run.returncode,
            "passed_cases": text.count(" PASSED"),
            "split_counts": splits,
            "cmd": cmd,
        }
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print("PASS", name, splits, flush=True)


if __name__ == "__main__":
    main()
