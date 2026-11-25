#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run existing exact-output scene goldens on the six-thread QKV topology."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run_study import ROOT, preflight

WRAPPER = """import copy,multiprocessing,runpy,sys,torch
multiprocessing.cpu_count=lambda:2
torch.set_num_threads(2)
from simpler_setup import SceneTestCase
original=SceneTestCase.run_module
def run(module_name):
    for value in list(vars(sys.modules[module_name]).values()):
        if isinstance(value,type) and issubclass(value,SceneTestCase) and value is not SceneTestCase:
            value.CASES=copy.deepcopy(value.CASES)
            for case in value.CASES:
                case.setdefault("config",{})["aicpu_thread_num"]=6
    original(module_name)
SceneTestCase.run_module=staticmethod(run)
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name="__main__")
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--scenes", nargs="+")
    ap.add_argument("--mode", type=int, default=1)
    ap.add_argument("--trace", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--card", type=int, default=4)
    args = ap.parse_args()
    hardware = preflight(args.card)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "hardware.json").write_text(json.dumps(hardware, indent=2))
    results = []
    for scene in args.scenes or (
        "qkv_sync_coexist",
        "spmd_multiblock_mix",
        "qkv_mix_chain",
        "spmd_multiblock_aiv",
        "dummy_task",
        "spmd_sync_start_early_dispatch",
    ):
        dest = out / scene
        dest.mkdir()
        (dest / "ascend").mkdir()
        env = os.environ.copy()
        env.update(
            {
                "SIMPLER_TMR_QKV_BATCH": str(args.mode),
                "SIMPLER_TMR_QKV_TRACE": str(args.trace),
                "PYTHONPATH": f"{ROOT}/python:{ROOT}",
                "ASCEND_PROCESS_LOG_PATH": str(dest / "ascend"),
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
            }
        )
        script = ROOT / f"tests/st/a2a3/tensormap_and_ringbuffer/{scene}/test_{scene}.py"
        cmd = [
            sys.executable,
            "-c",
            WRAPPER,
            str(script),
            "-p",
            "a2a3",
            "-d",
            str(args.device),
            "--manual",
            "include",
            "--log-level",
            "info",
            "--enable-chip-swimlane",
            "4",
        ]
        print("START", scene, flush=True)
        with (dest / "host.log").open("w") as log:
            run = subprocess.run(
                cmd,
                check=False,
                env=env,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        text = (dest / "host.log").read_text()
        device = "\n".join(p.read_text(errors="replace") for p in (dest / "ascend/debug/device-0").glob("*.log"))
        (dest / "feedback.txt").write_text("\n".join(line for line in device.splitlines() if "QKV_" in line) + "\n")
        results.append({"scene": scene, "returncode": run.returncode, "cmd": cmd})
        (out / "results.json").write_text(json.dumps(results, indent=2))
        assert run.returncode == 0 and "PASSED" in text, dest
        print("PASS", scene, flush=True)


if __name__ == "__main__":
    main()
