#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run a focused scene with bounded compilation and isolated diagnostic logs."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run_interleaved import ROOT

WRAPPER = """import multiprocessing,runpy,sys,torch
multiprocessing.cpu_count=lambda:2
torch.set_num_threads(2)
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name="__main__")
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path)
    parser.add_argument("--platform", choices=("a2a3", "a2a3sim"), default="a2a3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--trace", type=int, default=2)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "ascend").mkdir()
    env = os.environ.copy()
    env.update(
        SIMPLER_TMR_QKV_BATCH=str(args.mode),
        SIMPLER_TMR_QKV_TRACE=str(args.trace),
        ASCEND_PROCESS_LOG_PATH=str(folder / "ascend"),
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        PYTHONPATH=f"{ROOT}/python:{ROOT}",
    )
    command = [
        sys.executable,
        "-c",
        WRAPPER,
        str(args.script.resolve()),
        "-p",
        args.platform,
        "-d",
        "0",
        "--manual",
        "include",
        "--log-level",
        "info",
    ]
    (folder / "command.json").write_text(json.dumps(command, indent=2))
    with (folder / "host.log").open("w") as log:
        result = subprocess.run(
            command, cwd=ROOT, env=env, check=False, stdout=log, stderr=subprocess.STDOUT, timeout=900
        )
    (folder / "exit.json").write_text(json.dumps({"returncode": result.returncode}))
    if result.returncode or "PASSED" not in (folder / "host.log").read_text():
        raise RuntimeError(folder / "host.log")
    print("PASS", args.script, args.platform, flush=True)


if __name__ == "__main__":
    main()
