#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run frozen source variants sequentially on one card, with official Shared bookends."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v8", type=Path, required=True)
    parser.add_argument("--v11", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--physical-card", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    script = Path(__file__).with_name("run_study.py")
    assert (args.output / "official-before/summary.json").exists()
    for stage, runtime, shared in (
        ("v8", args.v8, False),
        ("v11", args.v11, False),
        ("v12", root, False),
        ("official-after", args.official, True),
    ):
        if (args.output / stage / "summary.json").exists():
            print(stage, "already complete", flush=True)
            continue
        command = [
            str(root / ".venv/bin/python"),
            str(script),
            stage,
            "--runtime",
            str(runtime),
            "--output",
            str(args.output / stage),
            "--physical-card",
            str(args.physical_card),
        ]
        if shared:
            command.append("--shared")
        with (args.output / f"{stage}-runner.log").open("w") as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(stage, "complete", flush=True)


if __name__ == "__main__":
    main()
