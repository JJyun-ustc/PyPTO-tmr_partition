# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run v8 and both ready-affinity builds, then recheck frozen official Shared."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--v8", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--physical-card", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    study = args.study.resolve()
    deadline = time.monotonic() + 1800
    for required in ["official-before/summary.json", "ready-no-pickup-artifacts/manifest.json"]:
        while not (study / required).exists():
            if time.monotonic() > deadline:
                raise TimeoutError(required)
            time.sleep(2)
    driver = Path(__file__).with_name("run_study.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}/python:{root}:/data/pypto-work/pypto-home/venv/lib/python3.10/site-packages"
    for stage, runtime in [
        ("v8", args.v8),
        ("ready-no-pickup", root),
        ("ready-pickup", root),
        ("official-after", args.official),
    ]:
        if stage == "ready-pickup":
            header = root / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/ready_affinity.h"
            text = header.read_text()
            assert "constexpr bool READY_REMOTE_PICKUP = false;" in text
            header.write_text(
                text.replace(
                    "constexpr bool READY_REMOTE_PICKUP = false;", "constexpr bool READY_REMOTE_PICKUP = true;"
                )
            )
            code = (
                "import multiprocessing; multiprocessing.cpu_count=lambda:2; "
                "from simpler_setup.runtime_builder import RuntimeBuilder; "
                "print(RuntimeBuilder('a2a3').get_binaries('tensormap_and_ringbuffer',build=True))"
            )
            with (study / "pickup-build.log").open("w") as log:
                subprocess.run(
                    [str(root / ".venv/bin/python"), "-u", "-c", code],
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            archive = study / "ready-pickup-artifacts"
            archive.mkdir()
            shutil.copytree(
                root / "src/a2a3/runtime/tensormap_and_ringbuffer",
                archive / "source/src/a2a3/runtime/tensormap_and_ringbuffer",
            )
            shutil.copytree(root / "tests/ut/cpp/a2a3/runtime", archive / "source/tests/ut/cpp/a2a3/runtime")
            shutil.copytree(root / "build/lib/a2a3/onboard/tensormap_and_ringbuffer", archive / "binaries")
            manifest = {
                str(p.relative_to(archive)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in archive.rglob("*")
                if p.is_file()
            }
            (archive / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        command = [
            sys.executable,
            "-u",
            str(driver),
            stage,
            "--runtime",
            str(runtime),
            "--physical-card",
            str(args.physical_card),
            "--output",
            str(study / stage),
        ]
        if stage == "official-after":
            command.append("--shared")
        subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
