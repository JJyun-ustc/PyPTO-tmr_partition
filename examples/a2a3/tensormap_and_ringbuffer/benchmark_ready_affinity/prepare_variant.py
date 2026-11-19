# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build a source-selected ready-affinity variant and archive its own artifacts."""

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import shutil
from pathlib import Path

from simpler_setup.runtime_builder import RuntimeBuilder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickup", choices=("enabled", "disabled"), required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    archive = args.archive.resolve()
    if archive.exists():
        parser.error("Archive already exists")
    header = root / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/ready_affinity.h"
    text = header.read_text()
    value = "true" if args.pickup == "enabled" else "false"

    text, count = re.subn(
        r"constexpr bool READY_REMOTE_PICKUP = (true|false);", f"constexpr bool READY_REMOTE_PICKUP = {value};", text
    )
    assert count == 1
    header.write_text(text)
    jobs = int(os.environ.get("PYPTO_BUILD_JOBS", "2"))
    multiprocessing.cpu_count = lambda: jobs

    RuntimeBuilder("a2a3").get_binaries("tensormap_and_ringbuffer", build=True)
    archive.mkdir(parents=True)
    shutil.copytree(
        root / "src/a2a3/runtime/tensormap_and_ringbuffer", archive / "source/src/a2a3/runtime/tensormap_and_ringbuffer"
    )
    shutil.copytree(root / "tests/ut/cpp/a2a3/runtime", archive / "source/tests/ut/cpp/a2a3/runtime")
    shutil.copytree(root / "build/lib/a2a3/onboard/tensormap_and_ringbuffer", archive / "binaries")
    manifest = {
        str(p.relative_to(archive)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in archive.rglob("*")
        if p.is_file()
    }
    (archive / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
