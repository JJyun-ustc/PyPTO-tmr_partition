"""Measure immutable source variants sequentially; runtime flags are unnecessary."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from reproduce_cluster_study import RUNTIME_ROOT, binary_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("stages", nargs="+")
    parser.add_argument("--jobs", type=int, required=True)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    for stage in args.stages:
        directory = args.root.resolve() / stage
        for source in (directory / "source/src").rglob("*"):
            if not source.is_file() or source.name == "compile_commands.json" or "__pycache__" in source.parts:
                continue
            destination = RUNTIME_ROOT / source.relative_to(directory / "source")
            if not destination.exists() or source.read_bytes() != destination.read_bytes():
                # Preserve archive timestamps, but always invalidate the active
                # build dependency when restoring an older source variant.
                shutil.copyfile(source, destination)
        with (directory / "experiment.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("run_control_stage.py")),
                    str(directory),
                    "--jobs",
                    str(args.jobs),
                ],
                cwd=RUNTIME_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        binaries = directory / "binaries"
        binaries.mkdir()
        for name in binary_hashes():
            shutil.copy2(RUNTIME_ROOT / "build/lib/a2a3/onboard/tensormap_and_ringbuffer" / name, binaries / name)
        print(stage, json.loads((directory / "run/summary.json").read_text())["Scale"], flush=True)


if __name__ == "__main__":
    main()
