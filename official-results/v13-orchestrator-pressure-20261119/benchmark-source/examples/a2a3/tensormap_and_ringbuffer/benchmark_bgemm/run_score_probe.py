"""Build an isolated diagnostic source probe; no probe run enters timing statistics."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from reproduce_cluster_study import RUNTIME_ROOT, verify_hardware


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--jobs", type=int, required=True)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    directory = args.directory.resolve()
    diagnostics = directory / "diagnostics"
    diagnostics.mkdir(exist_ok=False)
    verify_hardware(0, diagnostics)
    for source in (directory / "source/src").rglob("*"):
        if source.is_file() and source.name != "compile_commands.json" and "__pycache__" not in source.parts:
            destination = RUNTIME_ROOT / source.relative_to(directory / "source")
            if source.read_bytes() != destination.read_bytes():
                shutil.copyfile(source, destination)
    for component, name in (
        ("aicpu", "libaicpu_kernel.so"),
        ("host", "libhost_runtime.so"),
        ("aicore", "aicore_kernel.o"),
    ):
        cache = RUNTIME_ROOT / "build/cache/a2a3/onboard/tensormap_and_ringbuffer" / component
        with (directory / f"build-{component}.log").open("w") as log:
            subprocess.run(
                ["cmake", "--build", str(cache), "--parallel", str(args.jobs)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        shutil.copyfile(cache / name, RUNTIME_ROOT / "build/lib/a2a3/onboard/tensormap_and_ringbuffer" / name)
    logs = diagnostics / "device-logs"
    logs.mkdir()
    os.environ["ASCEND_PROCESS_LOG_PATH"] = str(logs)
    for case in ("Case0Adaptive", "ScaleAdaptive"):
        command = [
            sys.executable,
            str(Path(__file__).with_name("test_benchmark_bgemm.py")),
            "-p",
            "a2a3",
            "-d",
            "0",
            "--manual",
            "include",
            "--case",
            "Cluster" + case,
            "--enable-chip-swimlane",
            "4",
            "--enable-dep-gen",
            "--log-level",
            "info",
        ]
        with (diagnostics / f"Cluster{case}.log").open("w") as log:
            subprocess.run(command, cwd=RUNTIME_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(case, "diagnostic passed", flush=True)


if __name__ == "__main__":
    main()
