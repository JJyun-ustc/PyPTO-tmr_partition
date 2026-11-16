"""Build and measure one immutable source stage of the cluster control study."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import reproduce_cluster_study as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--physical-card", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer")
    root = study.RUNTIME_ROOT
    stage = args.stage.resolve()
    if (stage / "run").exists() or (stage / "diagnostics").exists():
        parser.error("Stage already contains results; choose a new output directory")
    stage.mkdir(parents=True, exist_ok=True)
    source = root / "src/a2a3/runtime/tensormap_and_ringbuffer"
    shutil.copytree(source, stage / "source" / source.relative_to(root), dirs_exist_ok=True)
    if not args.skip_build:
        for component, filename in (
            ("aicpu", "libaicpu_kernel.so"),
            ("host", "libhost_runtime.so"),
            ("aicore", "aicore_kernel.o"),
        ):
            cache = root / "build/cache/a2a3/onboard/tensormap_and_ringbuffer" / component
            with (stage / f"build-{component}.log").open("w") as log:
                subprocess.run(
                    ["cmake", "--build", str(cache), "--parallel", str(args.jobs)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            shutil.copy2(cache / filename, root / "build/lib/a2a3/onboard/tensormap_and_ringbuffer" / filename)
    command = [
        sys.executable,
        str(Path(__file__).with_name("reproduce_cluster_study.py")),
        "--physical-card",
        str(args.physical_card),
        "--device",
        str(args.device),
        "--output",
        str(stage / "run"),
    ]
    subprocess.run(command, cwd=root, check=True)
    diagnostics = stage / "diagnostics"
    diagnostics.mkdir()
    logs = diagnostics / "device-logs"
    logs.mkdir()
    os.environ["ASCEND_PROCESS_LOG_PATH"] = str(logs)
    for case in study.CASES:
        command = [
            sys.executable,
            str(Path(__file__).with_name("test_benchmark_bgemm.py")),
            "-p",
            "a2a3",
            "-d",
            str(args.device),
            "--case",
            f"Cluster{case}",
            "--manual",
            "include",
            "--enable-chip-swimlane",
            "4",
            "--enable-dep-gen",
            "--log-level",
            "info",
        ]
        with (diagnostics / f"Cluster{case}.log").open("w") as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"{case}: separate diagnostic passed", flush=True)
    (stage / "binary-hashes.json").write_text(json.dumps(study.binary_hashes(), indent=2) + "\n")


if __name__ == "__main__":
    main()
