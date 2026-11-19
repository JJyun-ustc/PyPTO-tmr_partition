"""Validate selected scores on predeclared BGEMM configurations using archived binaries."""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

from reproduce_cluster_study import RUNTIME_ROOT, binary_hashes, percentile, verify_hardware

CASES = ("HeavyShared", "HeavyAdaptive", "DeepShared", "DeepAdaptive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("stages", nargs="+")
    args = parser.parse_args()
    for stage in args.stages:
        directory = args.root.resolve() / stage
        output = directory / "holdout"
        output.mkdir(exist_ok=False)
        verify_hardware(0, output)
        for source in (directory / "source/src").rglob("*"):
            if source.is_file() and source.name != "compile_commands.json" and "__pycache__" not in source.parts:
                destination = RUNTIME_ROOT / source.relative_to(directory / "source")
                if source.read_bytes() != destination.read_bytes():
                    shutil.copyfile(source, destination)
        archived = directory / "binaries"
        for name in binary_hashes():
            shutil.copyfile(archived / name, RUNTIME_ROOT / "build/lib/a2a3/onboard/tensormap_and_ringbuffer" / name)
        before = binary_hashes()
        expected = json.loads((directory / "binary-hashes.json").read_text())
        assert before == expected
        logs = output / "device-logs"
        logs.mkdir()
        os.environ["ASCEND_PROCESS_LOG_PATH"] = str(logs)
        samples = {case: {} for case in CASES}
        for batch, cases, golden, rounds in (
            ("golden", CASES, True, 1),
            ("a", CASES, False, 15),
            ("b", tuple(reversed(CASES)), False, 15),
        ):
            for case in cases:
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("test_score_holdout.py")),
                    "-p",
                    "a2a3",
                    "-d",
                    "0",
                    "--manual",
                    "include",
                    "--case",
                    case,
                    "--rounds",
                    str(rounds),
                    "--log-level",
                    "timing",
                ]
                if not golden:
                    command.append("--skip-golden")
                path = output / f"{case}-{batch}.log"
                with path.open("w") as log:
                    subprocess.run(command, cwd=RUNTIME_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                content = path.read_text()
                assert f"TestScoreHoldout::{case} PASSED" in content, path
                if not golden:
                    values = [
                        int(v) / 1e6
                        for v in re.findall(r"name=chip\.run\.runner_run\.device_wall ts=\d+ dur=(\d+)\b", content)
                    ]
                    assert len(values) == rounds, path
                    samples[case][batch] = values
                print(stage, case, batch, "passed", flush=True)
        after = binary_hashes()
        assert before == after
        report = {"exclusive_lock": False, "binary_hashes": before, "cases": {}}
        for case, batches in samples.items():
            values = batches["a"] + batches["b"]
            report["cases"][case] = {
                "n": len(values),
                "median_ms": statistics.median(values),
                "p95_ms": percentile(values, 0.95),
                "samples_ms": batches,
                "batch_median_ms": {batch: statistics.median(v) for batch, v in batches.items()},
            }
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
