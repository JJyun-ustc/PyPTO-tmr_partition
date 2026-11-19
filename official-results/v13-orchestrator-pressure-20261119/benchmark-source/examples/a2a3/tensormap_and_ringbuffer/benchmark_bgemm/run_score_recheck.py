"""Recheck archived score binaries with forward/reverse stage order in one time window."""

import argparse
import json
import os
import shutil
import statistics
from pathlib import Path

from reproduce_cluster_study import CASES, RUNTIME_ROOT, binary_hashes, percentile, run_case, verify_hardware


def restore(directory):
    for source in (directory / "source/src").rglob("*"):
        if source.is_file() and source.name != "compile_commands.json" and "__pycache__" not in source.parts:
            target = RUNTIME_ROOT / source.relative_to(directory / "source")
            if source.read_bytes() != target.read_bytes():
                shutil.copyfile(source, target)
    expected = json.loads((directory / "binary-hashes.json").read_text())
    for name in expected:
        shutil.copyfile(
            directory / "binaries" / name, RUNTIME_ROOT / "build/lib/a2a3/onboard/tensormap_and_ringbuffer" / name
        )
    assert binary_hashes() == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    stages = ("no-pickup", "queue-future", "waves-clean")
    output = root / "interleaved-recheck"
    output.mkdir(exist_ok=False)
    verify_hardware(0, output)
    logs = output / "device-logs"
    logs.mkdir()
    os.environ["ASCEND_PROCESS_LOG_PATH"] = str(logs)
    samples = {stage: {case: {} for case in CASES} for stage in stages}
    try:
        for batch, order, cases in (("a", stages, CASES), ("b", tuple(reversed(stages)), tuple(reversed(CASES)))):
            for stage in order:
                restore(root / stage)
                directory = output / stage
                directory.mkdir(exist_ok=True)
                before = binary_hashes()
                for case in cases:
                    samples[stage][case][batch] = run_case(case, 15, False, directory / f"{case}-{batch}.log", 0, None)
                assert before == binary_hashes()
                print(stage, batch, "same archived binary verified", flush=True)
    finally:
        restore(root / "queue-future")
    report = {"order": {"a": list(stages), "b": list(reversed(stages))}, "exclusive_lock": False, "variants": {}}
    for stage, cases in samples.items():
        report["variants"][stage] = {}
        for case, batches in cases.items():
            values = batches["a"] + batches["b"]
            report["variants"][stage][case] = {
                "n": len(values),
                "median_ms": statistics.median(values),
                "p95_ms": percentile(values, 0.95),
                "samples_ms": batches,
                "batch_median_ms": {batch: statistics.median(v) for batch, v in batches.items()},
            }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
