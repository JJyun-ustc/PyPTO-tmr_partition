"""Compare staged control-path experiments, keeping diagnostics separate from timing."""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

STAGES = ("baseline", "groups", "deferred", "summary", "steal", "final")


def fields(line):
    return {
        key: [int(v) for v in value.split(",")] if "," in value else int(value)
        for key, value in re.findall(r"([a-z_][a-z_0-9]*)=([0-9,]+)", line)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report = {
        "date": "2026-11-16",
        "silicon": "910B2",
        "physical_card": 0,
        "logical_device": 0,
        "orchestrator_cpu": 7,
        "exclusive_lock": False,
        "qualification": (
            "Sequential builds, fresh same-binary Shared controls, 30 rounds including process-first rounds. "
            "Diagnostics are separate instrumented golden runs. Stages are cumulative, not independent ablations. "
            "The summary stage also replaces shared ledger writers and removes ordinary decision timers. "
            "Final repeats the combined policy after origin-race fix, formatting and control-bookkeeping cleanup."
        ),
        "variants": {},
    }
    compute_hashes = set()
    for stage in STAGES:
        directory = root / stage
        timing = json.loads((directory / "run/summary.json").read_text())
        before = (directory / "run/performance/binaries-before.sha256").read_text()
        after = (directory / "run/performance/binaries-after.sha256").read_text()
        assert before == after, stage
        assert all(row["n"] == 30 for row in timing["cases"].values())
        costs_path = directory / "control-costs.json"
        with costs_path.open("w") as output:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("analyze_control_costs.py")),
                    str(directory / "diagnostics"),
                ],
                stdout=output,
                check=True,
            )
        diagnostics = json.loads(costs_path.read_text())
        for case, row in diagnostics.items():
            capture = row.get("capture")
            if capture:
                destination = directory / "captures" / case
                destination.mkdir(parents=True, exist_ok=True)
                for name in ("chip_swimlane_records.json", "deps.json", "dispatch_identity.json"):
                    source = Path(capture).with_name(name)
                    if source.exists():
                        shutil.copy2(source, destination / name)
                row["capture_copy"] = str(destination.relative_to(root))
            content = Path(row["device_log"]).read_text()
            row["placement_final"] = [fields(line) for line in content.splitlines() if "AICPU_ADAPTIVE_FINAL" in line]
            row["groups"] = [fields(line) for line in content.splitlines() if "AICPU_PLACEMENT_GROUP" in line]
            row["ledger"] = [fields(line) for line in content.splitlines() if "AICPU_PLACEMENT_LEDGER" in line]
            row["pickup"] = [fields(line) for line in content.splitlines() if "AICPU_PLACEMENT_PICKUP" in line]
            for ledger in row["ledger"]:
                assert ledger["preferred"] == ledger["released"], (stage, case, ledger)
                assert ledger["queued"] == ledger["taken"], (stage, case, ledger)
                assert sum(ledger["preferred"]) == sum(ledger["queued"]) == sum(ledger["actual"])
        smoke = {}
        for host in sorted((directory / "run").glob("Cluster*Adaptive-smoke.log")):
            matches = re.findall(r"simpler_inner_[0-9a-f]+_(\d+)\.json", host.read_text())
            if not matches:
                raise ValueError(f"Missing smoke worker PID: {host}")
            paths = list((directory / "run/device-logs").rglob(f"device-{matches[0]}_*.log"))
            device = max(paths, key=lambda item: item.read_text().count("AICPU_ADAPTIVE_FINAL"))
            content = device.read_text()
            ledgers = [fields(line) for line in content.splitlines() if "AICPU_PLACEMENT_LEDGER" in line]
            for row in ledgers:
                assert row["preferred"] == row["released"]
                assert row["queued"] == row["taken"]
                assert sum(row["queued"]) == sum(row["actual"])
            pickup = [fields(line) for line in content.splitlines() if "AICPU_PLACEMENT_PICKUP" in line]
            smoke[host.stem] = {"device_log": str(device), "ledger": ledgers, "pickup": pickup}
        hashes = set()
        for logfile in (directory / "run/performance").glob("*.log"):
            hashes.update(re.findall(r"func_id=(\d+)\b[^\n]*code_fnv1a64=(0x[0-9a-f]+)", logfile.read_text()))
        assert len(hashes) == 2, (stage, hashes)
        compute_hashes.add(tuple(sorted(hashes)))
        report["variants"][stage] = {"timing": timing, "diagnostics": diagnostics, "smoke": smoke}
    assert len(compute_hashes) == 1, "Compute binaries differ between policy builds"
    report["compute_code_hashes"] = list(next(iter(compute_hashes)))
    baseline = report["variants"]["baseline"]["timing"]["cases"]
    for stage, variant in report["variants"].items():
        variant["automatic_vs_baseline_pct"] = {
            family: 100
            * (
                variant["timing"]["cases"][family + "Adaptive"]["median_ms"]
                / baseline[family + "Adaptive"]["median_ms"]
                - 1
            )
            for family in ("Case0", "Scale")
        }
        print(stage, {family: variant["timing"][family] for family in ("Case0", "Scale")})
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
