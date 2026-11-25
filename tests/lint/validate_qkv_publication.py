# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate the published QKV samples, feedback, fingerprints and document links."""

import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "official-results/qkv-ready-batch-20261125"


def read(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix == ".gz":
        payload = gzip.decompress(payload)
    return payload.decode()


def p95(values: list[float]) -> float:
    values = sorted(values)
    position = (len(values) - 1) * 0.95
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def main() -> None:
    manifest = json.loads((DATA / "publication-manifest.json").read_text())
    for row in manifest["files"]:
        path = DATA / row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["published_sha256"], path
    summary = json.loads((DATA / "summary.json").read_text())
    expected = json.loads((DATA / "kernel-hashes.json").read_text())
    topologies, orchestration = set(), set()
    retained = 0
    for mode in ("shared", "qkv"):
        formal = []
        for period in range(6):
            folder = DATA / mode / f"block{period}"
            samples = json.loads((folder / "samples.json").read_text())
            assert len(samples["all_ms"]) == 6
            assert samples["all_ms"][1:] == samples["formal_ms"]
            assert samples["all_ms"][0] == samples["warmup_ms"]
            retained += len(samples["all_ms"])
            formal += samples["formal_ms"]
            host = read(folder / "host.log.gz")
            actual = dict(re.findall(r"func_id=(\d+).*?code_fnv1a64=(0x[0-9a-f]+)", host))
            assert len(actual) == 37 and actual == expected
            topologies.update(re.findall(r"A2A3 AICPU ALLOWED_CPUS = (.*)", host))
            orchestration.update(re.findall(r"orch_hash=(0x[0-9a-f]+)", host))
        assert len(formal) == summary[mode]["n"] == 30
        assert formal == summary[mode]["all_ms"]
        assert statistics.median(formal) == summary[mode]["median_ms"]
        assert abs(p95(formal) - summary[mode]["p95_ms"]) < 1e-9
    assert retained == 72 and len(topologies) == len(orchestration) == 1
    feedback = json.loads((DATA / "feedback.json").read_text())
    formal = feedback["formal"]
    assert formal["candidates"] == formal["assigned"] + sum(formal["fallback_full_snapshot_noside_budget_oversize"])
    assert sum(formal["natural_batch_histogram"]) == formal["batches"]
    assert sum(formal["natural_batch_histogram"][1:]) == formal["multi_task_batches"]
    assert feedback["diagnostic_exact_decision_us"]["n"] == len(feedback["diagnostic_exact_decision_us"]["all"])
    failures = DATA / "attempts/measurement-timing-loglevel"
    for mode in ("shared", "qkv"):
        assert len(json.loads((failures / mode / "raw-device-wall.json").read_text())["all_ms"]) == 6
    for language in ("en", "zh"):
        for name in ("qkv-ready-batch.md", "algorithm-qkv-ready-batch.md"):
            doc = ROOT / f"docs/{language}/experiments/tmr-cluster/{name}"
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", doc.read_text()):
                if "://" not in target and not target.startswith("#"):
                    assert (doc.parent / target.split("#")[0]).exists(), (doc, target)
    print("PASS: 60 formal samples, 12 warmups, 12 excluded attempt rounds, 37 kernels, topology, feedback, hashes and bilingual links")


if __name__ == "__main__":
    main()
