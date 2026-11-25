#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Apply the predeclared quality and actual AICPU shadow budget gates."""

import argparse
import json
import re
from pathlib import Path

from round2_data import distribution


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    root = args.root
    folder = root / "shadow/improved/shadow"
    run = json.loads((folder / "result.json").read_text())
    assert run["passed"]
    text = (folder / "feedback.txt").read_text()
    rows = re.findall(r"QKV_R2_SHADOW candidates=16 ticks=(\d+) linear_ticks=(\d+)", text)
    assert len(rows) >= 30, ("insufficient actual AICPU shadow batches", len(rows))
    attention = [int(a) / 50 for a, b in rows]
    linear = [int(b) / 50 for a, b in rows]
    result = {
        "attention_us": distribution(attention),
        "linear_us": distribution(linear),
        "all_attention_us": attention,
        "all_linear_us": linear,
        "candidates": 16,
        "tasks": 8,
        "feature_construction_included": True,
        "clock_ticks_per_us": 50,
        "hardware": "Ascend910B2 AICPU",
        "warmup": "none excluded; all shadow batches retained",
    }
    (root / "models/aicpu-shadow.json").write_text(json.dumps(result, indent=2))
    gate = json.loads((root / "models/gate.json").read_text())
    gate.setdefault("python_wrapper_budget_pass", gate["cpu_budget_pass"])
    gate["actual_aicpu_budget_pass"] = result["attention_us"]["p95"] <= 50
    gate["cpu_budget_pass"] = gate["actual_aicpu_budget_pass"]
    gate["deploy"] = bool(gate["quality_pass"] and gate["native_cpu_budget_pass"] and gate["actual_aicpu_budget_pass"])
    gate["requires_actual_aicpu_shadow_before_dispatch"] = False
    gate["actual_aicpu_evidence"] = "aicpu-shadow.json"
    gate["warning"] = "proxy-time prediction quality does not guarantee a scheduling speedup"
    (root / "models/gate.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps({"shadow": result, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
