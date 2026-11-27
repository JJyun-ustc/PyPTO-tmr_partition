#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Estimate factual execution spans and conditional residual life; never action gains."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for path in sorted(args.folder.glob("period*/labelled.jsonl"))
        for line in path.read_text().splitlines()
    ]
    if not rows or any(r["proxy_kind"] != 7 for r in rows):
        raise ValueError("requires validated AICore timing labels")
    train = [r for r in rows if r["period"] < 4 and not r["warmup"]]
    test = [r for r in rows if r["period"] == 5 and not r["warmup"]]
    groups = defaultdict(list)
    for row in train:
        groups[row["kernel"], row["actual_side"]].append(row["span_us"])
    table = []
    for (kernel, side), values in sorted(groups.items()):
        remaining = []
        for elapsed in np.quantile(values, [0, 0.25, 0.5, 0.75, 0.9]):
            tail = [d - elapsed for d in values if d > elapsed]
            remaining.append(
                {
                    "elapsed_us": float(elapsed),
                    "survivors": len(tail),
                    "remaining_p50_us": float(np.median(tail)) if tail else None,
                }
            )
        table.append(
            {
                "kernel": kernel,
                "side": side,
                "n": len(values),
                "p50_us": float(np.median(values)),
                "p90_us": float(np.quantile(values, 0.9)),
                "conditional_remaining": remaining,
            }
        )
    errors = [abs(r["span_us"] - np.median(groups[r["kernel"], r["actual_side"]])) for r in test]
    result = {
        "label": "AICore post-ACK to post-kernel/pre-FIN dispatcher interval",
        "scope": "first tile per layer of kernels 31..34, block=1, controlled decode placement",
        "split": {"train_periods": [0, 1, 2, 3], "reserved_calibration_period": 4, "test_period": 5},
        "independent_unit": "complete run/period; layers are correlated, not independent replicates",
        "table": table,
        "test": {
            "n": len(errors),
            "mae_us": float(np.mean(errors)),
            "absolute_error_p95_us": float(np.quantile(errors, 0.95)),
        },
        "runtime_admitted": False,
        "reason": "factual execution-span calibration does not calibrate two-action gain or optimism",
    }
    (args.folder / "span-calibration.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
