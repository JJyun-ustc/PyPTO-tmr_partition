#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Post-hoc state-budget sensitivity using only successor indices observed at each epoch."""

import argparse
import json
from collections import Counter
from pathlib import Path

from analyze import read_trace


def calculate(rows):
    hist, observed = Counter(), 0
    for row in rows:
        info = row["visibility"]
        if not info["observed"]:
            continue
        observed += 1
        if info["state"] == 3:
            continue
        fanins = []
        for suffix in (0, 1):
            index, epoch = info[f"succ{suffix}"], info[f"reg{suffix}"]
            if index and epoch and epoch <= info["epoch"]:
                fanins.append(rows[index - 1]["visibility"]["fanin"])
        if fanins:
            hist[min(fanins)] += 1
    result = {
        "observed": observed,
        "denominator": "ordinary Shared queue-pop observations",
        "scope": "post-hoc sensitivity, not an online performance or action-ranking result",
        "prefix_rule": (
            "only recorded successor indices with 0 < registered_epoch <= captured_epoch; "
            "successor-index overflow excluded"
        ),
        "minimum_registered_fanin_histogram": dict(sorted(hist.items())),
        "budgets": [],
    }
    for budget in (4, 8, 16, 32, 64, 128, 256):
        count = sum(v for k, v in hist.items() if k <= budget)
        result["budgets"].append({"max_fanin": budget, "usable": count, "fraction": count / observed})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    rows = read_trace(args.folder)
    result = calculate(rows)
    (args.folder / "visibility-budgets-reproduced.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
