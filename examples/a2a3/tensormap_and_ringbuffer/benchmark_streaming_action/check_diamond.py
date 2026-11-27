#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify real early/late registered joins without treating future edges as visible."""

import argparse
import json
from pathlib import Path

from qwen_pair_analyze import launches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    runs = launches(args.folder)
    if len(runs) != 4:
        raise ValueError("expected four diamond launches")
    report = []
    for ordinal, run in enumerate(runs):
        late, slow = divmod(ordinal, 2)
        tasks = run["tasks"]
        if run["meta"]["dropped"] or len({t["task"] for t in tasks}) != len(tasks):
            raise ValueError("dropped or duplicate records")
        t = next(t for t in tasks if 1 in (t["k0"], t["k1"], t["k2"]))
        j = next(t for t in tasks if 2 in (t["k0"], t["k1"], t["k2"]))
        sibling = [t for t in tasks if 0 in (t["k0"], t["k1"], t["k2"])][1]
        info = run["visible"][t["task"]]
        expected = 0 if late else 2
        if info["state"] != expected or not info["observed"]:
            raise ValueError(f"visibility differs from submission protocol: {info}")
        if bool(sibling["fin"] > t["fin"]) != bool(slow):
            raise ValueError("fast/slow branch did not establish intended FIN order")
        report.append(
            {
                "late": bool(late),
                "slow": bool(slow),
                "observed_state": info["state"],
                "captured_epoch": info["epoch"],
                "t_claim": info["claim"],
                "t_fin": t["fin"],
                "g_fin": sibling["fin"],
                "j_registered": run["visible"][j["task"]]["registered"],
                "j_ready": j["ready"],
                "j_completed": j["completed"],
                "all_tasks": len(tasks),
                "scope": "Shared visibility, not active C/D action gain",
            }
        )
    (args.folder / "checked.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
