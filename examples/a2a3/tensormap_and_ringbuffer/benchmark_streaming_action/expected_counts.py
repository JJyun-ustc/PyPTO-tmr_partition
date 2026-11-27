#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Derive expected AICore block invocations from an independent complete trace."""

import argparse
import json
from pathlib import Path

from analyze import read_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = read_trace(args.trace)
    counts = {
        str(r["task"]): r["blocks"] * r["mask"].bit_count() for r in rows if any(r[k] >= 0 for k in ("k0", "k1", "k2"))
    }
    if not counts or not all(counts.values()):
        raise ValueError("invalid compute-task resource masks")
    args.output.write_text(json.dumps(counts, indent=2))
    print("tokens", len(counts), "block invocations", sum(counts.values()))


if __name__ == "__main__":
    main()
