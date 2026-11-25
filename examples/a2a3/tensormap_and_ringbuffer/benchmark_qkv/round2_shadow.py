#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Export and verify native CPU shadow inference of learned Q/K/V projections."""

import argparse
import gzip
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from round2_train import Attention


def literal(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return "{" + ",".join(format(float(v), ".17g") for v in a) + "}"
    return "{" + ",".join(literal(v) for v in a) + "}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", type=Path)
    args = ap.parse_args()
    out = args.models
    w = json.loads((out / "weights.json").read_text())
    state = w["attention"]
    rows = []
    with gzip.open(out / "real-action-labels.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == "test":
                rows.append(r)
            if len(rows) == 16:
                break
    x = (np.array([r["x"] for r in rows]) - w["mean"]) / w["std"]
    t = np.array([r["tokens"] for r in rows])
    assert len(x) == 16
    source = "#include <cmath>\n#include <chrono>\n#include <algorithm>\n#include <cstdio>\n#include <vector>\n"
    for name, value in [("x", x), ("tokens", t)] + [(k.replace(".", "_"), v) for k, v in state.items()]:
        shape = "".join(f"[{i}]" for i in np.asarray(value).shape)
        source += f"static const double {name}{shape}={literal(value)};\n"
    source += (
        "\n"
        "void infer(int i, double y[2]) {\n"
        "  double q[4]{}, k[3][4]{}, v[3][4]{}, logits[3]{}, z[4]{};\n"
        "  for (int d = 0; d < 4; ++d) {\n"
        "    q[d] = q_bias[d];\n"
        "    for (int j = 0; j < 12; ++j)\n"
        "      q[d] += q_weight[d][j] * x[i][j];\n"
        "    for (int a = 0; a < 3; ++a) {\n"
        "      k[a][d] = k_bias[d];\n"
        "      v[a][d] = v_bias[d];\n"
        "      for (int j = 0; j < 5; ++j) {\n"
        "        k[a][d] += k_weight[d][j] * tokens[i][a][j];\n"
        "        v[a][d] += v_weight[d][j] * tokens[i][a][j];\n"
        "      }\n"
        "      logits[a] += q[d] * k[a][d] / 2;\n"
        "    }\n"
        "  }\n"
        "  double top = *std::max_element(logits, logits + 3), sum = 0;\n"
        "  for (int a = 0; a < 3; ++a) {\n"
        "    logits[a] = std::exp(logits[a] - top);\n"
        "    sum += logits[a];\n"
        "  }\n"
        "  for (int a = 0; a < 3; ++a)\n"
        "    for (int d = 0; d < 4; ++d)\n"
        "      z[d] += logits[a] / sum * v[a][d];\n"
        "  for (int o = 0; o < 2; ++o) {\n"
        "    y[o] = skip_bias[o] + out_bias[o];\n"
        "    for (int j = 0; j < 12; ++j)\n"
        "      y[o] += skip_weight[o][j] * x[i][j];\n"
        "    for (int d = 0; d < 4; ++d)\n"
        "      y[o] += out_weight[o][d] * z[d];\n"
        "  }\n"
        "}\n"
        "int main() {\n"
        "  std::vector<double> ns;\n"
        "  volatile double sink = 0;\n"
        "  for (int n = 0; n < 11000; ++n) {\n"
        "    auto begin = std::chrono::steady_clock::now();\n"
        "    for (int i = 0; i < 16; ++i) {\n"
        "      double y[2];\n"
        "      infer(i, y);\n"
        "      sink += y[0] + y[1];\n"
        "    }\n"
        "    auto end = std::chrono::steady_clock::now();\n"
        "    if (n >= 1000)\n"
        "      ns.push_back(std::chrono::duration<double, std::micro>(end - "
        "begin).count());\n"
        "  }\n"
        "  std::sort(ns.begin(), ns.end());\n"
        '  printf("{\\"n\\":%zu,\\"median_us\\":%.9g,\\"p95_us\\":%.9g,\\"checksum\\":%.9g,"\n'
        '         "\\"predictions\\":[",\n'
        "         ns.size(), ns[ns.size() / 2], ns[size_t(.95 * (ns.size() - 1))],\n"
        "         double(sink));\n"
        "  for (int i = 0; i < 16; ++i) {\n"
        "    double y[2];\n"
        "    infer(i, y);\n"
        '    printf("%s[%.17g,%.17g]", i ? "," : "", y[0], y[1]);\n'
        "  }\n"
        '  puts("]}");\n'
        "}\n"
    )
    (out / "shadow.cpp").write_text(source)
    cmd = [
        "g++",
        "-std=c++17",
        "-O3",
        str(out / "shadow.cpp"),
        "-o",
        str(out / "shadow"),
    ]
    subprocess.run(cmd, check=True)
    result = json.loads(subprocess.check_output([str(out / "shadow")], text=True))
    model = Attention()
    model.load_state_dict(torch.load(out / "attention.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)).numpy()
    result["max_normalized_prediction_difference"] = float(np.max(np.abs(pred - np.array(result["predictions"]))))
    assert result["max_normalized_prediction_difference"] < 1e-4
    result.update(
        {
            "cmd": cmd,
            "candidates_per_batch": 16,
            "actual_aicpu_measured": False,
            "includes_feature_construction": False,
            "scope": "host CPU native shadow, normalized residual outputs",
        }
    )
    (out / "native-shadow.json").write_text(json.dumps(result, indent=2))
    gate = json.loads((out / "gate.json").read_text())
    gate["native_cpu_budget_pass"] = result["p95_us"] <= 50
    gate["deploy"] = bool(gate["quality_pass"] and gate["native_cpu_budget_pass"])
    gate["requires_actual_aicpu_shadow_before_dispatch"] = True
    (out / "gate.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
