#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fit frozen time calibration and two residual models using whole-run splits.

Only actual selected-side measurements supervise fitting. Counterfactual replay
labels are produced by round2_replay.py and never enter these real labels.
Start is an explicitly labelled register/promotion/doorbell proxy, not AICore time.
"""

import argparse
import gzip
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from round2_data import distribution, parse
from torch import nn

torch.set_num_threads(2)
np.random.seed(1234)
torch.manual_seed(1234)
FEATURES = [
    "blocks/cap",
    "queued/cap",
    "reserved/cap",
    "occupied",
    "unavailable",
    "age_ms",
    "cap/32",
    "mask/7",
    "kernel/36",
    "shape/2",
    "base_wait/100",
    "base_run/100",
]


def key(t):
    return ",".join(str(t[k]) for k in ("k0", "k1", "k2", "blocks", "mask", "sync"))


def split(t):
    return "train" if t["seq_len"] in (512, 1750) else "validation" if t["seq_len"] == 2625 else "test"


def duration(t):
    return (t["fin"] - t["start_proxy"]) / 50


def fit_calibration(tasks):
    rows = {}
    for t in tasks:
        if split(t) == "train" and t["shape"] != 3 and t["proxy_kind"] in (1, 2, 3) and t["fin"] > t["start_proxy"] > 0:
            rows.setdefault(key(t), []).append(t)
    table = {}
    for k, ts in rows.items():
        lengths = sorted(set(t["seq_len"] for t in ts))
        med = [statistics.median(duration(t) for t in ts if t["seq_len"] == s) for s in lengths]
        slope = (med[-1] - med[0]) / (lengths[-1] - lengths[0]) if len(lengths) > 1 else 0
        table[k] = {
            "intercept_us": float(med[0] - slope * lengths[0]),
            "per_seq_us": float(slope),
            "n": len(ts),
            "lengths": lengths,
        }
    return {
        "table": table,
        "fallback_us": statistics.median(duration(t) for ts in rows.values() for t in ts),
        "actual_start_available": False,
        "clock_ticks_per_us": 50,
        "runtime_label": "first register/promotion/doorbell proxy to final FIN, not device kernel start",
        "fit_runs": sorted({t["run_id"] for t in tasks if split(t) == "train"}),
        "wait_formula": "(queued/cap + batch_reserved/cap + occupied) * calibrated_runtime / max(1,blocks/cap)",
    }


def baseline(t, side, cal):
    row = cal["table"].get(key(t))
    run = max(0.02, row["intercept_us"] + row["per_seq_us"] * t["seq_len"]) if row else cal["fallback_us"]
    q, o, idle, cap, res = t["features"][side]
    wait = (q + res + o) * run / max(1, t["blocks"] / max(1, cap))
    return np.array([wait, run], dtype=np.float64)


def features(t, side, cal):
    q, o, idle, cap, res = t["features"][side]
    base = baseline(t, side, cal)
    x = [
        t["blocks"] / max(1, cap),
        q,
        res,
        o,
        1 - idle,
        min(1, max(0, (t["pop"] - t["ready"]) / 50000)),
        cap / 32,
        t["mask"] / 7,
        t["kernel"] / 36,
        t["shape"] / 2,
        base[0] / 100,
        base[1] / 100,
    ]
    # Tokens describe the three permanent queue/running/idle entries.
    tokens = [
        [q, 0, 1, 0, cap / 32],
        [o, 0.5, 0, 1, cap / 32],
        [1 - idle, 1, 0, 0, cap / 32],
    ]
    return x, tokens, base


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(12, 4)
        self.k = nn.Linear(5, 4)
        self.v = nn.Linear(5, 4)
        self.out = nn.Linear(4, 2)
        self.skip = nn.Linear(12, 2)

    def forward(self, x, tokens):
        q = self.q(x)
        k = self.k(tokens)
        v = self.v(tokens)
        w = torch.softmax((k * q[:, None, :]).sum(-1) / 2, dim=-1)
        return self.skip(x) + self.out((w[:, :, None] * v).sum(1))


def errors(pred, true):
    e = np.abs(pred - true)
    return {
        "n": len(true),
        "mae_wait_us": float(e[:, 0].mean()),
        "mae_run_us": float(e[:, 1].mean()),
        "composite_mae_us": float(e.mean()),
        "p95_absolute_us": np.percentile(e, 95, axis=0).tolist(),
    }


def load_runs(folder):
    tasks = []
    runs = []
    for path in sorted(folder.rglob("result.json")):
        result = json.loads(path.read_text())
        if result.get("returncode") != 0 or result.get("trace") != 2:
            continue
        data = parse(path.parent)
        (path.parent / "trace-integrity.json").write_text(json.dumps(data["integrity"], indent=2))
        tasks += data["tasks"]
        runs.append(data)
        with gzip.open(path.parent / "tasks.jsonl.gz", "wt") as f:
            for t in data["tasks"]:
                f.write(json.dumps(t) + "\n")
        (path.parent / "dag.json").write_text(json.dumps({"run": result, "edges": data["edges"]}))
    return tasks, runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    tasks, runs = load_runs(args.data)
    cal = fit_calibration(tasks)
    (out / "calibration.json").write_text(json.dumps(cal, indent=2))
    rows = []
    for t in tasks:
        side = t["selected"]
        if (
            side not in (0, 1)
            or t["features"][side][3] <= 0
            or t["proxy_kind"] not in (1, 2, 3)
            or not (t["fin"] > t["start_proxy"] >= t["ready"] > 0)
        ):
            continue
        x, tokens, base = features(t, side, cal)
        rows.append(
            {
                "task": t["task"],
                "run_id": t["run_id"],
                "split": split(t),
                "side": side,
                "x": x,
                "tokens": tokens,
                "base": base.tolist(),
                "y": [(t["start_proxy"] - t["ready"]) / 50, duration(t)],
            }
        )
    assert rows, "no actual-side training labels"
    (out / "splits.json").write_text(
        json.dumps(
            {
                s: {
                    "runs": sorted({r["run_id"] for r in rows if r["split"] == s}),
                    "n": sum(r["split"] == s for r in rows),
                }
                for s in ("train", "validation", "test")
            },
            indent=2,
        )
    )
    with gzip.open(out / "real-action-labels.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    X = np.array([r["x"] for r in rows])
    T = np.array([r["tokens"] for r in rows])
    Y = np.array([r["y"] for r in rows])
    B = np.array([r["base"] for r in rows])
    R = Y - B
    masks = {s: np.array([r["split"] == s for r in rows]) for s in ("train", "validation", "test")}
    for s, m in masks.items():
        assert m.any(), s
    mean = X[masks["train"]].mean(0)
    std = X[masks["train"]].std(0).clip(0.05)
    Z = (X - mean) / std
    ym = R[masks["train"]].mean(0)
    ys = R[masks["train"]].std(0).clip(1)
    A = np.column_stack([Z, np.ones(len(Z))])
    tr = masks["train"]
    reg = np.eye(13) * 1.0
    reg[-1, -1] = 0
    W = np.linalg.solve(A[tr].T @ A[tr] + reg, A[tr].T @ ((R[tr] - ym) / ys))
    linear = B + (A @ W) * ys + ym
    model = Attention()
    optim = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    z = torch.tensor(Z, dtype=torch.float32)
    tokens = torch.tensor(T, dtype=torch.float32)
    target = torch.tensor((R - ym) / ys, dtype=torch.float32)
    best = float("inf")
    state = None
    history = []
    g = torch.Generator().manual_seed(1234)
    indices = torch.tensor(np.where(tr)[0])
    val = torch.tensor(np.where(masks["validation"])[0])
    patience = 0
    for epoch in range(300):
        model.train()
        perm = indices[torch.randperm(len(indices), generator=g)]
        for ids in perm.split(256):
            optim.zero_grad()
            loss = (model(z[ids], tokens[ids]) - target[ids]).square().mean()
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            v = float((model(z[val], tokens[val]) - target[val]).abs().mean())
        history.append({"epoch": epoch, "validation_normalized_mae": v})
        if v < best - 1e-5:
            best = v
            state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 35:
            break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        att = B + model(z, tokens).numpy() * ys + ym
    np.savez(
        out / "linear.npz",
        weight=W,
        mean=mean,
        std=std,
        residual_mean=ym,
        residual_std=ys,
    )
    torch.save(state, out / "attention.pt")
    weights = {k: v.tolist() for k, v in state.items()}
    export = {
        "features": FEATURES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "residual_mean": ym.tolist(),
        "residual_std": ys.tolist(),
        "linear": W.tolist(),
        "attention": weights,
    }
    (out / "weights.json").write_text(json.dumps(export, indent=2))
    (out / "training-history.json").write_text(json.dumps(history, indent=2))
    metrics = {
        s: {
            "base": errors(B[m], Y[m]),
            "linear": errors(linear[m], Y[m]),
            "attention": errors(att[m], Y[m]),
        }
        for s, m in masks.items()
    }
    test = metrics["test"]
    quality = test["attention"]["composite_mae_us"] <= 0.98 * test["linear"]["composite_mae_us"] and all(
        test["attention"][k] <= test["linear"][k] for k in ("mae_wait_us", "mae_run_us")
    )
    # A genuine CPU shadow run of the trained projections; no scheduler actions.
    batch = torch.arange(8) % len(z)
    lat = []
    with torch.no_grad():
        for i in range(1100):
            start = time.perf_counter_ns()
            model(z[batch], tokens[batch])
            elapsed = (time.perf_counter_ns() - start) / 1000
            if i >= 100:
                lat.append(elapsed)
    shadow = distribution(lat)
    budget = shadow["p95"] <= 50
    gate = {
        "predeclared": "test composite MAE at least 2% better than ridge; neither target worse; batch8 p95<=50us",
        "quality_pass": bool(quality),
        "cpu_budget_pass": budget,
        "deploy": bool(quality and budget),
        "cpu_python_shadow_us": shadow,
        "warning": "Python CPU shadow, not AICPU latency; native benchmark must also pass before deployment",
    }
    # Python gate is only a necessary condition; no trained action is enabled here.
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "gate.json").write_text(json.dumps(gate, indent=2))
    bad = np.argsort(np.abs(att - Y).sum(1))[::-1][:30]
    (out / "failure-cases.json").write_text(
        json.dumps(
            [
                {**rows[i], "linear": linear[i].tolist(), "attention": att[i].tolist()}
                for i in bad
                if rows[i]["split"] == "test"
            ],
            indent=2,
        )
    )
    print(json.dumps({"metrics": metrics, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
