#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Resource-calendar DAG replay; counterfactuals remain separate from real labels.

Ready events depend on replayed producer completion, never recorded ready times.
MIX masks reserve cores within one physical group. Sync cohorts start together.
The model omits instruction-level synchronization and is checked against traces.
"""

import argparse
import heapq
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from round2_data import distribution
from round2_train import (
    Attention,
    features,
    fit_calibration,
    key,
    load_runs,
)


def run_compute(t, ready, groups, caps, cal, selected):
    sides = [selected] if selected in (0, 1) and not t["sync"] else [0, 1]
    row = cal.get("replay_runtime_table", {}).get(key(t)) or cal["table"].get(key(t))
    whole = max(0.02, row["intercept_us"] + row["per_seq_us"] * t["seq_len"]) if row else cal["fallback_us"]
    compat = sum(caps[c] * (2 if t["shape"] == 1 else 1) for c in sides)
    service = whole / max(1, (t["blocks"] + compat - 1) // compat) * cal.get("replay_service_factor", 1.0)
    starts = []
    ends = []
    cohort = []
    cohort_used = set()
    for block in range(t["blocks"]):
        options = []
        for c in sides:
            for g in range(caps[c]):
                if t["shape"] == 0:
                    options.append((groups[c][g, 0], c, g, [0]))
                elif t["shape"] == 1:
                    options.extend((groups[c][g, j], c, g, [j]) for j in (1, 2))
                else:
                    slots = [j for j in range(3) if t["mask"] & (1 << j)]
                    options.append((max(groups[c][g, slots]), c, g, slots))
        if t["sync"]:
            options = [v for v in options if all((v[1], v[2], j) not in cohort_used for j in v[3])]
        free, c, g, slots = min(options, key=lambda x: (x[0], x[1], x[2], x[3]))
        start = max(ready, float(free))
        end = start + service
        groups[c][g, slots] = end
        starts.append(start)
        ends.append(end)
        cohort.append((c, g, slots))
        if t["sync"]:
            cohort_used.update((c, g, j) for j in slots)
    if t["sync"]:
        # The measured workload's sync cohort fits the hardware groups.
        if t["blocks"] > compat:
            raise ValueError("oversize sync cohort unsupported in replay")
        common = max(starts)
        end = common + service
        start = common
        for c, g, slots in cohort:
            groups[c][g, slots] = end
    else:
        start = min(starts)
        end = max(ends)
    return start, end


def replay(data, cal, forced=None):
    tasks = {t["task"]: t for t in data["tasks"]}
    assert all(a in tasks and b in tasks for a, b in data["edges"]), (
        "incomplete trace cannot supervise counterfactual replay"
    )
    origin = min(t["created"] for t in tasks.values())
    succ = {k: [] for k in tasks}
    indeg = {k: 0 for k in tasks}
    parent_done = {k: 0.0 for k in tasks}
    for a, b in set(map(tuple, data["edges"])):
        if a in tasks and b in tasks and a != b:
            succ[a].append(b)
            indeg[b] += 1
    # Derive the actual asymmetric per-side topology from captured snapshots.
    caps = [max([int(t["features"][c][3]) for t in tasks.values() if t["shape"] == 0] + [1]) for c in (0, 1)]
    if sum(caps) <= 2:
        caps = [15, 9]
    groups = [np.zeros((cap, 3)) for cap in caps]
    cpus = [0.0] * 5
    events = []
    for k, n in indeg.items():
        if not n:
            heapq.heappush(events, ((tasks[k]["created"] - origin) / 50, k))
    completed = {}
    started = {}
    ready_times = {}
    while events:
        ready, k = heapq.heappop(events)
        t = tasks[k]
        ready_times[k] = ready
        if t["shape"] == 3 and t.get("proxy_kind") == 6:
            start = ready
            end = start + max(0.02, (t["fin"] - t["start_proxy"]) / 50)
        elif t["shape"] == 3:
            s = min(range(5), key=lambda i: cpus[i])
            start = max(ready, cpus[s])
            dur = max(0.02, (t["dummy_end"] - t["dummy_begin"]) / 50)
            end = start + dur
            cpus[s] = end
        else:
            selected = forced[1] if forced and forced[0] == k else t["selected"]
            start, end = run_compute(t, ready, groups, caps, cal, selected)
        started[k] = start
        completed[k] = end
        for b in succ[k]:
            parent_done[b] = max(parent_done[b], end)
            indeg[b] -= 1
            if indeg[b] == 0:
                heapq.heappush(
                    events,
                    (max(parent_done[b], (tasks[b]["created"] - origin) / 50), b),
                )
    assert len(completed) == len(tasks), (
        len(completed),
        len(tasks),
        "cycle/incomplete graph",
    )
    return {
        "makespan_us": max(completed.values()),
        "start": started,
        "finish": completed,
        "ready": ready_times,
    }


def predict(t, side, cal, w, model):
    x, tok, base = features(t, side, cal)
    z = (np.array(x) - w["mean"]) / w["std"]
    linear = base + (np.append(z, 1) @ np.array(w["linear"])) * w["residual_std"] + w["residual_mean"]
    with torch.no_grad():
        res = model(
            torch.tensor(z[None], dtype=torch.float32),
            torch.tensor([tok], dtype=torch.float32),
        ).numpy()[0]
    attention = base + res * w["residual_std"] + w["residual_mean"]
    return {
        "base": float(np.maximum(base, 0).sum()),
        "linear": float(np.maximum(linear, 0).sum()),
        "attention": float(np.maximum(attention, 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    args = ap.parse_args()
    cal = json.loads((args.models / "calibration.json").read_text())
    w = json.loads((args.models / "weights.json").read_text())
    for k in ("mean", "std", "residual_mean", "residual_std"):
        w[k] = np.array(w[k])
    model = Attention()
    model.load_state_dict(torch.load(args.models / "attention.pt", weights_only=True))
    model.eval()
    _, runs = load_runs(args.data)
    checks = []
    rankings = []
    # One global service-scale calibration, fitted to train runs only. This
    # corrects proxy-span inflation without fitting candidate ranking labels.
    train = [d for d in runs if d["run"]["seq_len"] in (512, 1750)]
    replay_fit = fit_calibration([t for d in runs for t in d["tasks"]])
    cal["replay_runtime_table"] = replay_fit["table"]
    cal["replay_fit_runs"] = replay_fit["fit_runs"]

    def observed(d):
        ts = d["tasks"]
        return (max(max(t["fin"], t["completed"]) for t in ts) - min(t["created"] for t in ts)) / 50

    target = statistics.median(observed(d) for d in train)
    low, high = 0.1, 1.2
    fit = []
    for iteration in range(6):
        factor = (low + high) / 2
        cal["replay_service_factor"] = factor
        prediction = statistics.median(replay(d, cal)["makespan_us"] for d in train)
        fit.append({"factor": factor, "train_median_us": prediction, "target_us": target})
        if prediction > target:
            high = factor
        else:
            low = factor
    cal["replay_service_factor"] = (low + high) / 2
    cal["replay_scale_fit"] = {
        "runs": [d["run"]["run_id"] for d in train],
        "iterations": fit,
        "method": "six bisection iterations on train median makespan; no holdout labels",
    }
    (args.models / "calibration.json").write_text(json.dumps(cal, indent=2))
    for data in runs:
        tasks = data["tasks"]
        r = replay(data, cal)
        origin = min(t["created"] for t in tasks)
        observed = (max(max(t["fin"], t["completed"]) for t in tasks) - origin) / 50
        start_errors = [
            abs(r["start"][t["task"]] - (t["start_proxy"] - origin) / 50)
            for t in tasks
            if t["start_proxy"] and t["proxy_kind"] in (1, 2, 3)
        ]
        checks.append(
            {
                "run_id": data["run"]["run_id"],
                "observed_us": observed,
                "replayed_us": r["makespan_us"],
                "relative_error_percent": 100 * (r["makespan_us"] / observed - 1),
                "start_proxy_absolute_us": distribution(start_errors),
            }
        )
        if data["run"]["seq_len"] != 3500:
            continue
        candidates = [
            t for t in tasks if t["selected"] in (0, 1) and t["features"][0][3] > 0 and t["features"][1][3] > 0
        ]
        # A fixed stratified subset, no ranking/target-dependent sampling.
        ids = np.linspace(0, len(candidates) - 1, min(16, len(candidates)), dtype=int)
        for i in ids:
            t = candidates[i]
            both = [replay(data, cal, (t["task"], side)) for side in (0, 1)]
            scores = [predict(t, side, cal, w, model) for side in (0, 1)]
            label = int(both[1]["finish"][t["task"]] < both[0]["finish"][t["task"]])
            rankings.append(
                {
                    "source": "calibrated_replay_not_real_action",
                    "run_id": t["run_id"],
                    "task": t["task"],
                    "actual_side": t["selected"],
                    "replay_best_side": label,
                    "focal_finish_us": [v["finish"][t["task"]] for v in both],
                    "global_makespan_us": [v["makespan_us"] for v in both],
                    "model_scores": scores,
                    "chosen": {m: int(scores[1][m] < scores[0][m]) for m in ("base", "linear", "attention")},
                    "ready_events_recomputed": True,
                }
            )
    (args.models / "replay-validation.json").write_text(json.dumps(checks, indent=2))
    (args.models / "replay-counterfactuals.json").write_text(json.dumps(rankings, indent=2))
    nontrivial = [r for r in rankings if abs(r["focal_finish_us"][0] - r["focal_finish_us"][1]) >= 0.02]
    totals = {
        m: {
            "n": len(rankings),
            "agreement": sum(r["chosen"][m] == r["replay_best_side"] for r in rankings) / max(1, len(rankings)),
            "non_tie_n": len(nontrivial),
            "non_tie_agreement": sum(r["chosen"][m] == r["replay_best_side"] for r in nontrivial)
            / max(1, len(nontrivial)),
        }
        for m in ("base", "linear", "attention")
    }
    (args.models / "replay-ranking.json").write_text(json.dumps(totals, indent=2))
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
