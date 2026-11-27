#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefix-only reference policy. Times are microseconds; this is not an NPU timing model."""

from dataclasses import dataclass, replace
from enum import Enum
from math import inf, isfinite
from statistics import median


class Visibility(str, Enum):
    UNKNOWN = "unknown"
    REGISTERING = "registering"
    USABLE = "usable"


@dataclass(frozen=True)
class Task:
    task_id: int
    resource: str
    spans: tuple  # calibrated span by execution side, not observed future duration
    ready: float | None = None
    preference: int = -1
    blocks: int = 1
    claimed: int = 0
    special: bool = False
    generation: int = 0

    @property
    def eligible(self):
        return self.resource in ("AIC", "AIV") and self.blocks == 1 and not self.claimed and not self.special


@dataclass(frozen=True)
class Join:
    task_id: int
    predecessors: tuple
    registered: bool = False
    completion_gated: bool = True


class Prefix:
    """Only submit/register methods can make graph information visible."""

    def __init__(self):
        self.tasks = {}
        self.joins = {}
        self.successors = {}
        self.epoch = 0

    def submit(self, task):
        if task.task_id in self.tasks:
            raise ValueError("duplicate task generation")
        self.tasks[task.task_id] = task
        self.epoch += 1

    def register_join(self, join):
        if join.task_id not in self.tasks:
            raise ValueError("unsubmitted successor")
        if any(p not in self.tasks for p in join.predecessors):
            raise ValueError("unsubmitted predecessor")
        old = self.joins.get(join.task_id)
        if old is not None and (old.registered or old.predecessors != join.predecessors):
            raise ValueError("immutable registration")
        self.joins[join.task_id] = join
        for p in join.predecessors:
            self.successors.setdefault(p, set()).add(join.task_id)
        self.epoch += 1

    def successor_state(self, task_id):
        known = self.successors.get(task_id, ())
        if not known:
            return Visibility.UNKNOWN
        return Visibility.USABLE if any(self.joins[j].registered for j in known) else Visibility.REGISTERING

    def usable_join(self, affected):
        # An overflowing index cannot silently erase a predecessor or successor.
        candidates = set().union(*(self.successors.get(x, set()) for x in affected))
        if len(candidates) > 2:
            return None
        for jid in sorted(candidates):
            j = self.joins[jid]
            if j.registered and j.completion_gated and len(j.predecessors) <= 4:
                return j
        return None


@dataclass(frozen=True)
class Slot:
    side: int
    resource: str
    available: float
    group: int


@dataclass(frozen=True)
class Snapshot:
    now: float
    slots: tuple
    local_slot: int
    summary_age: float = 0.0
    initialized: bool = True
    # Predicted finish or real release for visible committed/completed predecessors.
    predecessor_finish: tuple = ()


@dataclass(frozen=True)
class Decision:
    task_id: int | None
    gain: float
    reason: str
    default: int | None
    alternative: int | None
    endpoint_ids: tuple = ()
    a0: float | None = None
    a1: float | None = None
    epoch: int = 0


def conditional_remaining(spans, elapsed):
    alive = [x - elapsed for x in spans if x > elapsed]
    return median(alive) if alive else None


@dataclass(frozen=True)
class Gate:
    optimism: float | None = None
    independent_blocks: int = 0
    extra_cost: float = 0.0
    delta: float = 0.5

    def permits(self, gain):
        return (
            self.independent_blocks >= 30
            and self.optimism is not None
            and gain > self.optimism + self.extra_cost + self.delta
        )


def action_finish(tasks, first, snapshot):
    """Same physical first slot; place each known task once under B continuation."""
    slots = list(snapshot.slots)
    finish = {}
    first_slot = slots[snapshot.local_slot]
    order = [first] + sorted((t for t in tasks if t.task_id != first.task_id), key=lambda t: (t.ready, t.task_id))
    for i, task in enumerate(order):
        choices = (
            [snapshot.local_slot] if i == 0 else [k for k, slot in enumerate(slots) if slot.resource == task.resource]
        )
        if not choices:
            return None

        def completion(k):
            s = slots[k]
            span = task.spans[s.side]
            return max(snapshot.now, task.ready, s.available) + span if span is not None else inf

        # Default claiming is work-conserving: choose earliest start, not a
        # faster future slot that would leave an idle compatible slot unused.
        choices = [k for k in choices if task.spans[slots[k].side] is not None]
        if not choices:
            return None
        k = min(
            choices, key=lambda k: (max(snapshot.now, task.ready, slots[k].available), slots[k].side, slots[k].group)
        )
        end = completion(k)
        if not isfinite(end) or (i == 0 and first_slot.resource != task.resource):
            return None
        finish[task.task_id] = end
        slots[k] = replace(slots[k], available=end)
    return finish


def compare(prefix, heads, snapshot, policy="D", gate=Gate(), expiry=10.0, max_age=5.0):
    # heads are already legal published queue heads, at most one per logical lane.
    if len(heads) > 3:
        raise ValueError("head budget exceeded")
    for t in heads:
        if t.task_id not in prefix.tasks or prefix.tasks[t.task_id] != t or t.ready is None:
            raise ValueError("head is not visible and ready")
    ordered = sorted(heads, key=lambda t: (t.ready, t.preference + 1, t.generation, t.task_id))
    if not ordered:
        return Decision(None, 0, "empty", None, None, epoch=prefix.epoch)
    u = ordered[0]

    def default(reason, t=None):
        return Decision(u.task_id, 0, reason, u.task_id, t.task_id if t else None, epoch=prefix.epoch)

    if policy == "B":
        return default("structure")
    if not u.eligible:
        return default("ineligible_default")
    other = [t for t in ordered[1:] if t.eligible and t.resource == u.resource]
    if not other:
        return default("single_candidate")
    t = other[0]
    if max(snapshot.now - u.ready, snapshot.now - t.ready) >= expiry:
        return default("expired", t)
    if not snapshot.initialized:
        return default("uninitialized", t)
    if snapshot.summary_age > max_age:
        return default("stale", t)
    local = snapshot.slots[snapshot.local_slot]
    if local.available > snapshot.now or local.resource != u.resource:
        return Decision(None, 0, "no_idle_slot", u.task_id, t.task_id, epoch=prefix.epoch)
    tasks = [x for x in ordered if x.eligible and x.resource == u.resource]
    f0 = action_finish(tasks, u, snapshot)
    f1 = action_finish(tasks, t, snapshot)
    if f0 is None or f1 is None:
        return default("unknown_span_or_capacity", t)
    endpoints = tuple(x.task_id for x in tasks)
    if policy == "D":
        join = prefix.usable_join(endpoints)
        known = dict(snapshot.predecessor_finish)
        if join and all(p in f0 or p in known for p in join.predecessors):

            def join_ready(f):
                return max(f[p] if p in f else known[p] for p in join.predecessors)

            f0[join.task_id] = join_ready(f0)
            f1[join.task_id] = join_ready(f1)
            endpoints = tuple(p for p in endpoints if p not in join.predecessors) + (join.task_id,)
    m0, m1 = max(f0[p] for p in endpoints), max(f1[p] for p in endpoints)
    gain = m0 - m1
    chosen = t if gate.permits(gain) else u
    reason = "override" if chosen is t else ("uncalibrated_gate" if gate.independent_blocks < 30 else "below_gate")
    return Decision(chosen.task_id, gain, reason, u.task_id, t.task_id, endpoints, m0, m1, prefix.epoch)
