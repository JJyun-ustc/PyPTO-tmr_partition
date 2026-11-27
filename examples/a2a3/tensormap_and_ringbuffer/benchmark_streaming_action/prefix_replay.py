#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Discrete-event CPU validation. Future submission events never enter policy inputs."""

import heapq
from collections import deque
from dataclasses import dataclass, replace

from reference import Gate, Join, Prefix, Snapshot, Task, compare, conditional_remaining


@dataclass(frozen=True)
class SubmittedTask:
    task: Task
    submit: float
    predecessors: tuple = ()
    registration_delay: float = 0.0
    fixed_side: int = -1
    actual_spans: tuple | None = None


def validate_definitions(definitions):
    if len({d.task.task_id for d in definitions}) != len(definitions):
        raise ValueError("duplicate task identity")


def predicted_snapshot(prefix, slots, running, completed, now, local):
    predicted_slots = []
    known = dict(completed)
    reliable = True
    for k, slot in enumerate(slots):
        free = now
        if k in running:
            tid, start, _ = running[k]
            duration = prefix.tasks[tid].spans[slot.side]
            remaining = conditional_remaining([duration], now - start)
            if remaining is None:
                free = float("inf")
                if slot.resource == slots[local].resource:
                    reliable = False
            else:
                free = now + remaining
                known[tid] = free
        predicted_slots.append(replace(slot, available=free))
    return Snapshot(now, tuple(predicted_slots), local, initialized=reliable, predecessor_finish=tuple(known.items()))


def replay(definitions, policy, physical_slots, fixture_gate=Gate(0.0, 30)):
    """Perfect calibrated spans are a synthetic fixture, not board causal labels."""
    validate_definitions(definitions)
    future, serial = [], 0
    # The event engine holds the source graph; compare() receives Prefix only.
    for definition in definitions:
        heapq.heappush(future, (definition.submit, serial, "submit", definition))
        serial += 1
    prefix, visible, registered, completed = Prefix(), {}, set(), {}
    lanes = {slot.resource: [deque() for _ in range(3)] for slot in physical_slots}
    published, claims, finishes, decisions = set(), {}, {}, []
    slots = list(physical_slots)
    running = {}
    now = 0.0

    def event(when, kind, value):
        nonlocal serial
        heapq.heappush(future, (when, serial, kind, value))
        serial += 1

    def publish_ready():
        for tid, definition in visible.items():
            if tid in published or tid not in registered:
                continue
            if all(p in completed for p in definition.predecessors):
                task = replace(definition.task, ready=now)
                prefix.tasks[tid] = task
                lanes[task.resource][task.preference + 1].append(tid)
                published.add(tid)

    def process_event(kind, value):
        if kind == "submit":
            d = value
            prefix.submit(replace(d.task, ready=None))
            visible[d.task.task_id] = d
            if d.predecessors:
                prefix.register_join(Join(d.task.task_id, d.predecessors, False))
            event(now + d.registration_delay, "registered", d.task.task_id)
        elif kind == "registered":
            registered.add(value)
            d = visible[value]
            if d.predecessors:
                prefix.register_join(Join(value, d.predecessors, True))
        else:
            slot_index, tid = value
            if tid in completed:
                raise AssertionError("duplicate completion")
            completed[tid] = now
            running.pop(slot_index)

    while future:
        now = future[0][0]
        while future and future[0][0] == now:
            _, _, kind, value = heapq.heappop(future)
            process_event(kind, value)
            publish_ready()
        for slot_index, physical in enumerate(slots):
            if slot_index in running:
                continue
            heads = []
            for lane in lanes[physical.resource]:
                if not lane:
                    continue
                tid = lane[0]
                task, definition = prefix.tasks[tid], visible[tid]
                if task.resource == physical.resource and definition.fixed_side in (-1, physical.side):
                    heads.append(task)
            if not heads:
                continue
            if policy == "A":
                selected = min(heads, key=lambda t: (t.ready, t.task_id)).task_id
            else:
                snap = predicted_snapshot(prefix, slots, running, completed, now, slot_index)
                decision = compare(prefix, heads, snap, policy=policy, gate=fixture_gate)
                decisions.append({"time": now, "side": physical.side, "decision": decision})
                selected = decision.task_id
            if selected is None:
                continue
            task = prefix.tasks[selected]
            lane = lanes[task.resource][task.preference + 1]
            if lane.popleft() != selected or selected in claims:
                raise AssertionError("non-head or duplicate claim")
            if not all(p in completed for p in visible[selected].predecessors):
                raise AssertionError("dependency violation")
            claims[selected] = (now, physical.side)
            predicted = now + task.spans[physical.side]
            factual = visible[selected].actual_spans or task.spans
            finish = now + factual[physical.side]
            running[slot_index] = (selected, now, predicted)
            finishes[selected] = finish
            event(finish, "finish", (slot_index, selected))
    if any(q for queues in lanes.values() for q in queues) or running or len(completed) != len(definitions):
        raise AssertionError("unfinished work or queue not drained")
    return {
        "makespan_us": now,
        "claims": claims,
        "finishes": finishes,
        "decisions": decisions,
        "completed": completed,
        "published": len(published),
    }
