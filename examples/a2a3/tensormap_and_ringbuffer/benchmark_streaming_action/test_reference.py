#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Deterministic prefix, work-conservation and two-action objective checks."""

import unittest
from dataclasses import replace

from prefix_replay import SubmittedTask, replay
from reference import Gate, Join, Prefix, Slot, Snapshot, Task, Visibility, compare, conditional_remaining


class StreamingReferenceTest(unittest.TestCase):
    def fixture(self):
        p = Prefix()
        u = Task(1, "AIC", (2.0, 10.0), ready=0.0)
        t = Task(2, "AIC", (4.0, 14.0), ready=0.0, preference=0)
        p.submit(u)
        p.submit(t)
        s = Snapshot(0.0, (Slot(0, "AIC", 14.0, 0), Slot(1, "AIC", 0.0, 1)), 1)
        return p, [u, t], s

    def decision(self, p, h, s, **kw):
        return compare(p, h, s, gate=Gate(0.0, 30), **kw)

    def add_join(self, p, registered=True):
        p.submit(Task(3, "AIC", (1.0, 1.0)))
        p.submit(Task(4, "AIC", (1.0, 1.0)))
        p.register_join(Join(4, (2, 3), registered))

    def test_both_tasks_limit_gain(self):
        p, h, s = self.fixture()
        d = self.decision(p, h, s)
        self.assertEqual((d.a0, d.a1, d.gain, d.task_id), (24.0, 16.0, 8.0, 2))

    def test_slow_visible_sibling_masks_gain(self):
        p, h, s = self.fixture()
        self.add_join(p)
        d = self.decision(p, h, replace(s, predecessor_finish=((3, 30.0),)))
        self.assertEqual((d.a0, d.a1, d.gain, d.task_id), (30.0, 30.0, 0.0, 1))

    def test_fast_visible_sibling_keeps_net_two(self):
        p, h, s = self.fixture()
        self.add_join(p)
        d = self.decision(p, h, replace(s, predecessor_finish=((3, 12.0),)))
        self.assertEqual((d.a0, d.a1, d.gain), (24.0, 16.0, 8.0))

    def test_same_prefix_different_future(self):
        a, ha, sa = self.fixture()
        b, hb, sb = self.fixture()
        before_a = self.decision(a, ha, sa)
        before_b = self.decision(b, hb, sb)
        self.assertEqual(before_a, before_b)
        self.add_join(a)
        b.submit(Task(99, "AIV", (1.0, 1.0)))
        self.assertEqual(before_a, before_b)

    def test_diamond_submission_timing(self):
        p, h, s = self.fixture()
        self.assertEqual(self.decision(p, h, s).task_id, 2)
        self.add_join(p)
        s = replace(s, predecessor_finish=((3, 30.0),))
        self.assertEqual(self.decision(p, h, s).task_id, 1)
        self.assertEqual(self.decision(p, h, s, policy="C").task_id, 2)

    def test_incomplete_join_not_used(self):
        p, h, s = self.fixture()
        self.add_join(p, registered=False)
        self.assertEqual(p.successor_state(2), Visibility.REGISTERING)
        self.assertEqual(self.decision(p, h, replace(s, predecessor_finish=((3, 30.0),))).task_id, 2)
        p.register_join(Join(4, (2, 3), True))
        self.assertEqual(p.successor_state(2), Visibility.USABLE)

    def test_unknown_is_not_absent(self):
        p, _, _ = self.fixture()
        self.assertEqual(p.successor_state(2), Visibility.UNKNOWN)

    def test_single_preference_head_dispatches_with_stale_peer(self):
        p, h, s = self.fixture()
        self.assertEqual(compare(p, [h[1]], replace(s, summary_age=1000)).task_id, 2)

    def test_stale_summary_defaults(self):
        p, h, s = self.fixture()
        d = self.decision(p, h, replace(s, summary_age=6.0))
        self.assertEqual((d.task_id, d.reason), (1, "stale"))

    def test_missing_summary_defaults(self):
        p, h, s = self.fixture()
        self.assertEqual(self.decision(p, h, replace(s, initialized=False)).reason, "uninitialized")

    def test_expiry_vetoes_override_without_sleep(self):
        p, h, s = self.fixture()
        self.assertEqual(self.decision(p, h, replace(s, now=10)).reason, "expired")

    def test_pending_is_not_idle(self):
        p, h, s = self.fixture()
        slots = (s.slots[0], replace(s.slots[1], available=1.0))
        d = self.decision(p, h, replace(s, slots=slots))
        self.assertEqual((d.task_id, d.reason), (None, "no_idle_slot"))

    def test_conditional_remaining_and_censoring(self):
        self.assertEqual(conditional_remaining([6, 8, 10, 15, 17], 11), 5)
        self.assertIsNone(conditional_remaining([6, 8], 11))

    def test_insufficient_independent_blocks_never_override(self):
        p, h, s = self.fixture()
        d = compare(p, h, s, gate=Gate(0, 6))
        self.assertEqual((d.task_id, d.reason), (1, "uncalibrated_gate"))

    def test_optimism_and_extra_cost_are_paid(self):
        p, h, s = self.fixture()
        self.assertEqual(compare(p, h, s, gate=Gate(7.0, 30, 0.5)).task_id, 1)

    def test_illegal_unpublished_head_is_rejected(self):
        p, h, s = self.fixture()
        with self.assertRaises(ValueError):
            self.decision(p, h + [Task(99, "AIC", (1, 1), ready=0)], s)

    def test_sync_default_is_not_skipped(self):
        p, h, s = self.fixture()
        h[0] = replace(h[0], special=True)
        p.tasks[1] = h[0]
        d = self.decision(p, h, s)
        self.assertEqual((d.task_id, d.reason), (1, "ineligible_default"))

    def test_fanin_overflow_does_not_truncate_max(self):
        p, h, s = self.fixture()
        for i in range(3, 8):
            p.submit(Task(i, "AIC", (1, 1)))
        p.register_join(Join(7, (2, 3, 4, 5, 6), True))
        s = replace(s, predecessor_finish=tuple((i, 100.0) for i in range(3, 7)))
        self.assertEqual(self.decision(p, h, s).gain, 8.0)

    def test_empty(self):
        self.assertIsNone(compare(Prefix(), [], Snapshot(0, (), 0)).task_id)


class PrefixReplayTest(unittest.TestCase):
    def fixture(self, sibling=30.0, submit_join=0.0, alternate_future=False):
        tasks = [
            SubmittedTask(Task(0, "AIC", (14, 14), special=True), 0, fixed_side=0),
            SubmittedTask(Task(1, "AIC", (2, 10)), 0),
            SubmittedTask(Task(2, "AIC", (4, 14), preference=0), 0),
            SubmittedTask(Task(3, "AIV", (sibling, sibling), special=True), 0, fixed_side=0),
            SubmittedTask(Task(4, "AIC", (1, 1)), submit_join, predecessors=(1 if alternate_future else 2, 3)),
        ]
        slots = (Slot(0, "AIV", 0, 2), Slot(0, "AIC", 0, 0), Slot(1, "AIC", 0, 1))
        return tasks, slots

    def test_recomputed_ready_and_resources(self):
        definitions, slots = self.fixture(sibling=12)
        b = replay(definitions, "B", slots)
        c = replay(definitions, "C", slots)
        d = replay(definitions, "D", slots)
        self.assertEqual((b["makespan_us"], c["makespan_us"], d["makespan_us"]), (25, 16, 16))
        self.assertEqual((b["claims"][4][0], d["claims"][4][0]), (24, 14))
        self.assertEqual(len(d["claims"]), len(definitions))

    def test_slow_join_masks_local_override(self):
        definitions, slots = self.fixture()
        c = replay(definitions, "C", slots)
        d = replay(definitions, "D", slots)
        self.assertEqual(c["claims"][2][0], 0)
        self.assertEqual(d["claims"][1][0], 0)
        self.assertEqual((c["makespan_us"], d["makespan_us"]), (31, 31))

    def test_late_future_is_invisible(self):
        a, slots = self.fixture(submit_join=19)
        b, _ = self.fixture(submit_join=19, alternate_future=True)
        ra = replay(a, "D", slots)
        rb = replay(b, "D", slots)
        da = [x for x in ra["decisions"] if x["time"] < 19]
        db = [x for x in rb["decisions"] if x["time"] < 19]
        self.assertEqual(da, db)

    def test_actual_future_duration_is_not_a_policy_feature(self):
        definitions, slots = self.fixture()
        changed = [replace(d, actual_spans=(40, 40)) if d.task.task_id == 3 else d for d in definitions]
        a = replay(definitions, "D", slots)
        b = replay(changed, "D", slots)
        self.assertEqual([x for x in a["decisions"] if x["time"] == 0], [x for x in b["decisions"] if x["time"] == 0])

    def test_all_groups_dispatch_once(self):
        definitions, slots = self.fixture(sibling=12)
        for mode in ("A", "B", "C", "D"):
            result = replay(definitions, mode, slots)
            self.assertEqual(result["published"], 5)
            self.assertEqual(set(result["completed"]), set(range(5)))


if __name__ == "__main__":
    unittest.main()
