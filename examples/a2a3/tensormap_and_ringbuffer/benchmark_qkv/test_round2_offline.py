# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check replay dependency propagation and actual-side feature construction."""

import unittest

from round2_replay import replay
from round2_train import features, key


class ReplayTest(unittest.TestCase):
    def task(self, ident, kernel, side, created=100):
        return {
            "task": ident,
            "created": created,
            "selected": side,
            "shape": 0,
            "blocks": 2,
            "sync": 0,
            "mask": 1,
            "k0": kernel,
            "k1": -1,
            "k2": -1,
            "kernel": kernel,
            "seq_len": 512,
            "features": [[0, 0, 1, 2, 0], [0, 0, 1, 2, 0]],
            "ready": 100,
            "pop": 100,
        }

    def test_counterfactual_recomputes_successor_ready(self):
        a = self.task(1, 0, 0)
        b = self.task(2, 1, 0)
        c = self.task(3, 2, 0)
        cal = {
            "table": {key(t): {"intercept_us": 10, "per_seq_us": 0} for t in (a, b, c)},
            "fallback_us": 10,
        }
        data = {"tasks": [a, b, c], "edges": [[1, 3]]}
        left = replay(data, cal, (1, 0))
        right = replay(data, cal, (1, 1))
        self.assertEqual(left["ready"][3], left["finish"][1])
        self.assertEqual(right["ready"][3], right["finish"][1])
        self.assertEqual(left["makespan_us"], 30)
        self.assertEqual(right["makespan_us"], 20)
        self.assertEqual(a["selected"], 0)

    def test_sync_cohort_uses_distinct_groups_even_when_one_side_is_busy(self):
        running = self.task(1, 0, 0)
        sync = self.task(2, 1, -1)
        sync.update({"blocks": 4, "sync": 1})
        cal = {
            "table": {
                key(running): {"intercept_us": 100, "per_seq_us": 0},
                key(sync): {"intercept_us": 10, "per_seq_us": 0},
            },
            "fallback_us": 10,
        }
        r = replay({"tasks": [running, sync], "edges": []}, cal)
        self.assertEqual(r["start"][2], 100)
        self.assertEqual(r["finish"][2], 110)

    def test_incomplete_graph_is_rejected(self):
        t = self.task(1, 0, 0)
        with self.assertRaises(AssertionError):
            replay({"tasks": [t], "edges": [[999, 1]]}, {"table": {}, "fallback_us": 10})

    def test_inline_allocation_does_not_occupy_scheduler_cpu(self):
        a = self.task(1, -1, -1)
        a.update({"shape": 3, "proxy_kind": 6, "start_proxy": 100, "fin": 150})
        b = self.task(2, 0, 0)
        cal = {
            "table": {key(b): {"intercept_us": 10, "per_seq_us": 0}},
            "fallback_us": 10,
        }
        r = replay({"tasks": [a, b], "edges": [[1, 2]]}, cal)
        self.assertEqual(r["ready"][2], 1)
        self.assertEqual(r["makespan_us"], 11)

    def test_tokens_and_baseline_change_with_side(self):
        t = self.task(1, 0, 0)
        t["features"][0] = [2, 0.5, 0.25, 2, 1]
        cal = {
            "table": {key(t): {"intercept_us": 10, "per_seq_us": 0}},
            "fallback_us": 10,
        }
        x, tok, base = features(t, 0, cal)
        other = features(t, 1, cal)
        self.assertEqual(tok[0][0], 2)
        self.assertGreater(base[0], other[2][0])
        self.assertNotEqual(x, other[0])


if __name__ == "__main__":
    unittest.main()
