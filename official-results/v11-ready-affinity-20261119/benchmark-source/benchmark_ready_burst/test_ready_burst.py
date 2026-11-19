#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Delayed producer with multi-block successors, for ready-burst placement."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

SIZES = (1, 4, 16, 48, 96, 256)


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestReadyBurst(SceneTestCase):
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/block_pressure_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "BLOCK_WRITE",
                "source": "kernels/aiv/write_block.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.IN],
            }
        ],
    }
    CASES = [
        {
            "name": name + policy,
            "platforms": ["a2a3"],
            "manual": True,
            "config": {"aicpu_thread_num": 6},
            "params": {"tasks": 384, "mode": mode, "auto": int(policy == "Auto")},
        }
        for name, mode in (("Burst", 2),)
        for policy in ("Shared", "Auto")
    ]

    def generate_args(self, params):
        torch.set_num_threads(2)
        blocks = [16 if params["mode"] == 0 else SIZES[t % 6] for t in range(params["tasks"])]
        return TaskArgsBuilder(
            TensorArg("out", torch.zeros(16 * sum(blocks))),
            TensorArg("seed", torch.zeros(16)),
            TensorArg("config", torch.tensor([params["tasks"], params["mode"], params["auto"]], dtype=torch.int64)),
        )

    def compute_golden(self, args, params):
        offset = 0
        for t in range(params["tasks"]):
            blocks = 16 if params["mode"] == 0 else SIZES[t % 6]
            args.out[offset : offset + blocks * 16 : 16] = torch.arange(1, blocks + 1, dtype=torch.float32) + int(
                params["mode"] == 2
            )
            offset += blocks * 16


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
