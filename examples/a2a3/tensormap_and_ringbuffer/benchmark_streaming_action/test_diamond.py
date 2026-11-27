#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real submission-time visibility checks; Shared dispatch remains the baseline."""

from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

VECTOR = Path(__file__).resolve().parent.parent / "vector_example/kernels/aiv"


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestStreamingDiamond(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/diamond.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN],
        },
        "incores": [
            {"func_id": 0, "source": "kernels/hold.cpp", "core_type": "aiv", "signature": [D.INOUT]},
            {
                "func_id": 1,
                "source": str(VECTOR / "kernel_add_scalar.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": str(VECTOR / "kernel_add.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }
    CASES = [
        {
            "name": f"{'Late' if late else 'Early'}{'Slow' if slow else 'Fast'}",
            "platforms": ["a2a3"],
            "params": {"late": late, "slow": slow},
            "config": {"aicpu_thread_num": 6},
        }
        for late in (False, True)
        for slow in (False, True)
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("output", torch.zeros(4 * 16384, dtype=torch.float32)),
            TensorArg("control", torch.tensor([params["late"], params["slow"]], dtype=torch.int32)),
        )

    def compute_golden(self, args, params):
        for lane, value in enumerate((1, 2, 1, 3)):
            args.output[lane * 16384 : (lane + 1) * 16384] = value


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
