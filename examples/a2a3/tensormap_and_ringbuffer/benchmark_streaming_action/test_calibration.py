#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paired physical-placement calibration; no learned scheduling is active."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestStreamingCalibration(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/calibration.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [{"func_id": 0, "source": "kernels/reuse.cpp", "core_type": "aiv", "signature": [D.INOUT]}],
    }
    CASES = [
        {"name": "paired_locality", "platforms": ["a2a3sim", "a2a3"], "params": {}, "config": {"aicpu_thread_num": 6}}
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(TensorArg("output", torch.zeros(6 * 262144, dtype=torch.float32)))

    def compute_golden(self, args, params):
        sizes = (1024, 16384, 262144)
        for size, elements in enumerate(sizes):
            for reuse in range(2):
                lane = size * 2 + reuse
                args.output[lane * 262144 : lane * 262144 + elements] = 1 + (8 if reuse else 1)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
