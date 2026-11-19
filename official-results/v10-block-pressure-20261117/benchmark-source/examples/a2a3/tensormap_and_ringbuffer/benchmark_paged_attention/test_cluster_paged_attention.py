#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paired cluster placement study on online paged attention."""

import ctypes

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.goldens.paged_attention import compute_golden, generate_inputs


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestClusterPagedAttention(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/paged_attention_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "QK",
                "source": "../paged_attention/kernels/aic/aic_qk_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "SF",
                "source": "../paged_attention/kernels/aiv/aiv_softmax_prepare.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT, D.OUT, D.OUT],
            },
            {
                "func_id": 2,
                "name": "PV",
                "source": "../paged_attention/kernels/aic/aic_pv_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 3,
                "name": "UP",
                "source": "../paged_attention/kernels/aiv/aiv_online_update.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.IN, D.INOUT, D.INOUT, D.INOUT, D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": family + mode,
            "platforms": ["a2a3"],
            "manual": True,
            "config": {"aicpu_thread_num": 6},
            "params": {
                "batch": batch,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 128,
                "block_size": 128,
                "context_len": 4096 if family == "Uniform" else 8192,
                "context_lens_list": lengths,
                "max_model_len": 8192,
                "dtype": "bfloat16",
                "placement_auto": int(mode == "Auto"),
            },
        }
        for family, batch, lengths in (("Uniform", 16, None), ("Skew", 32, [128, 512, 2048, 8192]))
        for mode in ("Shared", "Auto")
    ]

    def generate_args(self, params):
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9102)
            inputs = generate_inputs(params)
        specs = [
            TensorArg(name, value) if isinstance(value, torch.Tensor) else Scalar(name, value) for name, value in inputs
        ]
        specs.append(Scalar("placement_auto", ctypes.c_uint64(params["placement_auto"])))
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        tensors = {s.name: s.value for s in args.specs if isinstance(s, TensorArg)}
        compute_golden(tensors, params)
        args.out[:] = tensors["out"]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
