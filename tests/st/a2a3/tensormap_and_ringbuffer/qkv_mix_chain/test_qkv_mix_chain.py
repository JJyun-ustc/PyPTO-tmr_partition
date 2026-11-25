#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sixteen short ordinary MIX chains covering masks 3, 5, 6 and 7."""

import copy
import importlib.util
import sys
from pathlib import Path

import torch

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

_source = Path(__file__).resolve().parent.parent / "spmd_multiblock_mix/test_spmd_multiblock_mix.py"
_spec = importlib.util.spec_from_file_location("qkv_mix_base", _source)
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestQkvMixChain(_base.TestSpmdMultiblockMix):
    CALLABLE = copy.deepcopy(_base.TestSpmdMultiblockMix.CALLABLE)
    CALLABLE["orchestration"]["source"] = str(
        Path(__file__).resolve().parent / "kernels/orchestration/qkv_mix_chain.cpp"
    )
    for _k in CALLABLE["incores"]:
        _k["source"] = str(_source.parent / _k["source"])
    CASES = [
        {
            "name": "MasksAndShortChains",
            "platforms": ["a2a3sim", "a2a3"],
            "params": {},
            "config": {"aicpu_thread_num": 6},
        }
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(TensorArg("output", torch.zeros(16 * 96, dtype=torch.float32)))

    def compute_golden(self, args, params):
        for lane in range(16):
            mask = (3, 5, 6, 7)[lane % 4]
            for slot in range(3):
                if mask & (1 << slot):
                    args.output[lane * 96 + (3 + slot) * 16] = 1.0


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
