# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the existing MIX/sync/AIV stress golden with both AICPU sides present."""
import copy
import importlib.util
from pathlib import Path
import sys

from simpler_setup import SceneTestCase, scene_test

_source = Path(__file__).resolve().parent.parent / "spmd_sync_start_stress/test_spmd_sync_start_stress.py"
_spec = importlib.util.spec_from_file_location("qkv_sync_stress_base", _source)
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestQkvSyncCoexist(_base.TestSpmdSyncStartStress):
    CASES = copy.deepcopy(_base.TestSpmdSyncStartStress.CASES)
    # Six active AICPU threads expose two sides; the default four fit one side.
    CASES[0]["config"] = {"aicpu_thread_num": 6}


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
