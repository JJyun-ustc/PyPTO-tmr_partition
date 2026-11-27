#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Controlled placement of an actual Qwen producer/consumer pair."""

import torch

from simpler_setup.goldens.qwen3_14b_decode import ParamSpec


def generate_orchestration(original, destination):
    text = original.read_text()
    text = text.replace("rt_submit_aic_task(", "pair_submit_aic(")
    text = text.replace("rt_submit_aiv_task(", "pair_submit_aiv(")
    text = text.replace(".expected_arg_count = 20", ".expected_arg_count = 21")
    helper = r"""
static int pair_producer_side, pair_consumer_side;
static uint32_t pair_seen_33, pair_seen_34;
static TaskOutputTensors pair_submit_aiv(int kernel, CoreTaskArgs &args) {
    if (kernel == 33 && (pair_seen_33++ % 17u) == 0) args.set_aicpu_cluster(pair_producer_side);
    return rt_submit_aiv_task(kernel, args);
}
static TaskOutputTensors pair_submit_aic(int kernel, CoreTaskArgs &args) {
    if (kernel == 34 && (pair_seen_34++ % 85u) == 0) args.set_aicpu_cluster(pair_consumer_side);
    return rt_submit_aic_task(kernel, args);
}
"""
    text = text.replace('#include "orchestration_api.h"', '#include "orchestration_api.h"\n' + helper, 1)
    entry = "void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {"
    setup = r"""
    const auto &control = orch_args.tensor(20).ref();
    uint32_t index[1] = {0};
    pair_producer_side = get_tensor_data<int32_t>(control, 1, index);
    index[0] = 1;
    pair_consumer_side = get_tensor_data<int32_t>(control, 1, index);
    pair_seen_33 = pair_seen_34 = 0;
"""
    if entry not in text:
        raise ValueError("unexpected Qwen orchestration entry")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text.replace(entry, entry + setup, 1))


def install(driver, state, manifest):
    original_specs = driver.param_specs
    original_upload = driver._upload_fixture
    original_spec = driver._chip_spec

    def specs(n_layers=40):
        return original_specs(n_layers) + [ParamSpec("pair_control", (2,), "INT32")]

    def spec(*args, **kwargs):
        result = original_spec(*args, **kwargs)
        result["orchestration"]["source"] = manifest["orchestration_source"]
        result["orchestration"]["signature"].append(driver.D.IN)
        return result

    def upload(child, signature, **kwargs):
        driver.param_specs = original_specs
        try:
            original_upload(child, signature[:-1], **kwargs)
        finally:
            driver.param_specs = specs
        child.add("pair_control", torch.zeros(2, dtype=torch.int32), driver.D.IN)
        state["child"] = child

    driver.param_specs = specs
    driver._chip_spec = spec
    driver._upload_fixture = upload


def update_control(worker, child, row):
    worker.copy_to(
        child.buffers["pair_control"], torch.tensor([row["producer_side"], row["consumer_side"]], dtype=torch.int32)
    )
