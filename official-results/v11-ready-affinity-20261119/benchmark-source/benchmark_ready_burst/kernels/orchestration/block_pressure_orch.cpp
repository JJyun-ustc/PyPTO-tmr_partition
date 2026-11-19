/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "orchestration_api.h"
extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 3};
}
__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &inputs) {
    const auto &out = inputs.tensor(0).ref();
    auto seed = inputs.tensor(1).ref();
    auto *cfg = inputs.tensor(2).ref().data_as<int64_t>();
    int tasks = static_cast<int>(cfg[0]);
    int mode = static_cast<int>(cfg[1]);
    bool automatic = cfg[2] != 0;
    SIMPLER_SCOPE_GUARD();
    uint32_t seed_shape[1]{16};
    if (mode == 2) {
        CoreTaskArgs root;
        TensorCreateInfo seed_info(seed_shape, 1, DataType::FLOAT32);
        root.add_output(seed_info);
        root.add_input(seed);
        root.add_scalar(int64_t{2000000});
        if (automatic) root.set_aicpu_cluster_auto();
        auto result = rt_submit_aiv_task(0, root);
        seed = result.get_ref(0);
    }
    uint32_t offset = 0;
    const int sizes[6]{1, 4, 16, 48, 96, 256};
    for (int t = 0; t < tasks; ++t) {
        int blocks = mode == 0 ? 16 : sizes[t % 6];
        uint32_t shape[1]{static_cast<uint32_t>(16 * blocks)};
        uint32_t offsets[1]{offset};
        auto dst = out.view(shape, offsets);
        CoreTaskArgs args;
        args.add_inout(dst);
        args.add_input(seed);
        args.add_scalar(int64_t{0});
        args.launch_spec.set_block_num(blocks);
        if (automatic) args.set_aicpu_cluster_auto();
        rt_submit_aiv_task(0, args);
        offset += shape[0];
    }
}
}
