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
    return OrchestrationConfig{.expected_arg_count = 1};
}
__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const auto &out = args.tensor(0).ref();
    const int masks[4] = {3, 5, 6, 7};
    for (int wave = 0; wave < 64; ++wave)
        for (int lane = 0; lane < 16; ++lane) {
            uint32_t shapes[1] = {96}, offsets[1] = {static_cast<uint32_t>(lane * 96)};
            auto view = out.view(shapes, offsets);
            int mask = masks[lane % 4];
            MixedKernels kernels;
            kernels.aic_kernel_id = (mask & 1) ? 0 : -1;
            kernels.aiv0_kernel_id = (mask & 2) ? 1 : -1;
            kernels.aiv1_kernel_id = (mask & 4) ? 2 : -1;
            CoreTaskArgs task;
            task.add_inout(view);
            task.add_scalar(int64_t(0));
            task.launch_spec.set_block_num(2);
            rt_submit_task(kernels, task);
        }
}
}
