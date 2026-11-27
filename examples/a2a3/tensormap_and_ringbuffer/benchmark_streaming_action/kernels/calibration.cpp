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
    const auto &output = args.tensor(0).ref();
    const int sizes[3] = {1024, 16384, 262144};
    for (int period = 0; period < 6; ++period) {
        for (int size = 0; size < 3; ++size) {
            for (int reuse = 0; reuse < 2; ++reuse) {
                for (int direction = 0; direction < 2; ++direction) {
                    for (int arm = 0; arm < 2; ++arm) {
                        int cross = (period & 1) ? 1 - arm : arm;
                        int lane = size * 2 + reuse;
                        uint32_t shape[1] = {static_cast<uint32_t>(sizes[size])};
                        uint32_t offset[1] = {static_cast<uint32_t>(lane * 262144)};
                        auto region = output.view(shape, offset);
                        for (int repeat = 0; repeat < 21; ++repeat) {
                            for (int consumer = 0; consumer < 2; ++consumer) {
                                CoreTaskArgs task;
                                task.add_inout(region);
                                task.add_scalar(int64_t(sizes[size]));
                                task.add_scalar(int64_t(consumer && reuse ? 8 : 1));
                                task.add_scalar(int64_t(consumer ? 0 : 1));
                                task.set_aicpu_cluster(consumer && cross ? 1 - direction : direction);
                                task.launch_spec.set_block_num(1);
                                rt_submit_aiv_task(0, task);
                            }
                        }
                        uint32_t first[1] = {0};
                        (void)get_tensor_data<float>(region, 1, first);
                    }
                }
            }
        }
    }
}
}
