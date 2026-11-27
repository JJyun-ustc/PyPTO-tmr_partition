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
    return OrchestrationConfig{.expected_arg_count = 2};
}
__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const auto &storage = args.tensor(0).ref();
    const auto &control = args.tensor(1).ref();
    uint32_t index[1] = {0};
    int late = get_tensor_data<int32_t>(control, 1, index);
    index[0] = 1;
    int slow = get_tensor_data<int32_t>(control, 1, index);
    uint32_t shape[1] = {16384}, offset[1] = {0};
    auto a = storage.view(shape, offset);
    offset[0] = 16384;
    auto t = storage.view(shape, offset);
    offset[0] = 32768;
    auto g = storage.view(shape, offset);
    offset[0] = 49152;
    auto j = storage.view(shape, offset);
    CoreTaskArgs producer;
    producer.add_inout(a);
    producer.add_scalar(int64_t(1000));
    rt_submit_aiv_task(0, producer);
    CoreTaskArgs branch;
    branch.add_input(a);
    branch.add_output(t);
    branch.add_scalar(1.0f);
    branch.add_scalar(int64_t(16384));
    rt_submit_aiv_task(1, branch);
    CoreTaskArgs sibling;
    sibling.add_inout(g);
    sibling.add_scalar(int64_t(slow ? 2000 : 200));
    rt_submit_aiv_task(0, sibling);
    if (late) {
        index[0] = 0;
        (void)get_tensor_data<float>(t, 1, index);
    }
    CoreTaskArgs join;
    join.add_input(t);
    join.add_input(g);
    join.add_output(j);
    rt_submit_aiv_task(2, join);
}
}
