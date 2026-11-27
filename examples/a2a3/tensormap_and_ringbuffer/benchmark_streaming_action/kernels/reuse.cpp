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


#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "pipe_sync.h"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *data = reinterpret_cast<__gm__ float *>(tensor->buffer.addr) + tensor->start_offset;
    const int64_t elements = args[1], repeats = args[2];
    using Shape = pto::Shape<1, 1, 1, 32, 32>;
    using Stride = pto::Stride<1, 1, 1, 32, 1>;
    using Global = pto::GlobalTensor<float, Shape, Stride>;
    using Tile = pto::Tile<pto::TileType::Vec, float, 32, 32, pto::BLayout::RowMajor, -1, -1>;
    Tile input(32, 32), output(32, 32);
    pto::TASSIGN(input, 0);
    pto::TASSIGN(output, 4096);
    for (int64_t repeat = 0; repeat < repeats; ++repeat) {
        for (int64_t offset = 0; offset < elements; offset += 1024) {
            Global region(data + offset);
            if (args[3]) {
                pto::TEXPANDS(output, 1.0f);
            } else {
                pto::TLOAD(input, region);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                pto::TADDS(output, input, 1.0f);
            }
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            pto::TSTORE(region, output);
            pipe_sync();
        }
    }
}
