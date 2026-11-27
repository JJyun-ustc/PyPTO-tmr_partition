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
#include "intrinsic.h"
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *data = reinterpret_cast<__gm__ float *>(tensor->buffer.addr) + tensor->start_offset;
    uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    uint64_t ticks = static_cast<uint64_t>(args[1]) * 50;
    while (static_cast<uint64_t>(get_sys_cnt()) - begin < ticks) {}
    using Shape = pto::Shape<1, 1, 1, 32, 32>;
    using Stride = pto::Stride<1, 1, 1, 32, 1>;
    using Global = pto::GlobalTensor<float, Shape, Stride>;
    using Tile = pto::Tile<pto::TileType::Vec, float, 32, 32, pto::BLayout::RowMajor, -1, -1>;
    Tile value(32, 32);
    pto::TASSIGN(value, 0);
    pto::TEXPANDS(value, 1.0f);
    for (int offset = 0; offset < 16384; offset += 1024) {
        Global out(data + offset);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pto::TSTORE(out, value);
        pipe_sync();
    }
}
