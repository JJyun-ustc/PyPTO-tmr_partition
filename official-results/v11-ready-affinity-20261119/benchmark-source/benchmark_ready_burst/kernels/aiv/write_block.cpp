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
#include "intrinsic.h"
#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *dst = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *src = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(dst->buffer.addr) + dst->start_offset;
    __gm__ float *seed = reinterpret_cast<__gm__ float *>(src->buffer.addr) + src->start_offset;
    int32_t block = get_block_idx(args);
    int32_t spins = static_cast<int32_t>(args[2]);
    volatile int32_t accumulator = 0;
    for (int32_t i = 0; i < spins; ++i)
        accumulator += i & 1;
    out[16 * block] = seed[0] + static_cast<float>(block + 1);
    dcci(&out[16 * block], 0, 0);
}
