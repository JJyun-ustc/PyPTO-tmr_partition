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
#pragma once
#include "qkv_model_weights.h"

namespace simpler::tmr {
// This frozen model is calibrated for the Qwen seq_len=3500 experiment only.
// Missing kernel/launch specifications use the fixed-formula allocator.
inline double qkv_learned_score(
    const QkvSnapshot &s, int c, int r, int blocks, int mask, const int *kernels, double age, double reservation,
    bool attention = true
) {
    double run = learned::runtime(kernels, blocks, mask, 0);
    if (run < 0) return INFINITY;
    double cap = s.capacity[c][r], q = s.queued[c][r] / cap, o = s.occupied[c][r], idle = s.idle[c][r] / cap;
    double reserved = reservation / cap, u = blocks / cap;
    double wait = (q + reserved + o) * run / (u > 1 ? u : 1);
    int kernel = kernels[0] >= 0 ? kernels[0] : kernels[1] >= 0 ? kernels[1] : kernels[2];
    int shape = r < 2 ? r : 2;
    double raw[12] = {u,         q,         reserved,     o,          1 - idle,    age,
                      cap / 32., mask / 7., kernel / 36., shape / 2., wait / 100., run / 100.};
    double x[12];
    for (int j = 0; j < 12; ++j)
        x[j] = (raw[j] - learned::mean[j]) / learned::std[j];
    double tokens[3][5] = {{q, 0, 1, 0, cap / 32.}, {o, .5, 0, 1, cap / 32.}, {1 - idle, 1, 0, 0, cap / 32.}};
    double residual[2];
    if (attention) learned::residual(x, tokens, residual);
    else
        for (int o = 0; o < 2; ++o) {
            double value = learned::linear[12][o];
            for (int j = 0; j < 12; ++j)
                value += x[j] * learned::linear[j][o];
            residual[o] = value * learned::residual_std[o] + learned::residual_mean[o];
        }
    double predicted_wait = wait + residual[0], predicted_run = run + residual[1];
    return (predicted_wait > 0 ? predicted_wait : 0) + (predicted_run > 0 ? predicted_run : 0);
}
inline int qkv_learned_choose(
    const QkvSnapshot &s, int r, int blocks, int mask, const int *kernels, double age, double reserved[2][QKV_CHANNELS],
    QkvBatchScorer &fallback
) {
    if (learned::runtime(kernels, blocks, mask, 0) < 0) return fallback.choose(s, r, blocks, age, reserved);
    int best = -1;
    double value = INFINITY;
    for (int c = 0; c < 2; ++c) {
        if (s.capacity[c][r] < blocks) continue;
        double score = qkv_learned_score(s, c, r, blocks, mask, kernels, age, reserved[c][r]);
        if (score < value || (score == value && best >= 0 && reserved[c][r] < reserved[best][r])) {
            best = c;
            value = score;
        }
    }
    if (best >= 0) reserved[best][r] += blocks;
    return best;
}
}  // namespace simpler::tmr
