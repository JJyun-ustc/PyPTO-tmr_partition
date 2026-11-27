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
#include <cmath>
#include <cstdint>

namespace simpler::tmr {
constexpr int STREAM_MAX_HEADS = 3;
constexpr int STREAM_MAX_SLOTS = 72;
struct StreamCandidate {
    uint64_t id{0};
    double ready_us{0}, span_us[2]{0, 0};
    int resource{0};
};
struct StreamSlot {
    double available_us{0};
    int side{0}, resource{0}, group{0};
};
struct StreamJoin {
    bool complete{false};
    int count{0};
    int candidate[4]{-1, -1, -1, -1};
    double committed_finish_us[4]{};
};
struct StreamPairResult {
    bool valid{false}, used_join{false};
    double default_end{0}, alternative_end{0}, gain{0};
};
inline bool stream_action_finish(
    const StreamCandidate *tasks, int n, int first, const StreamSlot *initial, int slots, int local, double now,
    double finish[STREAM_MAX_HEADS]
) {
    if (n < 2 || n > STREAM_MAX_HEADS || slots < 1 || slots > STREAM_MAX_SLOTS || local < 0 || local >= slots ||
        initial[local].available_us > now)
        return false;
    double free_at[STREAM_MAX_SLOTS];
    for (int i = 0; i < slots; ++i)
        free_at[i] = initial[i].available_us;
    // Input heads follow the common B ordering; both actions process each once.
    for (int step = 0; step < n; ++step) {
        int index = step == 0 ? first : (step <= first ? step - 1 : step);
        const auto &task = tasks[index];
        int best = -1;
        double end = INFINITY, earliest = INFINITY;
        for (int k = 0; k < slots; ++k) {
            if ((step == 0 && k != local) || initial[k].resource != task.resource) continue;
            double duration = task.span_us[initial[k].side];
            if (!(duration > 0) || !std::isfinite(duration)) continue;
            double start = now > free_at[k] ? now : free_at[k];
            if (task.ready_us > start) start = task.ready_us;
            double candidate = start + duration;
            if (start < earliest ||
                (start == earliest && best >= 0 &&
                 (initial[k].side < initial[best].side ||
                  (initial[k].side == initial[best].side && initial[k].group < initial[best].group)))) {
                earliest = start;
                end = candidate;
                best = k;
            }
        }
        if (best < 0) return false;
        finish[index] = end;
        free_at[best] = end;
    }
    return true;
}
inline StreamPairResult stream_compare(
    const StreamCandidate *tasks, int n, int alternative, const StreamSlot *slots, int count, int local, double now,
    const StreamJoin &join
) {
    StreamPairResult result;
    if (alternative <= 0 || alternative >= n) return result;
    double f[2][STREAM_MAX_HEADS]{};
    if (!stream_action_finish(tasks, n, 0, slots, count, local, now, f[0]) ||
        !stream_action_finish(tasks, n, alternative, slots, count, local, now, f[1]))
        return result;
    bool usable = join.complete && join.count > 0 && join.count <= 4;
    for (int p = 0; p < join.count && usable; ++p) {
        int i = join.candidate[p];
        if (i >= n || i < -1 || (i == -1 && !std::isfinite(join.committed_finish_us[p]))) usable = false;
    }
    double endpoints[2]{};
    for (int a = 0; a < 2; ++a) {
        for (int i = 0; i < n; ++i) {
            bool dominated = false;
            if (usable)
                for (int p = 0; p < join.count; ++p)
                    if (join.candidate[p] == i) dominated = true;
            if (!dominated && f[a][i] > endpoints[a]) endpoints[a] = f[a][i];
        }
        if (usable) {
            for (int p = 0; p < join.count; ++p) {
                double value = join.candidate[p] >= 0 ? f[a][join.candidate[p]] : join.committed_finish_us[p];
                if (value > endpoints[a]) endpoints[a] = value;
            }
        }
    }
    result.valid = true;
    result.used_join = usable;
    result.default_end = endpoints[0];
    result.alternative_end = endpoints[1];
    result.gain = endpoints[0] - endpoints[1];
    return result;
}
inline bool stream_admit(double gain, double optimism, int independent_blocks, double extra, double delta = 0.5) {
    return independent_blocks >= 30 && optimism >= 0 && gain > optimism + extra + delta;
}
}  // namespace simpler::tmr
