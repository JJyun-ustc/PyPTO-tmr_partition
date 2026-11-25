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

#include <atomic>
#include <cmath>
#include <cstdint>

namespace simpler::tmr {
constexpr int QKV_BATCH = 8;
constexpr int QKV_PENDING = 64;
constexpr int QKV_THREADS = 6;
// Fixed policy. No parameter search: a decision consumes at most 50us per batch.
constexpr uint64_t QKV_BUDGET_NS = 50000;
enum class QkvFallback : int { Full, Snapshot, NoSide, Budget, Oversize, Count };
struct QkvSnapshot {
    double capacity[2][2]{}, idle[2][2]{}, occupied[2][2]{}, queued[2][2]{};
};

// Three permanent entries per side/resource: queued, running+pending, idle.
// The mask keeps precisely the candidate side and resource. Six-dimensional
// Q=[side sign, resource sign, normalized blocks, normalized age, 1, 1].
// K=[side sign, resource sign, occupancy, entry_type/2, is_queue, is_running].
// V=[occupancy ratio, release pressure]. Running pressure is occupied slots,
// not remaining execution time; idle pressure also includes MIX/early/sync use.
inline double qkv_pressure(const QkvSnapshot &s, int c, int r, double blocks, double age) {
    double cap = s.capacity[c][r];
    if (cap <= 0) return INFINITY;
    const double occupation[3] = {s.queued[c][r] / cap, s.occupied[c][r], 1.0 - s.idle[c][r] / cap};
    double logits[3], max_logit = -INFINITY;
    for (int j = 0; j < 3; ++j) {
        double q[6] = {c ? 1.0 : -1.0, r ? 1.0 : -1.0, blocks / cap, age, 1, 1};
        double k[6] = {q[0], q[1], occupation[j], j / 2.0, double(j == 0), double(j == 1)};
        double dot = 0;
        for (int d = 0; d < 6; ++d) dot += q[d] * k[d];
        // Relation bias favors the running entry by a fixed dimensionless 1/4.
        logits[j] = dot / std::sqrt(6.0) + (j == 1 ? 0.25 : 0.0);
        if (logits[j] > max_logit) max_logit = logits[j];
    }
    double denominator = 0, z0 = 0, z1 = 0;
    for (int j = 0; j < 3; ++j) {
        double w = std::exp(logits[j] - max_logit);
        denominator += w;
        z0 += w * occupation[j];
        z1 += w * (j == 1 ? occupation[j] : 0);
    }
    return (0.5 * z0 + 0.5 * z1) / denominator;
}
inline int qkv_choose(const QkvSnapshot &s, int r, int blocks, double age, double reserved[2][2]) {
    int best = -1;
    double best_score = INFINITY;
    for (int c = 0; c < 2; ++c) {
        if (s.capacity[c][r] < blocks || s.capacity[c][r] <= 0) continue;
        double score = (s.queued[c][r] + reserved[c][r] + blocks) / s.capacity[c][r] +
                       qkv_pressure(s, c, r, blocks, age);
        if (score < best_score || (score == best_score && best >= 0 && reserved[c][r] < reserved[best][r])) {
            best = c;
            best_score = score;
        }
    }
    if (best >= 0) reserved[best][r] += blocks;
    return best;
}
// One S owns a row. All observations are atomic; the packed resource word is a
// lightweight versioned snapshot (one release/acquire word per resource).
// No FIN adds a global RMW: all resource paths already update CoreTracker.
struct alignas(64) QkvThread {
    std::atomic<uint64_t> resources[2]{};
    std::atomic<uint64_t> taken[2][2]{};
    std::atomic<uint64_t> shared_added[2]{}, shared_taken[2]{};
    uint64_t dispatched[2]{};
    void reset() {
        for (auto &x : resources) x.store(0, std::memory_order_relaxed);
        for (auto &x : shared_added) x.store(0, std::memory_order_relaxed);
        for (auto &x : shared_taken) x.store(0, std::memory_order_relaxed);
        for (auto &side : taken) for (auto &x : side) x.store(0, std::memory_order_relaxed);
        dispatched[0] = dispatched[1] = 0;
    }
};
struct QkvState {
    bool enabled{false};
    int side[QKV_THREADS]{}, schedulers{0};
    uint32_t capacity[2][2]{};
    QkvThread threads[QKV_THREADS];
    std::atomic<uint64_t> eligible{0}, fallback[static_cast<int>(QkvFallback::Count)]{};
    // Single allocator S0 owns assigned and all batch statistics. Readers only
    // read these after scheduling has quiesced.
    uint64_t assigned_tasks[2]{}, assigned[2][2]{}, batches[QKV_BATCH + 1]{}, ticks{0}, max_ticks{0};
    uint64_t tick_hist[32]{};  // logarithmic elapsed-ns bins; exact batch samples below
    uint64_t decision_ticks[4096]{};
    uint64_t decision_count{0};
    void reset() {
        enabled = false; schedulers = 0;
        eligible.store(0, std::memory_order_relaxed);
        for (auto &x : fallback) x.store(0, std::memory_order_relaxed);
        for (auto &t : threads) t.reset();
        for (auto &c : capacity) for (auto &x : c) x = 0;
        for (auto &c : assigned) for (auto &x : c) x = 0;
        for (auto &x : batches) x = 0;
        for (auto &x : tick_hist) x = 0;
        assigned_tasks[0] = assigned_tasks[1] = 0;
        ticks = max_ticks = decision_count = 0;
    }
    void fail(QkvFallback reason) { fallback[static_cast<int>(reason)].fetch_add(1, std::memory_order_relaxed); }
};
}  // namespace simpler::tmr
