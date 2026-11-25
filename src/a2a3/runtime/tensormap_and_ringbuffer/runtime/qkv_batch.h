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
#include "qkv_trace.h"

namespace simpler::tmr {
constexpr int QKV_BATCH = 8;
constexpr int QKV_CHANNELS = 10;
constexpr uint64_t QKV_SUMMARY_MAX_NS = 200000;
inline int qkv_channel(int shape, int mask) { return shape < 2 ? shape : 2 + mask; }
constexpr int QKV_PENDING = 64;
constexpr int QKV_THREADS = 6;
// Fixed policy. No parameter search: a decision consumes at most 50us per batch.
constexpr uint64_t QKV_BUDGET_NS = 50000;
enum class QkvFallback : int { Full, Snapshot, NoSide, Budget, Oversize, Uninitialized, Stale, Busy, Version, Count };
struct QkvSnapshot {
    double capacity[2][QKV_CHANNELS]{}, idle[2][QKV_CHANNELS]{}, occupied[2][QKV_CHANNELS]{}, queued[2][QKV_CHANNELS]{};
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
        for (int d = 0; d < 6; ++d)
            dot += q[d] * k[d];
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
inline int
qkv_choose(const QkvSnapshot &s, int r, int blocks, double age, double reserved[2][2], bool attention = true) {
    int best = -1;
    double best_score = INFINITY;
    for (int c = 0; c < 2; ++c) {
        if (s.capacity[c][r] < blocks || s.capacity[c][r] <= 0) continue;
        double score = (s.queued[c][r] + reserved[c][r] + blocks) / s.capacity[c][r] +
                       (attention ? qkv_pressure(s, c, r, blocks, age) : 0.0);
        if (score < best_score || (score == best_score && best >= 0 && reserved[c][r] < reserved[best][r])) {
            best = c;
            best_score = score;
        }
    }
    if (best >= 0) reserved[best][r] += blocks;
    return best;
}
struct QkvFastRow {
    int side, channel, blocks;
    double e0, e1, e2, q, o, idle;
    void init(const QkvSnapshot &s, int c, int r, int b) {
        side = c;
        channel = r;
        blocks = b;
        const double inv = 1.0 / std::sqrt(6.0), u = b / s.capacity[c][r];
        q = s.queued[c][r] / s.capacity[c][r];
        o = s.occupied[c][r];
        idle = 1 - s.idle[c][r] / s.capacity[c][r];
        // Remove the common side/resource dot terms and the common 1/sqrt(6).
        double logits[3] = {u * q * inv, u * o * inv + 0.25, u * idle * inv - inv};
        int top = 0;
        for (int j = 1; j < 3; ++j)
            if (logits[j] > logits[top]) top = j;
        double e[3];
        for (int j = 0; j < 3; ++j)
            e[j] = j == top ? 1.0 : std::exp(logits[j] - logits[top]);
        e0 = e[0];
        e1 = e[1];
        e2 = e[2];
    }
    double pressure(double age) const {
        double a = age == 0 ? 1 : std::exp(age / (2 * std::sqrt(6.0)));
        double w1 = e1 * a, w2 = e2 * a * a;
        return (0.5 * e0 * q + w1 * o + 0.5 * w2 * idle) / (e0 + w1 + w2);
    }
};
struct QkvBatchScorer {
    QkvFastRow rows[16];
    int count{0};
    double pressure(const QkvSnapshot &s, int c, int r, int b, double age) {
        for (int j = 0; j < count; ++j)
            if (rows[j].side == c && rows[j].channel == r && rows[j].blocks == b) return rows[j].pressure(age);
        rows[count].init(s, c, r, b);
        return rows[count++].pressure(age);
    }
    int choose(const QkvSnapshot &s, int r, int blocks, double age, double reserved[2][QKV_CHANNELS]) {
        int best = -1;
        double value = INFINITY;
        for (int c = 0; c < 2; ++c) {
            if (s.capacity[c][r] < blocks) continue;
            double score =
                (s.queued[c][r] + reserved[c][r] + blocks) / s.capacity[c][r] + pressure(s, c, r, blocks, age);
            if (score < value || (score == value && best >= 0 && reserved[c][r] < reserved[best][r])) {
                best = c;
                value = score;
            }
        }
        if (best >= 0) reserved[best][r] += blocks;
        return best;
    }
};
// One S owns a row. All observations are atomic; the packed resource word is a
// lightweight versioned snapshot (one release/acquire word per resource).
// No FIN adds a global RMW: all resource paths already update CoreTracker.
struct alignas(64) QkvThread {
    std::atomic<uint64_t> resources[QKV_CHANNELS]{};
    std::atomic<uint64_t> taken[2][QKV_CHANNELS]{};
    std::atomic<uint64_t> shared_added[QKV_CHANNELS]{}, shared_taken[QKV_CHANNELS]{};
    std::atomic<uint64_t> version{0}, stamp{0};
    uint64_t dispatched[2]{};
    void reset() {
        version.store(0);
        stamp.store(0);
        for (auto &x : resources)
            x.store(0, std::memory_order_relaxed);
        for (auto &x : shared_added)
            x.store(0, std::memory_order_relaxed);
        for (auto &x : shared_taken)
            x.store(0, std::memory_order_relaxed);
        for (auto &side : taken)
            for (auto &x : side)
                x.store(0, std::memory_order_relaxed);
        dispatched[0] = dispatched[1] = 0;
    }
};
struct QkvState {
    bool enabled{false};
    int mode{0};
    QkvTrace trace;
    std::atomic<uint32_t> active_channels{3};
    // One atomic word binds a hint to its timestamp (ticks in high bits).
    std::atomic<uint64_t> idle_hint[2][QKV_CHANNELS]{};
    std::atomic<uint64_t> shape_eligible[4]{};
    uint64_t shape_assigned[4]{};
    int side[QKV_THREADS]{}, schedulers{0};
    uint32_t capacity[2][QKV_CHANNELS]{};
    QkvThread threads[QKV_THREADS];
    std::atomic<uint64_t> eligible{0}, fallback[static_cast<int>(QkvFallback::Count)]{};
    // Single allocator S0 owns assigned and all batch statistics. Readers only
    // read these after scheduling has quiesced.
    uint64_t assigned_tasks[2]{}, assigned[2][QKV_CHANNELS]{}, batches[QKV_BATCH + 1]{}, ticks{0}, max_ticks{0};
    uint64_t tick_hist[32]{};  // logarithmic elapsed-ns bins; exact batch samples below
    uint64_t decision_ticks[4096]{};
    uint64_t decision_count{0};
    uint64_t learned_tasks{0}, learned_missing{0};
    void reset() {
        enabled = false;
        mode = 0;
        trace.reset();
        schedulers = 0;
        eligible.store(0, std::memory_order_relaxed);
        active_channels.store(3);
        for (auto &row : idle_hint)
            for (auto &x : row)
                x.store(0);
        for (auto &x : shape_eligible)
            x.store(0);
        for (auto &x : shape_assigned)
            x = 0;
        for (auto &x : fallback)
            x.store(0, std::memory_order_relaxed);
        for (auto &t : threads)
            t.reset();
        for (auto &c : capacity)
            for (auto &x : c)
                x = 0;
        for (auto &c : assigned)
            for (auto &x : c)
                x = 0;
        for (auto &x : batches)
            x = 0;
        for (auto &x : tick_hist)
            x = 0;
        assigned_tasks[0] = assigned_tasks[1] = 0;
        ticks = max_ticks = decision_count = 0;
        learned_tasks = learned_missing = 0;
    }
    void fail(QkvFallback reason) { fallback[static_cast<int>(reason)].fetch_add(1, std::memory_order_relaxed); }
};
}  // namespace simpler::tmr
