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
#include <cstdint>

namespace simpler::tmr {
constexpr int READY_THREADS = 6;
constexpr int READY_SHAPES = 4;
constexpr int READY_SCOPE_DEPTH = 64;
constexpr uint32_t READY_REFRESH_TASKS = 64;
constexpr uint32_t READY_FEEDBACK_LOOPS = 64;

// Scores use waves, scaled by 256. These are fixed policy prices, not durations.
constexpr uint64_t QUOTA_SCALE = 256;
constexpr uint64_t QUOTA_AFFINITY = 128;
constexpr uint64_t QUOTA_SPLIT_COST = 32;

enum class AffinityStrength : uint8_t { NEUTRAL, WEAK, STRONG };
inline uint64_t pack_votes(uint32_t a, uint32_t b) { return uint64_t(a) | (uint64_t(b) << 32); }
inline uint32_t side_vote(uint64_t votes, int side) { return static_cast<uint32_t>(votes >> (32 * side)); }
inline uint64_t distribution_vote(uint64_t a, uint64_t b) {
    if (!a && !b) return 0;
    if (!a) return pack_votes(0, 256);
    if (!b) return pack_votes(256, 0);
    uint32_t left = static_cast<uint32_t>(256 * a / (a + b));
    return pack_votes(left, 256 - left);
}
struct ReadyTaskState {
    uint32_t scope{0};
    int8_t preferred_cluster{-1};
    AffinityStrength strength{AffinityStrength::NEUTRAL};
    uint8_t hops{0};
    bool bound{false};
    int8_t ready_writer{READY_THREADS - 1};
    int8_t release_cluster{-1};
    int8_t predicted_cluster{-1};
    std::atomic<int32_t> dispatch_cluster{-1};
    uint64_t affinity_votes{0};
    int16_t quota[2]{};
    std::atomic<int16_t> next[2]{};
    int32_t claim(int side, int32_t count, int32_t &start) {
        int32_t end = side ? quota[0] + quota[1] : quota[0];
        int16_t current = next[side].load(std::memory_order_relaxed);
        while (current < end && count > 0) {
            int32_t n = end - current;
            if (n > count) n = count;
            if (next[side].compare_exchange_weak(
                    current, static_cast<int16_t>(current + n), std::memory_order_relaxed
                )) {
                start = current;
                return n;
            }
        }
        start = current;
        return 0;
    }
    std::atomic<uint32_t> dispatched[2]{};
    void reset() {
        scope = 0;
        preferred_cluster = release_cluster = predicted_cluster = -1;
        strength = AffinityStrength::NEUTRAL;
        hops = 0;
        bound = false;
        ready_writer = READY_THREADS - 1;
        dispatch_cluster.store(-1, std::memory_order_relaxed);
        affinity_votes = 0;
        quota[0] = quota[1] = 0;
        next[0].store(0, std::memory_order_relaxed);
        next[1].store(0, std::memory_order_relaxed);
        for (auto &count : dispatched)
            count.store(0, std::memory_order_relaxed);
    }
    uint64_t actual_votes() const {
        return distribution_vote(
            dispatched[0].load(std::memory_order_acquire), dispatched[1].load(std::memory_order_acquire)
        );
    }
};

struct AffinityVote {
    uint32_t known[2]{}, unknown{0};
    uint8_t hops{0};
    bool inferred{false};
    void add(const ReadyTaskState &parent, int strict_side, uint32_t scope) {
        uint64_t actual = parent.actual_votes();
        if (actual) {
            for (int c = 0; c < 2; ++c)
                known[c] += side_vote(actual, c);
            return;
        }
        int side = parent.dispatch_cluster.load(std::memory_order_acquire);
        if (side < 0 && !parent.scope) side = strict_side;
        uint32_t weight = 256;
        if (side < 0 && parent.scope == scope && parent.hops < 7 && parent.preferred_cluster >= 0) {
            side = static_cast<uint8_t>(parent.preferred_cluster);
            weight = 64;
            inferred = true;
            if (parent.hops + 1 > hops) hops = parent.hops + 1;
        }
        if (side >= 0 && side < 2) known[side] += weight;
        else ++unknown;
    }
    void finish(ReadyTaskState &task, uint32_t scope) const {
        task.scope = scope;
        task.hops = hops;
        task.affinity_votes = pack_votes(known[0], known[1]);
        if (known[0] == known[1]) return;
        task.preferred_cluster = known[0] > known[1] ? 0 : 1;
        task.strength =
            unknown || inferred || (known[0] && known[1]) ? AffinityStrength::WEAK : AffinityStrength::STRONG;
    }
};

struct ReadyTrace {
    uint64_t ticks[9]{}, calls[9]{};
};
struct AffinityBuilder {
    uint32_t next_scope{0}, scope_ids[READY_SCOPE_DEPTH]{};
    uint64_t tasks{0}, neutral{0}, weak{0}, strong{0};
    ReadyTrace trace;
    void reset() { *this = AffinityBuilder{}; }
    void open_scope(int depth) { scope_ids[depth] = ++next_scope; }
    void close_scope(int) {}
    void record(const ReadyTaskState &task) {
        ++tasks;
        if (task.strength == AffinityStrength::STRONG) ++strong;
        else if (task.strength == AffinityStrength::WEAK) ++weak;
        else ++neutral;
    }
};

// Each row has exactly one writer. Finished units use six units per logical
// block, so MIX subtasks retire fractions without another shared block counter.
struct ReadySample {
    uint64_t enqueued[2]{}, claimed[2]{}, finished[2]{}, capacity{0};
};
struct alignas(64) ReadyRow {
    std::atomic<uint64_t> enqueued[2]{}, claimed[2]{}, finished[2]{}, capacity{0};
    static void add(std::atomic<uint64_t> &counter, uint64_t n) {
        counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
    }
    ReadySample read() const {
        ReadySample s;
        for (int c = 0; c < 2; ++c) {
            s.enqueued[c] = enqueued[c].load(std::memory_order_relaxed);
            s.claimed[c] = claimed[c].load(std::memory_order_relaxed);
            s.finished[c] = finished[c].load(std::memory_order_relaxed);
        }
        s.capacity = capacity.load(std::memory_order_relaxed);
        return s;
    }
    void reset() {
        for (int c = 0; c < 2; ++c) {
            enqueued[c].store(0);
            claimed[c].store(0);
            finished[c].store(0);
        }
        capacity.store(0);
    }
};
struct alignas(64) ReadyWriter {
    ReadyRow rows[READY_SHAPES];
    uint64_t executed[READY_SHAPES][2]{};
    uint64_t refreshes{0};
    ReadyTrace trace;
    void reset() {
        for (auto &row : rows)
            row.reset();
        for (auto &shape : executed)
            for (auto &n : shape)
                n = 0;
        refreshes = 0;
        trace = ReadyTrace{};
    }
};
// Leaders publish S-only cumulative snapshots. O's row is read directly by O,
// so an O ready event cannot be counted both directly and through a snapshot.
struct alignas(64) ReadySummary {
    std::atomic<uint64_t> version{0};
    ReadyRow rows[READY_SHAPES];
    void reset() {
        version.store(0);
        for (auto &row : rows)
            row.reset();
    }
};
inline uint64_t positive(int64_t n) { return n > 0 ? uint64_t(n) : 0; }
struct ReadyFeedback {
    std::atomic<bool> enabled{false};
    int32_t schedulers[2]{};
    int32_t cluster[READY_THREADS]{-1, -1, -1, -1, -1, -1};
    ReadyWriter writers[READY_THREADS];
    ReadySummary summaries[2];
    ReadySample views[2][READY_SHAPES];
    uint64_t assigned[READY_SHAPES][2]{}, creations{0}, split_tasks{0};
    void reset() {
        enabled.store(false);
        schedulers[0] = schedulers[1] = 0;
        for (auto &c : cluster)
            c = -1;
        for (auto &w : writers)
            w.reset();
        for (auto &s : summaries)
            s.reset();
        for (auto &side : views)
            for (auto &v : side)
                v = ReadySample{};
        for (auto &shape : assigned)
            for (auto &n : shape)
                n = 0;
        creations = split_tasks = 0;
    }
    void publish_side(int leader) {
        auto &out = summaries[cluster[leader]];
        uint64_t v = out.version.load(std::memory_order_relaxed);
        out.version.store(v + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        for (int shape = 0; shape < READY_SHAPES; ++shape) {
            ReadySample sum;
            for (int t = 0; t < READY_THREADS - 1; ++t) {
                if (cluster[t] != cluster[leader]) continue;
                auto s = writers[t].rows[shape].read();
                sum.capacity += s.capacity;
                for (int c = 0; c < 2; ++c) {
                    sum.enqueued[c] += s.enqueued[c];
                    sum.claimed[c] += s.claimed[c];
                    sum.finished[c] += s.finished[c];
                }
            }
            auto &row = out.rows[shape];
            row.capacity.store(sum.capacity, std::memory_order_relaxed);
            for (int c = 0; c < 2; ++c) {
                row.enqueued[c].store(sum.enqueued[c], std::memory_order_relaxed);
                row.claimed[c].store(sum.claimed[c], std::memory_order_relaxed);
                row.finished[c].store(sum.finished[c], std::memory_order_relaxed);
            }
        }
        out.version.store(v + 2, std::memory_order_release);
        ++writers[leader].refreshes;
    }
    void refresh() {
        for (int side = 0; side < 2; ++side) {
            auto &s = summaries[side];
            uint64_t v = s.version.load(std::memory_order_acquire);
            if (v & 1) continue;
            ReadySample next[READY_SHAPES];
            for (int r = 0; r < READY_SHAPES; ++r)
                next[r] = s.rows[r].read();
            std::atomic_thread_fence(std::memory_order_acquire);
            if (v == s.version.load(std::memory_order_relaxed))
                for (int r = 0; r < READY_SHAPES; ++r)
                    views[side][r] = next[r];
        }
        ++writers[READY_THREADS - 1].refreshes;
    }
    void bind(int writer, int shape, const ReadyTaskState &task) {
        for (int c = 0; c < 2; ++c)
            ReadyRow::add(writers[writer].rows[shape].enqueued[c], task.quota[c]);
    }
    void predict(int shape, ReadyTaskState &task, uint64_t units, bool, bool allow_split = true) {
        constexpr int O = READY_THREADS - 1;
        if ((creations++ % READY_REFRESH_TASKS) == 0) refresh();
        double base[2]{}, cap[2]{};
        uint64_t votes = task.affinity_votes;
        double total = uint64_t(side_vote(votes, 0)) + side_vote(votes, 1);
        auto own = writers[O].rows[shape].read();
        for (int c = 0; c < 2; ++c) {
            uint64_t e = own.enqueued[c], claimed = 0, finished = 0;
            for (auto &side : views) {
                e += side[shape].enqueued[c];
                claimed += side[shape].claimed[c];
                finished += side[shape].finished[c];
            }
            double blocked = double(positive(int64_t(assigned[shape][c]) - int64_t(e)));
            double queued = double(positive(int64_t(e) - int64_t(claimed)));
            double inflight = double(positive(int64_t(claimed * 6) - int64_t(finished))) / 6;
            cap[c] = double(views[c][shape].capacity);
            if (!cap[c]) cap[c] = schedulers[c];
            // The unready task contributes a full marginal allocation below;
            // alpha discounts only the older, uncertain future reservations.
            double affinity = total ? double(side_vote(votes, c)) / total : 0.5;
            base[c] = cap[c] ?
                          (queued + inflight + 0.5 * blocked) * QUOTA_SCALE / cap[c] + QUOTA_AFFINITY * (1 - affinity) :
                          0;
        }
        int32_t n = static_cast<int32_t>(units), best = 0;
        if (!cap[0]) best = 0;
        else if (!cap[1]) best = n;
        else {
            double x = cap[0] * (n + cap[1] * (base[1] - base[0]) / QUOTA_SCALE) / (cap[0] + cap[1]);
            if (x < 0) x = 0;
            if (x > n) x = n;
            int32_t candidates[4]{0, n, static_cast<int32_t>(x), static_cast<int32_t>(x) + 1};
            double best_cost = 1e300;
            for (int32_t left : candidates) {
                if (left > n || (!allow_split && left && left < n)) continue;
                double a = left ? base[0] + QUOTA_SCALE * left / cap[0] : 0;
                double b = left < n ? base[1] + QUOTA_SCALE * (n - left) / cap[1] : 0;
                double cost = (a > b ? a : b) + (left && left < n ? QUOTA_SPLIT_COST : 0);
                if (cost < best_cost) {
                    best_cost = cost;
                    best = left;
                }
            }
        }
        if (best && best < n) ++split_tasks;
        task.quota[0] = best;
        task.quota[1] = n - best;
        task.next[0].store(0, std::memory_order_relaxed);
        task.next[1].store(best, std::memory_order_relaxed);
        task.predicted_cluster = best ? 0 : 1;
        for (int c = 0; c < 2; ++c)
            assigned[shape][c] += task.quota[c];
    }
};
}  // namespace simpler::tmr
