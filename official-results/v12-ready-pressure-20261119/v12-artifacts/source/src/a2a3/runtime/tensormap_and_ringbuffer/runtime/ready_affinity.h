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
constexpr uint32_t READY_REFRESH_TASKS = 16;
constexpr uint32_t READY_FEEDBACK_LOOPS = 64;
constexpr uint32_t READY_PROBE_LOOPS = 32;
constexpr uint32_t READY_PROBE_COOLDOWN = 128;
constexpr uint64_t READY_UNITS_PER_BLOCK = 6;
constexpr uint64_t READY_WAVE = 256;
// These are policy prices in waves, not measured nanoseconds or kernel durations.
constexpr uint64_t READY_REMOTE_PRICE = 128;
constexpr uint64_t READY_PUBLISH_PRICE = 32;
constexpr uint64_t READY_SWITCH_MARGIN = 16;
constexpr bool READY_REMOTE_PICKUP = true;

enum class AffinityStrength : uint8_t { NEUTRAL, WEAK, STRONG };
inline uint64_t pack_votes(uint32_t a, uint32_t b) { return uint64_t(a) | (uint64_t(b) << 32); }
inline uint32_t side_vote(uint64_t votes, int side) { return static_cast<uint32_t>(votes >> (32 * side)); }
inline uint64_t distribution_vote(uint64_t a, uint64_t b) {
    if (!a && !b) return 0;
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
    std::atomic<int32_t> dispatch_cluster{-1};
    uint64_t affinity_votes{0};
    std::atomic<uint64_t> completed_votes{0};
    std::atomic<uint32_t> dispatched[2]{};
    void reset() {
        scope = 0;
        preferred_cluster = release_cluster = -1;
        strength = AffinityStrength::NEUTRAL;
        hops = 0;
        bound = false;
        ready_writer = READY_THREADS - 1;
        dispatch_cluster.store(-1, std::memory_order_relaxed);
        affinity_votes = 0;
        completed_votes.store(0, std::memory_order_relaxed);
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
            side = parent.preferred_cluster;
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
    uint64_t ticks[8]{}, calls[8]{};
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

// Only the owning O/S writes a row. Completion units divide one block across its active subtasks.
struct alignas(128) ReadyRow {
    std::atomic<uint64_t> enqueued[2]{}, claimed[2]{}, accepted[2]{}, executed[2]{}, completed_units[2]{};
    std::atomic<uint64_t> resources{0};
    static void add(std::atomic<uint64_t> &counter, uint64_t n) {
        counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
    }
    int64_t queue_units(int c) const {
        return int64_t(READY_UNITS_PER_BLOCK) * (int64_t(enqueued[c].load(std::memory_order_relaxed)) -
                                                 int64_t(claimed[c].load(std::memory_order_relaxed)));
    }
    int64_t pressure_units(int c) const {
        return queue_units(c) + int64_t(READY_UNITS_PER_BLOCK) * int64_t(accepted[c].load(std::memory_order_relaxed)) -
               int64_t(completed_units[c].load(std::memory_order_relaxed));
    }
    void reset() {
        for (int c = 0; c < 2; ++c) {
            enqueued[c].store(0);
            claimed[c].store(0);
            accepted[c].store(0);
            executed[c].store(0);
            completed_units[c].store(0);
        }
        resources.store(0);
    }
};
static_assert(sizeof(ReadyRow) == 128);

// Cuts and signed bases describe the same collected rows; clamp only after applying local deltas.
struct alignas(128) ReadyHint {
    std::atomic<uint64_t> version{0}, resources{0};
    std::atomic<int64_t> work{0}, queued{0};
    std::atomic<int64_t> pressure_cut[READY_THREADS]{}, queue_cut[READY_THREADS]{};
    void reset() {
        version.store(0);
        resources.store(0);
        work.store(0);
        queued.store(0);
        for (int t = 0; t < READY_THREADS; ++t) {
            pressure_cut[t].store(0);
            queue_cut[t].store(0);
        }
    }
};
static_assert(sizeof(ReadyHint) == 128);
struct ReadyView {
    int64_t work[2]{}, queued[2]{}, pcut[2]{}, qcut[2]{};
    uint64_t capacity[2]{}, free[2]{}, refreshed_tick{0};
    uint32_t uses{READY_REFRESH_TASKS};
};
struct alignas(64) ReadyWriter {
    ReadyRow rows[READY_SHAPES];
    ReadyView views[READY_SHAPES];
    uint64_t epoch{0}, refreshes{0}, probes{0}, attempts{0}, remote_blocks{0}, remote_multi_blocks{0};
    uint64_t now_tick{0}, max_age_ticks{0};
    uint32_t probe_clock{0}, probe_next{0};
    ReadyTrace trace;
    void reset() {
        for (auto &row : rows)
            row.reset();
        for (auto &view : views)
            view = ReadyView{};
        epoch = refreshes = probes = attempts = remote_blocks = remote_multi_blocks = 0;
        now_tick = max_age_ticks = 0;
        probe_clock = probe_next = 0;
        trace = ReadyTrace{};
    }
};
inline uint64_t positive(int64_t n) { return n > 0 ? static_cast<uint64_t>(n) : 0; }
inline uint64_t ready_score(
    int side, uint64_t pressure_units, uint64_t capacity, uint64_t units, uint64_t votes, int release_side,
    uint64_t confidence
) {
    uint64_t waves = READY_WAVE * (pressure_units + READY_UNITS_PER_BLOCK * units) / (READY_UNITS_PER_BLOCK * capacity);
    if (waves < READY_WAVE) waves = READY_WAVE;
    uint64_t total = uint64_t(side_vote(votes, 0)) + side_vote(votes, 1);
    uint64_t remote = total ? READY_REMOTE_PRICE * confidence * side_vote(votes, 1 - side) / (256 * total) : 0;
    return waves + remote + (release_side >= 0 && release_side != side ? READY_PUBLISH_PRICE : 0);
}
inline int choose_ready_side(
    int preferred, AffinityStrength strength, int release_side, const uint64_t work[2], const uint64_t capacity[2],
    const uint64_t free[2], uint64_t units, uint64_t votes = 0, uint64_t confidence = 256
) {
    if (!capacity[0] && !capacity[1]) return preferred >= 0 ? preferred : 0;
    if (!capacity[0]) return 1;
    if (!capacity[1]) return 0;
    if (!votes && preferred >= 0)
        votes = preferred ? pack_votes(0, strength == AffinityStrength::STRONG ? 256 : 64) :
                            pack_votes(strength == AffinityStrength::STRONG ? 256 : 64, 0);
    uint64_t scores[2]{};
    for (int c = 0; c < 2; ++c)
        scores[c] = ready_score(c, work[c], capacity[c], units, votes, release_side, confidence);
    int anchor = preferred >= 0 ? preferred : (release_side >= 0 ? release_side : -1);
    if (votes) {
        uint32_t a = side_vote(votes, 0), b = side_vote(votes, 1);
        if (a != b) anchor = a > b ? 0 : 1;
    }
    if (anchor >= 0 && scores[anchor] <= scores[1 - anchor] + READY_SWITCH_MARGIN) return anchor;
    if (scores[0] != scores[1]) return scores[0] < scores[1] ? 0 : 1;
    if (free[0] != free[1]) return free[0] > free[1] ? 0 : 1;
    return 0;
}
inline bool may_remote_pickup(uint64_t local_work, uint64_t remote_work, int remote_capacity) {
    return !local_work && remote_capacity > 0 && remote_work >= uint64_t(remote_capacity);
}

struct ReadyFeedback {
    std::atomic<bool> enabled{false};
    int32_t schedulers[2]{};
    int32_t cluster[READY_THREADS]{-1, -1, -1, -1, -1, -1};
    ReadyHint hints[2][READY_SHAPES];
    ReadyWriter writers[READY_THREADS];
    void reset() {
        enabled.store(false);
        schedulers[0] = schedulers[1] = 0;
        for (auto &c : cluster)
            c = -1;
        for (auto &side : hints)
            for (auto &hint : side)
                hint.reset();
        for (auto &writer : writers)
            writer.reset();
    }
    // Event grouping is diagnostic only; cached feedback survives producer boundaries.
    void begin_batch(int writer) { ++writers[writer].epoch; }
    void refresh(int writer, int shape) {
        auto &owner = writers[writer];
        auto &view = owner.views[shape];
        bool expired = owner.max_age_ticks && owner.now_tick - view.refreshed_tick >= owner.max_age_ticks;
        if (view.uses < READY_REFRESH_TASKS && !expired) return;
        for (int side = 0; side < 2; ++side) {
            const auto &hint = hints[side][shape];
            uint64_t v = hint.version.load(std::memory_order_acquire);
            if (v & 1) continue;
            int64_t work = hint.work.load(std::memory_order_relaxed);
            int64_t queued = hint.queued.load(std::memory_order_relaxed);
            int64_t pcut = hint.pressure_cut[writer].load(std::memory_order_relaxed);
            int64_t qcut = hint.queue_cut[writer].load(std::memory_order_relaxed);
            uint64_t r = hint.resources.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (v != hint.version.load(std::memory_order_relaxed)) continue;
            view.work[side] = work;
            view.queued[side] = queued;
            view.pcut[side] = pcut;
            view.qcut[side] = qcut;
            view.capacity[side] = r & 0xffff;
            view.free[side] = r >> 16;
        }
        view.refreshed_tick = owner.now_tick;
        view.uses = 0;
        ++owner.refreshes;
    }
    uint64_t pressure(int writer, int shape, int side) const {
        const auto &owner = writers[writer];
        const auto &view = owner.views[shape];
        return positive(view.work[side] + owner.rows[shape].pressure_units(side) - view.pcut[side]);
    }
    uint64_t queued(int writer, int shape, int side) const {
        const auto &owner = writers[writer];
        const auto &view = owner.views[shape];
        return positive(view.queued[side] + owner.rows[shape].queue_units(side) - view.qcut[side]);
    }
    int bind(int writer, int shape, const ReadyTaskState &task, uint64_t units) {
        refresh(writer, shape);
        auto &owner = writers[writer];
        auto &view = owner.views[shape];
        ++view.uses;
        uint64_t work[2]{}, cap[2]{};
        for (int side = 0; side < 2; ++side) {
            work[side] = pressure(writer, shape, side);
            cap[side] = view.capacity[side] ? view.capacity[side] : schedulers[side];
        }
        uint64_t votes = task.completed_votes.load(std::memory_order_acquire);
        uint64_t confidence = votes || task.strength != AffinityStrength::WEAK ? 256 : 64;
        if (!votes) votes = task.affinity_votes;
        int release = task.release_cluster >= 0 ? task.release_cluster : cluster[writer];
        return choose_ready_side(
            task.preferred_cluster, task.strength, release, work, cap, view.free, units, votes, confidence
        );
    }
};
}  // namespace simpler::tmr
