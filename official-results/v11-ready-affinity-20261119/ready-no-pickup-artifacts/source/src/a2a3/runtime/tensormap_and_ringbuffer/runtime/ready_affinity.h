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
constexpr bool READY_REMOTE_PICKUP = false;

enum class AffinityStrength : uint8_t { NEUTRAL, WEAK, STRONG };
struct ReadyTaskState {
    uint32_t scope{0};
    int8_t preferred_cluster{-1};
    AffinityStrength strength{AffinityStrength::NEUTRAL};
    uint8_t hops{0};
    bool bound{false};
    int8_t ready_writer{READY_THREADS - 1};
    int8_t release_cluster{-1};
    std::atomic<int32_t> dispatch_cluster{-1};
    void reset() {
        scope = 0;
        preferred_cluster = release_cluster = -1;
        strength = AffinityStrength::NEUTRAL;
        hops = 0;
        bound = false;
        ready_writer = READY_THREADS - 1;
        dispatch_cluster.store(-1, std::memory_order_relaxed);
    }
};

struct AffinityVote {
    uint32_t known[2]{}, unknown{0};
    uint8_t hops{0};
    bool inferred{false};
    void add(const ReadyTaskState &parent, int strict_side, uint32_t scope) {
        int side = parent.dispatch_cluster.load(std::memory_order_acquire);
        if (side < 0 && !parent.scope) side = strict_side;
        if (side < 0) {
            if (parent.scope == scope && parent.hops < 7 && parent.preferred_cluster >= 0) {
                side = parent.preferred_cluster;
                inferred = true;
                if (parent.hops + 1 > hops) hops = parent.hops + 1;
            }
        }
        if (side >= 0 && side < 2) ++known[side];
        else ++unknown;
    }
    void finish(ReadyTaskState &task, uint32_t scope) const {
        task.scope = scope;
        task.hops = hops;
        if (!!known[0] == !!known[1]) return;
        task.preferred_cluster = known[0] ? 0 : 1;
        task.strength = unknown || inferred ? AffinityStrength::WEAK : AffinityStrength::STRONG;
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

// Each row has exactly one writer. Counters are logical block counts by queue side.
struct alignas(64) ReadyRow {
    std::atomic<uint64_t> enqueued[2]{}, claimed[2]{}, executed[2]{};
    std::atomic<uint64_t> resources{0};
    static void add(std::atomic<uint64_t> &counter, uint64_t n) {
        counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
    }
    void reset() {
        for (int c = 0; c < 2; ++c) {
            enqueued[c].store(0);
            claimed[c].store(0);
            executed[c].store(0);
        }
        resources.store(0);
    }
};
static_assert(sizeof(ReadyRow) == 64);

// One side leader publishes the snapshot and the inclusion cut for every writer.
struct alignas(128) ReadyHint {
    std::atomic<uint64_t> version{0}, work{0}, resources{0};
    std::atomic<uint64_t> enqueued_cut[READY_THREADS]{}, claimed_cut[READY_THREADS]{};
    void reset() {
        version.store(0);
        work.store(0);
        resources.store(0);
        for (int t = 0; t < READY_THREADS; ++t) {
            enqueued_cut[t].store(0);
            claimed_cut[t].store(0);
        }
    }
};
static_assert(sizeof(ReadyHint) == 128);
struct ReadyView {
    uint64_t work[2]{}, qcut[2]{}, ccut[2]{}, capacity[2]{}, free[2]{};
    uint64_t epoch{~uint64_t{0}};
    uint32_t uses{READY_REFRESH_TASKS};
};
struct alignas(64) ReadyWriter {
    ReadyRow rows[READY_SHAPES];
    ReadyView views[READY_SHAPES];
    uint64_t epoch{0}, refreshes{0}, probes{0}, attempts{0}, remote_blocks{0};
    uint32_t probe_clock{0}, probe_next{0};
    ReadyTrace trace;
    void reset() {
        for (auto &row : rows)
            row.reset();
        for (auto &view : views)
            view = ReadyView{};
        epoch = refreshes = probes = attempts = remote_blocks = 0;
        probe_clock = probe_next = 0;
        trace = ReadyTrace{};
    }
};

inline int choose_ready_side(
    int preferred, AffinityStrength strength, int release_side, const uint64_t work[2], const uint64_t capacity[2],
    const uint64_t free[2], uint64_t units
) {
    if (!capacity[0] && !capacity[1]) return preferred >= 0 ? preferred : 0;
    if (!capacity[0]) return 1;
    if (!capacity[1]) return 0;
    uint64_t p[2]{256 * (work[0] + units) / capacity[0], 256 * (work[1] + units) / capacity[1]};
    if (p[0] + 64 < p[1]) return 0;
    if (p[1] + 64 < p[0]) return 1;
    if (strength == AffinityStrength::STRONG && preferred >= 0) return preferred;
    if (!!free[0] != !!free[1]) return free[0] ? 0 : 1;
    if (preferred >= 0) return preferred;
    if (release_side >= 0) return release_side;
    return p[0] <= p[1] ? 0 : 1;
}
inline bool may_remote_pickup(uint64_t local_entries, uint64_t remote_entries, int remote_schedulers) {
    return !local_entries && remote_schedulers > 0 && remote_entries >= uint64_t(2 * remote_schedulers + 2);
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
    void begin_batch(int writer) { ++writers[writer].epoch; }
    int bind(int writer, int shape, const ReadyTaskState &task, uint64_t units) {
        auto &owner = writers[writer];
        auto &view = owner.views[shape];
        if (view.epoch != owner.epoch || view.uses >= READY_REFRESH_TASKS) {
            for (int side = 0; side < 2; ++side) {
                const auto &hint = hints[side][shape];
                uint64_t v = hint.version.load(std::memory_order_acquire);
                if (v & 1) continue;
                uint64_t work = hint.work.load(std::memory_order_relaxed);
                uint64_t q = hint.enqueued_cut[writer].load(std::memory_order_relaxed);
                uint64_t c = hint.claimed_cut[writer].load(std::memory_order_relaxed);
                uint64_t r = hint.resources.load(std::memory_order_relaxed);
                std::atomic_thread_fence(std::memory_order_acquire);
                if (v != hint.version.load(std::memory_order_relaxed)) continue;
                view.work[side] = work;
                view.qcut[side] = q;
                view.ccut[side] = c;
                view.capacity[side] = r & 0xffff;
                view.free[side] = r >> 16;
            }
            view.epoch = owner.epoch;
            view.uses = 0;
            ++owner.refreshes;
        }
        ++view.uses;
        uint64_t work[2]{}, cap[2]{};
        const auto &own = owner.rows[shape];
        for (int side = 0; side < 2; ++side) {
            uint64_t q = own.enqueued[side].load(std::memory_order_relaxed);
            uint64_t c = own.claimed[side].load(std::memory_order_relaxed);
            uint64_t total = view.work[side] + (q >= view.qcut[side] ? q - view.qcut[side] : 0);
            uint64_t removed = c >= view.ccut[side] ? c - view.ccut[side] : 0;
            work[side] = total > removed ? total - removed : 0;
            cap[side] = view.capacity[side] ? view.capacity[side] : schedulers[side];
        }
        return choose_ready_side(
            task.preferred_cluster, task.strength, task.release_cluster, work, cap, view.free, units
        );
    }
};
}  // namespace simpler::tmr
