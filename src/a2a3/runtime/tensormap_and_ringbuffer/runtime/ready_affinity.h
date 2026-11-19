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
constexpr uint32_t READY_PROBE_LOOPS = 32;
constexpr uint32_t READY_PROBE_COOLDOWN = 128;
// Fixed policy prices; no measured duration, calibration or price learning.
constexpr uint64_t READY_LOCAL_PRICE = 250;
constexpr uint64_t READY_REMOTE_PRICE = 750;
constexpr uint64_t READY_QUEUE_PRICE = 2000;
constexpr uint64_t READY_FUTURE_PRICE = 500;
constexpr bool READY_REMOTE_PICKUP = true;

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
    std::atomic<uint64_t> completed_votes{0};
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

struct ReadySample {
    uint64_t revoked[2]{}, enqueued[2]{}, claimed[2]{};
};
// R and E form one single-writer transaction. C is an independent monotone atomic observation.
struct alignas(64) ReadyRow {
    std::atomic<uint64_t> version{0}, capacity{0};
    std::atomic<uint64_t> revoked[2]{}, enqueued[2]{}, claimed[2]{};
    static void add(std::atomic<uint64_t> &counter, uint64_t n) {
        counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
    }
    void publish(int predicted, int queue, uint64_t units) {
        uint64_t v = version.load(std::memory_order_relaxed);
        version.store(v + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        add(revoked[predicted], units);
        add(enqueued[queue], units);
        version.store(v + 2, std::memory_order_release);
    }
    bool read(ReadySample &out) const {
        uint64_t v = version.load(std::memory_order_acquire);
        if (v & 1) return false;
        ReadySample next;
        for (int c = 0; c < 2; ++c) {
            next.revoked[c] = revoked[c].load(std::memory_order_relaxed);
            next.enqueued[c] = enqueued[c].load(std::memory_order_relaxed);
            next.claimed[c] = claimed[c].load(std::memory_order_relaxed);
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (v != version.load(std::memory_order_relaxed)) return false;
        out = next;
        return true;
    }
    int64_t queue(int c) const {
        return int64_t(enqueued[c].load(std::memory_order_relaxed)) -
               int64_t(claimed[c].load(std::memory_order_relaxed));
    }
    void reset() {
        version.store(0);
        capacity.store(0);
        for (int c = 0; c < 2; ++c) {
            revoked[c].store(0);
            enqueued[c].store(0);
            claimed[c].store(0);
        }
    }
};
static_assert(sizeof(ReadyRow) == 64);

struct ReadyHintSample {
    ReadySample schedulers, orchestrator;
    uint64_t capacity{0};
    int64_t own_cut[2]{};
};
// Each hint owns only its side's S rows. O is a separate sidecar on O's side, never part of S totals.
struct alignas(64) ReadyHint {
    std::atomic<uint64_t> version{0}, capacity{0};
    std::atomic<uint64_t> revoked[2]{}, enqueued[2]{}, claimed[2]{};
    std::atomic<uint64_t> o_revoked[2]{}, o_enqueued[2]{}, o_claimed[2]{};
    std::atomic<int64_t> queue_cut[READY_THREADS - 1][2]{};
    bool read(int writer, ReadyHintSample &out) const {
        uint64_t v = version.load(std::memory_order_acquire);
        if (v & 1) return false;
        ReadyHintSample next;
        next.capacity = capacity.load(std::memory_order_relaxed);
        for (int c = 0; c < 2; ++c) {
            next.schedulers.revoked[c] = revoked[c].load(std::memory_order_relaxed);
            next.schedulers.enqueued[c] = enqueued[c].load(std::memory_order_relaxed);
            next.schedulers.claimed[c] = claimed[c].load(std::memory_order_relaxed);
            next.orchestrator.revoked[c] = o_revoked[c].load(std::memory_order_relaxed);
            next.orchestrator.enqueued[c] = o_enqueued[c].load(std::memory_order_relaxed);
            next.orchestrator.claimed[c] = o_claimed[c].load(std::memory_order_relaxed);
            if (writer < READY_THREADS - 1) next.own_cut[c] = queue_cut[writer][c].load(std::memory_order_relaxed);
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (v != version.load(std::memory_order_relaxed)) return false;
        out = next;
        return true;
    }
    void reset() {
        version.store(0);
        capacity.store(0);
        for (int c = 0; c < 2; ++c) {
            revoked[c].store(0);
            enqueued[c].store(0);
            claimed[c].store(0);
            o_revoked[c].store(0);
            o_enqueued[c].store(0);
            o_claimed[c].store(0);
            for (auto &cut : queue_cut)
                cut[c].store(0);
        }
    }
};
static_assert(sizeof(ReadyHint) == 192);
struct ReadyView {
    ReadyHintSample sides[2];
};
struct alignas(64) ReadyWriter {
    ReadyRow rows[READY_SHAPES];
    ReadyView views[READY_SHAPES];
    ReadySample collected[READY_SHAPES][READY_THREADS];
    uint64_t executed[READY_SHAPES][2]{};
    uint64_t refreshes{0}, probes{0}, attempts{0}, remote_blocks{0}, remote_multi_blocks{0};
    uint32_t probe_clock{0}, probe_next{READY_PROBE_LOOPS};
    ReadyTrace trace;
    void reset() {
        for (auto &row : rows)
            row.reset();
        for (auto &view : views)
            view = ReadyView{};
        for (auto &shape : collected)
            for (auto &sample : shape)
                sample = ReadySample{};
        for (auto &shape : executed)
            for (auto &n : shape)
                n = 0;
        refreshes = probes = attempts = remote_blocks = remote_multi_blocks = 0;
        probe_clock = 0;
        probe_next = READY_PROBE_LOOPS;
        trace = ReadyTrace{};
    }
};
inline uint64_t positive(int64_t n) { return n > 0 ? uint64_t(n) : 0; }
inline int
light_ready_side(int predicted, uint64_t votes, bool actual, const uint64_t queue[2], const int32_t schedulers[2]) {
    if (!schedulers[predicted]) return schedulers[1 - predicted] ? 1 - predicted : predicted;
    int other = 1 - predicted;
    if (!schedulers[other]) return predicted;
    uint64_t a = side_vote(votes, predicted), b = side_vote(votes, other), total = a + b;
    if (!total || b <= a || (b - a) * (actual ? 4 : 2) <= total) return predicted;
    // More than two additional queued blocks per S, with an absolute floor, protects the candidate side.
    bool congested =
        queue[other] >= 4 && queue[other] * schedulers[predicted] >
                                 queue[predicted] * schedulers[other] + 2u * schedulers[predicted] * schedulers[other];
    return congested ? predicted : other;
}
struct ReadyFeedback {
    std::atomic<bool> enabled{false};
    int32_t schedulers[2]{};
    int32_t cluster[READY_THREADS]{-1, -1, -1, -1, -1, -1};
    ReadyHint hints[2][READY_SHAPES];
    ReadyWriter writers[READY_THREADS];
    uint64_t assigned[READY_SHAPES][2]{}, creations{0};
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
        for (auto &shape : assigned)
            for (auto &n : shape)
                n = 0;
        creations = 0;
    }
    void refresh(int writer, int shape) {
        auto &owner = writers[writer];
        for (int side = 0; side < 2; ++side)
            hints[side][shape].read(writer, owner.views[shape].sides[side]);
        ++owner.refreshes;
    }
    void refresh_all(int writer) {
        for (int shape = 0; shape < READY_SHAPES; ++shape)
            refresh(writer, shape);
    }
    void publish_side(int leader, int shape) {
        int side = cluster[leader];
        auto &owner = writers[leader];
        auto &cache = owner.collected[shape];
        ReadySample sum;
        uint64_t cap = 0;
        for (int t = 0; t < READY_THREADS - 1; ++t)
            if (cluster[t] == side) {
                writers[t].rows[shape].read(cache[t]);
                cap += writers[t].rows[shape].capacity.load(std::memory_order_relaxed);
                for (int c = 0; c < 2; ++c) {
                    sum.revoked[c] += cache[t].revoked[c];
                    sum.enqueued[c] += cache[t].enqueued[c];
                    sum.claimed[c] += cache[t].claimed[c];
                }
            }
        ReadySample o;
        if (cluster[READY_THREADS - 1] == side) {
            writers[READY_THREADS - 1].rows[shape].read(cache[READY_THREADS - 1]);
            o = cache[READY_THREADS - 1];
        }
        auto &hint = hints[side][shape];
        uint64_t v = hint.version.load(std::memory_order_relaxed);
        hint.version.store(v + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        hint.capacity.store(cap, std::memory_order_relaxed);
        for (int c = 0; c < 2; ++c) {
            hint.revoked[c].store(sum.revoked[c], std::memory_order_relaxed);
            hint.enqueued[c].store(sum.enqueued[c], std::memory_order_relaxed);
            hint.claimed[c].store(sum.claimed[c], std::memory_order_relaxed);
            hint.o_revoked[c].store(o.revoked[c], std::memory_order_relaxed);
            hint.o_enqueued[c].store(o.enqueued[c], std::memory_order_relaxed);
            hint.o_claimed[c].store(o.claimed[c], std::memory_order_relaxed);
            for (int t = 0; t < READY_THREADS - 1; ++t)
                hint.queue_cut[t][c].store(
                    cluster[t] == side ? int64_t(cache[t].enqueued[c]) - int64_t(cache[t].claimed[c]) : 0,
                    std::memory_order_relaxed
                );
        }
        hint.version.store(v + 2, std::memory_order_release);
    }
    int64_t queued(int writer, int shape, int c) const {
        const auto &owner = writers[writer];
        int64_t q = owner.rows[shape].queue(c);
        for (const auto &side : owner.views[shape].sides) {
            q += int64_t(side.schedulers.enqueued[c]) - int64_t(side.schedulers.claimed[c]);
            if (writer < READY_THREADS - 1)
                q += int64_t(side.orchestrator.enqueued[c]) - int64_t(side.orchestrator.claimed[c]) - side.own_cut[c];
        }
        return q;
    }
    int64_t blocked(int shape, int c) const {
        int64_t w = int64_t(assigned[shape][c]) -
                    int64_t(writers[READY_THREADS - 1].rows[shape].revoked[c].load(std::memory_order_relaxed));
        for (const auto &side : writers[READY_THREADS - 1].views[shape].sides)
            w -= int64_t(side.schedulers.revoked[c]);
        return w;
    }
    int predict(int shape, ReadyTaskState &task, uint64_t units, bool waiting) {
        constexpr int O = READY_THREADS - 1;
        if ((creations++ % READY_REFRESH_TASKS) == 0) refresh_all(O);
        uint64_t votes = task.affinity_votes;
        uint64_t total = uint64_t(side_vote(votes, 0)) + side_vote(votes, 1);
        uint64_t score[2]{};
        for (int c = 0; c < 2; ++c) {
            if (!schedulers[c]) {
                score[c] = ~uint64_t{0};
                continue;
            }
            uint64_t price;
            if (waiting && total)
                price = READY_LOCAL_PRICE + (READY_REMOTE_PRICE - READY_LOCAL_PRICE) * side_vote(votes, 1 - c) / total;
            else price = cluster[O] >= 0 && cluster[O] != c ? READY_REMOTE_PRICE : READY_LOCAL_PRICE;
            uint64_t q = positive(queued(O, shape, c)) + (waiting ? 0 : units);
            uint64_t cap = writers[O].views[shape].sides[c].capacity;
            if (!cap) cap = schedulers[c];
            uint64_t w = positive(blocked(shape, c)) + (waiting ? units : 0);
            if (w > cap) w = cap;
            score[c] = price + READY_QUEUE_PRICE * q / (2 * schedulers[c]) + READY_FUTURE_PRICE * w / cap;
        }
        int chosen = score[0] < score[1] ? 0 : 1;
        if (score[0] == score[1])
            chosen = assigned[shape][0] * schedulers[1] <= assigned[shape][1] * schedulers[0] ? 0 : 1;
        assigned[shape][chosen] += units;
        task.predicted_cluster = chosen;
        return chosen;
    }
    int bind(int writer, int shape, const ReadyTaskState &task, uint64_t units) {
        uint64_t votes = task.completed_votes.load(std::memory_order_acquire);
        bool actual = votes != 0;
        if (!actual) votes = task.affinity_votes;
        uint64_t q[2]{positive(queued(writer, shape, 0)), positive(queued(writer, shape, 1))};
        int side = light_ready_side(task.predicted_cluster, votes, actual, q, schedulers);
        writers[writer].rows[shape].publish(task.predicted_cluster, side, units);
        return side;
    }
};
}  // namespace simpler::tmr
