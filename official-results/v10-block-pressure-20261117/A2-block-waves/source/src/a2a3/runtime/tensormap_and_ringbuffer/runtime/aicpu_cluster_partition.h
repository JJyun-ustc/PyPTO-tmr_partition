/*
 * Copyright (c) PyPTO Contributors.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include <atomic>
#include <cstdint>

namespace simpler::tmr {
constexpr int PARTITION_THREADS = 6;
constexpr int PARTITION_SHAPES = 4;
constexpr int PARTITION_SCOPE_DEPTH = 64;
constexpr int PARTITION_METRICS = 6;  // queued wait, O0/O1 publication, S0/S1 notification, idle pickup
constexpr uint32_t PARTITION_FROM_O = 2u;
constexpr uint32_t PARTITION_READY = 4u;
constexpr uint32_t PARTITION_OBSERVED = 8u;
constexpr uint32_t PARTITION_IDLE_AT_PUBLISH = 16u;
constexpr uint64_t PARTITION_CALIBRATION_NODES = 256;
constexpr uint64_t PARTITION_BATCH_NODES = 64;
constexpr uint32_t PARTITION_GROUP_LIMIT = 8;
constexpr uint32_t PARTITION_GROUP_SLOTS = 64;
constexpr uint64_t PARTITION_GROUP_LIFETIME = 32;
constexpr uint32_t PARTITION_STEAL_CHECK_LOOPS = 32;
constexpr uint32_t PARTITION_STEAL_COOLDOWN_LOOPS = 128;
constexpr uint64_t PARTITION_QUEUE_WEIGHT_NUM = 1;
constexpr uint64_t PARTITION_QUEUE_WEIGHT_DEN = 2;
constexpr uint64_t PARTITION_RESOURCE_WEIGHT_NUM = 0;
constexpr uint64_t PARTITION_RESOURCE_WEIGHT_DEN = 4;
constexpr uint64_t PARTITION_EXTRA_REMOTE_WAIT_NS = 0;
constexpr uint64_t PARTITION_FUTURE_WEIGHT_DEN = 4;

struct AdaptiveTaskState {
    uint32_t scope{0}, group{0};
    uint64_t submit_tick{0}, release_tick{0};
    uint32_t ready_depth{0};
    int8_t release_cluster{-1};
    std::atomic<int32_t> dispatch_cluster{-1};
    uint8_t reserved{0};
    // O owns group identity and preference; queue target is separate in ChipTask.
    int8_t preferred_cluster{-1};
    int8_t ready_writer{PARTITION_THREADS - 1};
    void reset() {
        scope = submit_tick = release_tick = 0;
        ready_depth = reserved = 0;
        release_cluster = preferred_cluster = -1;
        group = 0;
        ready_writer = PARTITION_THREADS - 1;
        dispatch_cluster.store(-1, std::memory_order_relaxed);
    }
};
static_assert(sizeof(AdaptiveTaskState) == 40);

struct PartitionSample {
    uint64_t observed{0};
    uint64_t count[PARTITION_METRICS]{}, ns[PARTITION_METRICS]{};
};

// One S writes each cumulative bucket. Scope changes cannot evict late samples.
struct alignas(64) PartitionBucket {
    std::atomic<uint64_t> version{0}, observed{0};
    std::atomic<uint64_t> count[PARTITION_METRICS]{}, ns[PARTITION_METRICS]{};
    void reset() {
        version.store(0, std::memory_order_relaxed);
        observed.store(0, std::memory_order_relaxed);
        for (int i = 0; i < PARTITION_METRICS; ++i) {
            count[i].store(0, std::memory_order_relaxed);
            ns[i].store(0, std::memory_order_relaxed);
        }
    }
    void add(int metric, uint64_t value, bool first = true) {
        version.fetch_add(1, std::memory_order_acq_rel);
        if (first) observed.fetch_add(1, std::memory_order_relaxed);
        if (metric >= 0 && metric < PARTITION_METRICS) {
            count[metric].fetch_add(1, std::memory_order_relaxed);
            ns[metric].fetch_add(value, std::memory_order_relaxed);
        }
        version.fetch_add(1, std::memory_order_release);
    }
    bool read(PartitionSample &out) const {
        uint64_t begin = version.load(std::memory_order_acquire);
        if (begin & 1) return false;
        out.observed = observed.load(std::memory_order_relaxed);
        for (int i = 0; i < PARTITION_METRICS; ++i) {
            out.count[i] = count[i].load(std::memory_order_relaxed);
            out.ns[i] = ns[i].load(std::memory_order_relaxed);
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        return begin == version.load(std::memory_order_acquire);
    }
};
struct PartitionTrace {
    uint64_t ticks[8]{}, calls[8]{};
};
// One publisher owns each row, including the dedicated O row. Monotone
// counters use load/store, with no cross-S fetch_add or cache-line sharing.
struct PartitionSummarySample {
    uint64_t released[2]{}, queued[2]{}, taken[2]{};
    uint64_t claimed[2]{}, retired[2]{};
};
struct alignas(64) PartitionSummary {
    std::atomic<uint64_t> version{0};
    std::atomic<uint64_t> resources{0};  // capacity/free/idle: three 8-bit fields
    std::atomic<uint64_t> released[2]{}, queued[2]{}, taken[2]{};
    alignas(64) std::atomic<uint64_t> claimed[2]{};
    std::atomic<uint64_t> retired[2]{};
    static void add(std::atomic<uint64_t> &counter, uint64_t units) {
        counter.store(counter.load(std::memory_order_relaxed) + units, std::memory_order_relaxed);
    }
    void begin_write() { version.fetch_add(1, std::memory_order_acq_rel); }
    void end_write() { version.fetch_add(1, std::memory_order_release); }
    bool read(PartitionSummarySample &out) const {
        uint64_t before = version.load(std::memory_order_acquire);
        if (before & 1) return false;
        PartitionSummarySample sample;
        for (int c = 0; c < 2; ++c) {
            sample.released[c] = released[c].load(std::memory_order_relaxed);
            sample.queued[c] = queued[c].load(std::memory_order_relaxed);
            sample.taken[c] = taken[c].load(std::memory_order_relaxed);
            sample.claimed[c] = claimed[c].load(std::memory_order_relaxed);
            sample.retired[c] = retired[c].load(std::memory_order_relaxed);
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (before != version.load(std::memory_order_relaxed)) return false;
        out = sample;
        return true;
    }
    void reset() {
        version.store(0, std::memory_order_relaxed);
        resources.store(0, std::memory_order_relaxed);
        for (int c = 0; c < 2; ++c) {
            released[c].store(0, std::memory_order_relaxed);
            queued[c].store(0, std::memory_order_relaxed);
            taken[c].store(0, std::memory_order_relaxed);
            claimed[c].store(0, std::memory_order_relaxed);
            retired[c].store(0, std::memory_order_relaxed);
        }
    }
};
static_assert(sizeof(PartitionSummary) == 128);
struct alignas(64) PartitionThread {
    std::atomic<int32_t> cluster{-1};
    PartitionSummary summary[PARTITION_SHAPES];
    std::atomic<uint64_t> started[PARTITION_SHAPES]{};
    PartitionBucket samples[PARTITION_SHAPES];
    uint64_t steal_checks{0}, steal_attempts{0}, remote_blocks{0};
    alignas(64) PartitionTrace trace;
    void reset() {
        trace = PartitionTrace{};
        steal_checks = steal_attempts = remote_blocks = 0;
        cluster.store(-1, std::memory_order_relaxed);
        for (int i = 0; i < PARTITION_SHAPES; ++i) {
            summary[i].reset();
            started[i].store(0, std::memory_order_relaxed);
            samples[i].reset();
        }
    }
};
// Side queue hints occupy independent cache lines and have one designated writer.
struct alignas(64) PartitionReadyCounter {
    std::atomic<uint64_t> value{0};
    uint64_t load(std::memory_order order = std::memory_order_seq_cst) const { return value.load(order); }
    void store(uint64_t n, std::memory_order order = std::memory_order_seq_cst) { value.store(n, order); }
};
static_assert(sizeof(PartitionReadyCounter) == 64);
struct PartitionShared {
    std::atomic<bool> enabled{false};
    std::atomic<bool> calibration_done{false};
    int32_t schedulers[2]{}, orchestrator_cluster{-1};
    // Only the lowest-index S on each side writes its queue hint.
    // Low 32 bits: entries; high bits: that leader has idle resource.
    PartitionReadyCounter side_queue[2][PARTITION_SHAPES]{};
    PartitionReadyCounter side_work[2][PARTITION_SHAPES]{};
    PartitionReadyCounter side_capacity[2][PARTITION_SHAPES]{};
    PartitionThread threads[PARTITION_THREADS];
    void reset() {
        enabled.store(false, std::memory_order_relaxed);
        calibration_done.store(false, std::memory_order_relaxed);
        schedulers[0] = schedulers[1] = 0;
        orchestrator_cluster = -1;
        for (auto &cluster : side_queue)
            for (auto &hint : cluster)
                hint.store(0, std::memory_order_relaxed);
        for (auto &thread : threads)
            thread.reset();
        for (int c = 0; c < 2; ++c)
            for (int s = 0; s < PARTITION_SHAPES; ++s) {
                side_work[c][s].store(0, std::memory_order_relaxed);
                side_capacity[c][s].store(0, std::memory_order_relaxed);
            }
    }
};
struct PartitionLoad {
    uint64_t blocked[2]{}, ready[2]{}, inflight[2]{};
    uint32_t capacity[2]{}, free[2]{}, idle[2]{}, schedulers[2]{};
};
// Work is in logical-block equivalents; one capacity wave is 256 units.
inline uint64_t partition_future_wave(uint64_t waiting, uint64_t capacity) {
    return 64 * (waiting < capacity ? waiting : capacity) / capacity;
}
inline uint64_t partition_wave(uint64_t ready, uint64_t inflight, uint64_t waiting, uint64_t units, uint64_t capacity) {
    if (!capacity) return UINT64_MAX / 4;
    return 256 * (ready + inflight + units) / capacity + partition_future_wave(waiting, capacity);
}
inline int partition_block_ready_side(
    int preferred, int source, const uint64_t work[2], const uint64_t capacity[2], uint64_t units
) {
    int c = source >= 0 && source < 2 ? source : preferred;
    if (c < 0 || c > 1) c = capacity[0] ? 0 : 1;
    if (!capacity[c]) return capacity[1 - c] ? 1 - c : c;
    int other = 1 - c;
    if (capacity[other] && partition_wave(work[c], 0, 0, units, capacity[c]) >
                               partition_wave(work[other], 0, 0, units, capacity[other]) + 64)
        return other;
    return c;
}
inline int partition_ready_side(int preferred, int source, const uint64_t depth[2], const int32_t schedulers[2]) {
    int c = source >= 0 && source < 2 ? source : preferred;
    if (c < 0 || c > 1) c = schedulers[0] ? 0 : 1;
    if (!schedulers[c]) return schedulers[1 - c] ? 1 - c : c;
    int other = 1 - c;
    if (schedulers[other] && depth[c] >= 4 &&
        depth[c] * schedulers[other] > depth[other] * schedulers[c] + 2u * schedulers[c] * schedulers[other])
        return other;
    return c;
}
// Queue depth is advisory. The actual pop and range claim remain atomic.
inline bool partition_may_steal(uint64_t local, uint64_t remote, int local_s, int remote_s) {
    return local == 0 && local_s > 0 && remote_s > 0 && remote >= static_cast<uint64_t>(2 * remote_s + 2);
}
struct PartitionDecision {
    int32_t cluster{-1};
    uint64_t score_ns[2]{};
    bool learned{false};
};
struct PartitionWeights {
    uint64_t queue[2]{2000, 2000};
    uint64_t publish[2][2]{{250, 750}, {750, 250}};
    uint64_t dependency[2][2]{{250, 750}, {750, 250}};
    uint64_t idle_pickup[2]{};
};
// Bounded O-local table: no shared member counter and no scope-wide grouping.
struct PartitionGroup {
    uint64_t id{0}, scope{0}, born{0};
    uint32_t members{0};
};
struct ClusterPartitioner {
    PartitionSummarySample summary_cache[PARTITION_THREADS][PARTITION_SHAPES]{};
    PartitionGroup groups[PARTITION_GROUP_SLOTS]{};
    uint64_t next_group{0}, inherited_decisions{0}, full_decisions{0};
    bool can_inherit(const AdaptiveTaskState &producer, uint64_t scope) const {
        if (!producer.group) return false;
        const auto &g = groups[producer.group % PARTITION_GROUP_SLOTS];
        return g.id == producer.group && g.scope == scope && g.members < PARTITION_GROUP_LIMIT &&
               decision_count() - g.born < PARTITION_GROUP_LIFETIME;
    }
    void join_group(AdaptiveTaskState &task, uint64_t parent, uint64_t scope) {
        if (!parent) {
            parent = ++next_group;
            groups[parent % PARTITION_GROUP_SLOTS] = {parent, scope, decision_count(), 0};
            ++full_decisions;
        } else {
            ++inherited_decisions;
        }
        ++groups[parent % PARTITION_GROUP_SLOTS].members;
        task.group = parent;
    }
    PartitionDecision inherit(int cluster) {
        PartitionDecision out;
        out.cluster = cluster;
        out.learned = decision_count() >= PARTITION_CALIBRATION_NODES;
        ++decisions[cluster];
        if (out.learned) ++learned_decisions;
        else ++cold_decisions;
        return out;
    }

    PartitionTrace trace;
    PartitionWeights weights[PARTITION_SHAPES];
    uint64_t assigned[2][PARTITION_SHAPES]{};
    uint64_t next_scope{0}, last_closed{0}, scope_ids[PARTITION_SCOPE_DEPTH]{},
        scope_assignments[PARTITION_SCOPE_DEPTH]{};
    uint64_t observed[PARTITION_SHAPES][2]{};
    uint64_t decisions[2]{}, updates[2]{}, channel_updates[PARTITION_SHAPES][2][PARTITION_METRICS]{};
    uint64_t idle_choices{0}, cold_decisions{0}, learned_decisions{0}, calibrations{0}, snapshots{0};
    uint64_t decision_ticks{0}, max_decision_ticks{0};
    uint64_t load_batch[PARTITION_SHAPES]{~uint64_t{0}, ~uint64_t{0}, ~uint64_t{0}, ~uint64_t{0}};
    PartitionLoad cached_load[PARTITION_SHAPES]{};
    void reset() { *this = ClusterPartitioner{}; }
    uint64_t open_scope(int depth) {
        scope_ids[depth] = ++next_scope;
        scope_assignments[depth] = 0;
        return next_scope;
    }
    void close_scope(int depth) {
        if (scope_assignments[depth]) last_closed = scope_ids[depth];
    }
    uint64_t decision_count() const { return decisions[0] + decisions[1]; }
    // Calibration takes one coherent sample snapshot after a fixed submission
    // prefix. Busy buckets and sparse channels retain priors; late samples do
    // not extend calibration or delay the transition to scored placement.
    void calibrate_once(PartitionShared &shared) {
        if (calibrations || decision_count() < PARTITION_CALIBRATION_NODES) return;
        shared.calibration_done.store(true, std::memory_order_release);
        ++calibrations;
        PartitionSample totals[PARTITION_SHAPES][2]{};
        for (const auto &thread : shared.threads) {
            int c = thread.cluster.load(std::memory_order_relaxed);
            if (c < 0 || c > 1) continue;
            for (int shape = 0; shape < PARTITION_SHAPES; ++shape) {
                PartitionSample sample;
                if (!thread.samples[shape].read(sample)) continue;
                auto &total = totals[shape][c];
                total.observed += sample.observed;
                for (int m = 0; m < PARTITION_METRICS; ++m) {
                    total.count[m] += sample.count[m];
                    total.ns[m] += sample.ns[m];
                }
            }
        }
        for (int shape = 0; shape < PARTITION_SHAPES; ++shape) {
            for (int c = 0; c < 2; ++c) {
                const auto &sample = totals[shape][c];
                observed[shape][c] = sample.observed;
                auto &w = weights[shape];
                // Fit the receiver baseline before subtracting it from queue wait.
                for (int k = 0; k < PARTITION_METRICS; ++k) {
                    int m = k == 0 ? 5 : k - 1;
                    if (sample.count[m] < (m == 0 ? 32u : 16u)) continue;
                    uint64_t value = sample.ns[m] / sample.count[m];
                    if (m == 5) {
                        w.idle_pickup[c] = value;
                    } else {
                        uint64_t &coefficient =
                            m == 0 ? w.queue[c] : (m <= 2 ? w.publish[m - 1][c] : w.dependency[m - 3][c]);
                        if (m == 0) value = value > w.idle_pickup[c] ? value - w.idle_pickup[c] : 20;
                        if (value < 20) value = 20;
                        if (value > 32000) value = 32000;
                        if (value < coefficient / 2) value = coefficient / 2;
                        if (value > coefficient * 2) value = coefficient * 2;
                        coefficient = value;
                    }
                    ++channel_updates[shape][c][m];
                    ++updates[c];
                }
            }
        }
    }
    bool needs_snapshot(int shape) const { return load_batch[shape] != decision_count() / PARTITION_BATCH_NODES; }
    // Each shape uses one advisory snapshot per 64-decision epoch. Subsequent
    // submissions add local reservations; scheduler progress is incorporated at
    // the next epoch. The snapshot does not constitute a resource reservation.
    PartitionLoad load(const PartitionShared &shared, int shape) {
        if (needs_snapshot(shape)) {
            auto &cached = cached_load[shape];
            cached = PartitionLoad{};
            uint64_t released[2]{}, queued[2]{}, taken[2]{}, claimed[2]{}, retired[2]{};
            for (int t = 0; t < PARTITION_THREADS; ++t) {
                const auto &thread = shared.threads[t];
                const auto &row = thread.summary[shape];
                auto &sample = summary_cache[t][shape];
                // A racing row retains its last coherent totals; readers never spin.
                row.read(sample);
                for (int side = 0; side < 2; ++side) {
                    released[side] += sample.released[side];
                    queued[side] += sample.queued[side];
                    taken[side] += sample.taken[side];
                    claimed[side] += sample.claimed[side];
                    retired[side] += sample.retired[side];
                }
                int c = thread.cluster.load(std::memory_order_relaxed);
                if (c < 0 || c > 1) continue;
                uint64_t packed = row.resources.load(std::memory_order_relaxed);
                cached.capacity[c] += packed & 255u;
                cached.free[c] += (packed >> 8) & 255u;
                cached.idle[c] += (packed >> 16) & 255u;
            }
            for (int c = 0; c < 2; ++c) {
                cached.schedulers[c] = shared.schedulers[c];
                cached.blocked[c] = assigned[c][shape] > released[c] ? assigned[c][shape] - released[c] : 0;
                cached.ready[c] = queued[c] > claimed[c] ? queued[c] - claimed[c] : 0;
                cached.inflight[c] = claimed[c] > retired[c] ? claimed[c] - retired[c] : 0;
            }
            load_batch[shape] = decision_count() / PARTITION_BATCH_NODES;
            ++snapshots;
        }
        return cached_load[shape];
    }
    void commit(int cluster, int shape, uint32_t units) {
        assigned[cluster][shape] += units;
        auto &load = cached_load[shape];
        // New O assignments remain local future reservations until refresh.
        // Refresh rebuilds from totals; it never adds the old local delta again.
        load.blocked[cluster] += units;
    }
    // Future reservations are block units, capped at one capacity wave per side.
    // They do not consume free/idle slots. Pending new tasks include their own
    // blocks before assignment; ready publication removes existing preferences
    // through assigned - released, independently of dispatch snapshots.
    static uint64_t
    future_reservation_cost(const PartitionLoad &load, int c, uint32_t pending_units, uint64_t queue_ns) {
        uint64_t budget = load.capacity[c] ? load.capacity[c] : load.schedulers[c];
        if (!budget) return 0;
        uint64_t reserved = load.blocked[c] < budget ? load.blocked[c] : budget;
        uint64_t room = budget - reserved;
        reserved += pending_units < room ? pending_units : room;
        return queue_ns * reserved / budget / PARTITION_FUTURE_WEIGHT_DEN;
    }
    PartitionDecision choose(
        const PartitionLoad &load, uint32_t count, const uint32_t sources[2], int shape, int o_cluster,
        uint32_t units = 1
    ) {
        PartitionDecision out;
        out.learned = decision_count() >= PARTITION_CALIBRATION_NODES;
        auto &w = weights[shape];
        uint64_t handoff[2]{};
        for (int c = 0; c < 2; ++c) {
            if (!load.schedulers[c]) continue;
            if (count) {
                for (int source = 0; source < 2; ++source)
                    if ((sources[source] || sources[0] + sources[1] < count) && w.dependency[source][c] > handoff[c])
                        handoff[c] = w.dependency[source][c];
            } else {
                int source = o_cluster >= 0 && o_cluster < 2 ? o_cluster : c;
                handoff[c] = w.publish[source][c];
            }
            uint64_t cap = load.capacity[c] ? load.capacity[c] : load.schedulers[c];
            out.score_ns[c] = partition_wave(load.ready[c], load.inflight[c], load.blocked[c], units, cap);
            if (out.cluster < 0) {
                out.cluster = c;
                continue;
            }
            int previous = out.cluster;
            uint64_t previous_cap = load.capacity[previous] ? load.capacity[previous] : load.schedulers[previous];
            bool much_lighter = out.score_ns[c] + 64 < out.score_ns[previous];
            bool close =
                out.score_ns[c] <= out.score_ns[previous] + 64 && out.score_ns[previous] <= out.score_ns[c] + 64;
            bool tie =
                handoff[c] == handoff[previous] && assigned[c][shape] * previous_cap < assigned[previous][shape] * cap;
            if (much_lighter || (close && (handoff[c] < handoff[previous] || tie))) out.cluster = c;
        }
        if (out.cluster >= 0) {
            ++decisions[out.cluster];
            if (out.learned) ++learned_decisions;
            else ++cold_decisions;
            if (load.idle[out.cluster]) ++idle_choices;
        }
        return out;
    }
};
}  // namespace simpler::tmr
