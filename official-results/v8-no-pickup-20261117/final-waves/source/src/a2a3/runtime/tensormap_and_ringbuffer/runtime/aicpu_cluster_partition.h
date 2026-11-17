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
constexpr uint32_t PARTITION_FROM_O = 2u;
constexpr uint32_t PARTITION_READY = 4u;
constexpr uint64_t PARTITION_BATCH_NODES = 64;
constexpr uint32_t PARTITION_GROUP_LIMIT = 8;
constexpr uint32_t PARTITION_GROUP_SLOTS = 64;
constexpr uint64_t PARTITION_GROUP_LIFETIME = 32;
constexpr uint64_t PARTITION_WAVE_SCALE = 1024;
constexpr uint64_t PARTITION_AFFINITY_WAVES = 1;

struct AdaptiveTaskState {
    uint32_t scope{0}, group{0};
    uint64_t submit_tick{0}, release_tick{0};
    uint32_t reserved_depth{0};
    int8_t release_cluster{-1};
    std::atomic<int32_t> dispatch_cluster{-1};
    uint8_t reserved{0};
    // O owns group identity and preference; queue target is separate in ChipTask.
    int8_t preferred_cluster{-1};
    int8_t ready_writer{PARTITION_THREADS - 1};
    void reset() {
        scope = submit_tick = release_tick = 0;
        reserved_depth = reserved = 0;
        release_cluster = preferred_cluster = -1;
        group = 0;
        ready_writer = PARTITION_THREADS - 1;
        dispatch_cluster.store(-1, std::memory_order_relaxed);
    }
};
static_assert(sizeof(AdaptiveTaskState) == 40);

struct PartitionTrace {
    uint64_t ticks[7]{}, calls[7]{};
};
// One publisher owns each row, including the dedicated O row. Monotone
// counters use load/store, with no cross-S fetch_add or cache-line sharing.
struct PartitionSummarySample {
    uint64_t released[2]{}, queued[2]{}, taken[2]{};
};
struct alignas(64) PartitionSummary {
    std::atomic<uint64_t> version{0};
    std::atomic<uint64_t> resources{0};  // capacity/free/idle: three 8-bit fields
    std::atomic<uint64_t> released[2]{}, queued[2]{}, taken[2]{};
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
        }
    }
};
static_assert(sizeof(PartitionSummary) == 64);
struct alignas(64) PartitionThread {
    std::atomic<int32_t> cluster{-1};
    PartitionSummary summary[PARTITION_SHAPES];
    std::atomic<uint64_t> started[PARTITION_SHAPES]{};
    alignas(64) PartitionTrace trace;
    void reset() {
        trace = PartitionTrace{};
        cluster.store(-1, std::memory_order_relaxed);
        for (int i = 0; i < PARTITION_SHAPES; ++i) {
            summary[i].reset();
            started[i].store(0, std::memory_order_relaxed);
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
    int32_t schedulers[2]{}, orchestrator_cluster{-1};
    // Only the lowest-index S on each side writes its queue hint.
    // Low 32 bits: entries; high bits: that leader has idle resource.
    PartitionReadyCounter side_queue[2][PARTITION_SHAPES]{};
    PartitionThread threads[PARTITION_THREADS];
    void reset() {
        enabled.store(false, std::memory_order_relaxed);
        schedulers[0] = schedulers[1] = 0;
        orchestrator_cluster = -1;
        for (auto &cluster : side_queue)
            for (auto &hint : cluster)
                hint.store(0, std::memory_order_relaxed);
        for (auto &thread : threads)
            thread.reset();
    }
};
struct PartitionLoad {
    uint64_t blocked[2]{}, ready[2]{};
    uint32_t capacity[2]{}, free[2]{}, idle[2]{}, schedulers[2]{};
};
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
struct PartitionDecision {
    int32_t cluster{-1};
    uint64_t pressure[2]{};  // fixed-point capacity waves, never nanoseconds
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
        ++decisions[cluster];
        return out;
    }

    PartitionTrace trace;
    uint64_t assigned[2][PARTITION_SHAPES]{};
    uint64_t next_scope{0}, last_closed{0}, scope_ids[PARTITION_SCOPE_DEPTH]{},
        scope_assignments[PARTITION_SCOPE_DEPTH]{};
    uint64_t decisions[2]{}, snapshots{0};
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
    bool needs_snapshot(int shape) const {
        return load_batch[shape] != decision_count() / PARTITION_BATCH_NODES;
    }
    // Each shape uses one advisory snapshot per 64-decision epoch. Subsequent
    // submissions add local reservations; scheduler progress is incorporated at
    // the next epoch. The snapshot does not constitute a resource reservation.
    PartitionLoad load(const PartitionShared &shared, int shape) {
        if (needs_snapshot(shape)) {
            auto &cached = cached_load[shape];
            cached = PartitionLoad{};
            uint64_t released[2]{}, queued[2]{}, taken[2]{};
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
                uint64_t waiting = queued[c] > taken[c] ? queued[c] - taken[c] : 0;
                uint64_t depth = shared.side_queue[c][shape].load(std::memory_order_relaxed) & 0xffffffffu;
                cached.ready[c] = waiting > depth ? waiting : depth;
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
    PartitionDecision choose(
        const PartitionLoad &load, uint32_t count, const uint32_t sources[2], int o_cluster,
        uint32_t units = 1
    ) {
        PartitionDecision out;
        // Every numerator term is a block-equivalent workload. One capacity
        // wave is the maximum extra pressure accepted for source affinity.
        for (int c = 0; c < 2; ++c) {
            if (!load.schedulers[c]) continue;
            uint64_t cap = load.capacity[c] ? load.capacity[c] : load.schedulers[c];
            uint64_t future = load.blocked[c] < cap ? load.blocked[c] : cap;
            uint64_t available = load.free[c] < cap ? load.free[c] : cap;
            uint64_t unavailable = load.capacity[c] ? cap - available : 0;
            out.pressure[c] = PARTITION_WAVE_SCALE * (load.ready[c] + future + unavailable + units) / cap;
            if (out.cluster < 0 || out.pressure[c] < out.pressure[out.cluster] ||
                (out.pressure[c] == out.pressure[out.cluster] &&
                 decisions[c] * load.schedulers[out.cluster] < decisions[out.cluster] * load.schedulers[c]))
                out.cluster = c;
        }
        int affinity = -1;
        if (count && sources[0] != sources[1]) affinity = sources[1] > sources[0] ? 1 : 0;
        else if (!count && o_cluster >= 0 && o_cluster < 2) affinity = o_cluster;
        if (out.cluster >= 0 && affinity >= 0 && load.schedulers[affinity] &&
            out.pressure[affinity] <= out.pressure[out.cluster] + PARTITION_WAVE_SCALE * PARTITION_AFFINITY_WAVES) out.cluster = affinity;
        if (out.cluster >= 0) {
            ++decisions[out.cluster];
        }
        return out;
    }
};
}  // namespace simpler::tmr
