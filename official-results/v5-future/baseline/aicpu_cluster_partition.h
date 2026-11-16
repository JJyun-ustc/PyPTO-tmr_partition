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
constexpr uint32_t PARTITION_MOVED = 1u;
constexpr uint32_t PARTITION_FROM_O = 2u;
constexpr uint32_t PARTITION_READY = 4u;
constexpr uint32_t PARTITION_OBSERVED = 8u;
constexpr uint32_t PARTITION_IDLE_AT_PUBLISH = 16u;
constexpr uint64_t PARTITION_POLL_NODES = 64;
constexpr uint64_t PARTITION_UPDATE_NODES = 256;
constexpr uint64_t PARTITION_UPDATE_NS = 100000;
constexpr uint64_t PARTITION_WINDOW_SAMPLES = 512;
constexpr uint64_t PARTITION_QUEUE_WEIGHT_NUM = 1;
constexpr uint64_t PARTITION_QUEUE_WEIGHT_DEN = 2;
constexpr uint64_t PARTITION_RESOURCE_WEIGHT_NUM = 1;
constexpr uint64_t PARTITION_RESOURCE_WEIGHT_DEN = 4;
constexpr uint64_t PARTITION_EXTRA_REMOTE_WAIT_NS = 128;

struct AdaptiveTaskState {
    uint64_t scope{0}, submit_tick{0}, release_tick{0};
    uint32_t ready_depth{0};
    int32_t release_cluster{-1};
    std::atomic<int32_t> dispatch_cluster{-1};
    uint32_t reserved{0};
    void reset() {
        scope = submit_tick = release_tick = 0;
        ready_depth = reserved = 0;
        release_cluster = -1;
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
    uint64_t ticks[6]{}, calls[6]{};
};
struct alignas(64) PartitionThread {
    std::atomic<int32_t> cluster{-1};
    std::atomic<uint32_t> capacity[PARTITION_SHAPES]{}, free_slots[PARTITION_SHAPES]{}, idle_slots[PARTITION_SHAPES]{};
    std::atomic<uint64_t> started[PARTITION_SHAPES]{}, incoming_moves[PARTITION_SHAPES]{};
    PartitionBucket samples[PARTITION_SHAPES];
    alignas(64) PartitionTrace trace;
    void reset() {
        trace = PartitionTrace{};
        cluster.store(-1, std::memory_order_relaxed);
        for (int i = 0; i < PARTITION_SHAPES; ++i) {
            capacity[i].store(0, std::memory_order_relaxed);
            free_slots[i].store(0, std::memory_order_relaxed);
            idle_slots[i].store(0, std::memory_order_relaxed);
            started[i].store(0, std::memory_order_relaxed);
            incoming_moves[i].store(0, std::memory_order_relaxed);
            samples[i].reset();
        }
    }
};
// A publication cell has independent ownership traffic from other resource shapes.
struct alignas(64) PartitionReadyCounter {
    std::atomic<uint64_t> value{0};
    uint64_t load(std::memory_order order = std::memory_order_seq_cst) const { return value.load(order); }
    void store(uint64_t n, std::memory_order order = std::memory_order_seq_cst) { value.store(n, order); }
    uint64_t fetch_add(uint64_t n, std::memory_order order = std::memory_order_seq_cst) {
        return value.fetch_add(n, order);
    }
};
static_assert(sizeof(PartitionReadyCounter) == 64);
struct PartitionShared {
    std::atomic<bool> enabled{false};
    int32_t schedulers[2]{}, orchestrator_cluster{-1};
    PartitionReadyCounter ready_published[2][PARTITION_SHAPES]{};
    std::atomic<uint32_t> idle_mask[2][PARTITION_SHAPES]{};
    PartitionThread threads[PARTITION_THREADS];
    void reset() {
        enabled.store(false, std::memory_order_relaxed);
        schedulers[0] = schedulers[1] = 0;
        orchestrator_cluster = -1;
        for (auto &cluster : ready_published)
            for (auto &count : cluster)
                count.store(0, std::memory_order_relaxed);
        for (auto &cluster : idle_mask)
            for (auto &mask : cluster)
                mask.store(0, std::memory_order_relaxed);
        for (auto &thread : threads)
            thread.reset();
    }
};
struct PartitionLoad {
    uint64_t blocked[2]{}, ready[2]{};
    uint32_t capacity[2]{}, free[2]{}, idle[2]{}, schedulers[2]{};
};
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
struct PartitionWindow {
    uint64_t count[PARTITION_METRICS]{}, ns[PARTITION_METRICS]{};
    void append(int metric, uint64_t n, uint64_t sum) {
        if (!n) return;
        uint64_t room = PARTITION_WINDOW_SAMPLES - count[metric];
        // A full window keeps recent observations without making update cadence depend on scope size.
        if (n >= room) {
            uint64_t retain = n < PARTITION_WINDOW_SAMPLES ? n : PARTITION_WINDOW_SAMPLES;
            uint64_t keep = PARTITION_WINDOW_SAMPLES - retain;
            ns[metric] = (count[metric] ? ns[metric] * keep / count[metric] : 0) + sum * retain / n;
            count[metric] = (count[metric] ? keep : 0) + retain;
        } else {
            count[metric] += n;
            ns[metric] += sum;
        }
    }
};
struct ClusterPartitioner {
    PartitionTrace trace;
    PartitionWeights weights[PARTITION_SHAPES];
    uint64_t assigned[2][PARTITION_SHAPES]{};
    uint64_t next_scope{0}, last_closed{0}, scope_ids[PARTITION_SCOPE_DEPTH]{},
        scope_assignments[PARTITION_SCOPE_DEPTH]{};
    PartitionSample consumed[PARTITION_THREADS][PARTITION_SHAPES]{};
    PartitionWindow pending[PARTITION_SHAPES][2];
    uint64_t observed[PARTITION_SHAPES][2]{};
    uint64_t decisions[2]{}, updates[2]{}, channel_updates[PARTITION_SHAPES][2][PARTITION_METRICS]{};
    uint64_t idle_choices{0}, cold_decisions{0}, learned_decisions{0}, polls{0};
    uint64_t decision_ticks{0}, max_decision_ticks{0};
    uint64_t last_poll_decision{0}, last_update_decision{0}, last_update_ns{0};
    uint64_t load_refresh[PARTITION_SHAPES]{~uint64_t{0}, ~uint64_t{0}, ~uint64_t{0}, ~uint64_t{0}};
    PartitionLoad cached_load[PARTITION_SHAPES]{};
    uint64_t cached_started[PARTITION_SHAPES][2]{}, cached_moved[PARTITION_SHAPES][2]{};
    void reset() { *this = ClusterPartitioner{}; }
    uint64_t open_scope(int depth) {
        scope_ids[depth] = ++next_scope;
        scope_assignments[depth] = 0;
        return next_scope;
    }
    void close_scope(int depth) {
        if (scope_assignments[depth]) last_closed = scope_ids[depth];
    }
    static uint64_t smooth(uint64_t prior, uint64_t value) {
        if (value < 20) value = 20;
        if (value > 32000) value = 32000;
        if (value < prior / 2) value = prior / 2;
        if (value > prior * 2) value = prior * 2;
        return (3 * prior + value) / 4;
    }
    void poll(const PartitionShared &shared, uint64_t now_ns) {
        uint64_t count = decisions[0] + decisions[1];
        if (count - last_poll_decision < PARTITION_POLL_NODES) return;
        last_poll_decision = count;
        ++polls;
        for (int t = 0; t < PARTITION_THREADS; ++t) {
            int c = shared.threads[t].cluster.load(std::memory_order_relaxed);
            if (c < 0 || c > 1) continue;
            for (int shape = 0; shape < PARTITION_SHAPES; ++shape) {
                PartitionSample sample;
                if (!shared.threads[t].samples[shape].read(sample)) continue;
                auto &previous = consumed[t][shape];
                observed[shape][c] += sample.observed - previous.observed;
                for (int m = 0; m < PARTITION_METRICS; ++m)
                    pending[shape][c].append(m, sample.count[m] - previous.count[m], sample.ns[m] - previous.ns[m]);
                previous = sample;
            }
        }
        if (count - last_update_decision < PARTITION_UPDATE_NODES ||
            (last_update_ns && now_ns - last_update_ns < PARTITION_UPDATE_NS))
            return;
        bool changed = false;
        for (int shape = 0; shape < PARTITION_SHAPES; ++shape) {
            for (int c = 0; c < 2; ++c) {
                auto &window = pending[shape][c];
                for (int k = 0; k < PARTITION_METRICS; ++k) {
                    int m = k == 0 ? 5 : k - 1;
                    uint64_t minimum = m == 0 ? 32 : 16;
                    if (window.count[m] < minimum) continue;
                    auto &w = weights[shape];
                    if (m == 5) {
                        uint64_t sample = window.ns[m] / window.count[m];
                        w.idle_pickup[c] = w.idle_pickup[c] ? smooth(w.idle_pickup[c], sample) : sample;
                    } else {
                        uint64_t &coefficient =
                            m == 0 ? w.queue[c] : (m <= 2 ? w.publish[m - 1][c] : w.dependency[m - 3][c]);
                        uint64_t sample = window.ns[m] / window.count[m];
                        if (m == 0) sample = sample > w.idle_pickup[c] ? sample - w.idle_pickup[c] : 20;
                        coefficient = smooth(coefficient, sample);
                    }
                    window.count[m] = window.ns[m] = 0;
                    ++channel_updates[shape][c][m];
                    ++updates[c];
                    changed = true;
                }
            }
        }
        if (changed) {
            last_update_decision = count;
            last_update_ns = now_ns;
        }
    }
    PartitionLoad load(const PartitionShared &shared, int shape, const uint64_t queue_depth[2]) {
        uint64_t count = decisions[0] + decisions[1];
        if (load_refresh[shape] == ~uint64_t{0} || count - load_refresh[shape] >= 8 ||
            cached_load[shape].capacity[0] + cached_load[shape].capacity[1] == 0) {
            auto &cached = cached_load[shape];
            cached = PartitionLoad{};
            for (int c = 0; c < 2; ++c)
                cached_started[shape][c] = cached_moved[shape][c] = 0;
            for (const auto &thread : shared.threads) {
                int c = thread.cluster.load(std::memory_order_relaxed);
                if (c < 0 || c > 1) continue;
                cached.capacity[c] += thread.capacity[shape].load(std::memory_order_relaxed);
                cached.free[c] += thread.free_slots[shape].load(std::memory_order_relaxed);
                cached.idle[c] += thread.idle_slots[shape].load(std::memory_order_relaxed);
                cached_started[shape][c] += thread.started[shape].load(std::memory_order_relaxed);
                cached_moved[shape][c] += thread.incoming_moves[shape].load(std::memory_order_relaxed);
            }
            load_refresh[shape] = count;
        }
        PartitionLoad result = cached_load[shape];
        for (int c = 0; c < 2; ++c) {
            result.schedulers[c] = shared.schedulers[c];
            uint64_t published = shared.ready_published[c][shape].load(std::memory_order_relaxed);
            result.blocked[c] = assigned[c][shape] > published ? assigned[c][shape] - published : 0;
            uint64_t supply = published + cached_moved[shape][c];
            uint64_t used = cached_started[shape][c] + cached_moved[shape][1 - c];
            uint64_t waiting = supply > used ? supply - used : 0;
            result.ready[c] = waiting > queue_depth[c] ? waiting : queue_depth[c];
        }
        return result;
    }
    bool mature(int shape, const PartitionLoad &load) const {
        if (decisions[0] + decisions[1] < 128) return false;
        uint64_t total = 0;
        bool fitted = false;
        for (int c = 0; c < 2; ++c) {
            total += observed[shape][c];
            if (load.schedulers[c] && observed[shape][c] < 16) return false;
            for (int m = 0; m < PARTITION_METRICS; ++m)
                fitted = fitted || channel_updates[shape][c][m] != 0;
        }
        return total >= 64 && fitted;
    }
    PartitionDecision choose(
        const PartitionLoad &load, uint32_t count, const uint32_t sources[2], int shape, int o_cluster,
        uint32_t units = 1
    ) {
        PartitionDecision out;
        out.learned = mature(shape, load);
        auto &w = weights[shape];
        for (int c = 0; c < 2; ++c) {
            if (!load.schedulers[c]) continue;
            if (!out.learned) {
                uint64_t capacity = load.capacity[c] ? load.capacity[c] : load.schedulers[c];
                out.score_ns[c] = 1024 * (load.blocked[c] + load.ready[c] + units) / capacity;
            } else {
                uint64_t cap = load.capacity[c];
                uint64_t free = load.free[c] < cap ? load.free[c] : cap;
                uint64_t idle = load.idle[c] < cap ? load.idle[c] : cap;
                uint64_t resource = cap ? w.queue[c] * (2 * (cap - free) + cap - idle) / cap : w.queue[c];
                uint64_t dependency = 0;
                if (count) {
                    for (int p = 0; p < 2; ++p)
                        if (sources[p] && w.dependency[p][c] > dependency) dependency = w.dependency[p][c];
                    if (sources[0] + sources[1] < count)
                        for (int p = 0; p < 2; ++p)
                            if (w.dependency[p][c] > dependency) dependency = w.dependency[p][c];
                } else {
                    int source = o_cluster >= 0 && o_cluster < 2 ? o_cluster : c;
                    dependency = w.publish[source][c];
                }
                uint64_t queue = w.queue[c] * load.ready[c] / load.schedulers[c];
                uint64_t extra_remote = count ? sources[1 - c] : 0;
                if (extra_remote && dependency == w.dependency[1 - c][c]) --extra_remote;
                out.score_ns[c] = dependency + PARTITION_QUEUE_WEIGHT_NUM * queue / PARTITION_QUEUE_WEIGHT_DEN +
                                  PARTITION_RESOURCE_WEIGHT_NUM * resource / PARTITION_RESOURCE_WEIGHT_DEN +
                                  PARTITION_EXTRA_REMOTE_WAIT_NS * extra_remote;
            }
            if (out.cluster < 0 || out.score_ns[c] < out.score_ns[out.cluster] ||
                (out.score_ns[c] == out.score_ns[out.cluster] &&
                 decisions[c] * load.schedulers[out.cluster] < decisions[out.cluster] * load.schedulers[c]))
                out.cluster = c;
        }
        if (!out.learned && out.cluster >= 0 && sources[0] != sources[1]) {
            int affinity = sources[1] > sources[0] ? 1 : 0;
            if (load.schedulers[affinity] && out.score_ns[affinity] <= out.score_ns[out.cluster] + 256)
                out.cluster = affinity;
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
