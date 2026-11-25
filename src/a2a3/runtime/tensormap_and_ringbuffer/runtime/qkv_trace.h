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
constexpr uint32_t QKV_TRACE_TASKS = 32768;
constexpr uint32_t QKV_TRACE_EDGES = 131072;
// All timestamps are AICPU counter ticks. Actual kernel start is unavailable;
// start_proxy distinguishes idle publication, promotion and gated release.
struct QkvTaskTrace {
    uint64_t task{0}, created{0}, ready{0}, pop{0}, publish{0}, completed{0};
    std::atomic<uint64_t> dispatch{0}, start_proxy{0}, first_fin{0}, fin{0};
    std::atomic<int> proxy_kind{0}, dispatch_side{-1}, scheduler{-1};
    int kernel[3]{}, blocks{0}, shape{0}, mask{0}, sync{0}, selected{-1}, fanout{0};
    uint64_t dummy_begin{0}, dummy_end{0};
    double features[2][5]{};  // queued/capacity, occupied, idle fraction, capacity, reservations/capacity
    uint64_t summary_age{0};
    void init(uint64_t id, uint64_t now) {
        task = id;
        created = now;
        ready = pop = publish = completed = dummy_begin = dummy_end = summary_age = 0;
        dispatch.store(0);
        start_proxy.store(0);
        first_fin.store(0);
        fin.store(0);
        proxy_kind.store(0);
        dispatch_side.store(-1);
        scheduler.store(-1);
        selected = -1;
        fanout = 0;
        for (auto &side : features)
            for (auto &x : side)
                x = 0;
    }
    static bool first(std::atomic<uint64_t> &value, uint64_t tick) {
        uint64_t zero = 0;
        return value.compare_exchange_strong(zero, tick, std::memory_order_relaxed);
    }
    void start(uint64_t tick, int kind) {
        if (first(start_proxy, tick)) proxy_kind.store(kind, std::memory_order_relaxed);
    }
};
struct QkvTraceEdge {
    uint64_t producer{0}, consumer{0};
};
struct QkvTrace {
    int level{0};                                  // 0 off, 1 deterministic 1/32 plus MIX/DUMMY, 2 full DAG diagnostics
    uint32_t count{0}, edge_count{0}, dropped{0};  // creation/wiring has one O writer
    QkvTaskTrace tasks[QKV_TRACE_TASKS];
    QkvTraceEdge edges[QKV_TRACE_EDGES];
    uint64_t phase_ticks[4]{}, service_calls{0};
    uint64_t shadow_ticks[1024]{}, shadow_linear_ticks[1024]{}, shadow_count{0};
    uint64_t phase_samples[4096][4]{};
    uint64_t s0_last_dispatch{0}, s0_max_gap{0}, s0_gaps[32]{};
    std::atomic<uint64_t> early_candidate{0}, early_hit{0}, early_fallback{0}, early_release{0};
    uint64_t early_staged[6]{};
    uint64_t early_slots[6][4]{};  // S0: pop, snapshot, score, publication
    uint64_t service_hist[4][32]{};
    void reset() {
        level = 0;
        count = edge_count = dropped = 0;
        service_calls = 0;
        shadow_count = 0;
        s0_last_dispatch = s0_max_gap = 0;
        for (auto &x : s0_gaps)
            x = 0;
        early_candidate.store(0);
        early_hit.store(0);
        early_fallback.store(0);
        early_release.store(0);
        for (auto &x : early_staged)
            x = 0;
        for (auto &row : early_slots)
            for (auto &x : row)
                x = 0;
        for (auto &v : phase_ticks)
            v = 0;
        for (auto &row : service_hist)
            for (auto &v : row)
                v = 0;
    }
};
}  // namespace simpler::tmr
