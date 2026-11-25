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
/**
 * tensormap_and_ringbuffer scheduler implementation
 *
 * Implements scheduler state management, ready queues, and task lifecycle.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "scheduler.h"
#include "qkv_learned.h"
#include <inttypes.h>
#include <cstdio>
#include <stdlib.h>
#include "common/unified_log.h"

#if SIMPLER_DFX
// Weak fallbacks for host/UT builds that don't link the scope_stats collector.
extern "C" __attribute__((weak, visibility("hidden"))) bool is_scope_stats_enabled() { return false; }
extern "C" __attribute__((weak, visibility("hidden"))) void scope_stats_note_heap_wrap(int) {}
#endif

// =============================================================================
// Scheduler Profiling Counters
// =============================================================================

#if SIMPLER_SCHED_PROFILING
#include "common/platform_config.h"

uint64_t g_sched_lock_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanout_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanin_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_self_consumed_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_lock_wait_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_push_wait_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_pop_wait_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_lock_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanout_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanin_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_self_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_pop_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_complete_count[PLATFORM_MAX_AICPU_THREADS] = {};

SchedProfilingData scheduler_get_profiling(int thread_idx) {
    SchedProfilingData d;
    d.lock_cycle = std::exchange(g_sched_lock_cycle[thread_idx], 0);
    d.fanout_cycle = std::exchange(g_sched_fanout_cycle[thread_idx], 0);
    d.fanin_cycle = std::exchange(g_sched_fanin_cycle[thread_idx], 0);
    d.self_consumed_cycle = std::exchange(g_sched_self_consumed_cycle[thread_idx], 0);
    d.lock_wait_cycle = std::exchange(g_sched_lock_wait_cycle[thread_idx], 0);
    d.push_wait_cycle = std::exchange(g_sched_push_wait_cycle[thread_idx], 0);
    d.pop_wait_cycle = std::exchange(g_sched_pop_wait_cycle[thread_idx], 0);
    d.lock_atomic_count = std::exchange(g_sched_lock_atomic_count[thread_idx], 0);
    d.fanout_atomic_count = std::exchange(g_sched_fanout_atomic_count[thread_idx], 0);
    d.fanin_atomic_count = std::exchange(g_sched_fanin_atomic_count[thread_idx], 0);
    d.self_atomic_count = std::exchange(g_sched_self_atomic_count[thread_idx], 0);
    d.pop_atomic_count = std::exchange(g_sched_pop_atomic_count[thread_idx], 0);
    d.complete_count = std::exchange(g_sched_complete_count[thread_idx], 0);
    return d;
}
#endif

// =============================================================================
// Async wait diagnostics
// =============================================================================

void AsyncWaitList::log_diagnostics(AICoreCompletionMailbox *aicore_mailbox, const char *reason) {
    struct PendingDetail {
        uint64_t task_id;
        uint64_t addr;
        uint64_t backend_cookie;
        uint32_t expected_value;
        uint32_t observed_value;
        int32_t entry_idx;
        int32_t condition_idx;
        int32_t waiting_completion_count;
        int32_t condition_count;
        int32_t completion_type;
        AsyncEngine engine;
        bool normal_done;
        bool retired;
        bool observed_available;
    };

    constexpr int32_t kMaxDetails = 16;
    constexpr int32_t kRunningReserve = kMaxDetails / 2;
    PendingDetail details[kMaxDetails];
    bool entry_selected[MAX_ASYNC_WAITS] = {};
    int32_t first_selected_condition[MAX_ASYNC_WAITS];
    for (int32_t i = 0; i < MAX_ASYNC_WAITS; i++)
        first_selected_condition[i] = -1;

    uint64_t mailbox_head = 0;
    uint64_t mailbox_tail = 0;
    auto sample_mailbox = [&]() {
        if (aicore_mailbox == nullptr) return;
        mailbox_head = aicore_mailbox->head.load(std::memory_order_acquire);
        mailbox_tail = aicore_mailbox->tail.load(std::memory_order_acquire);
    };
    auto mailbox_pending = [&]() {
        return mailbox_head >= mailbox_tail ? mailbox_head - mailbox_tail : 0;
    };

    constexpr int32_t kLockRetries = 1024;
    bool locked = false;
    for (int32_t i = 0; i < kLockRetries && !locked; i++) {
        locked = try_lock();
        if (!locked) SPIN_WAIT_HINT();
    }
    if (!locked) {
        sample_mailbox();
        LOG_WARN(
            "[ASYNC_WAIT reason=%s] status=skipped cause=list_busy retries=%d mailbox_head=%llu mailbox_tail=%llu "
            "mailbox_pending=%llu mpsc_skipped=%llu",
            reason, kLockRetries, static_cast<unsigned long long>(mailbox_head),
            static_cast<unsigned long long>(mailbox_tail), static_cast<unsigned long long>(mailbox_pending()),
            static_cast<unsigned long long>(mpsc_skipped_count.load(std::memory_order_relaxed))
        );
        return;
    }

    // Keep the mailbox counters and wait-list entries in the same diagnostic
    // window. Sampling before the consumer lock can combine two different
    // states and make head-tail appear to underflow.
    sample_mailbox();

    int32_t entry_count = count;
    if (entry_count < 0) entry_count = 0;
    if (entry_count > MAX_ASYNC_WAITS) entry_count = MAX_ASYNC_WAITS;

    int32_t task_body_pending_entries = 0;
    int32_t completion_pending_entries = 0;
    int32_t settled_entries = 0;
    int32_t pending_conditions = 0;
    int32_t detail_candidates = entry_count;
    for (int32_t entry_idx = 0; entry_idx < entry_count; entry_idx++) {
        const AsyncWaitEntry &entry = entries[entry_idx];
        int32_t waiting = entries[entry_idx].waiting_completion_count;
        bool task_body_pending = !entry.normal_done;
        bool completion_pending = waiting > 0;
        if (task_body_pending) task_body_pending_entries++;
        if (completion_pending) {
            completion_pending_entries++;
            pending_conditions += waiting;
        }
        if (!task_body_pending && !completion_pending) {
            settled_entries++;
        }

        int32_t condition_count = entry.condition_count;
        if (condition_count < 0) condition_count = 0;
        if (condition_count > MAX_COMPLETIONS_PER_TASK) condition_count = MAX_COMPLETIONS_PER_TASK;
        int32_t unsatisfied = 0;
        for (int32_t condition_idx = 0; condition_idx < condition_count; condition_idx++) {
            if (!entry.conditions[condition_idx].satisfied) unsatisfied++;
        }
        if (unsatisfied > 1) detail_candidates += unsatisfied - 1;
    }

    int32_t detail_count = 0;
    auto capture = [&](int32_t entry_idx, const CompletionCondition *cond, int32_t condition_idx) {
        if (detail_count >= kMaxDetails) return;
        const AsyncWaitEntry &entry = entries[entry_idx];
        PendingDetail &detail = details[detail_count++];
        detail.task_id = TaskId::to_uint64(entry.task_token);
        detail.addr = cond != nullptr ? cond->addr : 0;
        detail.backend_cookie = cond != nullptr ? cond->backend_cookie : 0;
        detail.expected_value = cond != nullptr ? cond->expected_value : 0;
        detail.observed_value = 0;
        detail.entry_idx = entry_idx;
        detail.condition_idx = condition_idx;
        detail.waiting_completion_count = entry.waiting_completion_count;
        detail.condition_count = entry.condition_count;
        detail.completion_type = cond != nullptr ? cond->completion_type : -1;
        detail.engine = cond != nullptr ? cond->engine : ASYNC_ENGINE_SDMA;
        detail.normal_done = entry.normal_done;
        detail.retired = cond != nullptr && cond->retired;
        detail.observed_available =
            cond != nullptr && cond->completion_type == COMPLETION_TYPE_COUNTER && cond->counter_addr != nullptr;
        if (detail.observed_available) {
            uintptr_t counter_line = mailbox_cache_line(cond->counter_addr);
            cache_invalidate_range(reinterpret_cast<const void *>(counter_line), sizeof(uint32_t));
            detail.observed_value = *cond->counter_addr;
        }
    };

    auto capture_entry = [&](int32_t entry_idx) {
        if (entry_selected[entry_idx] || detail_count >= kMaxDetails) return;
        const AsyncWaitEntry &entry = entries[entry_idx];
        int32_t condition_count = entry.condition_count;
        if (condition_count < 0) condition_count = 0;
        if (condition_count > MAX_COMPLETIONS_PER_TASK) condition_count = MAX_COMPLETIONS_PER_TASK;
        for (int32_t condition_idx = 0; condition_idx < condition_count; condition_idx++) {
            const CompletionCondition &cond = entry.conditions[condition_idx];
            if (cond.satisfied) continue;
            first_selected_condition[entry_idx] = condition_idx;
            entry_selected[entry_idx] = true;
            capture(entry_idx, &cond, condition_idx);
            return;
        }
        entry_selected[entry_idx] = true;
        capture(entry_idx, nullptr, -1);
    };

    // Keep both classes visible when the list is larger than the detail cap:
    // before-normal-done entries include running candidates, while
    // normal-done entries are blocked only by their completion conditions.
    for (int32_t entry_idx = 0; entry_idx < entry_count && detail_count < kRunningReserve; entry_idx++) {
        if (!entries[entry_idx].normal_done) capture_entry(entry_idx);
    }
    for (int32_t entry_idx = 0; entry_idx < entry_count && detail_count < kMaxDetails; entry_idx++) {
        if (entries[entry_idx].normal_done && entries[entry_idx].waiting_completion_count > 0) {
            capture_entry(entry_idx);
        }
    }
    for (int32_t entry_idx = 0; entry_idx < entry_count && detail_count < kMaxDetails; entry_idx++) {
        capture_entry(entry_idx);
    }

    // If fewer than 16 tasks are listed, use the remaining rows for additional
    // pending conditions belonging to those tasks.
    for (int32_t entry_idx = 0; entry_idx < entry_count && detail_count < kMaxDetails; entry_idx++) {
        const AsyncWaitEntry &entry = entries[entry_idx];
        int32_t condition_count = entry.condition_count;
        if (condition_count < 0) condition_count = 0;
        if (condition_count > MAX_COMPLETIONS_PER_TASK) condition_count = MAX_COMPLETIONS_PER_TASK;
        for (int32_t condition_idx = 0; condition_idx < condition_count && detail_count < kMaxDetails;
             condition_idx++) {
            const CompletionCondition &cond = entry.conditions[condition_idx];
            if (cond.satisfied || condition_idx == first_selected_condition[entry_idx]) continue;
            capture(entry_idx, &cond, condition_idx);
        }
    }

    uint64_t mpsc_skipped = mpsc_skipped_count.load(std::memory_order_relaxed);
    unlock();

    LOG_WARN(
        "[ASYNC_WAIT reason=%s] status=captured wait_count=%d task_body_pending_entries=%d "
        "completion_pending_entries=%d settled_entries=%d pending_conditions=%d details=%d truncated=%d "
        "mailbox_head=%llu mailbox_tail=%llu mailbox_pending=%llu mpsc_skipped=%llu",
        reason, entry_count, task_body_pending_entries, completion_pending_entries, settled_entries, pending_conditions,
        detail_count, detail_candidates > detail_count ? 1 : 0, static_cast<unsigned long long>(mailbox_head),
        static_cast<unsigned long long>(mailbox_tail), static_cast<unsigned long long>(mailbox_pending()),
        static_cast<unsigned long long>(mpsc_skipped)
    );

    for (int32_t i = 0; i < detail_count; i++) {
        const PendingDetail &detail = details[i];
        char observed[32];
        if (detail.observed_available) {
            snprintf(observed, sizeof(observed), "%u", detail.observed_value);
        } else {
            snprintf(observed, sizeof(observed), "unavailable");
        }
        LOG_WARN(
            "[ASYNC_WAIT reason=%s entry=%d cond=%d] task_token=%llu normal_done=%u waiting=%d conditions=%d "
            "engine=%s type=%d addr=0x%llx cookie=0x%llx expected=%u observed=%s retired=%u",
            reason, detail.entry_idx, detail.condition_idx, static_cast<unsigned long long>(detail.task_id),
            static_cast<unsigned>(detail.normal_done), detail.waiting_completion_count, detail.condition_count,
            detail.condition_idx >= 0 ? async_engine_name(detail.engine) : "none", detail.completion_type,
            static_cast<unsigned long long>(detail.addr), static_cast<unsigned long long>(detail.backend_cookie),
            detail.expected_value, observed, static_cast<unsigned>(detail.retired)
        );
    }
}

// =============================================================================
// Debug Utilities
// =============================================================================

void SchedulerState::print_stats() {
    SchedulerState *sched = this;
    LOG_DEBUG("=== Scheduler Statistics ===");
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        if (sched->ring_sched_states[r].last_task_alive > 0) {
            LOG_DEBUG("Ring %d:", r);
            LOG_DEBUG("  last_task_alive: %d", sched->ring_sched_states[r].last_task_alive);
            auto &dp = sched->ring_sched_states[r].dep_pool;
            if (dp.top > 0) {
                LOG_DEBUG(
                    "  dep_pool: top=%d tail=%d used=%d high_water=%d capacity=%d", dp.top, dp.tail, dp.top - dp.tail,
                    dp.high_water, dp.capacity
                );
            }
        }
    }
#if SIMPLER_SCHED_PROFILING
    LOG_DEBUG("tasks_completed:   %lld", (long long)sched->tasks_completed.load(std::memory_order_relaxed));
    LOG_DEBUG("tasks_consumed:    %lld", (long long)sched->tasks_consumed.load(std::memory_order_relaxed));
#endif
    LOG_DEBUG("============================");
}

void SchedulerState::print_queues() {
    SchedulerState *sched = this;
    LOG_DEBUG("=== Ready Queues ===");

    const char *shape_names[] = {"AIC", "AIV", "MIX"};

    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        LOG_DEBUG("  %s: count=%" PRIu64, shape_names[i], sched->ready_queues[i].size());
        for (int cluster = 0; cluster < 2; cluster++) {
            LOG_DEBUG(
                "  cluster %d %s: count=%" PRIu64, cluster, shape_names[i],
                sched->cluster_ready_queues[cluster][i].size() + sched->auto_ready_queues[cluster][i].size()
            );
        }
    }
    LOG_DEBUG("  DUMMY: count=%" PRIu64, sched->dummy_ready_queue.size());
    for (int cluster = 0; cluster < 2; cluster++) {
        LOG_DEBUG("  cluster %d DUMMY: count=%" PRIu64, cluster, sched->cluster_dummy_queues[cluster].size());
    }

    LOG_DEBUG("====================");
}

void SchedulerState::service_qkv_batch() {
    using namespace simpler::tmr;
    auto &q = this->qkv;
    ChipTaskSlotState *batch[QKV_BATCH];
    // pop_batch consumes only currently published tickets; no wait to fill 8.
    bool timing = q.trace.level && ((q.decision_count & 7) == 0) && qkv_pending.size();
    uint64_t pop_begin = timing ? get_sys_cnt_aicpu() : 0;
    int n = this->qkv_pending.pop_batch(batch, QKV_BATCH);
    if (!n) return;
    uint64_t begin = get_sys_cnt_aicpu();
    uint64_t phase[4] = {timing ? begin - pop_begin : 0, 0, 0, 0};
    ++q.batches[n];
    QkvSnapshot snapshot;
    bool improved = q.mode >= 3 && q.mode != 6;
    uint32_t active = improved ? q.active_channels.load(std::memory_order_acquire) : 3;
    bool valid = true;
    QkvFallback invalid_reason = QkvFallback::Snapshot;
    uint64_t oldest = begin;
    for (int t = 0; t < q.schedulers; ++t) {
        int c = q.side[t];
        uint64_t version = q.threads[t].version.load(std::memory_order_acquire);
        for (int r = 0; r < QKV_CHANNELS; ++r) {
            if (!(active & (1u << r))) continue;
            uint64_t word = q.threads[t].resources[r].load(std::memory_order_acquire);
            double cap = word & 255, idle = (word >> 16) & 255, pending_free = (word >> 8) & 255;
            if (!cap) {
                valid = false;
                if (improved) invalid_reason = QkvFallback::Uninitialized;
            }
            snapshot.capacity[c][r] += cap;
            snapshot.idle[c][r] += idle;
            snapshot.occupied[c][r] += 2 * cap - 2 * idle - pending_free;
        }
        if (improved) {
            uint64_t stamp = q.threads[t].stamp.load(std::memory_order_acquire);
            if (stamp < oldest) oldest = stamp;
            if (!stamp || !version) {
                valid = false;
                invalid_reason = QkvFallback::Uninitialized;
            } else if ((begin > stamp ? begin - stamp : 0) * 1000000000ULL / PLATFORM_PROF_SYS_CNT_FREQ >
                       QKV_SUMMARY_MAX_NS) {
                valid = false;
                invalid_reason = QkvFallback::Stale;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            if ((version & 1) || version != q.threads[t].version.load(std::memory_order_relaxed)) {
                valid = false;
                invalid_reason = QkvFallback::Version;
            }
        }
    }
    for (int c = 0; c < 2; ++c)
        for (int r = 0; r < QKV_CHANNELS; ++r) {
            if (!(active & (1u << r))) continue;
            uint64_t taken = 0;
            for (int t = 0; t < q.schedulers; ++t)
                taken += q.threads[t].taken[c][r].load(std::memory_order_relaxed);
            snapshot.queued[c][r] = q.assigned[c][r] > taken ? q.assigned[c][r] - taken : 0;
            if (snapshot.capacity[c][r] > 0) snapshot.occupied[c][r] /= 2 * snapshot.capacity[c][r];
        }
    for (int r = 0; r < QKV_CHANNELS; ++r) {
        if (!(active & (1u << r))) continue;
        uint64_t added = 0, taken = 0;
        for (int t = 0; t < QKV_THREADS; ++t) {
            added += q.threads[t].shared_added[r].load(std::memory_order_relaxed);
            taken += q.threads[t].shared_taken[r].load(std::memory_order_relaxed);
        }
        double backlog = added > taken ? added - taken : 0;
        double total = snapshot.capacity[0][r] + snapshot.capacity[1][r];
        if (total > 0)
            for (int c = 0; c < 2; ++c)
                snapshot.queued[c][r] += backlog * snapshot.capacity[c][r] / total;
    }
    if (timing) phase[1] = get_sys_cnt_aicpu() - begin;
    double reserved[2][QKV_CHANNELS]{}, legacy_reserved[2][2]{};
    QkvBatchScorer scorer;
    uint64_t shadow_elapsed = 0;
    // Shadow covers eight tasks / sixteen candidate scores, without publishing.
    if (valid && timing && q.trace.level == 2 && q.mode == 3 && q.trace.shadow_count < 1024) {
        ChipTaskSlotState *sample = nullptr;
        for (int i = 0; i < n; ++i)
            if (learned::runtime(
                    batch[i]->task->kernel_id, batch[i]->logical_block_num, batch[i]->active_mask.core_mask(), 0
                ) > 0) {
                sample = batch[i];
                break;
            }
        if (sample) {
            int channel =
                qkv_channel(static_cast<int>(sample->active_mask.to_shape()), sample->active_mask.core_mask());
            volatile double checksum = 0;
            uint64_t shadow_begin = get_sys_cnt_aicpu();
            for (int k = 0; k < 8; ++k)
                for (int side = 0; side < 2; ++side)
                    checksum += qkv_learned_score(
                        snapshot, side, channel, sample->logical_block_num, sample->active_mask.core_mask(),
                        sample->task->kernel_id, double(k) * .00001, 0
                    );
            shadow_elapsed = get_sys_cnt_aicpu() - shadow_begin;
            q.trace.shadow_ticks[q.trace.shadow_count] = shadow_elapsed;
            uint64_t linear_begin = get_sys_cnt_aicpu();
            for (int k = 0; k < 8; ++k)
                for (int side = 0; side < 2; ++side)
                    checksum += qkv_learned_score(
                        snapshot, side, channel, sample->logical_block_num, sample->active_mask.core_mask(),
                        sample->task->kernel_id, double(k) * .00001, 0, false
                    );
            uint64_t linear_elapsed = get_sys_cnt_aicpu() - linear_begin;
            q.trace.shadow_linear_ticks[q.trace.shadow_count++] = linear_elapsed;
            shadow_elapsed += linear_elapsed;
            (void)checksum;
        }
    }
    uint64_t decision_begin = begin + shadow_elapsed;
    for (int i = 0; i < n; ++i) {
        auto *slot = batch[i];
        int shape = static_cast<int>(slot->active_mask.to_shape());
        int r = qkv_channel(shape, slot->active_mask.core_mask());
        auto *record = qkv_record(*slot);
        if (record) {
            record->pop = begin;
            record->summary_age = begin > oldest ? begin - oldest : 0;
            for (int side = 0; side < 2; ++side) {
                double cap = snapshot.capacity[side][r];
                record->features[side][0] = cap ? snapshot.queued[side][r] / cap : 0;
                record->features[side][1] = snapshot.occupied[side][r];
                record->features[side][2] = cap ? snapshot.idle[side][r] / cap : 0;
                record->features[side][3] = cap;
                record->features[side][4] = cap ? reserved[side][r] / cap : 0;
            }
        }
        uint64_t score_begin = timing ? get_sys_cnt_aicpu() : 0;
        int c = -1;
        if (!valid) q.fail(invalid_reason);
        else if (improved && snapshot.idle[0][r] - reserved[0][r] < slot->logical_block_num &&
                 snapshot.idle[1][r] - reserved[1][r] < slot->logical_block_num)
            q.fail(QkvFallback::Busy);
        else if ((get_sys_cnt_aicpu() - decision_begin) * 1000000000ULL / PLATFORM_PROF_SYS_CNT_FREQ >= QKV_BUDGET_NS)
            q.fail(QkvFallback::Budget);
        else {
            // Wait age saturates at 1ms to keep all Q features dimensionless.
            double age = double(begin - slot->payload->placement.submit_tick) / (PLATFORM_PROF_SYS_CNT_FREQ / 1000.0);
            if (age > 1) age = 1;
            if (q.mode == 4) {
                if (learned::runtime(slot->task->kernel_id, slot->logical_block_num, slot->active_mask.core_mask(), 0) >
                    0)
                    ++q.learned_tasks;
                else ++q.learned_missing;
                c = qkv_learned_choose(
                    snapshot, r, slot->logical_block_num, slot->active_mask.core_mask(), slot->task->kernel_id, age,
                    reserved, scorer
                );
            } else if (improved) c = scorer.choose(snapshot, r, slot->logical_block_num, age, reserved);
            else {
                c = qkv_choose(snapshot, r, slot->logical_block_num, age, legacy_reserved, q.mode != 2 && q.mode != 6);
                for (int side = 0; side < 2; ++side)
                    for (int type = 0; type < 2; ++type)
                        reserved[side][type] = legacy_reserved[side][type];
            }
            if (c < 0) q.fail(QkvFallback::NoSide);
            else if ((get_sys_cnt_aicpu() - decision_begin) * 1000000000ULL / PLATFORM_PROF_SYS_CNT_FREQ >=
                     QKV_BUDGET_NS) {
                // A score that crosses the deadline has no publication rights.
                reserved[c][r] -= slot->logical_block_num;
                c = -1;
                q.fail(QkvFallback::Budget);
            }
        }
        if (q.mode == 6) c = -1;  // Controlled Shared publication after the unchanged base decision.
        uint64_t publish_begin = timing ? get_sys_cnt_aicpu() : 0;
        if (timing) phase[2] += publish_begin - score_begin;
        if (c < 0) {
            qkv_trace_publish(*slot);
            slot->payload->placement.reserved = 0;
            this->note_qkv_shared(*slot, 0);
            this->ready_queues[shape].push(slot);
        } else {
            ++q.assigned_tasks[c];
            ++q.shape_assigned[shape];
            q.assigned[c][r] += slot->logical_block_num;
            slot->payload->placement.reserved = 2;
            slot->task->target_aicpu_cluster = c;
            this->has_cluster_targets.store(true, std::memory_order_release);
            qkv_trace_publish(*slot);
            this->cluster_ready_queues[c][shape].push(slot);
        }
        if (improved)
            for (int side = 0; side < 2; ++side) {
                double available = snapshot.idle[side][r] - reserved[side][r];
                uint64_t hint = static_cast<uint64_t>(available > 0 ? available : 0);
                q.idle_hint[side][r].store((oldest << 8) | hint, std::memory_order_release);
            }
        if (timing) phase[3] += get_sys_cnt_aicpu() - publish_begin;
    }
    if (timing) {
        uint64_t index = q.trace.service_calls++;
        for (int part = 0; part < 4; ++part) {
            q.trace.phase_ticks[part] += phase[part];
            if (index < 4096) q.trace.phase_samples[index][part] = phase[part];
        }
    }
    uint64_t elapsed = get_sys_cnt_aicpu() - begin;
    uint64_t ns = elapsed * 1000000000ULL / PLATFORM_PROF_SYS_CNT_FREQ;
    int bin = 0;
    while (ns > 1 && bin < 31) {
        ns >>= 1;
        ++bin;
    }
    ++q.tick_hist[bin];
    q.ticks += elapsed;
    if (elapsed > q.max_ticks) q.max_ticks = elapsed;
    if (q.decision_count < 4096) q.decision_ticks[q.decision_count] = elapsed;
    ++q.decision_count;
}
