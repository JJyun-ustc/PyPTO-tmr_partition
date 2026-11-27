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
 * Unit tests for SchedulerState from scheduler.h
 *
 * Tests task state transitions, fanin/fanout logic, subtask completion.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include "utils/device_arena.h"
#include "scheduler/scheduler_types.h"
#include "scheduler/scheduler.h"
#include "qkv_learned.h"
#include "streaming_action.h"

TEST(AsyncWaitDiagnosticsTest, SeparatesTaskBodyAndCompletionPendingAndGuardsMailboxUnderflow) {
    auto wait_list = std::make_unique<AsyncWaitList>();
    auto mailbox = std::make_unique<AICoreCompletionMailbox>();
    wait_list->reset_for_reuse();
    wait_list->count = 3;

    wait_list->entries[0].task_token = TaskId::make(0, 0);
    wait_list->entries[0].normal_done = false;
    wait_list->entries[0].waiting_completion_count = 0;

    wait_list->entries[1].task_token = TaskId::make(0, 1);
    wait_list->entries[1].normal_done = true;
    wait_list->entries[1].waiting_completion_count = 1;
    wait_list->entries[1].condition_count = 1;
    wait_list->entries[1].conditions[0].satisfied = false;

    wait_list->entries[2].task_token = TaskId::make(0, 2);
    wait_list->entries[2].normal_done = true;
    wait_list->entries[2].waiting_completion_count = 0;

    mailbox->head.store(1, std::memory_order_relaxed);
    mailbox->tail.store(2, std::memory_order_relaxed);
    testing::internal::CaptureStderr();
    wait_list->log_diagnostics(mailbox.get(), "unit_test");
    std::string output = testing::internal::GetCapturedStderr();

    EXPECT_NE(output.find("task_body_pending_entries=1"), std::string::npos);
    EXPECT_NE(output.find("completion_pending_entries=1"), std::string::npos);
    EXPECT_NE(output.find("settled_entries=1"), std::string::npos);
    EXPECT_NE(output.find("mailbox_head=1 mailbox_tail=2 mailbox_pending=0"), std::string::npos);
}

TEST(AsyncWaitDiagnosticsTest, CapsDetailRowsAtSixteen) {
    auto wait_list = std::make_unique<AsyncWaitList>();
    wait_list->reset_for_reuse();
    wait_list->count = 20;
    for (int32_t i = 0; i < wait_list->count; i++) {
        wait_list->entries[i].task_token = TaskId::make(0, static_cast<uint32_t>(i));
        wait_list->entries[i].normal_done = false;
    }

    testing::internal::CaptureStderr();
    wait_list->log_diagnostics(nullptr, "detail_cap");
    std::string output = testing::internal::GetCapturedStderr();
    const std::string marker = "[ASYNC_WAIT reason=detail_cap entry=";
    size_t detail_count = 0;
    for (size_t pos = 0; (pos = output.find(marker, pos)) != std::string::npos; pos += marker.size())
        detail_count++;

    EXPECT_EQ(detail_count, 16u);
    EXPECT_NE(output.find("details=16 truncated=1"), std::string::npos);
}

TEST(SyncStartDrainAttemptTest, LateAckCannotSatisfyNextAttemptBarrier) {
    std::atomic<uint64_t> ack_tokens[3]{};
    uint64_t old_attempt = 41;
    uint64_t next_attempt = sync_start_drain_next_attempt(old_attempt);
    uint64_t old_subtree = sync_start_drain_ack_subtree_token(old_attempt);
    uint64_t next_subtree = sync_start_drain_ack_subtree_token(next_attempt);

    ack_tokens[1].store(old_subtree, std::memory_order_relaxed);
    ack_tokens[2].store(old_subtree, std::memory_order_relaxed);
    EXPECT_EQ(ack_tokens[1].load(std::memory_order_relaxed), old_subtree);
    EXPECT_EQ(ack_tokens[2].load(std::memory_order_relaxed), old_subtree);

    // A late child token from attempt 41 cannot satisfy attempt 42's tree barrier.
    ack_tokens[1].store(next_subtree, std::memory_order_relaxed);
    ack_tokens[2].store(old_subtree, std::memory_order_relaxed);
    EXPECT_EQ(ack_tokens[1].load(std::memory_order_relaxed), next_subtree);
    EXPECT_NE(ack_tokens[2].load(std::memory_order_relaxed), next_subtree);

    ack_tokens[2].store(next_subtree, std::memory_order_relaxed);
    ack_tokens[0].store(old_subtree, std::memory_order_relaxed);
    EXPECT_NE(ack_tokens[0].load(std::memory_order_relaxed), next_subtree);
    ack_tokens[0].store(next_subtree, std::memory_order_relaxed);
    EXPECT_EQ(ack_tokens[0].load(std::memory_order_relaxed), next_subtree);

    EXPECT_EQ(sync_start_drain_next_attempt(SYNC_START_DRAIN_ATTEMPT_MASK), 1u);
}

class SchedulerStateTest : public ::testing::Test {
protected:
    SchedulerState sched;
    SharedMemoryHandle *sm_handle = nullptr;
    DeviceArena sm_arena;
    DeviceArena sched_arena;

    // Each init_slot()'d slot gets a distinct zeroed payload from this pool,
    // mirroring orch::prepare_task's bind_buffers: every production slot has a
    // payload, and the scheduler's release/propagate paths dereference it.
    static constexpr int kSlotPayloadPoolSize = 16;
    TaskPayload slot_payload_pool_[kSlotPayloadPoolSize];
    TaskDescriptor slot_task_pool_[kSlotPayloadPoolSize];
    int slot_payload_pool_idx_ = 0;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        auto layout = SchedulerState::reserve_layout(sched_arena);
        ASSERT_NE(sched_arena.commit(), nullptr);
        ASSERT_TRUE(sched.init_data_from_layout(layout, sched_arena, sm_handle->header));
        sched.wire_arena_pointers(layout, sched_arena);
    }

    void TearDown() override {
        sched.destroy();
        sched_arena.release();
        sm_arena.release();
    }

    void init_slot(
        ChipTaskSlotState &slot, ChipTaskState state, int32_t fanin_count, int32_t fanout_count, uint8_t ring_id = 0
    ) {
        memset(&slot, 0, sizeof(slot));
        slot.task_state.store(state);
        slot.fanin_count = fanin_count;
        slot.fanin_refcount.store(0);
        slot.fanout_count = fanout_count;
        slot.fanout_refcount.store(0);
        slot.fanout_lock.store(0);
        slot.fanout_head = nullptr;
        slot.ring_id = ring_id;
        slot.active_mask = ActiveMask(SUBTASK_MASK_AIC);
        slot.completed_subtasks.store(0);
        slot.total_required_subtasks = 1;
        slot.logical_block_num = 1;
        int pool_index = slot_payload_pool_idx_++ % kSlotPayloadPoolSize;
        TaskPayload &slot_pl = slot_payload_pool_[pool_index];
        memset(&slot_pl, 0, sizeof(slot_pl));
        slot_pl.placement.reset();
        slot.payload = &slot_pl;
        TaskDescriptor &task = slot_task_pool_[pool_index];
        memset(&task, 0, sizeof(task));
        task.target_aicpu_cluster = -1;
        slot.task = &task;
    }

    void init_ring_slot(
        SharedMemoryRingHeader &ring, int32_t task_id, ChipTaskState state, uint8_t ring_id, uint32_t fanout_count = 1,
        uint32_t fanout_refcount = 1
    ) {
        ChipTaskSlotState &slot = ring.get_slot_state_by_task_id(task_id);
        TaskPayload &payload = ring.get_payload_by_task_id(task_id);
        TaskDescriptor &task = ring.get_task_by_task_id(task_id);
        memset(&slot, 0, sizeof(slot));
        memset(&payload, 0, sizeof(payload));
        memset(&task, 0, sizeof(task));
        task.target_aicpu_cluster = -1;
        slot.task_state.store(state, std::memory_order_relaxed);
        slot.fanin_count = 0;
        slot.fanin_refcount.store(0, std::memory_order_relaxed);
        slot.fanout_count = fanout_count;
        slot.fanout_refcount.store(fanout_refcount, std::memory_order_relaxed);
        slot.fanout_lock.store(0, std::memory_order_relaxed);
        slot.fanout_head = nullptr;
        slot.ring_id = ring_id;
        slot.active_mask = ActiveMask(SUBTASK_MASK_AIC);
        slot.completed_subtasks.store(0, std::memory_order_relaxed);
        slot.total_required_subtasks = 1;
        slot.logical_block_num = 1;
        slot.lifecycle_flags.store(COMPLETION_DONE, std::memory_order_relaxed);
        slot.payload = &payload;
        slot.task = &task;
    }

    void setup_ring_for_reclaim_race(int32_t ring_id, int32_t current_task_index, int32_t blocked_task_id) {
        SharedMemoryRingHeader &ring = sm_handle->header->rings[ring_id];
        SchedulerState::RingSchedState &ring_sched = sched.ring_sched_states[ring_id];

        ring.fc.current_task_index.store(current_task_index, std::memory_order_release);
        ring.fc.last_task_alive.store(0, std::memory_order_release);
        ring_sched.last_task_alive = 0;
        ring_sched.advance_lock.store(0, std::memory_order_release);
        sched.advance_pending_mask.store(0, std::memory_order_release);

        for (int32_t task_id = 0; task_id < current_task_index; task_id++) {
            ChipTaskState state = task_id < blocked_task_id ? CHIP_TASK_CONSUMED : CHIP_TASK_COMPLETED;
            init_ring_slot(ring, task_id, state, static_cast<uint8_t>(ring_id));
        }
    }

    void setup_contended_head_case(int32_t ring_id, int32_t head_task_id) {
        SharedMemoryRingHeader &ring = sm_handle->header->rings[ring_id];
        SchedulerState::RingSchedState &ring_sched = sched.ring_sched_states[ring_id];
        ring.fc.current_task_index.store(head_task_id + 2, std::memory_order_release);
        ring.fc.last_task_alive.store(head_task_id, std::memory_order_release);
        ring_sched.last_task_alive = head_task_id;
        ring_sched.advance_lock.store(0, std::memory_order_release);
        sched.advance_pending_mask.store(0, std::memory_order_release);
        init_ring_slot(ring, head_task_id, CHIP_TASK_COMPLETED, static_cast<uint8_t>(ring_id));
        init_ring_slot(ring, head_task_id + 1, CHIP_TASK_COMPLETED, static_cast<uint8_t>(ring_id), 1, 0);
    }
};

// =============================================================================
// check_and_handle_consumed
// =============================================================================

TEST_F(SchedulerStateTest, ConsumedNotReady) {
    alignas(64) ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_COMPLETED, 1, 2);
    slot.fanout_refcount.store(1);  // 1 != 2

    sched.check_and_handle_consumed(slot);
    EXPECT_EQ(slot.task_state.load(), CHIP_TASK_COMPLETED);
}

TEST_F(SchedulerStateTest, ConsumedTransition) {
    alignas(64) ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_COMPLETED, 1, 2);
    slot.fanout_refcount.store(2);  // matches fanout_count

    sched.check_and_handle_consumed(slot);
    EXPECT_EQ(slot.task_state.load(), CHIP_TASK_CONSUMED);
}

TEST_F(SchedulerStateTest, ConsumedHeadAdvancesAfterContendedAdvanceLock) {
    constexpr int32_t ring_id = CHIP_MAX_RING_DEPTH - 1;
    constexpr int32_t head_task_id = 0;
    setup_ring_for_reclaim_race(ring_id, /*current_task_index=*/1, head_task_id);

    SharedMemoryRingHeader &ring = sm_handle->header->rings[ring_id];
    SchedulerState::RingSchedState &ring_sched = sched.ring_sched_states[ring_id];
    ChipTaskSlotState &head = ring.get_slot_state_by_task_id(head_task_id);
    uint32_t pending_bit = SchedulerState::ring_advance_pending_bit(ring_id);

    ring_sched.advance_lock.store(1, std::memory_order_release);
    std::thread unlocker([&]() {
        while ((sched.advance_pending_mask.load(std::memory_order_acquire) & pending_bit) == 0) {
            std::this_thread::yield();
        }
        ring_sched.advance_lock.store(0, std::memory_order_release);
    });

    sched.check_and_handle_consumed(head);
    unlocker.join();

    EXPECT_TRUE(sched.drain_pending_ring_advances());
    EXPECT_EQ(head.task_state.load(std::memory_order_acquire), CHIP_TASK_CONSUMED);
    EXPECT_EQ(ring.fc.last_task_alive.load(std::memory_order_acquire), 1)
        << "a CONSUMED ring head must not remain pinned after advance_lock contention clears";
}

TEST_F(SchedulerStateTest, ConsumedNotCompletedState) {
    alignas(64) ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_PENDING, 1, 1);
    slot.fanout_refcount.store(1);

    sched.check_and_handle_consumed(slot);
    // CAS fails because state is PENDING, not COMPLETED
    EXPECT_EQ(slot.task_state.load(), CHIP_TASK_PENDING);
}

TEST_F(SchedulerStateTest, ConsumedIdempotent) {
    alignas(64) ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_CONSUMED, 1, 1);
    slot.fanout_refcount.store(1);

    sched.check_and_handle_consumed(slot);
    EXPECT_EQ(slot.task_state.load(), CHIP_TASK_CONSUMED);
}

TEST_F(SchedulerStateTest, ContendedConsumedHeadSetsPendingAndIdleDrainAdvances) {
    constexpr int32_t ring_id = CHIP_MAX_RING_DEPTH - 1;
    constexpr int32_t head_task_id = 17;
    setup_contended_head_case(ring_id, head_task_id);

    SharedMemoryRingHeader &ring = sm_handle->header->rings[ring_id];
    SchedulerState::RingSchedState &ring_sched = sched.ring_sched_states[ring_id];
    ChipTaskSlotState &head = ring.get_slot_state_by_task_id(head_task_id);
    uint32_t pending_bit = SchedulerState::ring_advance_pending_bit(ring_id);

    ring_sched.advance_lock.store(1, std::memory_order_release);
    sched.check_and_handle_consumed(head);

    EXPECT_EQ(head.task_state.load(std::memory_order_acquire), CHIP_TASK_CONSUMED);
    EXPECT_EQ(ring.fc.last_task_alive.load(std::memory_order_acquire), head_task_id);
    EXPECT_NE(sched.advance_pending_mask.load(std::memory_order_acquire) & pending_bit, 0u);

    EXPECT_FALSE(sched.drain_pending_ring_advances());
    EXPECT_EQ(ring.fc.last_task_alive.load(std::memory_order_acquire), head_task_id);
    EXPECT_NE(sched.advance_pending_mask.load(std::memory_order_acquire) & pending_bit, 0u);

    ring_sched.advance_lock.store(0, std::memory_order_release);
    EXPECT_TRUE(sched.drain_pending_ring_advances());
    EXPECT_EQ(ring_sched.last_task_alive, head_task_id + 1);
    EXPECT_EQ(ring.fc.last_task_alive.load(std::memory_order_acquire), head_task_id + 1);
    EXPECT_EQ(sched.advance_pending_mask.load(std::memory_order_acquire) & pending_bit, 0u);
}

TEST_F(SchedulerStateTest, ContendedConsumedHeadIdleDrainStress) {
    const int32_t head_task_ids[] = {0, 1, 127, CHIP_TASK_WINDOW_SIZE - 2, CHIP_TASK_WINDOW_SIZE + 3};

    for (int32_t ring_id = 0; ring_id < CHIP_MAX_RING_DEPTH; ring_id++) {
        for (int32_t head_task_id : head_task_ids) {
            SCOPED_TRACE(::testing::Message() << "ring_id=" << ring_id << " head_task_id=" << head_task_id);
            setup_contended_head_case(ring_id, head_task_id);

            SharedMemoryRingHeader &ring = sm_handle->header->rings[ring_id];
            SchedulerState::RingSchedState &ring_sched = sched.ring_sched_states[ring_id];
            ChipTaskSlotState &head = ring.get_slot_state_by_task_id(head_task_id);

            ring_sched.advance_lock.store(1, std::memory_order_release);
            sched.check_and_handle_consumed(head);
            ring_sched.advance_lock.store(0, std::memory_order_release);

            EXPECT_TRUE(sched.drain_pending_ring_advances());
            EXPECT_EQ(ring_sched.last_task_alive, head_task_id + 1);
            EXPECT_EQ(ring.fc.last_task_alive.load(std::memory_order_acquire), head_task_id + 1);
        }
    }
}

// =============================================================================
// release_producer
// =============================================================================

TEST_F(SchedulerStateTest, ReleaseProducerIncrements) {
    alignas(64) ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_COMPLETED, 1, 3);

    sched.release_producer(slot);
    EXPECT_EQ(slot.fanout_refcount.load(), 1);

    sched.release_producer(slot);
    EXPECT_EQ(slot.fanout_refcount.load(), 2);
}

TEST_F(SchedulerStateTest, ReleaseProducerTriggersConsumed) {
    alignas(64) ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_COMPLETED, 1, 2);
    slot.fanout_refcount.store(1);  // One away

    sched.release_producer(slot);
    EXPECT_EQ(slot.task_state.load(), CHIP_TASK_CONSUMED);
}

// =============================================================================
// on_subtask_complete
// =============================================================================

TEST_F(SchedulerStateTest, SubtaskCompleteSingle) {
    alignas(64) ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_PENDING, 1, 1);
    slot.total_required_subtasks = 1;
    slot.completed_subtasks.store(0);

    EXPECT_TRUE(sched.on_subtask_complete(slot));
}

TEST_F(SchedulerStateTest, SubtaskCompleteMultiBlock) {
    alignas(64) ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_PENDING, 1, 1);
    slot.total_required_subtasks = 6;  // 3 cores * 2 blocks
    slot.completed_subtasks.store(0);

    for (int i = 0; i < 5; i++) {
        EXPECT_FALSE(sched.on_subtask_complete(slot));
    }
    EXPECT_TRUE(sched.on_subtask_complete(slot));
}

// =============================================================================
// on_scope_end
// =============================================================================

TEST_F(SchedulerStateTest, ScopeEndBatchRelease) {
    constexpr int N = 4;
    alignas(64) ChipTaskSlotState slots[N];
    ChipTaskSlotState *ptrs[N];

    for (int i = 0; i < N; i++) {
        init_slot(slots[i], CHIP_TASK_COMPLETED, 1, 2);
        ptrs[i] = &slots[i];
    }

    sched.on_scope_end(ptrs, N);

    for (int i = 0; i < N; i++) {
        // on_scope_end releases the owning-scope ref via release_producer_scope,
        // which adds FANOUT_SCOPE_BIT (bit31) to fanout_refcount.
        EXPECT_EQ(slots[i].fanout_refcount.load(), FANOUT_SCOPE_BIT);
    }
}

// =============================================================================
// get_ready_tasks_batch: drains the shared ready queue
// =============================================================================

TEST_F(SchedulerStateTest, GetReadyTasksBatchDrainsSharedQueue) {
    alignas(64) ChipTaskSlotState slot_a, slot_b;
    // fanin_count = 1 so a single release_fanin_and_check_ready call drives each
    // slot to ready (new_refcount 0->1 == fanin_count) and enqueues it.
    init_slot(slot_a, CHIP_TASK_PENDING, 1, 1);
    init_slot(slot_b, CHIP_TASK_PENDING, 1, 1);

    // Route both slots into the global ready queue via the src API.
    ASSERT_TRUE(sched.release_fanin_and_check_ready(slot_a));
    ASSERT_TRUE(sched.release_fanin_and_check_ready(slot_b));

    ChipTaskSlotState *out[4];
    int count = sched.get_ready_tasks_batch(sched.ready_queues, ResourceShape::AIC, out, 4);

    EXPECT_EQ(count, 2);
    // Shared queue is FIFO, so slot_a (pushed first) comes first.
    EXPECT_EQ(out[0], &slot_a);
    EXPECT_EQ(out[1], &slot_b);
}

TEST_F(SchedulerStateTest, AutomaticSingleBlockAndStrictTasksHaveSeparateQueues) {
    alignas(64) ChipTaskSlotState automatic, strict;
    init_slot(automatic, CHIP_TASK_PENDING, 1, 1);
    init_slot(strict, CHIP_TASK_PENDING, 1, 1);
    automatic.task->target_aicpu_cluster = 0;
    strict.task->target_aicpu_cluster = 0;
    automatic.payload->placement.scope = 1;
    sched.placement_feedback.calibration_done.store(true);
    sched.push_ready_routed(&automatic);
    sched.push_ready_routed(&strict);

    ChipTaskSlotState *out[2];
    EXPECT_EQ(sched.get_ready_tasks_batch(sched.ready_queues, ResourceShape::AIC, out, 2), 0);
    EXPECT_EQ(sched.get_ready_tasks_batch(sched.cluster_ready_queues[1], ResourceShape::AIC, out, 2), 0);
    ASSERT_EQ(sched.get_ready_tasks_batch(sched.auto_ready_queues[0], ResourceShape::AIC, out, 2), 1);
    EXPECT_EQ(out[0], &automatic);
    ASSERT_EQ(sched.get_ready_tasks_batch(sched.cluster_ready_queues[0], ResourceShape::AIC, out, 2), 1);
    EXPECT_EQ(out[0], &strict);
}

TEST_F(SchedulerStateTest, FrozenCalibrationKeepsReadyAccountingWithoutSampleStamp) {
    alignas(64) ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_PENDING, 1, 1);
    slot.task->target_aicpu_cluster = 1;
    slot.logical_block_num = 16;
    slot.payload->placement.scope = 1;
    sched.placement_feedback.calibration_done.store(true);
    sched.stamp_ready_queue(slot);
    sched.stamp_ready_queue(slot);
    EXPECT_EQ(sched.placement_feedback.threads[simpler::tmr::PARTITION_THREADS - 1].summary[0].queued[1].load(), 16u);
    EXPECT_EQ(slot.payload->ready_queue_timestamp.load(), 0u);
    EXPECT_EQ(slot.payload->placement.release_tick, 0u);
}

TEST(CoreTrackerTest, MixPartiallyRunningClusterAdmittedAsPerCorePlacement) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr int32_t cluster_offset = 0;
    tracker.change_core_state(cluster_offset + 1);  // AIV0 running (unrelated task), AIC/AIV1 idle
    tracker.clear_pending_occupied(cluster_offset + 1);

    EXPECT_TRUE(tracker.is_aic_core_idle(cluster_offset));
    EXPECT_FALSE(tracker.is_aiv0_core_idle(cluster_offset));
    EXPECT_TRUE(tracker.is_aiv1_core_idle(cluster_offset));

    // A 1C2V MIX task is admitted on this partial cluster as a PENDING placement.
    // Per-core dispatch then puts the idle AIC/AIV1 on their running slots (marked
    // running so the completion poller tracks them) and the busy AIV0 on its pending
    // slot, executing after the in-flight AIV-only task.
    constexpr uint8_t used_mask = SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0 | SUBTASK_MASK_AIV1;
    EXPECT_EQ(tracker.classify_mix_cluster(cluster_offset, used_mask), CoreTracker::MixPlacement::PENDING);

    // Not all used cores are idle, so the IDLE phase skips this cluster; it is
    // consumed by the PENDING phase.
    auto idle = tracker.get_idle_core_offset_states(ResourceShape::MIX);
    EXPECT_FALSE(idle.has_value());
}

TEST(CoreTrackerTest, MixPendingAcceptsFullyRunningClusterWithFreePendingSlots) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr int32_t cluster_offset = 0;
    tracker.change_core_state(cluster_offset);
    tracker.change_core_state(cluster_offset + 1);
    tracker.change_core_state(cluster_offset + 2);
    tracker.clear_pending_occupied(cluster_offset);
    tracker.clear_pending_occupied(cluster_offset + 1);
    tracker.clear_pending_occupied(cluster_offset + 2);

    auto pending = tracker.get_pending_core_offset_states(ResourceShape::MIX);
    EXPECT_TRUE(pending.has_value());
    EXPECT_EQ(pending.count(), 1);
}

TEST(CoreTrackerTest, MixPendingRejectsFullyRunningClusterWithOccupiedPendingSlot) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr int32_t cluster_offset = 0;
    tracker.change_core_state(cluster_offset);
    tracker.change_core_state(cluster_offset + 1);
    tracker.change_core_state(cluster_offset + 2);
    tracker.set_pending_occupied(cluster_offset + 1);

    auto pending = tracker.get_pending_core_offset_states(ResourceShape::MIX);
    EXPECT_FALSE(pending.has_value());
}

TEST(CoreTrackerTest, MixIdleAndPendingDoNotDoubleAdmitFullyIdleCluster) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    auto idle = tracker.get_idle_core_offset_states(ResourceShape::MIX);
    EXPECT_TRUE(idle.has_value());
    EXPECT_EQ(idle.count(), 1);

    auto pending = tracker.get_pending_core_offset_states(ResourceShape::MIX);
    EXPECT_FALSE(pending.has_value());
}

TEST(CoreTrackerTest, MixClassifyIgnoresUnusedBusyCoreForRunningPlacement) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr int32_t cluster_offset = 0;
    tracker.change_core_state(cluster_offset + 2);  // AIV1 running, unused by this 1c1v task

    auto placement = tracker.classify_mix_cluster(cluster_offset, SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0);
    EXPECT_EQ(placement, CoreTracker::MixPlacement::RUNNING);
}

TEST(CoreTrackerTest, MixClassifyAllowsPendingForUsedRunningCoresOnly) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr int32_t cluster_offset = 0;
    tracker.change_core_state(cluster_offset);
    tracker.change_core_state(cluster_offset + 1);
    tracker.set_pending_occupied(cluster_offset + 2);  // Unused AIV1 must not block this 1c1v task

    auto placement = tracker.classify_mix_cluster(cluster_offset, SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0);
    EXPECT_EQ(placement, CoreTracker::MixPlacement::PENDING);
}

TEST(CoreTrackerTest, MixClassifyAdmitsMixedUsedCoresAsPending) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr int32_t cluster_offset = 0;
    tracker.change_core_state(cluster_offset + 1);  // AIV0 running while AIC is idle

    // Mixed used-core state (AIC idle, AIV0 running) is admitted as PENDING; the
    // idle AIC takes its running slot and the busy AIV0 takes its pending slot.
    auto placement = tracker.classify_mix_cluster(cluster_offset, SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0);
    EXPECT_EQ(placement, CoreTracker::MixPlacement::PENDING);
}

TEST(CoreTrackerTest, MixClassifyRejectsOccupiedPendingSlotInUsedMask) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr int32_t cluster_offset = 0;
    tracker.change_core_state(cluster_offset);
    tracker.change_core_state(cluster_offset + 1);
    tracker.set_pending_occupied(cluster_offset + 1);

    auto placement = tracker.classify_mix_cluster(cluster_offset, SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0);
    EXPECT_EQ(placement, CoreTracker::MixPlacement::REJECT);
}

TEST(CoreTrackerTest, MixRunningClusterHelpersUseActiveMask) {
    CoreTracker tracker;
    tracker.init(2);
    tracker.set_cluster(0, 0, 1, 2);
    tracker.set_cluster(1, 3, 4, 5);

    tracker.change_core_state(2);
    tracker.change_core_state(5);

    constexpr uint8_t used_mask = SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0;

    EXPECT_EQ(tracker.get_idle_core_offset_states(ResourceShape::MIX).count(), 0);
    EXPECT_EQ(tracker.count_mix_running_clusters(used_mask), 2);
    EXPECT_EQ(tracker.get_mix_running_cluster_offset_states(used_mask).count(), 2);
}

TEST(CoreTrackerTest, MixRunningClusterHelpersRejectOccupiedUsedPendingSlot) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr uint8_t used_mask = SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0;
    tracker.set_pending_occupied(1);

    EXPECT_EQ(tracker.count_mix_running_clusters(used_mask), 0);
}

TEST(CoreTrackerTest, CountAvailableBlocksAivIncludesFreePendingSlots) {
    CoreTracker tracker;
    tracker.init(2);
    tracker.set_cluster(0, 0, 1, 2);
    tracker.set_cluster(1, 3, 4, 5);

    tracker.change_core_state(1);
    tracker.change_core_state(4);

    constexpr uint8_t aiv_mask = SUBTASK_MASK_AIV0;
    constexpr int32_t exact_fit = 4;
    EXPECT_EQ(tracker.count_available_blocks(ResourceShape::AIV, aiv_mask, false), 2);
    EXPECT_EQ(tracker.count_available_blocks(ResourceShape::AIV, aiv_mask, true), exact_fit);

    tracker.set_pending_occupied(4);
    EXPECT_EQ(tracker.count_available_blocks(ResourceShape::AIV, aiv_mask, true), exact_fit - 1);
}

TEST(CoreTrackerTest, CountAvailableBlocksMixCountsLogicalClusters) {
    CoreTracker tracker;
    tracker.init(2);
    tracker.set_cluster(0, 0, 1, 2);
    tracker.set_cluster(1, 3, 4, 5);

    tracker.change_core_state(4);

    constexpr uint8_t used_mask = SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0 | SUBTASK_MASK_AIV1;
    EXPECT_EQ(tracker.count_available_blocks(ResourceShape::MIX, used_mask, false), 1);
    EXPECT_EQ(tracker.count_available_blocks(ResourceShape::MIX, used_mask, true), 2);
}

TEST(CoreTrackerTest, CountAvailableBlocksMixRejectsOccupiedUsedPendingSlot) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr uint8_t used_mask = SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0;
    tracker.change_core_state(1);
    tracker.set_pending_occupied(1);

    EXPECT_EQ(tracker.count_available_blocks(ResourceShape::MIX, used_mask, false), 0);
    EXPECT_EQ(tracker.count_available_blocks(ResourceShape::MIX, used_mask, true), 0);
}

TEST(CoreTrackerTest, CountAvailableBlocksMixIgnoresUnavailableUnusedCore) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr uint8_t used_mask = SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0;
    tracker.change_core_state(2);
    tracker.set_pending_occupied(2);

    EXPECT_EQ(tracker.count_available_blocks(ResourceShape::MIX, used_mask, false), 1);
    EXPECT_EQ(tracker.count_available_blocks(ResourceShape::MIX, used_mask, true), 1);
}

TEST_F(SchedulerStateTest, DeferredBindingUsesActualProducerAndAccountsExactlyOnce) {
    alignas(64) ChipTaskSlotState producer, consumer;
    init_slot(producer, CHIP_TASK_COMPLETED, 1, 1);
    init_slot(consumer, CHIP_TASK_PENDING, 1, 1);
    sched.placement_feedback.schedulers[0] = 2;
    sched.placement_feedback.schedulers[1] = 3;
    sched.placement_feedback.calibration_done.store(true);
    producer.payload->placement.preferred_cluster = 0;
    producer.payload->placement.dispatch_cluster.store(1);
    consumer.payload->placement.scope = 1;
    consumer.payload->placement.preferred_cluster = 0;
    consumer.task->target_aicpu_cluster = -1;
    // Sampling is frozen (tick=0); actual release origin is still required.
    sched.record_ready_origin(consumer, &producer, 0, 2);
    sched.stamp_ready_queue(consumer);
    sched.stamp_ready_queue(consumer);
    EXPECT_EQ(consumer.payload->placement.preferred_cluster, 0);
    EXPECT_EQ(consumer.task->target_aicpu_cluster, 1);
    auto &row = sched.placement_feedback.threads[2].summary[0];
    EXPECT_EQ(row.released[0].load(), 1u);
    EXPECT_EQ(row.queued[1].load(), 1u);
    EXPECT_EQ(row.queued[0].load(), 0u);
    // A subsequent foreign pickup changes execution affinity, never queue debit.
    consumer.payload->placement.dispatch_cluster.store(0);
    EXPECT_EQ(consumer.task->target_aicpu_cluster, 1);
    EXPECT_EQ(&sched.ready_queue_for(consumer, ResourceShape::AIC, false), &sched.auto_ready_queues[1][0]);
}

TEST_F(SchedulerStateTest, AutomaticMultiblockAndStrictTargetsCannotEnterBorrowableQueues) {
    alignas(64) ChipTaskSlotState wide, strict;
    init_slot(wide, CHIP_TASK_PENDING, 1, 1);
    init_slot(strict, CHIP_TASK_PENDING, 1, 1);
    wide.logical_block_num = 16;
    wide.payload->placement.scope = 1;
    wide.task->target_aicpu_cluster = 1;
    strict.task->target_aicpu_cluster = 1;
    EXPECT_EQ(&sched.ready_queue_for(wide, ResourceShape::AIV, false), &sched.cluster_ready_queues[1][1]);
    EXPECT_EQ(&sched.ready_queue_for(strict, ResourceShape::AIV, false), &sched.cluster_ready_queues[1][1]);
}

TEST_F(SchedulerStateTest, UnknownReleaseTimestampDoesNotChangeSchedulerOriginToOrchestrator) {
    alignas(64) ChipTaskSlotState producer, consumer;
    init_slot(producer, CHIP_TASK_COMPLETED, 1, 1);
    init_slot(consumer, CHIP_TASK_PENDING, 1, 1);
    sched.placement_feedback.schedulers[0] = 2;
    sched.placement_feedback.schedulers[1] = 3;
    sched.placement_feedback.orchestrator_cluster = 1;
    producer.payload->placement.dispatch_cluster.store(0);
    consumer.payload->placement.scope = 1;
    consumer.payload->placement.preferred_cluster = 1;
    // A shared producer can start its fanout walk before feedback is enabled.
    // Its missing sample timestamp does not make it an O-origin publication.
    sched.record_ready_origin(consumer, &producer, 0, 2);
    sched.stamp_ready_queue(consumer);
    EXPECT_EQ(consumer.payload->placement.reserved & simpler::tmr::PARTITION_FROM_O, 0u);
    EXPECT_EQ(consumer.payload->placement.release_cluster, 0);
    EXPECT_EQ(consumer.task->target_aicpu_cluster, 0);
}

// QKV owns a ready slot only after the lifecycle CAS. Each test exercises the
// production queues and production allocator, including the Shared escape.
TEST_F(SchedulerStateTest, QkvSingletonAndDuplicateReady) {
    sched.qkv.enabled = true;
    sched.qkv.schedulers = 2;
    for (int c = 0; c < 2; ++c) {
        sched.qkv.side[c] = c;
        for (int r = 0; r < 2; ++r) {
            sched.qkv.capacity[c][r] = 8;
            sched.qkv.threads[c].resources[r].store(8 | (8 << 16));
        }
    }
    ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_PENDING, 0, 0);
    EXPECT_TRUE(sched.route_ready_once(slot));
    EXPECT_FALSE(sched.route_ready_once(slot));
    EXPECT_EQ(sched.qkv_pending.size(), 1u);
    sched.service_qkv_batch();
    EXPECT_EQ(sched.qkv.batches[1], 1u);
    EXPECT_EQ(sched.qkv_pending.size(), 0u);
    EXPECT_EQ(sched.cluster_ready_queues[0][0].size(), 1u);
    EXPECT_EQ(sched.ready_queues[0].size(), 0u);
    int start = -1;
    EXPECT_EQ(slot.claim_block_range(1, 8, start), 1);
    EXPECT_EQ(start, 0);
    EXPECT_EQ(slot.claim_block_range(1, 8, start), 0);
}

TEST_F(SchedulerStateTest, QkvFullBatchUpdatesReservations) {
    sched.qkv.enabled = true;
    sched.qkv.schedulers = 2;
    for (int c = 0; c < 2; ++c) {
        sched.qkv.side[c] = c;
        for (int r = 0; r < 2; ++r) {
            sched.qkv.capacity[c][r] = 8;
            sched.qkv.threads[c].resources[r].store(8 | (8 << 16));
        }
    }
    ChipTaskSlotState slots[8];
    for (auto &slot : slots) {
        init_slot(slot, CHIP_TASK_PENDING, 0, 0);
        EXPECT_TRUE(sched.route_ready_once(slot));
    }
    sched.service_qkv_batch();
    EXPECT_EQ(sched.qkv.batches[8], 1u);
    EXPECT_EQ(sched.cluster_ready_queues[0][0].size(), 4u);
    EXPECT_EQ(sched.cluster_ready_queues[1][0].size(), 4u);
    EXPECT_EQ(slots[0].task->target_aicpu_cluster, 0);
    EXPECT_EQ(slots[1].task->target_aicpu_cluster, 1);
}

TEST_F(SchedulerStateTest, QkvQueueFullFallsBackOnce) {
    sched.qkv.enabled = true;
    sched.qkv.capacity[0][0] = sched.qkv.capacity[1][0] = 8;
    // Occupy queue tickets with a sentinel; the tested task is never inserted.
    ChipTaskSlotState sentinel;
    for (int i = 0; i < simpler::tmr::QKV_PENDING; ++i)
        EXPECT_TRUE(sched.qkv_pending.push(&sentinel));
    ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_PENDING, 0, 0);
    EXPECT_TRUE(sched.route_ready_once(slot));
    EXPECT_FALSE(sched.route_ready_once(slot));
    EXPECT_EQ(sched.ready_queues[0].size(), 1u);
    EXPECT_EQ(sched.qkv.fallback[0].load(), 1u);
    EXPECT_EQ(sched.qkv_pending.size(), 64u);
}

TEST_F(SchedulerStateTest, QkvUnavailableSnapshotAndNoCompatibleSide) {
    sched.qkv.enabled = true;
    sched.qkv.schedulers = 2;
    sched.qkv.side[0] = 0;
    sched.qkv.side[1] = 1;
    sched.qkv.capacity[0][0] = sched.qkv.capacity[1][0] = 8;
    ChipTaskSlotState slots[2];
    for (auto &slot : slots) {
        init_slot(slot, CHIP_TASK_PENDING, 0, 0);
        slot.logical_block_num = 2;
    }
    EXPECT_TRUE(sched.route_ready_once(slots[0]));
    sched.service_qkv_batch();
    EXPECT_EQ(sched.qkv.fallback[1].load(), 1u);
    // A valid summary whose compatible capacity is smaller than this task.
    for (int c = 0; c < 2; ++c)
        for (int r = 0; r < 2; ++r)
            sched.qkv.threads[c].resources[r].store(1 | (1 << 16));
    EXPECT_TRUE(sched.route_ready_once(slots[1]));
    sched.service_qkv_batch();
    EXPECT_EQ(sched.qkv.fallback[2].load(), 1u);
    EXPECT_EQ(sched.ready_queues[0].size(), 2u);
    EXPECT_EQ(sched.qkv_pending.size(), 0u);
}

TEST_F(SchedulerStateTest, QkvOversizeAndSyncPreserveOriginalQueues) {
    sched.qkv.enabled = true;
    sched.qkv.capacity[0][0] = sched.qkv.capacity[1][0] = 8;
    ChipTaskSlotState wide, sync;
    init_slot(wide, CHIP_TASK_PENDING, 0, 0);
    wide.logical_block_num = 9;
    EXPECT_TRUE(sched.route_ready_once(wide));
    init_slot(sync, CHIP_TASK_PENDING, 0, 0);
    sync.task_attrs.set_sync_start();
    EXPECT_TRUE(sched.route_ready_once(sync));
    EXPECT_EQ(sched.qkv_pending.size(), 0u);
    EXPECT_EQ(sched.ready_queues[0].size(), 1u);
    EXPECT_EQ(sched.ready_sync_queues[0].size(), 1u);
    EXPECT_EQ(sched.qkv.fallback[4].load(), 1u);
}

TEST(QkvFormula, KvStateOverridesLowerBaseLoadAndBatchChangesNextChoice) {
    simpler::tmr::QkvSnapshot s;
    for (int c = 0; c < 2; ++c)
        for (int r = 0; r < 2; ++r) {
            s.capacity[c][r] = 8;
            s.idle[c][r] = 8;
        }
    double reserved[2][2]{};
    EXPECT_EQ(simpler::tmr::qkv_choose(s, 0, 1, 0, reserved), 0);
    EXPECT_EQ(simpler::tmr::qkv_choose(s, 0, 1, 0, reserved), 1);
    reserved[0][0] = reserved[1][0] = 0;
    s.queued[1][0] = 1;  // base load prefers side 0 by 1/8
    s.occupied[0][0] = 1;
    s.idle[0][0] = 0;
    EXPECT_EQ(simpler::tmr::qkv_choose(s, 0, 1, 0, reserved), 1);
    s.occupied[0][0] = 0;
    s.idle[0][0] = 8;
    reserved[0][0] = reserved[1][0] = 0;
    EXPECT_EQ(simpler::tmr::qkv_choose(s, 0, 1, 0, reserved), 0);
}

TEST_F(SchedulerStateTest, QkvConcurrentLocalRangeClaimsAreUnique) {
    sched.qkv.enabled = true;
    sched.qkv.schedulers = 2;
    for (int c = 0; c < 2; ++c) {
        sched.qkv.side[c] = c;
        for (int r = 0; r < 2; ++r) {
            sched.qkv.capacity[c][r] = 8;
            sched.qkv.threads[c].resources[r].store(8 | (8 << 16));
        }
    }
    ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_PENDING, 0, 0);
    slot.logical_block_num = 8;
    EXPECT_TRUE(sched.route_ready_once(slot));
    sched.service_qkv_batch();
    auto &queue = sched.cluster_ready_queues[0][0];
    std::atomic<int> seen[8]{};
    std::atomic<int> done{0};
    std::thread workers[4];
    for (auto &worker : workers)
        worker = std::thread([&] {
            while (done.load(std::memory_order_acquire) < 8) {
                auto *next = queue.pop();
                if (!next) continue;
                int start = -1;
                int count = next->claim_block_range(8, 1, start);
                if (!count) continue;
                // Requeue before recording completion, as production dispatch does.
                if (start + count < 8) queue.push(next);
                seen[start].fetch_add(1, std::memory_order_relaxed);
                done.fetch_add(1, std::memory_order_release);
            }
        });
    for (auto &worker : workers)
        worker.join();
    for (auto &count : seen)
        EXPECT_EQ(count.load(), 1);
    EXPECT_EQ(queue.size(), 0u);
    EXPECT_EQ(sched.qkv_pending.size(), 0u);
}

TEST_F(SchedulerStateTest, QkvStrictPredicateAndEarlyRemainderStayOnOriginalPaths) {
    sched.qkv.enabled = true;
    sched.qkv.capacity[0][0] = sched.qkv.capacity[1][0] = 8;
    ChipTaskSlotState strict, predicated, early;
    init_slot(strict, CHIP_TASK_PENDING, 0, 0);
    strict.task->target_aicpu_cluster = 1;
    EXPECT_TRUE(sched.route_ready_once(strict));
    EXPECT_EQ(sched.cluster_ready_queues[1][0].size(), 1u);
    init_slot(predicated, CHIP_TASK_PENDING, 0, 0);
    predicated.task_attrs.set_predicate();
    EXPECT_TRUE(sched.route_ready_once(predicated));
    init_slot(early, CHIP_TASK_PENDING, 0, 0);
    early.logical_block_num = 2;
    early.next_block_idx.store(1);
    early.payload->early_dispatch_state.store(EARLY_DISPATCH_STAGING);
    // No staged mask: this represents an early remainder after a prior claim.
    EXPECT_TRUE(sched.route_ready_once(early));
    EXPECT_FALSE(sched.route_ready_once(early));
    EXPECT_EQ(sched.qkv_pending.size(), 0u);
    EXPECT_EQ(sched.qkv.eligible.load(), 0u);
    EXPECT_EQ(early.payload->placement.reserved, 0);
}

TEST(QkvFormula, ReducedLogitsPreservePressureAndSequentialChoices) {
    using namespace simpler::tmr;
    uint32_t rng = 1234;
    auto next = [&]() {
        rng = rng * 1664525u + 1013904223u;
        return rng;
    };
    for (int trial = 0; trial < 2000; ++trial) {
        QkvSnapshot snapshot;
        for (int c = 0; c < 2; ++c) {
            snapshot.capacity[c][0] = 8 + next() % 25;
            snapshot.queued[c][0] = next() % 100;
            snapshot.occupied[c][0] = double(next() % 101) / 100;
            snapshot.idle[c][0] = next() % (int(snapshot.capacity[c][0]) + 1);
        }
        QkvBatchScorer fast;
        double old_reserved[2][2]{}, new_reserved[2][QKV_CHANNELS]{};
        for (int i = 0; i < 8; ++i) {
            int blocks = 1 + next() % 8;
            double age = double(next() % 1001) / 1000;
            for (int c = 0; c < 2; ++c) {
                QkvFastRow row;
                row.init(snapshot, c, 0, blocks);
                EXPECT_NEAR(row.pressure(age), qkv_pressure(snapshot, c, 0, blocks, age), 1e-12);
            }
            EXPECT_EQ(
                qkv_choose(snapshot, 0, blocks, age, old_reserved), fast.choose(snapshot, 0, blocks, age, new_reserved)
            );
        }
    }
}

TEST_F(SchedulerStateTest, QkvImprovedMixMasksUseDistinctCompatibleGroupLedgers) {
    using namespace simpler::tmr;
    sched.qkv.enabled = true;
    sched.qkv.mode = 3;
    sched.qkv.schedulers = 2;
    auto stamp = get_sys_cnt_aicpu();
    for (int c = 0; c < 2; ++c) {
        sched.qkv.side[c] = c;
        sched.qkv.threads[c].stamp.store(stamp);
        sched.qkv.threads[c].version.store(2);
        for (int r = 0; r < QKV_CHANNELS; ++r) {
            sched.qkv.capacity[c][r] = 8;
            sched.qkv.threads[c].resources[r].store(8 | (8 << 8) | (8 << 16));
            sched.qkv.idle_hint[c][r].store((stamp << 8) | 8);
        }
    }
    ChipTaskSlotState slots[4];
    const int masks[4] = {3, 5, 6, 7};
    for (int i = 0; i < 4; ++i) {
        init_slot(slots[i], CHIP_TASK_PENDING, 0, 0);
        slots[i].active_mask = ActiveMask(masks[i]);
        EXPECT_TRUE(sched.route_ready_once(slots[i]));
        EXPECT_FALSE(sched.route_ready_once(slots[i]));
    }
    EXPECT_EQ(sched.qkv_pending.size(), 4u);
    sched.service_qkv_batch();
    EXPECT_EQ(sched.qkv_pending.size(), 0u);
    EXPECT_EQ(sched.qkv.shape_assigned[2], 4u);
    EXPECT_EQ(sched.cluster_ready_queues[0][2].size(), 4u);
    for (int mask : masks)
        EXPECT_EQ(sched.qkv.assigned[0][qkv_channel(2, mask)], 1u);
}

TEST_F(SchedulerStateTest, QkvImprovedMissingStaleAndBusyHintsFallBackExactlyOnce) {
    using namespace simpler::tmr;
    sched.qkv.enabled = true;
    sched.qkv.mode = 3;
    sched.qkv.capacity[0][0] = sched.qkv.capacity[1][0] = 8;
    ChipTaskSlotState slots[3];
    for (int i = 0; i < 3; ++i) {
        init_slot(slots[i], CHIP_TASK_PENDING, 0, 0);
        uint64_t tick = get_sys_cnt_aicpu();
        for (int c = 0; c < 2; ++c) {
            uint64_t hint = i == 0 ? 0 : (i == 1 ? ((tick - 100000) << 8) | 8 : tick << 8);
            sched.qkv.idle_hint[c][0].store(hint);
        }
        EXPECT_TRUE(sched.route_ready_once(slots[i]));
        EXPECT_FALSE(sched.route_ready_once(slots[i]));
    }
    EXPECT_EQ(sched.qkv_pending.size(), 0u);
    EXPECT_EQ(sched.ready_queues[0].size(), 3u);
    EXPECT_EQ(sched.qkv.fallback[int(QkvFallback::Uninitialized)].load(), 1u);
    EXPECT_EQ(sched.qkv.fallback[int(QkvFallback::Stale)].load(), 1u);
    EXPECT_EQ(sched.qkv.fallback[int(QkvFallback::Busy)].load(), 1u);
}

TEST(QkvLearned, FrozenProjectionMatchesTorchAndMissingSpecUsesFormula) {
    using namespace simpler::tmr;
    QkvSnapshot s;
    for (int c = 0; c < 2; ++c) {
        s.capacity[c][1] = 30;
        s.queued[c][1] = 30;
        s.occupied[c][1] = .5;
        s.idle[c][1] = 3;
    }
    int kernels[3] = {-1, 4, -1};
    const double expected_pytorch_score = 136.84021918183097;
    EXPECT_NEAR(qkv_learned_score(s, 0, 1, 1, 2, kernels, .1, 6), expected_pytorch_score, 1e-4);
    int missing[3] = {999, -1, -1};
    EXPECT_LT(learned::runtime(missing, 1, 1, 0), 0);
    QkvBatchScorer formula, fallback;
    double a[2][QKV_CHANNELS]{}, b[2][QKV_CHANNELS]{};
    EXPECT_EQ(qkv_learned_choose(s, 1, 1, 2, missing, .1, a, fallback), formula.choose(s, 1, 1, .1, b));
    EXPECT_EQ(a[0][1], b[0][1]);
}

TEST(StreamingQueueTest, ObservationDoesNotClaimAndOldTicketCannotConsumeRecycledSlot) {
    ChipReadyQueue queue{};
    ChipReadyQueueSlot slots[2];
    queue.slots = slots;
    queue.capacity = 2;
    queue.mask = 1;
    for (int i = 0; i < 2; ++i)
        slots[i].sequence.store(i);
    ChipTaskSlotState items[4]{};
    ASSERT_TRUE(queue.push(&items[0]));
    ChipReadyQueue::HeadToken old;
    ASSERT_TRUE(queue.observe_head(old));
    EXPECT_EQ(queue.size(), 1u);
    EXPECT_EQ(queue.consume_exact(old), &items[0]);
    ASSERT_TRUE(queue.push(&items[1]));
    EXPECT_EQ(queue.pop(), &items[1]);
    ASSERT_TRUE(queue.push(&items[2]));
    EXPECT_EQ(queue.consume_exact(old), nullptr);
    EXPECT_EQ(queue.pop(), &items[2]);
    EXPECT_EQ(queue.pop(), nullptr);
}

TEST(StreamingQueueTest, CompetingConsumersAcquireExactlyOneTicket) {
    ChipReadyQueue queue{};
    ChipReadyQueueSlot slots[2];
    queue.slots = slots;
    queue.capacity = 2;
    queue.mask = 1;
    for (int i = 0; i < 2; ++i)
        slots[i].sequence.store(i);
    ChipTaskSlotState item{};
    ASSERT_TRUE(queue.push(&item));
    ChipReadyQueue::HeadToken token;
    ASSERT_TRUE(queue.observe_head(token));
    std::atomic<int> wins{0};
    std::thread a([&] {
        if (queue.consume_exact(token)) ++wins;
    });
    std::thread b([&] {
        if (queue.consume_exact(token)) ++wins;
    });
    a.join();
    b.join();
    EXPECT_EQ(wins.load(), 1);
    EXPECT_EQ(queue.size(), 0u);
}

TEST(StreamingQueueTest, FullFallbackPublishesOnlyToOneQueue) {
    ChipReadyQueue queues[2]{};
    ChipReadyQueueSlot slots[2][2];
    ChipTaskSlotState items[3]{};
    for (int q = 0; q < 2; ++q) {
        queues[q].slots = slots[q];
        queues[q].capacity = 2;
        queues[q].mask = 1;
        for (int i = 0; i < 2; ++i)
            slots[q][i].sequence.store(i);
    }
    ASSERT_TRUE(queues[1].push(&items[0]));
    ASSERT_TRUE(queues[1].push(&items[1]));
    if (!queues[1].push(&items[2])) ASSERT_TRUE(queues[0].push(&items[2]));
    EXPECT_EQ(queues[0].pop(), &items[2]);
    EXPECT_EQ(queues[0].pop(), nullptr);
    EXPECT_EQ(queues[1].pop(), &items[0]);
    EXPECT_EQ(queues[1].pop(), &items[1]);
    EXPECT_EQ(queues[1].pop(), nullptr);
}

TEST(StreamingQueueTest, ConcurrentObservationSurvivesWrapAndOrdinaryConsumers) {
    ChipReadyQueue queue{};
    ChipReadyQueueSlot slots[8];
    ChipTaskSlotState items[8]{};
    auto payloads = std::make_unique<TaskPayload[]>(8);
    queue.slots = slots;
    queue.capacity = 8;
    queue.mask = 7;
    queue.observe_metadata = 1;
    for (int i = 0; i < 8; ++i) {
        slots[i].sequence.store(i);
        items[i].payload = &payloads[i];
        payloads[i].placement.group = i + 1;
    }
    constexpr int total = 20000;
    std::atomic<int> consumed{0}, bad_record{0};
    std::thread producer([&] {
        for (int i = 0; i < total; ++i)
            while (!queue.push(&items[i % 8])) {}
    });
    std::thread observer([&] {
        while (consumed.load() < total) {
            ChipReadyQueue::HeadToken token;
            if (!queue.observe_head(token)) continue;
            auto *owned = queue.consume_exact(token);
            if (owned) {
                if (owned->payload->placement.group != token.record) ++bad_record;
                ++consumed;
            }
        }
    });
    std::thread ordinary([&] {
        while (consumed.load() < total)
            if (queue.pop()) ++consumed;
    });
    producer.join();
    observer.join();
    ordinary.join();
    EXPECT_EQ(consumed.load(), total);
    EXPECT_EQ(bad_record.load(), 0);
    EXPECT_EQ(queue.size(), 0u);
}

TEST(StreamingActionTest, VisibleDiamondChangesChoiceButUnknownAndIncompleteDoNot) {
    using namespace simpler::tmr;
    StreamCandidate tasks[2] = {{1, 0, {2, 10}, 0}, {2, 0, {4, 14}, 0}};
    StreamSlot slots[2] = {{14, 0, 0}, {0, 1, 0}};
    StreamJoin unknown;
    auto local = stream_compare(tasks, 2, 1, slots, 2, 1, 0, unknown);
    ASSERT_TRUE(local.valid);
    EXPECT_DOUBLE_EQ(local.default_end, 24);
    EXPECT_DOUBLE_EQ(local.alternative_end, 16);
    EXPECT_TRUE(stream_admit(local.gain, 0, 30, 0));
    StreamJoin pending;
    pending.count = 2;
    pending.candidate[0] = 1;
    pending.committed_finish_us[1] = 30;
    auto incomplete = stream_compare(tasks, 2, 1, slots, 2, 1, 0, pending);
    EXPECT_DOUBLE_EQ(incomplete.gain, local.gain);
    pending.complete = true;
    auto slow = stream_compare(tasks, 2, 1, slots, 2, 1, 0, pending);
    EXPECT_DOUBLE_EQ(slow.gain, 0);
    EXPECT_TRUE(slow.used_join);
    EXPECT_FALSE(stream_admit(slow.gain, 0, 30, 0));
    pending.committed_finish_us[1] = 12;
    auto fast = stream_compare(tasks, 2, 1, slots, 2, 1, 0, pending);
    EXPECT_DOUBLE_EQ(fast.gain, 8);
    EXPECT_FALSE(stream_admit(fast.gain, 0, 6, 0));
}

TEST_F(SchedulerStateTest, StreamingFastReadyAndNotificationHaveOnePublicationOwner) {
    sched.qkv.mode = 7;
    alignas(64) ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_PENDING, 1, 1);
    slot.task->task_id = TaskId::make(0, 1);
    sched.qkv_trace_create(slot);
    sched.qkv_registration_complete(slot);
    sched.push_ready_routed(&slot);
    sched.push_ready_routed(&slot);
    EXPECT_FALSE(sched.route_ready_once(slot));
    EXPECT_EQ(sched.ready_queues[0].size(), 1u);
    EXPECT_EQ(sched.qkv.eligible.load(), 1u);
    EXPECT_GT(sched.qkv.trace.tasks[0].registered_epoch.load(), sched.qkv.trace.tasks[0].submit_epoch);
    EXPECT_EQ(sched.ready_queues[0].pop(), &slot);
    int32_t start = -1;
    EXPECT_EQ(slot.claim_block_range(1, 1, start), 1);
    EXPECT_EQ(slot.claim_block_range(1, 1, start), 0);
}

TEST_F(SchedulerStateTest, StreamingPreservesStrictAndSyncQueues) {
    sched.qkv.mode = 7;
    alignas(64) ChipTaskSlotState strict, sync;
    init_slot(strict, CHIP_TASK_PENDING, 1, 1);
    init_slot(sync, CHIP_TASK_PENDING, 1, 1);
    strict.task->target_aicpu_cluster = 1;
    sync.task_attrs.set_sync_start();
    sched.push_ready_routed(&strict);
    sched.push_ready_routed(&sync);
    EXPECT_EQ(sched.cluster_ready_queues[1][0].pop(), &strict);
    EXPECT_EQ(sched.ready_sync_queues[0].pop(), &sync);
    EXPECT_EQ(sched.ready_queues[0].size(), 0u);
    EXPECT_EQ(sched.qkv.eligible.load(), 0u);
}

TEST_F(SchedulerStateTest, StreamingPreferenceCannotConsumeLegacyAutomaticWork) {
    sched.qkv.mode = 7;
    alignas(64) ChipTaskSlotState legacy;
    init_slot(legacy, CHIP_TASK_PENDING, 1, 1);
    legacy.payload->placement.scope = 1;
    legacy.task->target_aicpu_cluster = 1;
    ASSERT_TRUE(sched.auto_ready_queues[1][0].push(&legacy));
    ChipReadyQueue::HeadToken token;
    EXPECT_FALSE(sched.streaming_lane(ResourceShape::AIC, 2).observe_head(token));
    EXPECT_EQ(sched.auto_ready_queues[1][0].size(), 1u);
}
