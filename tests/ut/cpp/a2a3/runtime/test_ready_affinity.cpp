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
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include "ready_affinity.h"

using namespace simpler::tmr;
namespace {
void setup(ReadyFeedback &f) {
    f.reset();
    f.schedulers[0] = f.schedulers[1] = 1;
    f.cluster[0] = 0;
    f.cluster[1] = 1;
    for (int c = 0; c < 2; ++c) {
        f.writers[c].rows[0].capacity.store(8);
        f.publish_side(c);
    }
}
TEST(OQuota, BalancedCapacitySplitsLargeTask) {
    ReadyFeedback f;
    setup(f);
    ReadyTaskState t;
    f.predict(0, t, 32, true);
    EXPECT_EQ(t.quota[0], 16);
    EXPECT_EQ(t.quota[1], 16);
}
TEST(OQuota, CapacityRatioAndPressureChangeAllocation) {
    ReadyFeedback f;
    setup(f);
    f.writers[1].rows[0].capacity.store(16);
    f.publish_side(1);
    ReadyTaskState t;
    f.predict(0, t, 24, true);
    EXPECT_EQ(t.quota[0], 8);
    EXPECT_EQ(t.quota[1], 16);
    setup(f);
    f.writers[0].rows[0].enqueued[0].store(16);
    f.assigned[0][0] = 16;
    f.publish_side(0);
    t.reset();
    f.predict(0, t, 32, true);
    EXPECT_EQ(t.quota[0], 8);
    EXPECT_EQ(t.quota[1], 24);
}
TEST(OQuota, ClaimingDoesNotEraseRunningPressure) {
    ReadyFeedback f;
    setup(f);
    f.assigned[0][0] = 16;
    f.writers[0].rows[0].enqueued[0].store(16);
    f.writers[0].rows[0].claimed[0].store(16);
    f.publish_side(0);
    ReadyTaskState t;
    f.predict(0, t, 32, true);
    EXPECT_EQ(t.quota[0], 8);
    setup(f);
    f.assigned[0][0] = 16;
    auto &r = f.writers[0].rows[0];
    r.enqueued[0].store(16);
    r.claimed[0].store(16);
    r.finished[0].store(96);
    f.publish_side(0);
    t.reset();
    f.predict(0, t, 32, true);
    EXPECT_EQ(t.quota[0], 16);
}
TEST(OQuota, SingleBlockAndStrongAffinityRemainWhole) {
    ReadyFeedback f;
    setup(f);
    ReadyTaskState t;
    t.affinity_votes = pack_votes(256, 0);
    f.predict(0, t, 1, true);
    EXPECT_EQ(t.quota[0], 1);
    EXPECT_EQ(t.quota[1], 0);
}
TEST(OQuota, NoSchedulerMeansNoAllocation) {
    ReadyFeedback f;
    f.reset();
    f.schedulers[1] = 2;
    ReadyTaskState t;
    f.predict(0, t, 32, true);
    EXPECT_EQ(t.quota[0], 0);
    EXPECT_EQ(t.quota[1], 32);
}
TEST(OQuota, ReadyPublicationDoesNotChangeQuotaAndIsSeparateFromReservation) {
    ReadyFeedback f;
    setup(f);
    ReadyTaskState t;
    f.predict(0, t, 32, true);
    f.bind(2, 0, t);
    EXPECT_EQ(f.writers[2].rows[0].enqueued[0].load(), 16);
    EXPECT_EQ(f.writers[2].rows[0].enqueued[1].load(), 16);
    EXPECT_EQ(f.assigned[0][0], 16);
    EXPECT_EQ(t.quota[0], 16);
}
TEST(OQuota, OrchestratorReadyRowIsNeverDoubleCounted) {
    ReadyFeedback f;
    setup(f);
    f.cluster[5] = 0;
    f.assigned[0][0] = 16;
    f.writers[5].rows[0].enqueued[0].store(16);
    f.publish_side(0);
    ReadyTaskState t;
    f.predict(0, t, 32, true);
    EXPECT_EQ(t.quota[0], 8);
}
TEST(OQuota, SnapshotReplacementAndBusyVersionPreserveLastView) {
    ReadyFeedback f;
    setup(f);
    f.refresh();
    EXPECT_EQ(f.views[0][0].capacity, 8);
    f.summaries[0].version.store(3);
    f.summaries[0].rows[0].capacity.store(100);
    f.refresh();
    EXPECT_EQ(f.views[0][0].capacity, 8);
    f.summaries[0].version.store(4);
    f.refresh();
    f.refresh();
    EXPECT_EQ(f.views[0][0].capacity, 100);
}
TEST(OQuota, ConcurrentRangesCoverEveryBlockOnceWithoutCrossingQuota) {
    ReadyTaskState t;
    t.quota[0] = 333;
    t.quota[1] = 667;
    t.next[1].store(333);
    std::atomic<int> seen[1000]{};
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i)
        threads.emplace_back([&, i] {
            int side = i % 2, start;
            for (;;) {
                int n = t.claim(side, 7, start);
                if (!n) break;
                EXPECT_GE(start, side ? 333 : 0);
                EXPECT_LE(start + n, side ? 1000 : 333);
                for (int j = start; j < start + n; ++j)
                    ++seen[j];
            }
        });
    for (auto &th : threads)
        th.join();
    for (auto &v : seen)
        EXPECT_EQ(v.load(), 1);
}
TEST(OQuota, ConservationForAllSmallTaskSizesAndAsymmetricCapacity) {
    for (int n = 1; n <= 128; ++n) {
        ReadyFeedback f;
        setup(f);
        f.writers[1].rows[0].capacity.store(12);
        f.publish_side(1);
        ReadyTaskState t;
        f.predict(0, t, n, true);
        EXPECT_EQ(t.quota[0] + t.quota[1], n);
        EXPECT_GE(t.quota[0], 0);
        EXPECT_GE(t.quota[1], 0);
        int start;
        int left = t.claim(0, n, start);
        EXPECT_EQ(start, 0);
        EXPECT_EQ(left, t.quota[0]);
        int right = t.claim(1, n, start);
        EXPECT_EQ(start, left);
        EXPECT_EQ(right, t.quota[1]);
    }
}
TEST(OQuota, CreationAffinityUsesActualDistributionWithoutWaitVotes) {
    ReadyTaskState parent, child;
    parent.dispatched[0].store(6);
    parent.dispatched[1].store(2);
    AffinityVote vote;
    vote.add(parent, -1, 1);
    vote.finish(child, 1);
    EXPECT_EQ(side_vote(child.affinity_votes, 0), 192);
    EXPECT_EQ(side_vote(child.affinity_votes, 1), 64);
}
TEST(OQuota, PredicatedTasksKeepOneRange) {
    ReadyFeedback f;
    setup(f);
    ReadyTaskState t;
    f.predict(0, t, 128, true, false);
    EXPECT_TRUE(t.quota[0] == 0 || t.quota[1] == 0);
    EXPECT_EQ(t.quota[0] + t.quota[1], 128);
}
TEST(OQuota, ResetDropsOldCursorsAndSnapshots) {
    ReadyFeedback f;
    setup(f);
    ReadyTaskState t;
    f.predict(0, t, 32, true);
    int start;
    t.claim(0, 8, start);
    t.reset();
    EXPECT_EQ(t.next[0].load(), 0);
    EXPECT_EQ(t.next[1].load(), 0);
    EXPECT_EQ(t.quota[0] + t.quota[1], 0);
    f.reset();
    EXPECT_EQ(f.views[0][0].capacity, 0);
    EXPECT_EQ(f.assigned[0][0], 0);
    EXPECT_EQ(f.split_tasks, 0);
}
}  // namespace
