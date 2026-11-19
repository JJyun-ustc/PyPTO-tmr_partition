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

#include "ready_affinity.h"

using namespace simpler::tmr;

TEST(ReadyAffinity, MultipleActualParentsFormStrongConsensus) {
    ReadyTaskState a, b, task;
    a.dispatch_cluster.store(1);
    b.dispatch_cluster.store(1);
    AffinityVote vote;
    vote.add(a, -1, 3);
    vote.add(b, -1, 3);
    vote.finish(task, 3);
    EXPECT_EQ(task.preferred_cluster, 1);
    EXPECT_EQ(task.strength, AffinityStrength::STRONG);
}
TEST(ReadyAffinity, ConflictingActualParentsRemainNeutral) {
    ReadyTaskState a, b, task;
    a.dispatch_cluster.store(0);
    b.dispatch_cluster.store(1);
    AffinityVote vote;
    vote.add(a, -1, 3);
    vote.add(b, -1, 3);
    vote.finish(task, 3);
    EXPECT_EQ(task.preferred_cluster, -1);
}
TEST(ReadyAffinity, InferredPreferenceCannotBecomeStrong) {
    ReadyTaskState parent, task;
    parent.scope = 3;
    parent.preferred_cluster = 0;
    parent.strength = AffinityStrength::STRONG;
    AffinityVote vote;
    vote.add(parent, -1, 3);
    vote.finish(task, 3);
    EXPECT_EQ(task.strength, AffinityStrength::WEAK);
    EXPECT_EQ(task.hops, 1);
}
TEST(ReadyAffinity, InferredPreferenceExpiresAcrossScopeAndHopLimit) {
    ReadyTaskState parent, task;
    parent.scope = 3;
    parent.preferred_cluster = 0;
    AffinityVote cross_scope;
    cross_scope.add(parent, -1, 4);
    cross_scope.finish(task, 4);
    EXPECT_EQ(task.preferred_cluster, -1);
    parent.hops = 7;
    AffinityVote old;
    old.add(parent, -1, 3);
    old.finish(task, 3);
    EXPECT_EQ(task.preferred_cluster, -1);
}
TEST(ReadyAffinity, ActualExecutionOverridesOldPreference) {
    ReadyTaskState parent, task;
    parent.scope = 3;
    parent.preferred_cluster = 0;
    parent.dispatch_cluster.store(1);
    AffinityVote vote;
    vote.add(parent, -1, 3);
    vote.finish(task, 3);
    EXPECT_EQ(task.preferred_cluster, 1);
    EXPECT_EQ(task.strength, AffinityStrength::STRONG);
}
TEST(ReadyAffinity, UnknownParentWeakensConsensus) {
    ReadyTaskState known, unknown, task;
    known.dispatch_cluster.store(0);
    AffinityVote vote;
    vote.add(known, -1, 3);
    vote.add(unknown, -1, 3);
    vote.finish(task, 3);
    EXPECT_EQ(task.strength, AffinityStrength::WEAK);
}
TEST(ReadyAffinity, CurrentMultiblockSizeAffectsSideChoice) {
    uint64_t work[2]{0, 8}, cap[2]{10, 20}, free[2]{1, 1};
    EXPECT_EQ(choose_ready_side(-1, AffinityStrength::NEUTRAL, -1, work, cap, free, 1), 0);
    EXPECT_EQ(choose_ready_side(-1, AffinityStrength::NEUTRAL, -1, work, cap, free, 48), 1);
}
TEST(ReadyAffinity, PressureCanOverrideStrongAffinity) {
    uint64_t work[2]{100, 0}, cap[2]{10, 10}, free[2]{1, 1};
    EXPECT_EQ(choose_ready_side(0, AffinityStrength::STRONG, 0, work, cap, free, 1), 1);
}
TEST(ReadyAffinity, StrongConsensusOverridesLastNotifierWhenBalanced) {
    uint64_t work[2]{0, 0}, cap[2]{10, 10}, free[2]{1, 1};
    EXPECT_EQ(choose_ready_side(0, AffinityStrength::STRONG, 1, work, cap, free, 1), 0);
}
TEST(ReadyAffinity, FreeResourceHintOverridesWeakAffinity) {
    uint64_t work[2]{0, 0}, cap[2]{10, 10}, free[2]{0, 1};
    EXPECT_EQ(choose_ready_side(0, AffinityStrength::WEAK, 0, work, cap, free, 1), 1);
}
TEST(ReadyAffinity, CapacityZeroExcludesUnavailableSide) {
    uint64_t work[2]{0, 0}, cap[2]{0, 10}, free[2]{0, 1};
    EXPECT_EQ(choose_ready_side(0, AffinityStrength::STRONG, 0, work, cap, free, 48), 1);
}
TEST(ReadyAffinity, BatchOwnPublicationsAreImmediatelyVisible) {
    ReadyFeedback feedback;
    feedback.schedulers[0] = feedback.schedulers[1] = 1;
    ReadyTaskState task;
    task.preferred_cluster = 0;
    int first = feedback.bind(2, 0, task, 10);
    EXPECT_EQ(first, 0);
    ReadyRow::add(feedback.writers[2].rows[0].enqueued[0], 10);
    EXPECT_EQ(feedback.bind(2, 0, task, 10), 1);
    EXPECT_EQ(feedback.writers[2].refreshes, 1);
}
TEST(ReadyAffinity, SnapshotWatermarksDoNotDoubleCountOwnHistory) {
    ReadyFeedback feedback;
    feedback.schedulers[0] = feedback.schedulers[1] = 1;
    auto &row = feedback.writers[2].rows[0];
    row.enqueued[0].store(100);
    row.claimed[0].store(100);
    feedback.hints[0][0].enqueued_cut[2].store(100);
    feedback.hints[0][0].claimed_cut[2].store(100);
    ReadyTaskState task;
    task.preferred_cluster = 0;
    EXPECT_EQ(feedback.bind(2, 0, task, 1), 0);
    ReadyRow::add(row.enqueued[0], 8);
    EXPECT_EQ(feedback.bind(2, 0, task, 1), 1);
    ReadyRow::add(row.claimed[0], 8);
    EXPECT_EQ(feedback.bind(2, 0, task, 1), 0);
}
TEST(ReadyAffinity, RacingSnapshotRetainsPreviousViewWithoutSpinning) {
    ReadyFeedback feedback;
    feedback.schedulers[0] = feedback.schedulers[1] = 1;
    ReadyTaskState task;
    task.preferred_cluster = 0;
    EXPECT_EQ(feedback.bind(2, 0, task, 1), 0);
    feedback.hints[0][0].version.store(1);
    feedback.hints[0][0].work.store(1000);
    feedback.begin_batch(2);
    EXPECT_EQ(feedback.bind(2, 0, task, 1), 0);
}
TEST(ReadyAffinity, RemoteClaimDebitsQueueSideAndRecordsExecutionSeparately) {
    ReadyRow row;
    ReadyRow::add(row.enqueued[0], 1);
    ReadyRow::add(row.claimed[0], 1);
    ReadyRow::add(row.executed[1], 1);
    EXPECT_EQ(row.enqueued[0].load(), row.claimed[0].load());
    EXPECT_EQ(row.enqueued[1].load(), 0);
    EXPECT_EQ(row.executed[1].load(), 1);
}
TEST(ReadyAffinity, RemoteProbeRequiresLocalEmptyAndRemoteBacklog) {
    EXPECT_FALSE(may_remote_pickup(1, 100, 2));
    EXPECT_FALSE(may_remote_pickup(0, 5, 2));
    EXPECT_TRUE(may_remote_pickup(0, 6, 2));
}
