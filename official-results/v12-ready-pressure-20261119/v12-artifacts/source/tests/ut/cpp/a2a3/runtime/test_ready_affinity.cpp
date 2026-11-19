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

TEST(ReadyAffinity, MajoritySurvivesOneRemoteParent) {
    ReadyTaskState a, b, task;
    a.dispatched[0].store(1);
    b.dispatched[1].store(1);
    AffinityVote vote;
    for (int i = 0; i < 100; ++i)
        vote.add(a, -1, 3);
    vote.add(b, -1, 3);
    vote.finish(task, 3);
    EXPECT_EQ(task.preferred_cluster, 0);
    EXPECT_GT(side_vote(task.affinity_votes, 0), side_vote(task.affinity_votes, 1));
}
TEST(ReadyAffinity, SplitProducerUsesActualDistribution) {
    ReadyTaskState parent, task;
    parent.preferred_cluster = 0;
    parent.dispatched[0].store(2);
    parent.dispatched[1].store(6);
    AffinityVote vote;
    vote.add(parent, -1, 3);
    vote.finish(task, 3);
    EXPECT_EQ(task.preferred_cluster, 1);
    EXPECT_EQ(side_vote(task.affinity_votes, 0), 64);
    EXPECT_EQ(side_vote(task.affinity_votes, 1), 192);
}
TEST(ReadyAffinity, SmallQueueSavingDoesNotOverrideHandoffCost) {
    uint64_t pressure[2]{24, 0}, cap[2]{10, 10}, free[2]{1, 1};
    EXPECT_EQ(choose_ready_side(0, AffinityStrength::STRONG, 0, pressure, cap, free, 10), 0);
    pressure[0] = 180;
    EXPECT_EQ(choose_ready_side(0, AffinityStrength::STRONG, 0, pressure, cap, free, 10), 1);
}
TEST(ReadyAffinity, InflightWorkPreventsLargeTaskChoosingBusySide) {
    uint64_t pressure[2]{0, 6000}, cap[2]{20, 28}, free[2]{20, 0};
    EXPECT_EQ(choose_ready_side(-1, AffinityStrength::NEUTRAL, -1, pressure, cap, free, 256), 0);
}
TEST(ReadyAffinity, ZeroCapacityExcludesSide) {
    uint64_t pressure[2]{0, 0}, cap[2]{0, 10}, free[2]{0, 1};
    EXPECT_EQ(choose_ready_side(0, AffinityStrength::STRONG, 0, pressure, cap, free, 48), 1);
}
TEST(ReadyAffinity, CacheSurvivesConsecutiveProducerCompletions) {
    ReadyFeedback f;
    f.schedulers[0] = f.schedulers[1] = 1;
    ReadyTaskState task;
    for (int i = 0; i < 16; ++i) {
        f.begin_batch(2);
        (void)f.bind(2, 0, task, 1);
    }
    EXPECT_EQ(f.writers[2].refreshes, 1);
    (void)f.bind(2, 0, task, 1);
    EXPECT_EQ(f.writers[2].refreshes, 2);
}
TEST(ReadyAffinity, OwnBurstPublicationsChangeNextDecision) {
    ReadyFeedback f;
    f.schedulers[0] = f.schedulers[1] = 1;
    ReadyTaskState task;
    task.preferred_cluster = 0;
    EXPECT_EQ(f.bind(2, 0, task, 10), 0);
    ReadyRow::add(f.writers[2].rows[0].enqueued[0], 10);
    EXPECT_EQ(f.bind(2, 0, task, 10), 1);
    EXPECT_EQ(f.writers[2].refreshes, 1);
}
TEST(ReadyAffinity, LocalClaimTransfersQueueToInflightWithoutLosingPressure) {
    ReadyRow r;
    ReadyRow::add(r.enqueued[0], 8);
    EXPECT_EQ(r.pressure_units(0), 48);
    ReadyRow::add(r.claimed[0], 8);
    ReadyRow::add(r.accepted[0], 8);
    EXPECT_EQ(r.queue_units(0), 0);
    EXPECT_EQ(r.pressure_units(0), 48);
    ReadyRow::add(r.completed_units[0], 18);
    EXPECT_EQ(r.pressure_units(0), 30);
}
TEST(ReadyAffinity, RemoteRangeMovesPressureToActualSide) {
    ReadyRow r;
    ReadyRow::add(r.enqueued[0], 16);
    ReadyRow::add(r.claimed[0], 8);
    ReadyRow::add(r.accepted[1], 8);
    EXPECT_EQ(r.queue_units(0), 48);
    EXPECT_EQ(r.queue_units(1), 0);
    EXPECT_EQ(r.pressure_units(0), 48);
    EXPECT_EQ(r.pressure_units(1), 48);
    ReadyRow::add(r.completed_units[1], 48);
    EXPECT_EQ(r.pressure_units(1), 0);
}
TEST(ReadyAffinity, MixCompletionFractionsDoNotTripleCountBlocks) {
    ReadyRow r;
    r.accepted[0].store(1);
    for (int i = 0; i < 3; ++i)
        ReadyRow::add(r.completed_units[0], 6 / 3);
    EXPECT_EQ(r.pressure_units(0), 0);
    r.accepted[0].store(2);
    for (int i = 0; i < 2; ++i)
        ReadyRow::add(r.completed_units[0], 6 / 2);
    EXPECT_EQ(r.pressure_units(0), 0);
}
TEST(ReadyAffinity, RefreshWatermarkDoesNotDoubleCountOwnQueueAndInflight) {
    ReadyFeedback f;
    f.schedulers[0] = f.schedulers[1] = 1;
    auto &row = f.writers[2].rows[0];
    row.enqueued[0].store(10);
    row.claimed[0].store(4);
    row.accepted[0].store(4);
    row.completed_units[0].store(6);
    auto &hint = f.hints[0][0];
    hint.work.store(54);
    hint.queued.store(36);
    hint.pressure_cut[2].store(54);
    hint.queue_cut[2].store(36);
    f.refresh(2, 0);
    EXPECT_EQ(f.pressure(2, 0, 0), 54);
    EXPECT_EQ(f.queued(2, 0, 0), 36);
    row.completed_units[0].store(12);
    EXPECT_EQ(f.pressure(2, 0, 0), 48);
    hint.work.store(48);
    hint.pressure_cut[2].store(48);
    f.writers[2].views[0].uses = 16;
    f.refresh(2, 0);
    EXPECT_EQ(f.pressure(2, 0, 0), 48);
}
TEST(ReadyAffinity, SignedRemoteClaimCutPreservesCredit) {
    ReadyFeedback f;
    auto &r = f.writers[2].rows[0];
    r.claimed[0].store(4);
    auto &h = f.hints[0][0];
    h.work.store(24);
    h.queued.store(24);
    h.pressure_cut[2].store(-24);
    h.queue_cut[2].store(-24);
    f.refresh(2, 0);
    EXPECT_EQ(f.pressure(2, 0, 0), 24);
    r.claimed[0].store(8);
    EXPECT_EQ(f.pressure(2, 0, 0), 0);
}
TEST(ReadyAffinity, RacingSnapshotRetainsOldViewWithoutSpinning) {
    ReadyFeedback f;
    f.refresh(2, 0);
    auto &v = f.writers[2].views[0];
    v.uses = 16;
    f.hints[0][0].version.store(1);
    f.hints[0][0].work.store(1000);
    f.refresh(2, 0);
    EXPECT_EQ(f.pressure(2, 0, 0), 0);
}
TEST(ReadyAffinity, CacheExpiresByLowFrequencyClock) {
    ReadyFeedback f;
    auto &owner = f.writers[2];
    owner.max_age_ticks = 100;
    f.refresh(2, 0);
    owner.now_tick = 101;
    f.refresh(2, 0);
    EXPECT_EQ(owner.refreshes, 2);
}
TEST(ReadyAffinity, CompletedExecutionOverridesCreationPreference) {
    ReadyFeedback f;
    f.schedulers[0] = f.schedulers[1] = 10;
    ReadyTaskState task;
    task.preferred_cluster = 0;
    task.affinity_votes = pack_votes(256, 0);
    task.completed_votes.store(pack_votes(0, 256));
    EXPECT_EQ(f.bind(2, 0, task, 1), 1);
}
TEST(ReadyAffinity, RemoteProbeUsesBlocksRatherThanQueueEntryCount) {
    EXPECT_FALSE(may_remote_pickup(1, 100, 12));
    EXPECT_FALSE(may_remote_pickup(0, 11, 12));
    EXPECT_TRUE(may_remote_pickup(0, 256, 12));
}
