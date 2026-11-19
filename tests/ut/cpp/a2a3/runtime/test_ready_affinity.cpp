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

static void topology(ReadyFeedback &f) {
    f.schedulers[0] = 2;
    f.schedulers[1] = 3;
    for (int t = 0; t < 6; ++t)
        f.cluster[t] = t < 2 ? 0 : 1;
}
TEST(ReadyAffinity, PublishMovesPredictionAndQueueInOneTransaction) {
    ReadyRow row;
    row.publish(0, 1, 16);
    ReadySample snapshot;
    ASSERT_TRUE(row.read(snapshot));
    EXPECT_EQ(snapshot.revoked[0], 16);
    EXPECT_EQ(snapshot.enqueued[1], 16);
    EXPECT_EQ(snapshot.enqueued[0], 0);
}
TEST(ReadyAffinity, RacingPublicationRetainsPreviousSnapshot) {
    ReadyRow row;
    ReadySample old;
    old.enqueued[0] = 7;
    row.version.store(1);
    row.revoked[0].store(100);
    EXPECT_FALSE(row.read(old));
    EXPECT_EQ(old.enqueued[0], 7);
    EXPECT_EQ(old.revoked[0], 0);
}
TEST(ReadyAffinity, ConcurrentReaderNeverObservesHalfPublication) {
    ReadyRow row;
    std::atomic<bool> done{false};
    std::atomic<int> bad{0};
    std::thread writer([&] {
        for (int i = 0; i < 10000; ++i)
            row.publish(0, 1, 3);
        done.store(true);
    });
    do {
        ReadySample snap;
        if (row.read(snap) && snap.revoked[0] != snap.enqueued[1]) ++bad;
    } while (!done.load());
    writer.join();
    EXPECT_EQ(bad.load(), 0);
}
TEST(ReadyAffinity, LeadersOwnWritersAndPreserveBothQueueSides) {
    ReadyFeedback f;
    topology(f);
    f.writers[0].rows[0].publish(0, 1, 10);
    f.writers[2].rows[0].publish(1, 0, 20);
    f.writers[5].rows[0].publish(1, 1, 30);
    f.publish_side(0, 0);
    f.publish_side(2, 0);
    f.refresh(5, 0);
    EXPECT_EQ(f.queued(5, 0, 0), 20);
    EXPECT_EQ(f.queued(5, 0, 1), 40);
    EXPECT_EQ(f.writers[5].views[0].sides[0].schedulers.enqueued[1], 10);
    EXPECT_EQ(f.writers[5].views[0].sides[1].schedulers.enqueued[0], 20);
    EXPECT_EQ(f.writers[5].views[0].sides[1].schedulers.enqueued[1], 0);
}
TEST(ReadyAffinity, ReservationAndQueueUseCurrentOAndSnapshotS) {
    ReadyFeedback f;
    topology(f);
    ReadyTaskState task;
    f.predict(0, task, 10, true);
    int p = task.predicted_cluster;
    EXPECT_EQ(f.blocked(0, p), 10);
    f.writers[2].rows[0].publish(p, 1, 10);
    EXPECT_EQ(f.blocked(0, p), 10);
    f.publish_side(2, 0);
    f.refresh(5, 0);
    EXPECT_EQ(f.blocked(0, p), 0);
    EXPECT_EQ(f.queued(5, 0, 1), 10);
    f.writers[5].rows[0].publish(0, 0, 7);
    f.assigned[0][0] += 7;
    EXPECT_EQ(f.blocked(0, 0), 0);
    EXPECT_EQ(f.queued(5, 0, 0), 7);
}
TEST(ReadyAffinity, OwnDeltasSurviveRefreshWithoutDoubleCounting) {
    ReadyFeedback f;
    topology(f);
    auto &row = f.writers[2].rows[0];
    row.publish(1, 0, 8);
    f.publish_side(2, 0);
    f.refresh(2, 0);
    EXPECT_EQ(f.queued(2, 0, 0), 8);
    row.publish(0, 0, 4);
    EXPECT_EQ(f.queued(2, 0, 0), 12);
    f.publish_side(2, 0);
    f.refresh(2, 0);
    EXPECT_EQ(f.queued(2, 0, 0), 12);
    ReadyRow::add(row.claimed[0], 8);
    EXPECT_EQ(f.queued(2, 0, 0), 4);
    f.publish_side(2, 0);
    f.refresh(2, 0);
    EXPECT_EQ(f.queued(2, 0, 0), 4);
}
TEST(ReadyAffinity, NegativeRemoteClaimRowRetainsDebitAcrossRefresh) {
    ReadyFeedback f;
    topology(f);
    f.writers[0].rows[0].publish(0, 0, 8);
    f.writers[2].rows[0].claimed[0].store(4);
    f.publish_side(0, 0);
    f.publish_side(2, 0);
    f.refresh(2, 0);
    EXPECT_EQ(f.queued(2, 0, 0), 4);
    f.writers[2].rows[0].claimed[0].store(8);
    EXPECT_EQ(f.queued(2, 0, 0), 0);
}
TEST(ReadyAffinity, OrchestratorSidecarIsVisibleExactlyOnceToSchedulers) {
    ReadyFeedback f;
    topology(f);
    f.writers[5].rows[0].publish(0, 1, 9);
    f.publish_side(0, 0);
    f.publish_side(2, 0);
    f.refresh(0, 0);
    f.refresh(5, 0);
    EXPECT_EQ(f.queued(0, 0, 1), 9);
    EXPECT_EQ(f.queued(5, 0, 1), 9);
    EXPECT_EQ(f.writers[0].views[0].sides[0].orchestrator.enqueued[1], 0);
    EXPECT_EQ(f.writers[0].views[0].sides[1].orchestrator.enqueued[1], 9);
}
TEST(ReadyAffinity, SnapshotReplacementDoesNotAccumulateOldTotals) {
    ReadyFeedback f;
    topology(f);
    f.writers[0].rows[0].publish(0, 1, 9);
    f.publish_side(0, 0);
    f.refresh(5, 0);
    f.refresh(5, 0);
    EXPECT_EQ(f.queued(5, 0, 1), 9);
}
TEST(ReadyAffinity, FailedHintReadPreservesBothReservationAndQueue) {
    ReadyFeedback f;
    topology(f);
    auto &hint = f.hints[0][0];
    hint.revoked[0].store(3);
    hint.enqueued[1].store(3);
    f.refresh(5, 0);
    hint.version.store(1);
    hint.revoked[0].store(100);
    f.refresh(5, 0);
    EXPECT_EQ(f.writers[5].views[0].sides[0].schedulers.revoked[0], 3);
    EXPECT_EQ(f.queued(5, 0, 1), 3);
}
TEST(ReadyAffinity, ORefreshEvery64CreationsRatherThanEveryReady) {
    ReadyFeedback f;
    topology(f);
    for (int i = 0; i < 64; ++i) {
        ReadyTaskState t;
        f.predict(0, t, 1, true);
    }
    EXPECT_EQ(f.writers[5].refreshes, 4);
    ReadyTaskState t;
    f.predict(0, t, 1, true);
    EXPECT_EQ(f.writers[5].refreshes, 8);
}
TEST(ReadyAffinity, ReadyBindingNeverRefreshesFeedbackOrScoresPressure) {
    ReadyFeedback f;
    topology(f);
    ReadyTaskState t;
    t.predicted_cluster = 0;
    for (int i = 0; i < 64; ++i)
        EXPECT_EQ(f.bind(2, 0, t, 1), 0);
    EXPECT_EQ(f.writers[2].refreshes, 0);
    EXPECT_EQ(f.writers[2].rows[0].revoked[0].load(), 64);
}
TEST(ReadyAffinity, ActualVoteThresholdIsStrictlyGreaterThanQuarter) {
    int32_t sched[2]{2, 3};
    uint64_t q[2]{};
    EXPECT_EQ(light_ready_side(0, pack_votes(3, 5), true, q, sched), 0);
    EXPECT_EQ(light_ready_side(0, pack_votes(2, 6), true, q, sched), 1);
}
TEST(ReadyAffinity, InferredVoteThresholdIsStrictlyGreaterThanHalf) {
    int32_t sched[2]{2, 3};
    uint64_t q[2]{};
    EXPECT_EQ(light_ready_side(0, pack_votes(1, 3), false, q, sched), 0);
    EXPECT_EQ(light_ready_side(0, pack_votes(1, 4), false, q, sched), 1);
    EXPECT_EQ(light_ready_side(0, 0, true, q, sched), 0);
}
TEST(ReadyAffinity, CongestionProtectsPredictionFromAffinityCorrection) {
    int32_t sched[2]{2, 3};
    uint64_t q[2]{0, 8};
    EXPECT_EQ(light_ready_side(0, pack_votes(0, 256), true, q, sched), 0);
    q[0] = 8;
    EXPECT_EQ(light_ready_side(0, pack_votes(0, 256), true, q, sched), 1);
}
TEST(ReadyAffinity, CurrentOLocalReadyPublicationsAffectNextPrediction) {
    ReadyFeedback f;
    topology(f);
    ReadyTaskState a, b;
    int first = f.predict(0, a, 10, false);
    f.bind(5, 0, a, 10);
    int next = f.predict(0, b, 10, false);
    EXPECT_NE(first, next);
    EXPECT_EQ(f.assigned[0][first] + f.assigned[0][next], 20);
}
TEST(ReadyAffinity, QueuedDifferenceRemainsSignedUntilScore) {
    ReadyFeedback f;
    topology(f);
    f.writers[2].rows[0].claimed[0].store(9);
    f.publish_side(2, 0);
    f.refresh(5, 0);
    EXPECT_EQ(f.queued(5, 0, 0), -9);
    EXPECT_EQ(positive(f.queued(5, 0, 0)), 0);
}
