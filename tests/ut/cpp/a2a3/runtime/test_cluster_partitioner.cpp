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

#include "aicpu_cluster_partition.h"

using namespace simpler::tmr;
namespace {
PartitionLoad idle_load() {
    PartitionLoad load;
    load.schedulers[0] = load.schedulers[1] = 2;
    load.capacity[0] = load.capacity[1] = 20;
    load.free[0] = load.free[1] = 20;
    load.idle[0] = load.idle[1] = 20;
    return load;
}
void mature(ClusterPartitioner &model) {
    model.decisions[0] = model.decisions[1] = 128;
}
void samples(PartitionBucket &bucket, int metric, int count, uint64_t ns) {
    for (int i = 0; i < count; ++i)
        bucket.add(metric, ns);
}
}  // namespace
TEST(ClusterPartitioner, BlockedReservationsDoNotMakeIdleResourcesBusy) {
    ClusterPartitioner model;
    mature(model);
    auto load = idle_load();
    load.blocked[0] = 1000;
    uint32_t sources[2]{};
    auto result = model.choose(load, 0, sources, 0, 0);
    EXPECT_TRUE(result.learned);
    // Even a large blocked backlog contributes at most queue/4, not busy slots.
    EXPECT_EQ(result.score_ns[0], 750u);
    EXPECT_EQ(result.score_ns[1], 750u);
    EXPECT_EQ(result.cluster, 0);
}
TEST(ClusterPartitioner, DependencyCostDependsOnSourceAndDestination) {
    ClusterPartitioner model;
    mature(model);
    auto load = idle_load();
    model.weights[0].dependency[0][0] = 800;
    model.weights[0].dependency[0][1] = 100;
    model.weights[0].dependency[1][0] = 50;
    model.weights[0].dependency[1][1] = 900;
    uint32_t source0[2]{1, 0}, source1[2]{0, 1};
    EXPECT_EQ(model.choose(load, 1, source0, 0, 1).cluster, 1);
    EXPECT_EQ(model.choose(load, 1, source1, 0, 1).cluster, 0);
    EXPECT_EQ(model.choose(load, 8, source0, 0, 1).score_ns[0], 825u);
}
TEST(ClusterPartitioner, PublicationAndNotificationAreSeparatePaths) {
    ClusterPartitioner model;
    mature(model);
    auto load = idle_load();
    model.weights[0].publish[1][0] = 100;
    model.weights[0].publish[1][1] = 900;
    model.weights[0].dependency[0][0] = 800;
    model.weights[0].dependency[0][1] = 50;
    uint32_t none[2]{}, source0[2]{1, 0};
    EXPECT_EQ(model.choose(load, 0, none, 0, 1).cluster, 0);
    EXPECT_EQ(model.choose(load, 1, source0, 0, 1).cluster, 1);
}
TEST(ClusterPartitioner, QueueAndFutureDoNotDoubleCountResourceAndExtraEdges) {
    ClusterPartitioner model;
    mature(model);
    auto load = idle_load();
    load.ready[0] = 2;
    load.free[0] = 10;
    load.idle[0] = 10;
    uint32_t source0[2]{1, 0};
    auto decision = model.choose(load, 1, source0, 0, 1);
    EXPECT_EQ(decision.score_ns[0], 1275u);
    EXPECT_EQ(decision.score_ns[1], 775u);
    uint32_t three_remote[2]{3, 0};
    auto multi = model.choose(load, 3, three_remote, 0, 1);
    EXPECT_EQ(multi.score_ns[1], 775u);
    EXPECT_EQ(multi.score_ns[0], 1275u);
    model.weights[0].dependency[1][1] = 1000;
    uint32_t mixed[2]{2, 1};
    EXPECT_EQ(model.choose(load, 3, mixed, 0, 1).score_ns[1], 1025u);
}
TEST(ClusterPartitioner, FutureReservationsNormalizeCapacityAndBoundDistantWork) {
    auto load = idle_load();
    load.capacity[1] = 40;
    load.blocked[0] = 10;
    load.blocked[1] = 20;
    EXPECT_EQ(ClusterPartitioner::future_reservation_cost(load, 0, 0, 2000), 250u);
    EXPECT_EQ(ClusterPartitioner::future_reservation_cost(load, 1, 0, 2000), 250u);
    // A new wide blocked task is charged in block units, with the same bound.
    EXPECT_EQ(ClusterPartitioner::future_reservation_cost(load, 0, 4, 2000), 350u);
    EXPECT_EQ(ClusterPartitioner::future_reservation_cost(load, 0, 96, 2000), 500u);
    load.blocked[0] = ~uint64_t{0};
    EXPECT_EQ(ClusterPartitioner::future_reservation_cost(load, 0, 96, 2000), 500u);
    load.capacity[0] = 0;
    EXPECT_EQ(ClusterPartitioner::future_reservation_cost(load, 0, 1, 2000), 500u);
    load.schedulers[0] = 0;
    EXPECT_EQ(ClusterPartitioner::future_reservation_cost(load, 0, 1, 2000), 0u);
}
TEST(ClusterPartitioner, BlockedFanoutCanUsePeerWithoutMarkingResourcesBusy) {
    ClusterPartitioner model;
    mature(model);
    model.weights[0].queue[0] = model.weights[0].queue[1] = 8000;
    auto load = idle_load();
    uint32_t sources[2]{1, 0};
    uint32_t assigned[2]{};
    // All successors wait on one cluster-0 producer; none becomes ready yet.
    for (int i = 0; i < 24; ++i) {
        auto decision = model.choose(load, 1, sources, 0, 0);
        ++assigned[decision.cluster];
        ++load.blocked[decision.cluster];
    }
    EXPECT_GT(assigned[1], 0u);
    EXPECT_EQ(load.ready[0] + load.ready[1], 0u);
    EXPECT_EQ(load.free[0], 20u);
    EXPECT_EQ(load.idle[0], 20u);
}
TEST(ClusterPartitioner, ReadyPublicationTransfersFutureReservationToReadyPressure) {
    PartitionShared shared;
    shared.reset();
    shared.schedulers[0] = shared.schedulers[1] = 2;
    shared.threads[0].cluster.store(0);
    shared.threads[0].summary[0].resources.store(20);
    ClusterPartitioner model;
    model.assigned[0][0] = 20;
    auto blocked = model.load(shared, 0);
    EXPECT_EQ(ClusterPartitioner::future_reservation_cost(blocked, 0, 0, 2000), 500u);
    shared.threads[0].summary[0].queued[0].store(20);
    shared.threads[0].summary[0].released[0].store(20);
    model.decisions[0] = PARTITION_BATCH_NODES;
    auto ready = model.load(shared, 0);
    EXPECT_EQ(ready.blocked[0], 0u);
    EXPECT_EQ(ready.ready[0], 20u);
    EXPECT_EQ(ClusterPartitioner::future_reservation_cost(ready, 0, 0, 2000), 0u);
}
TEST(ClusterPartitioner, FixedPrefixCalibratesOnceThenFreezes) {
    PartitionShared shared;
    shared.reset();
    shared.threads[0].cluster.store(0);
    ClusterPartitioner model;
    samples(shared.threads[0].samples[0], 5, 16, 100);
    samples(shared.threads[0].samples[0], 0, 32, 1200);
    model.decisions[0] = 255;
    model.calibrate_once(shared);
    EXPECT_EQ(model.calibrations, 0u);
    EXPECT_FALSE(shared.calibration_done.load());
    model.decisions[0] = 256;
    model.calibrate_once(shared);
    EXPECT_TRUE(shared.calibration_done.load());
    EXPECT_EQ(model.weights[0].idle_pickup[0], 100u);
    EXPECT_EQ(model.weights[0].queue[0], 1100u);
    // Samples arriving after the prefix do not refit or prolong calibration.
    samples(shared.threads[0].samples[0], 0, 64, 6000);
    model.decisions[0] = 1024;
    model.calibrate_once(shared);
    EXPECT_EQ(model.weights[0].queue[0], 1100u);
    EXPECT_EQ(model.calibrations, 1u);
    EXPECT_EQ(model.channel_updates[0][0][0], 1u);
}
TEST(ClusterPartitioner, ScoredTransitionDoesNotWaitForShapeMaturity) {
    PartitionShared shared;
    shared.reset();
    ClusterPartitioner model;
    auto load = idle_load();
    uint32_t none[2]{};
    model.decisions[0] = 255;
    EXPECT_FALSE(model.choose(load, 0, none, 3, 1).learned);
    model.calibrate_once(shared);
    // This shape has no observations or fitted coefficients, but uses priors.
    EXPECT_TRUE(model.choose(load, 0, none, 3, 1).learned);
    EXPECT_EQ(model.observed[3][0] + model.observed[3][1], 0u);
    EXPECT_EQ(model.weights[3].queue[0], 2000u);
}
TEST(ClusterPartitioner, SparseAndRacingChannelsRetainPriorsWithoutRetry) {
    PartitionShared shared;
    shared.reset();
    shared.threads[0].cluster.store(0);
    shared.threads[1].cluster.store(1);
    samples(shared.threads[0].samples[0], 3, 15, 1000);
    samples(shared.threads[1].samples[0], 3, 16, 1000);
    shared.threads[1].samples[0].version.fetch_add(1);
    ClusterPartitioner model;
    model.decisions[0] = 256;
    model.calibrate_once(shared);
    EXPECT_EQ(model.weights[0].dependency[0][0], 250u);
    EXPECT_EQ(model.weights[0].dependency[0][1], 750u);
    shared.threads[1].samples[0].version.fetch_add(1);
    model.calibrate_once(shared);
    EXPECT_EQ(model.weights[0].dependency[0][1], 750u);
    EXPECT_EQ(model.calibrations, 1u);
}
TEST(ClusterPartitioner, CalibrationFitsOnlyObservedChannelsAndBoundsOutliers) {
    PartitionShared shared;
    shared.reset();
    shared.threads[0].cluster.store(1);
    samples(shared.threads[0].samples[1], 0, 32, 6000);
    samples(shared.threads[0].samples[1], 3, 16, 10000);
    ClusterPartitioner model;
    model.decisions[0] = 256;
    model.calibrate_once(shared);
    EXPECT_EQ(model.weights[1].queue[1], 4000u);
    EXPECT_EQ(model.weights[1].dependency[0][1], 1500u);
    EXPECT_EQ(model.weights[1].publish[0][1], 750u);
    EXPECT_EQ(model.weights[0].queue[1], 2000u);
}
TEST(ClusterPartitioner, FixedBatchReusesSnapshotAndReservesPendingWorkLocally) {
    PartitionShared shared;
    shared.reset();
    shared.schedulers[0] = shared.schedulers[1] = 2;
    shared.threads[0].cluster.store(0);
    shared.threads[0].summary[0].resources.store(20);
    ClusterPartitioner model;
    auto initial = model.load(shared, 0);
    EXPECT_EQ(initial.ready[0] + initial.blocked[0], 0u);
    // O commits a blocked multi-block task; concurrent S publication and
    // dispatch remain invisible until the next refresh epoch.
    model.commit(0, 0, 4);
    shared.threads[0].summary[0].queued[0].store(4);
    shared.threads[0].summary[0].released[0].store(4);
    shared.threads[0].summary[0].taken[0].store(4);
    model.decisions[0] = 1;
    auto cached = model.load(shared, 0);
    EXPECT_EQ(cached.blocked[0], 4u);
    EXPECT_EQ(cached.ready[0], 0u);
    EXPECT_EQ(model.snapshots, 1u);
    model.decisions[0] = PARTITION_BATCH_NODES;
    auto refreshed = model.load(shared, 0);
    EXPECT_EQ(refreshed.blocked[0] + refreshed.ready[0], 0u);
    EXPECT_EQ(model.snapshots, 2u);
}
TEST(ClusterPartitioner, BatchReadyReservationsAreCountedOnceAndRefreshInBlockUnits) {
    PartitionShared shared;
    shared.reset();
    shared.schedulers[0] = shared.schedulers[1] = 2;
    shared.threads[0].cluster.store(0);
    ClusterPartitioner model;
    (void)model.load(shared, 0);
    model.commit(0, 0, 16);
    auto cached = model.load(shared, 0);
    EXPECT_EQ(cached.ready[0], 0u);
    EXPECT_EQ(cached.blocked[0], 16u);
    shared.threads[0].summary[0].queued[0].store(16);
    shared.threads[0].summary[0].released[0].store(16);
    shared.threads[0].summary[0].taken[0].store(6);
    shared.side_queue[0][0].store(1);  // One entry, ten undispatched blocks.
    model.decisions[0] = PARTITION_BATCH_NODES;
    EXPECT_EQ(model.load(shared, 0).ready[0], 10u);
}
TEST(ClusterPartitioner, ShapesHaveIndependentSnapshotsWithinGlobalBatch) {
    PartitionShared shared;
    shared.reset();
    ClusterPartitioner model;
    (void)model.load(shared, 0);
    (void)model.load(shared, 0);
    (void)model.load(shared, 1);
    (void)model.load(shared, 1);
    EXPECT_EQ(model.snapshots, 2u);
    model.decisions[0] = 7;
    EXPECT_FALSE(model.needs_snapshot(0));
    model.decisions[0] = PARTITION_BATCH_NODES;
    EXPECT_TRUE(model.needs_snapshot(0));
    EXPECT_TRUE(model.needs_snapshot(1));
}
TEST(ClusterPartitioner, ReadyLedgerHasNoTransfersAndSeparatesBlockedWork) {
    PartitionShared shared;
    shared.reset();
    shared.schedulers[0] = shared.schedulers[1] = 2;
    shared.threads[0].cluster.store(0);
    ClusterPartitioner model;
    model.assigned[0][0] = 100;
    shared.threads[0].summary[0].queued[0].store(40);
    shared.threads[0].summary[0].released[0].store(40);
    shared.threads[0].summary[0].taken[0].store(10);
    auto load = model.load(shared, 0);
    EXPECT_EQ(load.blocked[0], 60u);
    EXPECT_EQ(load.ready[0], 30u);
    EXPECT_EQ(load.ready[1], 0u);
}
TEST(ClusterPartitioner, ColdStartUsesBoundedAffinityAndBalancesFutureWork) {
    ClusterPartitioner model;
    auto load = idle_load();
    uint32_t source1[2]{0, 1};
    auto first = model.choose(load, 1, source1, 0, 1);
    EXPECT_FALSE(first.learned);
    EXPECT_EQ(first.cluster, 1);
    load.blocked[1] = 1000;
    EXPECT_EQ(model.choose(load, 1, source1, 0, 1).cluster, 0);
}
TEST(ClusterPartitioner, ConcurrentSnapshotsRemainCoherent) {
    PartitionBucket bucket;
    std::atomic<bool> done{false};
    std::thread writer([&] {
        for (int i = 0; i < 20000; ++i)
            bucket.add(3, 7);
        done.store(true);
    });
    while (!done.load()) {
        PartitionSample sample;
        if (bucket.read(sample)) {
            EXPECT_EQ(sample.ns[3], 7 * sample.count[3]);
            EXPECT_EQ(sample.observed, sample.count[3]);
        }
    }
    writer.join();
    PartitionSample sample;
    ASSERT_TRUE(bucket.read(sample));
    EXPECT_EQ(sample.count[3], 20000u);
}

TEST(ClusterPartitioner, ReadyBindingFollowsActualReleaseWithCongestionHysteresis) {
    int32_t schedulers[2]{2, 3};
    uint64_t empty[2]{};
    EXPECT_EQ(partition_ready_side(0, 1, empty, schedulers), 1);
    uint64_t balanced[2]{4, 6};
    EXPECT_EQ(partition_ready_side(0, 1, balanced, schedulers), 1);
    uint64_t congested[2]{0, 15};
    EXPECT_EQ(partition_ready_side(0, 1, congested, schedulers), 0);
    EXPECT_EQ(partition_ready_side(1, -1, empty, schedulers), 1);
    schedulers[0] = 0;
    EXPECT_EQ(partition_ready_side(0, 0, empty, schedulers), 1);
}

TEST(ClusterPartitioner, GroupsBoundFanoutLifetimeScopeAndSlotReuse) {
    ClusterPartitioner model;
    AdaptiveTaskState parent;
    model.join_group(parent, 0, 7);
    EXPECT_TRUE(model.can_inherit(parent, 7));
    EXPECT_FALSE(model.can_inherit(parent, 8));
    for (uint32_t i = 1; i < PARTITION_GROUP_LIMIT; ++i) {
        AdaptiveTaskState child;
        model.join_group(child, parent.group, 7);
    }
    EXPECT_FALSE(model.can_inherit(parent, 7));
    model.join_group(parent, 0, 7);
    model.decisions[0] += PARTITION_GROUP_LIFETIME;
    EXPECT_FALSE(model.can_inherit(parent, 7));
    model.join_group(parent, 0, 7);
    for (uint32_t i = 0; i < PARTITION_GROUP_SLOTS; ++i) {
        AdaptiveTaskState other;
        model.join_group(other, 0, 7);
    }
    EXPECT_FALSE(model.can_inherit(parent, 7));
}

TEST(ClusterPartitioner, RacedSummaryRetainsLastCoherentLedgerWithoutDoubleReservation) {
    PartitionShared shared;
    shared.schedulers[0] = shared.schedulers[1] = 2;
    ClusterPartitioner model;
    model.assigned[0][0] = 16;
    auto &row = shared.threads[0].summary[0];
    row.begin_write();
    row.released[0].store(4);
    row.queued[1].store(4);
    row.end_write();
    auto first = model.load(shared, 0);
    EXPECT_EQ(first.blocked[0], 12u);
    EXPECT_EQ(first.ready[1], 4u);
    row.begin_write();
    row.released[0].store(16);
    row.queued[1].store(16);
    model.decisions[0] += PARTITION_BATCH_NODES;
    auto racing = model.load(shared, 0);
    EXPECT_EQ(racing.blocked[0], 12u);
    EXPECT_EQ(racing.ready[1], 4u);
    row.end_write();
    model.decisions[0] += PARTITION_BATCH_NODES;
    auto final = model.load(shared, 0);
    EXPECT_EQ(final.blocked[0], 0u);
    EXPECT_EQ(final.ready[1], 16u);
}

TEST(ClusterPartitioner, FinalQueueAccountingIsIndependentOfPreference) {
    PartitionShared shared;
    shared.schedulers[0] = 2;
    shared.schedulers[1] = 3;
    ClusterPartitioner model;
    model.assigned[0][0] = 16;
    // O prefers 0, readiness binds to 1, and a cluster-1 S dispatches six.
    auto &publisher = shared.threads[2].summary[0];
    publisher.released[0].store(16);
    publisher.queued[1].store(16);
    shared.threads[2].summary[0].taken[1].store(6);
    shared.threads[2].started[0].store(6);
    auto load = model.load(shared, 0);
    EXPECT_EQ(load.blocked[0] + load.blocked[1], 0u);
    EXPECT_EQ(load.ready[0], 0u);
    EXPECT_EQ(load.ready[1], 10u);
}

TEST(ClusterPartitioner, BusyHintAloneDoesNotOverrideHandoffWithEmptyQueues) {
    ClusterPartitioner model;
    mature(model);
    auto load = idle_load();
    uint32_t source0[2]{1, 0};
    EXPECT_EQ(model.choose(load, 1, source0, 0, 1).cluster, 0);
    load.free[0] = load.idle[0] = 0;
    EXPECT_EQ(model.choose(load, 1, source0, 0, 1).cluster, 0);
}
