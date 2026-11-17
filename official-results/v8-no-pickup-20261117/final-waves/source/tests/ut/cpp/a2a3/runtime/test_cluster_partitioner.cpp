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
}  // namespace

TEST(ClusterPartitioner, ReadyPublicationTransfersFutureReservationToReadyPressure) {
    PartitionShared shared;
    shared.reset();
    shared.schedulers[0] = shared.schedulers[1] = 2;
    shared.threads[0].cluster.store(0);
    shared.threads[0].summary[0].resources.store(20);
    ClusterPartitioner model;
    model.assigned[0][0] = 20;
    auto blocked = model.load(shared, 0);
    EXPECT_EQ(blocked.blocked[0], 20u);
    EXPECT_EQ(blocked.ready[0], 0u);
    shared.threads[0].summary[0].queued[0].store(20);
    shared.threads[0].summary[0].released[0].store(20);
    model.decisions[0] = PARTITION_BATCH_NODES;
    auto ready = model.load(shared, 0);
    EXPECT_EQ(ready.blocked[0], 0u);
    EXPECT_EQ(ready.ready[0], 20u);
    EXPECT_EQ(ready.blocked[0], 0u);
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


TEST(ClusterPartitioner, WavesNormalizeBacklogByCapacity) {
    ClusterPartitioner model;
    auto load = idle_load();
    load.capacity[0] = load.free[0] = 10;
    load.ready[0] = 9;
    load.ready[1] = 19;
    uint32_t unknown[2]{};
    auto result = model.choose(load, 1, unknown, 1);
    EXPECT_EQ(result.pressure[0], result.pressure[1]);
}
TEST(ClusterPartitioner, SourceAffinityStopsBeyondOneWave) {
    ClusterPartitioner model;
    auto load = idle_load();
    uint32_t source0[2]{1, 0};
    load.ready[0] = 20;
    EXPECT_EQ(model.choose(load, 1, source0, 1).cluster, 0);
    load.ready[0] = 21;
    EXPECT_EQ(model.choose(load, 1, source0, 1).cluster, 1);
}
TEST(ClusterPartitioner, UnavailableDispatchCapacityInfluencesUnknownSource) {
    ClusterPartitioner model;
    auto load = idle_load();
    load.free[0] = 0;
    uint32_t unknown[2]{};
    EXPECT_EQ(model.choose(load, 1, unknown, 0).cluster, 1);
    load.free[0] = 100;
    auto clamped = model.choose(load, 1, unknown, 0);
    EXPECT_EQ(clamped.pressure[0], clamped.pressure[1]);
}
TEST(ClusterPartitioner, DistantReservationsAreBoundedWithoutMakingResourcesBusy) {
    ClusterPartitioner model;
    auto load = idle_load();
    uint32_t unknown[2]{};
    load.blocked[0] = 20;
    auto one_wave = model.choose(load, 1, unknown, 1);
    load.blocked[0] = ~uint64_t{0};
    auto distant = model.choose(load, 1, unknown, 1);
    EXPECT_EQ(one_wave.pressure[0], distant.pressure[0]);
    EXPECT_EQ(load.free[0], 20u);
}
TEST(ClusterPartitioner, WideTaskCanExceedOriginAffinityBudget) {
    ClusterPartitioner model;
    auto load = idle_load();
    load.capacity[0] = load.free[0] = 10;
    uint32_t none[2]{};
    EXPECT_EQ(model.choose(load, 0, none, 0, 96).cluster, 1);
}
TEST(ClusterPartitioner, MissingCapacityAndMissingSideAreHandledWithoutDivisionByZero) {
    ClusterPartitioner model;
    PartitionLoad load;
    load.schedulers[1] = 3;
    uint32_t none[2]{};
    EXPECT_EQ(model.choose(load, 0, none, 0).cluster, 1);
    load.schedulers[1] = 0;
    EXPECT_EQ(model.choose(load, 0, none, 0).cluster, -1);
}
TEST(ClusterPartitioner, MixedSourceHasNoInventedOriginPreference) {
    ClusterPartitioner model;
    auto load = idle_load();
    load.ready[0] = 10;
    uint32_t mixed[2]{1, 1};
    EXPECT_EQ(model.choose(load, 2, mixed, 0).cluster, 1);
}
TEST(ClusterPartitioner, ShardedConcurrentLedgerSnapshotsStayCoherent) {
    PartitionSummary row;
    std::atomic<bool> done{false};
    std::thread writer([&] {
        for (uint64_t i = 1; i <= 10000; ++i) {
            row.begin_write();
            row.released[0].store(i, std::memory_order_relaxed);
            row.queued[1].store(i, std::memory_order_relaxed);
            row.end_write();
        }
        done.store(true, std::memory_order_release);
    });
    while (!done.load(std::memory_order_acquire)) {
        PartitionSummarySample sample;
        if (row.read(sample)) EXPECT_EQ(sample.released[0], sample.queued[1]);
    }
    writer.join();
    PartitionSummarySample sample;
    ASSERT_TRUE(row.read(sample));
    EXPECT_EQ(sample.released[0], 10000u);
    EXPECT_EQ(sample.queued[1], 10000u);
}
