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
void mature(ClusterPartitioner &model, int shape = 0) {
    model.decisions[0] = model.decisions[1] = 128;
    model.observed[shape][0] = model.observed[shape][1] = 32;
    model.channel_updates[shape][0][0] = 1;
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
TEST(ClusterPartitioner, QueueResourceAndExtraRemoteEdgesHaveSeparateWeights) {
    ClusterPartitioner model;
    mature(model);
    auto load = idle_load();
    load.ready[0] = 2;
    load.free[0] = 10;
    load.idle[0] = 10;
    uint32_t source0[2]{1, 0};
    auto decision = model.choose(load, 1, source0, 0, 1);
    EXPECT_EQ(decision.score_ns[0], 2025u);
    EXPECT_EQ(decision.score_ns[1], 775u);
    uint32_t three_remote[2]{3, 0};
    auto multi = model.choose(load, 3, three_remote, 0, 1);
    EXPECT_EQ(multi.score_ns[1], 1031u);
    EXPECT_EQ(multi.score_ns[0], 2025u);
    model.weights[0].dependency[1][1] = 1000;
    uint32_t mixed[2]{2, 1};
    EXPECT_EQ(model.choose(load, 3, mixed, 0, 1).score_ns[1], 1281u);
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
    shared.threads[0].capacity[0].store(20);
    ClusterPartitioner model;
    model.assigned[0][0] = 20;
    uint64_t depths[2]{};
    auto blocked = model.load(shared, 0, depths);
    EXPECT_EQ(ClusterPartitioner::future_reservation_cost(blocked, 0, 0, 2000), 500u);
    shared.ready_published[0][0].store(20);
    auto ready = model.load(shared, 0, depths);
    EXPECT_EQ(ready.blocked[0], 0u);
    EXPECT_EQ(ready.ready[0], 20u);
    EXPECT_EQ(ClusterPartitioner::future_reservation_cost(ready, 0, 0, 2000), 0u);
}
TEST(ClusterPartitioner, IdlePickupBaselineReducesQueuedCoefficient) {
    PartitionShared shared;
    shared.reset();
    shared.threads[0].cluster.store(0);
    ClusterPartitioner model;
    samples(shared.threads[0].samples[0], 5, 16, 100);
    samples(shared.threads[0].samples[0], 0, 32, 1200);
    model.decisions[0] = 256;
    model.poll(shared, 1000000);
    EXPECT_EQ(model.weights[0].idle_pickup[0], 100u);
    EXPECT_EQ(model.weights[0].queue[0], 1775u);
}
TEST(ClusterPartitioner, AccumulatesAcrossSmallScopesAndAcceptsLateSamples) {
    PartitionShared shared;
    shared.reset();
    shared.threads[0].cluster.store(0);
    ClusterPartitioner model;
    for (int burst = 1; burst <= 8; ++burst) {
        for (int i = 0; i < 100; ++i) {
            model.open_scope(0);
            model.scope_assignments[0] = 1;
            model.close_scope(0);
        }
        samples(shared.threads[0].samples[0], 0, 4, 1000);
        model.decisions[0] = burst * 64;
        model.poll(shared, burst * 100000);
        if (burst < 8) EXPECT_EQ(model.weights[0].queue[0], 2000u);
    }
    EXPECT_EQ(model.weights[0].queue[0], 1750u);
    EXPECT_EQ(model.observed[0][0], 32u);
    EXPECT_EQ(model.updates[0], 1u);
}
TEST(ClusterPartitioner, SampleThresholdAndTimeLimitBothApply) {
    PartitionShared shared;
    shared.reset();
    shared.threads[0].cluster.store(0);
    ClusterPartitioner model;
    samples(shared.threads[0].samples[0], 3, 16, 1000);
    model.decisions[0] = 256;
    model.poll(shared, 1000000);
    EXPECT_EQ(model.updates[0], 1u);
    samples(shared.threads[0].samples[0], 3, 16, 1000);
    model.decisions[0] = 512;
    model.poll(shared, 1050000);
    EXPECT_EQ(model.updates[0], 1u);
    model.decisions[0] = 576;
    model.poll(shared, 1100000);
    EXPECT_EQ(model.updates[0], 2u);
    model.decisions[0] = 832;
    model.poll(shared, 2100000);
    EXPECT_EQ(model.updates[0], 2u);
}
TEST(ClusterPartitioner, QueueSamplesCannotChangeChannelCoefficients) {
    PartitionShared shared;
    shared.reset();
    shared.threads[0].cluster.store(1);
    ClusterPartitioner model;
    samples(shared.threads[0].samples[1], 0, 64, 6000);
    model.decisions[0] = 256;
    model.poll(shared, 1000000);
    EXPECT_EQ(model.weights[1].queue[1], 2500u);
    EXPECT_EQ(model.weights[0].queue[1], 2000u);
    EXPECT_EQ(model.weights[1].dependency[0][1], 750u);
    EXPECT_EQ(model.weights[1].publish[0][1], 750u);
}
TEST(ClusterPartitioner, ReadyCountersAndTransfersExcludeBlockedWork) {
    PartitionShared shared;
    shared.reset();
    shared.schedulers[0] = shared.schedulers[1] = 2;
    shared.threads[0].cluster.store(0);
    shared.threads[2].cluster.store(1);
    ClusterPartitioner model;
    model.assigned[0][0] = 100;
    shared.ready_published[0][0].store(40);
    shared.threads[0].started[0].store(10);
    shared.threads[2].incoming_moves[0].store(5);
    shared.threads[2].started[0].store(5);
    uint64_t queue_depth[2]{};
    auto load = model.load(shared, 0, queue_depth);
    EXPECT_EQ(load.blocked[0], 60u);
    EXPECT_EQ(load.ready[0], 25u);
    EXPECT_EQ(load.blocked[1], 0u);
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
    model.decisions[0] = model.decisions[1] = 128;
    model.observed[0][0] = model.observed[0][1] = 32;
    EXPECT_FALSE(model.mature(0, load));
    model.channel_updates[0][0][3] = 1;
    EXPECT_TRUE(model.mature(0, load));
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
