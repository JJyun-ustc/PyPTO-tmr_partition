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
    EXPECT_LT(result.score_ns[0], result.score_ns[1]);
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
    EXPECT_EQ(model.choose(load, 8, source0, 0, 1).score_ns[0], 800u);
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

TEST(ClusterPartitioner, PartialBatchesSurviveUntilFlushAndRemainCumulative) {
    PartitionBucket bucket;
    samples(bucket, 4, 3, 9);
    PartitionSample snapshot;
    ASSERT_TRUE(bucket.read(snapshot));
    EXPECT_EQ(snapshot.observed, 0u);
    bucket.flush();
    ASSERT_TRUE(bucket.read(snapshot));
    EXPECT_EQ(snapshot.observed, 3u);
    EXPECT_EQ(snapshot.count[4], 3u);
    EXPECT_EQ(snapshot.ns[4], 27u);
    samples(bucket, 4, 5, 9);
    ASSERT_TRUE(bucket.read(snapshot));
    EXPECT_EQ(snapshot.observed, 8u);
    EXPECT_EQ(snapshot.count[4], 8u);
    EXPECT_EQ(snapshot.ns[4], 72u);
}
