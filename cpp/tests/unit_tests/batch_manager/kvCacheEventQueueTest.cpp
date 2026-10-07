/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/batch_manager/kvCacheEventManager.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <gtest/gtest.h>

namespace kv = tensorrt_llm::batch_manager::kv_cache_manager;
namespace ex = tensorrt_llm::executor;

TEST(KVCacheEventQueueTest, FlushPreservesEventOrderAndDrains)
{
    kv::KVCacheEventManager queue{8};
    EXPECT_TRUE(queue.getEvents(std::chrono::milliseconds{0}).empty());
    queue.enqueueCreatedEvent({4, 2}, 32);
    queue.enqueueUpdatedEvent(ex::KVCacheUpdatedData{123}.priorityUpdated(1, 7), 64);
    queue.flush();
    auto events = queue.getEvents(std::chrono::seconds{5});
    ASSERT_EQ(events.size(), 2);
    EXPECT_EQ(events[0].eventId, 0);
    EXPECT_EQ(events[0].windowSize, 32);
    EXPECT_EQ(std::get<ex::KVCacheCreatedData>(events[0].data).numBlocksPerCacheLevel, (std::vector<int32_t>{4, 2}));
    EXPECT_EQ(events[1].eventId, 1);
    EXPECT_EQ(events[1].windowSize, 64);
    auto const& update = std::get<ex::KVCacheUpdatedData>(events[1].data);
    EXPECT_EQ(update.blockHash, 123);
    ASSERT_TRUE(update.priority);
    EXPECT_EQ(update.priority->oldValue, 1);
    EXPECT_EQ(update.priority->newValue, 7);
    EXPECT_TRUE(queue.getEvents(std::chrono::milliseconds{0}).empty());
}

TEST(KVCacheEventQueueTest, BoundedQueueKeepsNewestEvents)
{
    kv::KVCacheEventManager queue{3};
    for (int32_t index = 0; index < 6; ++index)
    {
        queue.enqueueCreatedEvent({index}, 32);
    }
    queue.flush();
    auto events = queue.getEvents(std::chrono::seconds{5});
    ASSERT_EQ(events.size(), 3);
    for (size_t index = 0; index < events.size(); ++index)
    {
        EXPECT_EQ(events[index].eventId, index + 3);
        EXPECT_EQ(std::get<ex::KVCacheCreatedData>(events[index].data).numBlocksPerCacheLevel[0], index + 3);
    }
}

TEST(KVCacheEventQueueTest, AttentionDpGatherPreservesPayloadsAndPerRankOrder)
{
#if ENABLE_MULTI_DEVICE
    auto const& comm = tensorrt_llm::mpi::MpiComm::session();
    auto const rank = comm.getRank();
    auto const size = comm.getSize();
    if (size < 2)
    {
        GTEST_SKIP() << "Requires at least two MPI ranks";
    }

    {
        kv::KVCacheEventManager queue{8, rank, size};
        queue.enqueueCreatedEvent({rank + 1, rank + 2}, 32);
        queue.enqueueUpdatedEvent(ex::KVCacheUpdatedData{static_cast<size_t>(rank + 100)}.priorityUpdated(1, 7), 64);
        queue.flush();

        if (rank == 0)
        {
            std::deque<ex::KVCacheEvent> gathered;
            auto const expectedCount = static_cast<size_t>(2 * size);
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
            while (gathered.size() < expectedCount && std::chrono::steady_clock::now() < deadline)
            {
                auto events = queue.getEvents(std::chrono::milliseconds{100});
                gathered.insert(gathered.end(), events.begin(), events.end());
            }
            EXPECT_EQ(gathered.size(), expectedCount);
            std::vector<size_t> nextEventIds(size, 0);
            for (auto const& event : gathered)
            {
                EXPECT_TRUE(event.attentionDpRank.has_value());
                if (!event.attentionDpRank || *event.attentionDpRank < 0 || *event.attentionDpRank >= size)
                {
                    ADD_FAILURE() << "Invalid attention DP rank";
                    continue;
                }
                auto const sourceRank = *event.attentionDpRank;
                EXPECT_EQ(event.eventId, nextEventIds[sourceRank]++);
                if (event.eventId == 0)
                {
                    EXPECT_EQ(event.windowSize, 32);
                    auto const* created = std::get_if<ex::KVCacheCreatedData>(&event.data);
                    EXPECT_NE(created, nullptr);
                    if (created)
                    {
                        EXPECT_EQ(
                            created->numBlocksPerCacheLevel, (std::vector<int32_t>{sourceRank + 1, sourceRank + 2}));
                    }
                }
                else
                {
                    EXPECT_EQ(event.eventId, 1);
                    EXPECT_EQ(event.windowSize, 64);
                    auto const* updated = std::get_if<ex::KVCacheUpdatedData>(&event.data);
                    EXPECT_NE(updated, nullptr);
                    if (updated)
                    {
                        EXPECT_EQ(updated->blockHash, sourceRank + 100);
                        EXPECT_TRUE(updated->priority.has_value());
                        if (updated->priority)
                        {
                            EXPECT_EQ(updated->priority->oldValue, 1);
                            EXPECT_EQ(updated->priority->newValue, 7);
                        }
                    }
                }
            }
            for (auto const nextEventId : nextEventIds)
            {
                EXPECT_EQ(nextEventId, 2);
            }
            EXPECT_TRUE(queue.getEvents(std::chrono::milliseconds{0}).empty());
        }
        // Keep exchange threads alive until rank zero has drained every rank's events.
        comm.barrier();
    }
    comm.barrier();
#else
    GTEST_SKIP() << "Multi-device support is disabled";
#endif
}
