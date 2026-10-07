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
