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

#include "tensorrt_llm/executor/kvCacheEventSource.h"

#include <gtest/gtest.h>

namespace ex = tensorrt_llm::executor;

namespace
{
class EventSource : public ex::detail::KVCacheEventSource
{
public:
    std::deque<ex::KVCacheEvent> getLatestEvents(std::optional<std::chrono::milliseconds> timeout) override
    {
        lastTimeout = timeout;
        return std::exchange(events, {});
    }

    std::optional<std::chrono::milliseconds> lastTimeout;
    std::deque<ex::KVCacheEvent> events;
};
} // namespace

TEST(KVCacheEventReaderTest, ForwardsTimeoutAndDrainsSource)
{
    auto source = std::make_shared<EventSource>();
    source->events.emplace_back(7, ex::KVCacheRemovedData{{11, 12}}, 32, 2);
    auto reader = ex::detail::KVCacheEventManagerAccess::create(source);
    auto events = reader.getLatestEvents(std::chrono::milliseconds{25});
    ASSERT_EQ(events.size(), 1);
    EXPECT_EQ(events.front().eventId, 7);
    EXPECT_EQ(events.front().windowSize, 32);
    EXPECT_EQ(events.front().attentionDpRank, 2);
    EXPECT_EQ(std::get<ex::KVCacheRemovedData>(events.front().data).blockHashes, (std::vector<ex::IdType>{11, 12}));
    EXPECT_EQ(source->lastTimeout, std::chrono::milliseconds{25});
    EXPECT_TRUE(reader.getLatestEvents(std::chrono::milliseconds{0}).empty());
    EXPECT_EQ(source->lastTimeout, std::chrono::milliseconds{0});
    EXPECT_TRUE(reader.getLatestEvents().empty());
    EXPECT_EQ(source->lastTimeout, std::nullopt);
}

TEST(KVCacheEventReaderTest, RetainsSourceForReaderLifetime)
{
    std::weak_ptr<EventSource> weak;
    {
        auto source = std::make_shared<EventSource>();
        weak = source;
        auto reader = ex::detail::KVCacheEventManagerAccess::create(source);
        source.reset();
        EXPECT_FALSE(weak.expired());
        EXPECT_TRUE(reader.getLatestEvents(std::chrono::milliseconds{0}).empty());
    }
    EXPECT_TRUE(weak.expired());
}
