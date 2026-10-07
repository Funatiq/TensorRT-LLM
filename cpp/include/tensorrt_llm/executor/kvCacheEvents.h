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

#pragma once

#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/common.h"

#include <chrono>
#include <deque>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
class BaseKVCacheManager;
} // namespace tensorrt_llm::batch_manager::kv_cache_manager

namespace tensorrt_llm::executor
{
namespace detail
{
class KVCacheEventSource;
class KVCacheEventManagerAccess;
} // namespace detail

struct KVCacheCreatedData
{
    /// @brief The amount of blocks at each cache level
    std::vector<SizeType32> numBlocksPerCacheLevel;
};

/// @brief An entry for a single block stored into the tree
struct KVCacheStoredBlockData
{

    KVCacheStoredBlockData(IdType blockHash, tensorrt_llm::runtime::VecUniqueTokens tokens,
        std::optional<tensorrt_llm::runtime::LoraTaskIdType> loraId, SizeType32 cacheLevel, SizeType32 priority,
        std::vector<MmKey> mmKeys = {}, std::optional<std::string> cacheSalt = std::nullopt)
        : blockHash{blockHash}
        , tokens{std::move(tokens)}
        , loraId{loraId}
        , cacheLevel{cacheLevel}
        , priority{priority}
        , mmKeys{std::move(mmKeys)}
        , cacheSalt{std::move(cacheSalt)}
    {
    }

    /// @brief The hash of the block
    IdType blockHash;
    /// @brief The unique tokens of the block
    tensorrt_llm::runtime::VecUniqueTokens tokens;
    /// @brief The Lora task id of the block
    std::optional<tensorrt_llm::runtime::LoraTaskIdType> loraId;
    /// @brief The cache level of the block
    SizeType32 cacheLevel;
    /// @brief The priority of the block
    SizeType32 priority;
    /// @brief The multimodal keys of the block
    std::vector<MmKey> mmKeys;
    /// @brief The original cache salt string of the block, if any
    std::optional<std::string> cacheSalt;
};

struct KVCacheStoredData
{
    /// @brief The parent of this sequence of stored blocks
    std::optional<IdType> parentHash;
    /// @brief A sequence of blocks. The parent of block `i` is block `i-1`
    std::vector<KVCacheStoredBlockData> blocks;
};

struct KVCacheRemovedData
{
    /// @brief The hashes of blocks being removed
    std::vector<IdType> blockHashes;
};

template <typename T>
struct KVCacheEventDiff
{
    T oldValue;
    T newValue;
};

struct KVCacheUpdatedData
{

    explicit KVCacheUpdatedData(IdType blockHash)
        : blockHash{blockHash} {};

    explicit KVCacheUpdatedData(IdType blockHash, std::optional<KVCacheEventDiff<SizeType32>> cacheLevel,
        std::optional<KVCacheEventDiff<SizeType32>> priority)
        : blockHash{blockHash}
        , cacheLevel{cacheLevel}
        , priority{priority} {};

    KVCacheUpdatedData& cacheLevelUpdated(SizeType32 oldValue, SizeType32 newValue)
    {
        cacheLevel = KVCacheEventDiff<SizeType32>{oldValue, newValue};
        return *this;
    }

    KVCacheUpdatedData& priorityUpdated(SizeType32 oldValue, SizeType32 newValue)
    {
        priority = KVCacheEventDiff<SizeType32>{oldValue, newValue};
        return *this;
    }

    /// @brief The hash of the updated block
    IdType blockHash;
    /// @brief The updated value of the cacheLevel field
    std::optional<KVCacheEventDiff<SizeType32>> cacheLevel = std::nullopt;
    /// @brief The updated value of the priority field
    std::optional<KVCacheEventDiff<SizeType32>> priority = std::nullopt;
};

using KVCacheEventData = std::variant<KVCacheCreatedData, KVCacheStoredData, KVCacheRemovedData, KVCacheUpdatedData>;

struct KVCacheEvent
{
    KVCacheEvent(IdType eventId, KVCacheEventData data, SizeType32 windowSize,
        std::optional<SizeType32> attentionDpRank = std::nullopt)
        : eventId{eventId}
        , data{std::move(data)}
        , windowSize{windowSize}
        , attentionDpRank{attentionDpRank}
    {
    }

    /// @brief The unique id of this event
    IdType eventId;
    /// @brief The data corresponding to this event
    KVCacheEventData data;
    /// @brief The sliding window size
    SizeType32 windowSize;
    /// @brief The attention DP rank of the event, if applicable
    std::optional<SizeType32> attentionDpRank;
};

/// @brief Exposes a limited set of KV cache manager functionalities
class KVCacheEventManager
{
public:
    KVCacheEventManager(
        std::shared_ptr<tensorrt_llm::batch_manager::kv_cache_manager::BaseKVCacheManager> kvCacheManager);

    /// @brief Get the latest KV Cache events.
    /// @param timeout The maximum time to wait for new events. If nullopt, will only return when new events are
    /// available, or when the executor instance has shutdown.
    std::deque<KVCacheEvent> getLatestEvents(std::optional<std::chrono::milliseconds> timeout = std::nullopt);

private:
    friend class detail::KVCacheEventManagerAccess;

    struct EventSourceTag
    {
    };

    KVCacheEventManager(std::shared_ptr<detail::KVCacheEventSource> source, EventSourceTag);
    std::shared_ptr<detail::KVCacheEventSource> mSource;
};

} // namespace tensorrt_llm::executor
