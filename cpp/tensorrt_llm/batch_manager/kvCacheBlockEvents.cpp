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
#include "tensorrt_llm/batch_manager/kvCacheManager.h"

namespace tle = tensorrt_llm::executor;

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

void KVCacheEventManager::enqueueStoredEvent(std::vector<BlockPtr> const& blocks, SizeType32 windowSize)
{
    if (blocks.empty())
    {
        return;
    }

    flushRemovedEvents(windowSize);

    auto const parentBlock = blocks.front()->getPrevBlock();
    auto const parent = (parentBlock != nullptr && parentBlock->getBlockId() >= 0)
        ? std::optional<size_t>(parentBlock->getHash())
        : std::nullopt;

    tle::KVCacheStoredData data{parent, {}};

    for (auto const& block : blocks)
    {
        data.blocks.emplace_back(block->getHash(), block->getUniqueTokens(), block->getBlockKey().loraTaskId,
            block->isPrimary() ? kPrimaryLevel : kSecondaryLevel, block->getPriority(), block->getExtraKeys(),
            block->getBlockKey().cacheSalt);
    }

    enqueueEvent({mEventId++, data, windowSize, mAttentionDpRank});
}

void KVCacheEventManager::enqueueRemovedEvent(BlockPtr const& block, SizeType32 windowSize)
{
    auto& latestRemovedEvent = mLatestRemovedEvents[windowSize];
    if (latestRemovedEvent != std::nullopt)
    {
        latestRemovedEvent->blockHashes.push_back(block->getHash());
    }
    else
    {
        latestRemovedEvent = tle::KVCacheRemovedData{{block->getHash()}};
    }
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
