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

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/executor/kvCacheEventSource.h"

namespace tensorrt_llm::executor
{
namespace
{

class LegacyKVCacheEventSource : public detail::KVCacheEventSource
{
public:
    explicit LegacyKVCacheEventSource(std::shared_ptr<batch_manager::kv_cache_manager::BaseKVCacheManager> manager)
        : mManager{std::move(manager)}
    {
    }

    std::deque<KVCacheEvent> getLatestEvents(std::optional<std::chrono::milliseconds> timeout) override
    {
        return mManager->getLatestEvents(timeout);
    }

private:
    std::shared_ptr<batch_manager::kv_cache_manager::BaseKVCacheManager> mManager;
};

} // namespace

KVCacheEventManager::KVCacheEventManager(std::shared_ptr<batch_manager::kv_cache_manager::BaseKVCacheManager> manager)
    : KVCacheEventManager{std::make_shared<LegacyKVCacheEventSource>(std::move(manager)), EventSourceTag{}}
{
}

} // namespace tensorrt_llm::executor
