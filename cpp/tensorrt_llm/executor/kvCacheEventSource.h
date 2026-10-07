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

#include "tensorrt_llm/executor/kvCacheEvents.h"

namespace tensorrt_llm::executor::detail
{

class KVCacheEventSource
{
public:
    virtual ~KVCacheEventSource() = default;
    virtual std::deque<KVCacheEvent> getLatestEvents(std::optional<std::chrono::milliseconds> timeout) = 0;
};

class KVCacheEventManagerAccess
{
public:
    static KVCacheEventManager create(std::shared_ptr<KVCacheEventSource> source)
    {
        return KVCacheEventManager{std::move(source), KVCacheEventManager::EventSourceTag{}};
    }
};

} // namespace tensorrt_llm::executor::detail
