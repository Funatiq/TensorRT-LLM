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

#include "tensorrt_llm/batch_manager/cacheBufferKind.h"
#include "tensorrt_llm/batch_manager/requestInfo.h"
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/executor/transferAgent.h"

namespace tensorrt_llm::executor::kv_cache
{

struct RequestAndBufferInfo
{
    std::string mAgentName;
    std::string mAddress;
    batch_manager::RequestInfo mRequestInfo;
    std::vector<MemoryDesc> mBufferDescs;
    std::optional<std::string> mMetadata;
    int mValidConnectionIdx;
    std::vector<uint8_t> mBufferKinds;

    static void serialize(RequestAndBufferInfo const& requestAndBufferInfo, std::ostream& os)
    {
        namespace su = executor::serialize_utils;
        su::serialize(requestAndBufferInfo.mAgentName, os);
        su::serialize(requestAndBufferInfo.mAddress, os);
        batch_manager::RequestInfo::serialize(requestAndBufferInfo.mRequestInfo, os);
        su::serialize(requestAndBufferInfo.mBufferDescs.size(), os);
        for (auto const& bufferDesc : requestAndBufferInfo.mBufferDescs)
        {
            MemoryDesc::serialize(bufferDesc, os);
        }
        su::serialize(requestAndBufferInfo.mMetadata, os);
        su::serialize(requestAndBufferInfo.mValidConnectionIdx, os);
        su::serialize(requestAndBufferInfo.mBufferKinds.size(), os);
        for (auto kind : requestAndBufferInfo.mBufferKinds)
        {
            su::serialize(kind, os);
        }
    }

    static RequestAndBufferInfo deserialize(std::istream& is)
    {
        namespace su = executor::serialize_utils;
        auto agentName = su::deserialize<decltype(mAgentName)>(is);
        auto address = su::deserialize<decltype(mAddress)>(is);
        auto requestInfo = batch_manager::RequestInfo::deserialize(is);
        auto bufferDescsSize = su::deserialize<decltype(mBufferDescs.size())>(is);
        std::vector<MemoryDesc> bufferDescs;
        bufferDescs.reserve(bufferDescsSize);
        for (size_t i = 0; i < bufferDescsSize; i++)
        {
            bufferDescs.emplace_back(MemoryDesc::deserialize(is));
        }
        auto metadata = su::deserialize<decltype(mMetadata)>(is);
        auto validConnectionIdx = su::deserialize<decltype(mValidConnectionIdx)>(is);
        auto bufferKindsSize = su::deserialize<size_t>(is);
        std::vector<uint8_t> bufferKinds;
        bufferKinds.reserve(bufferKindsSize);
        for (size_t i = 0; i < bufferKindsSize; i++)
        {
            bufferKinds.push_back(su::deserialize<uint8_t>(is));
        }
        return RequestAndBufferInfo{
            agentName, address, requestInfo, bufferDescs, metadata, validConnectionIdx, bufferKinds};
    }

    static size_t serializedSize(RequestAndBufferInfo const& requestAndBufferInfo)
    {
        namespace su = executor::serialize_utils;
        size_t totalSize = 0;
        totalSize += su::serializedSize(requestAndBufferInfo.mAgentName);
        totalSize += su::serializedSize(requestAndBufferInfo.mAddress);
        totalSize += batch_manager::RequestInfo::serializedSize(requestAndBufferInfo.mRequestInfo);
        totalSize += su::serializedSize(requestAndBufferInfo.mBufferDescs.size());
        for (auto const& bufferDesc : requestAndBufferInfo.mBufferDescs)
        {
            totalSize += MemoryDesc::serializedSize(bufferDesc);
        }
        totalSize += su::serializedSize(requestAndBufferInfo.mMetadata);
        totalSize += su::serializedSize(requestAndBufferInfo.mValidConnectionIdx);
        totalSize += su::serializedSize(requestAndBufferInfo.mBufferKinds.size());
        totalSize += requestAndBufferInfo.mBufferKinds.size() * su::serializedSize(uint8_t{});
        return totalSize;
    }
};

struct ReadySignalInfo
{
    std::string mAgentName;
    DataContext mContext;

    bool mIsReady;

    static void serialize(ReadySignalInfo const& readySignalInfo, std::ostream& os)
    {
        namespace su = executor::serialize_utils;
        su::serialize(readySignalInfo.mAgentName, os);
        su::serialize(readySignalInfo.mContext.getTag(), os);
        su::serialize(readySignalInfo.mIsReady, os);
    }

    static ReadySignalInfo deserialize(std::istream& is)
    {
        namespace su = executor::serialize_utils;
        auto agentName = su::deserialize<decltype(mAgentName)>(is);
        auto contextTag = su::deserialize<decltype(mContext.getTag())>(is);
        DataContext context{contextTag};
        auto isReady = su::deserialize<decltype(mIsReady)>(is);
        return ReadySignalInfo{agentName, context, isReady};
    }

    static size_t serializedSize(ReadySignalInfo const& readySignalInfo)
    {
        namespace su = executor::serialize_utils;
        return su::serializedSize(readySignalInfo.mAgentName) + su::serializedSize(readySignalInfo.mContext.getTag())
            + su::serializedSize(readySignalInfo.mIsReady);
    }
};

struct NotificationSyncInfo
{

    std::string mAgentName;
    DataContext mContext;

    static void serialize(NotificationSyncInfo const& notificationSyncInfo, std::ostream& os)
    {
        namespace su = executor::serialize_utils;
        su::serialize(notificationSyncInfo.mAgentName, os);
        su::serialize(notificationSyncInfo.mContext.getTag(), os);
    }

    static NotificationSyncInfo deserialize(std::istream& is)
    {
        namespace su = executor::serialize_utils;
        auto agentName = su::deserialize<decltype(mAgentName)>(is);
        auto contextTag = su::deserialize<decltype(mContext.getTag())>(is);
        DataContext context{contextTag};
        return NotificationSyncInfo{agentName, context};
    }

    static size_t serializedSize(NotificationSyncInfo const& notificationSyncInfo)
    {
        namespace su = executor::serialize_utils;
        return su::serializedSize(notificationSyncInfo.mAgentName)
            + su::serializedSize(notificationSyncInfo.mContext.getTag());
    }
};

struct NotificationInfo
{

    std::variant<RequestAndBufferInfo, NotificationSyncInfo, ReadySignalInfo> mInfo;

    static void serialize(NotificationInfo const& notificationInfo, std::ostream& os)
    {
        namespace su = executor::serialize_utils;
        su::serialize(notificationInfo.mInfo.index(), os);
        if (std::holds_alternative<RequestAndBufferInfo>(notificationInfo.mInfo))
        {
            RequestAndBufferInfo::serialize(std::get<RequestAndBufferInfo>(notificationInfo.mInfo), os);
        }
        else if (std::holds_alternative<NotificationSyncInfo>(notificationInfo.mInfo))
        {
            NotificationSyncInfo::serialize(std::get<NotificationSyncInfo>(notificationInfo.mInfo), os);
        }
        else if (std::holds_alternative<ReadySignalInfo>(notificationInfo.mInfo))
        {
            ReadySignalInfo::serialize(std::get<ReadySignalInfo>(notificationInfo.mInfo), os);
        }
        else
        {
            TLLM_THROW("Unknown variant type");
        }
    }

    static NotificationInfo deserialize(std::istream& is)
    {
        namespace su = executor::serialize_utils;
        auto variantIdx = su::deserialize<std::size_t>(is);
        constexpr std::size_t requestAndBufferInfoIdx{0};
        constexpr std::size_t notificationSyncInfoIdx{1};
        constexpr std::size_t readySignalInfoIdx{2};
        if (variantIdx == requestAndBufferInfoIdx)
        {
            return NotificationInfo{RequestAndBufferInfo::deserialize(is)};
        }
        else if (variantIdx == notificationSyncInfoIdx)
        {
            return NotificationInfo{NotificationSyncInfo::deserialize(is)};
        }
        else if (variantIdx == readySignalInfoIdx)
        {
            return NotificationInfo{ReadySignalInfo::deserialize(is)};
        }
        else
        {
            TLLM_THROW("Unknown variant type");
        }
    }

    static size_t serializedSize(NotificationInfo const& notificationInfo)
    {
        namespace su = executor::serialize_utils;
        size_t totalSize = 0;
        totalSize += su::serializedSize(notificationInfo.mInfo.index());
        if (std::holds_alternative<RequestAndBufferInfo>(notificationInfo.mInfo))
        {
            totalSize += RequestAndBufferInfo::serializedSize(std::get<RequestAndBufferInfo>(notificationInfo.mInfo));
        }
        else if (std::holds_alternative<NotificationSyncInfo>(notificationInfo.mInfo))
        {
            totalSize += NotificationSyncInfo::serializedSize(std::get<NotificationSyncInfo>(notificationInfo.mInfo));
        }
        else if (std::holds_alternative<ReadySignalInfo>(notificationInfo.mInfo))
        {
            totalSize += ReadySignalInfo::serializedSize(std::get<ReadySignalInfo>(notificationInfo.mInfo));
        }
        else
        {
            TLLM_THROW("Unknown variant type");
        }
        return totalSize;
    }
};
} // namespace tensorrt_llm::executor::kv_cache
