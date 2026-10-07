/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "agentName.h"

#include "tensorrt_llm/batch_manager/requestInfo.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/executor/cache_transmission/cacheTransmissionInfo.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/transferAgent.h"
#include <map>

namespace tensorrt_llm::batch_manager
{
class BaseTransBufferManager;
} // namespace tensorrt_llm::batch_manager

namespace tensorrt_llm::executor::kv_cache
{

class AgentConnectionManager;

class AgentConnection : public Connection
{
public:
    AgentConnection(
        std::string mAgentName, std::string mRemoteAgentName, AgentConnectionManager* mAgentConnectionManager);
    void send(DataContext const& ctx, void const* data, size_t size) const override;
    void recv(DataContext const& ctx, void* data, size_t size) const override;
    void sendRequestAndBufferInfo(batch_manager::RequestInfo& requestInfo,
        std::vector<std::optional<size_t>> const& cacheBufferIds, int validConnectionIdx,
        std::atomic<bool> const* perRequestCancel = nullptr);
    void setSenderState(std::vector<MemoryDesc> cacheReceiverBufferDescs, int validSegmentIdx,
        std::vector<std::pair<size_t, size_t>> offsetRatios, std::vector<uint8_t> bufferKinds);
    void setHasLoadRemoteAgent(bool hasLoadRemoteAgent);
    [[nodiscard]] bool hasLoadRemoteAgent() const;
    void sendReadySignal(DataContext const& ctx, bool isReady) const;
    bool recvReadySignal(DataContext const& ctx) const;
    std::optional<bool> recvReadySignalWithStatus(DataContext const& ctx) const;

    void activateBuffer(uint8_t kind) const override;
    [[nodiscard]] std::optional<size_t> getPreAssignedBufferId(uint8_t kind) const override;

private:
    std::string mAgentName;
    std::string mRemoteAgentName;

    struct SenderState
    {
        std::vector<MemoryDesc> mCacheReceiverBufferDescs;
        int validSegmentIdx{0};
        /// Per-buffer offset ratios. Index corresponds to mCacheReceiverBufferDescs / mActiveBufferIdx.
        std::vector<std::pair<size_t, size_t>> mOffsetRatios;
        mutable size_t mActiveBufferIdx{0};
        [[nodiscard]] MemoryDesc const& activeBufferDesc() const;
        [[nodiscard]] std::pair<size_t, size_t> const& activeOffsetRatio() const;
        void setActiveBufferIdx(size_t bufferIdx) const;
        SenderState() = default;
    };

    AgentConnectionManager* mAgentConnectionManager;

    std::vector<batch_manager::BaseTransBufferManager*> const& mCacheTransBufferManagers;
    std::vector<std::optional<size_t>> mCacheBufferIds;
    std::vector<uint8_t> mBufferKinds;
    mutable SenderState mSenderState;
    bool mNeedSendMetadata{true};
    bool mHasLoadRemoteAgent{false};
};

class AgentConnectionManager : public ConnectionManager
{
public:
    AgentConnectionManager(std::vector<batch_manager::BaseTransBufferManager*> cacheTransBufferManagers,
        CacheState cacheState, std::string const& backendType,
        std::optional<CacheState::RnnCacheState> rnnCacheState = std::nullopt);
    ~AgentConnectionManager();
    AgentConnection* recvConnect(DataContext const& ctx, void* data, size_t size) override;
    [[nodiscard]] std::vector<Connection const*> getConnections(CommState const& state) override;
    [[nodiscard]] CommState const& getCommState() const override;
    AgentConnection const* recvConnectionAndRequestInfo(
        batch_manager::RequestInfo& requestInfo, std::atomic<bool> const& terminateFlag);
    [[nodiscard]] std::vector<batch_manager::BaseTransBufferManager*> const& getCacheTransBufferManagers() const;
    [[nodiscard]] std::vector<uint8_t> const& getBufferKinds() const;
    void updateUnhandledNotifications();
    [[nodiscard]] BaseTransferAgent* getAgent() const;
    AgentConnection* connect(std::string const& remoteAgentName, std::string const& address,
        std::optional<std::string> metadata = std::nullopt, bool isSender = false);
    int getDeviceId() const;
    [[nodiscard]] std::string const& getAgentName() const;

    template <typename NotificationType>
    bool waitForNotification(
        std::string const& remoteAgentName, NotificationType& expectedInfo, std::atomic<bool> const& terminateFlag);
    bool waitForSyncInfo(
        std::string const& remoteAgentName, NotificationSyncInfo& syncInfo, std::atomic<bool> const& terminateFlag);
    bool waitForReadySignal(
        std::string const& remoteAgentName, ReadySignalInfo& readySignalInfo, std::atomic<bool> const& terminateFlag);
    [[nodiscard]] bool isRunning() const override;

private:
    std::map<std::string, std::shared_ptr<AgentConnection>> mConnections;
    std::mutex mConnectionsMutex;
    /// Connection info for dynamically discovered agents that are not listed in mCommState.
    std::map<std::string, std::string> mRemoteConnectionInfo;
    std::mutex mRemoteConnectionInfoMutex;
    CommState mCommState;
    CacheState mCacheState;
    std::optional<CacheState::RnnCacheState> mRnnCacheState;
    std::vector<batch_manager::BaseTransBufferManager*> mCacheTransBufferManagers;
    std::vector<uint8_t> mBufferKinds;
    std::mutex mNotificationMutex;
    std::unordered_map<std::string, std::list<std::string>> mUnhandledNotifications;
    std::unique_ptr<BaseTransferAgent> m_Agent;
    int mDeviceId;
    int mRank{0};
    int mWorldSize{1};
    std::string mAgentName;
    MemoryDescs mRegMemDescs;
    std::atomic<bool> mIsRunning{true};
};

} // namespace tensorrt_llm::executor::kv_cache
