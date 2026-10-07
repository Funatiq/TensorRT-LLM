/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/batch_manager/cacheTransBuffer.h"
#include "tensorrt_llm/batch_manager/cacheTransceiverComm.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/rnnCacheTransBuffer.h"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/pgUtils.h"
#include <atomic>
#include <cstddef>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <torch/custom_class.h>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using SizeType32 = tensorrt_llm::runtime::SizeType32;

namespace tensorrt_llm::batch_manager
{

class ContextProgress;
class BaseCacheTransceiver;

namespace kv_cache_manager
{
class BaseKVCacheManager;
} // namespace kv_cache_manager

class CacheSender;
class CacheReceiver;
class ContextTransferCoordinator;

class CacheTransceiverFactory
{
public:
    static std::unique_ptr<BaseCacheTransceiver> createCacheTransceiver(
        kv_cache_manager::BaseKVCacheManager* cacheManager, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig,
        executor::kv_cache::CacheState::AttentionType attentionType
        = executor::kv_cache::CacheState::AttentionType::kDEFAULT,
        std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig = std::nullopt);
};

struct RequestStatuses
{
    /// Requests that have completed their transfer successfully.
    std::unordered_set<LlmRequest::RequestIdType> completedRequestIds;
    /// Requests that have encountered an error during their transfer.
    std::unordered_set<LlmRequest::RequestIdType> errorRequestIds;
};

class BaseCacheTransceiver
{
public:
    virtual ~BaseCacheTransceiver() = default;
    // Async entry points take shared_ptr so the async worker holds a strong
    // reference for the transfer lifetime; see CacheTransceiver::mSenderFutures.
    virtual void respondAndSendAsync(std::shared_ptr<LlmRequest> llmRequest) = 0;
    virtual void respondAndSendLayerWise(
        RequestVector const& requests, std::shared_ptr<ContextProgress> const& progress)
        = 0;

    virtual void requestAndReceiveSync(std::shared_ptr<LlmRequest> llmRequest) = 0;
    virtual void requestAndReceiveAsync(std::shared_ptr<LlmRequest> llmRequest) = 0;

    /// Check all requests transferring context, and return the requests that have completed or encountered an error.
    virtual RequestStatuses checkContextTransferStatus(
        std::optional<int> const& atLeastRequestNum = std::nullopt, bool markComplete = false)
        = 0;

    virtual void checkGenTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) = 0;

    [[nodiscard]] virtual bool checkGenTransferComplete() const = 0;

    virtual bool cancelRequest(std::shared_ptr<LlmRequest> llmRequest) = 0;

    /// Get the serialized DataTransceiverState (CacheState + CommState) for this transceiver.
    [[nodiscard]] virtual std::vector<char> getSerializedDataTransceiverState() const
    {
        return {};
    }

    [[nodiscard]] virtual bool hasPoisonedTransferBuffer() const
    {
        return false;
    }
};

class CacheTransceiver : public BaseCacheTransceiver
{
public:
    CacheTransceiver(kv_cache_manager::BaseKVCacheManager* cacheManager,
        executor::kv_cache::CacheState::ModelConfig const& cacheStateModelCfg, runtime::WorldConfig const& worldConfig,
        std::vector<SizeType32> const& attentionLayerNumPerPP, tensorrt_llm::DataType dataType,
        executor::kv_cache::CacheState::AttentionType attentionType
        = executor::kv_cache::CacheState::AttentionType::kDEFAULT,
        std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig = std::nullopt,
        std::vector<SizeType32> const& rnnLayerNumPerPP = {}, std::vector<SizeType32> const& indexerLayerNumPerPP = {});

    CacheTransceiver(kv_cache_manager::BaseKVCacheManager* cacheManager, std::vector<SizeType32> numKvHeadsPerLayer,
        SizeType32 sizePerHead, SizeType32 tokensPerBlock, runtime::WorldConfig const& worldConfig,
        std::vector<SizeType32> const& attentionLayerNumPerPP, tensorrt_llm::DataType dataType,
        executor::kv_cache::CacheState::AttentionType attentionType
        = executor::kv_cache::CacheState::AttentionType::kDEFAULT,
        std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig = std::nullopt,
        std::vector<SizeType32> const& rnnLayerNumPerPP = {}, std::vector<SizeType32> const& indexerLayerNumPerPP = {})
        : CacheTransceiver(cacheManager,
            executor::kv_cache::CacheState::ModelConfig{numKvHeadsPerLayer, sizePerHead, tokensPerBlock}, worldConfig,
            attentionLayerNumPerPP, dataType, attentionType, cacheTransceiverConfig, rnnLayerNumPerPP,
            indexerLayerNumPerPP)
    {
    }

    virtual ~CacheTransceiver();

    void respondAndSendAsync(std::shared_ptr<LlmRequest> llmRequest) override;

    void respondAndSendLayerWise(
        RequestVector const& requests, std::shared_ptr<ContextProgress> const& progress) override;

    void requestAndReceiveSync(std::shared_ptr<LlmRequest> llmRequest) override;
    void requestAndReceiveAsync(std::shared_ptr<LlmRequest> llmRequest) override;

    RequestStatuses checkContextTransferStatus(
        std::optional<int> const& atLeastRequestNum = std::nullopt, bool markComplete = false) override;

    void checkGenTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) override;

    [[nodiscard]] bool checkGenTransferComplete() const override;

    virtual bool cancelRequest(std::shared_ptr<LlmRequest> llmRequest) override;

    [[nodiscard]] std::vector<char> getSerializedDataTransceiverState() const override;

    [[nodiscard]] bool hasPoisonedTransferBuffer() const override;
    /// Return a human-readable dump of transceiver state for debugging hangs.
    std::string getStatusDump() const;

private:
    struct StatusSnapshot
    {
        size_t senderAsyncActive{0};
        size_t requesterAsyncActive{0};
        size_t timedOutSenders{0};
        size_t timedOutRequesters{0};
        size_t cancelingSenders{0};
        size_t cancelingRequesters{0};
        size_t completedSenders{0};
        size_t completedRequesters{0};
        size_t failedSenders{0};
        size_t failedRequesters{0};
        size_t sendersAwaitingConsensus{0};
        size_t requestersAwaitingConsensus{0};
    };

    class SyncRequesterStatusGuard
    {
    public:
        explicit SyncRequesterStatusGuard(CacheTransceiver& transceiver);
        ~SyncRequesterStatusGuard() noexcept;

        SyncRequesterStatusGuard(SyncRequesterStatusGuard const&) = delete;
        SyncRequesterStatusGuard& operator=(SyncRequesterStatusGuard const&) = delete;

    private:
        CacheTransceiver& mTransceiver;
    };

    void initializeCommState();

    void setContextState(LlmRequest* llmRequest);

    // Append one row per completed request to the gen-side transfer summary CSV. Opens the file
    // lazily on first use; expects timing to already be synced across ranks by the caller.
    void writeGenTransferSummary(std::vector<LlmRequest*> const& completedRequests);
    void publishStatusSnapshot() noexcept;

    std::unique_ptr<CacheSender> mCacheSender;
    std::unique_ptr<CacheReceiver> mCacheReceiver;
    // shared_ptr (not raw LlmRequest*) so the futures hold a strong reference for
    // the transfer lifetime; otherwise Python's _terminate_request can drop the
    // request while a C++ status check still dereferences it.
    std::vector<std::pair<std::shared_ptr<LlmRequest>, std::future<void>>> mSenderFutures;
    std::vector<std::pair<std::shared_ptr<LlmRequest>, std::future<void>>> mRequesterFutures;
    // Dedup timeout logs separately from accepted cancellation requests so a
    // backend that initially declines cancellation is retried on later polls.
    std::unordered_set<LlmRequest::RequestIdType> mTimedOutSenderIds;
    std::unordered_set<LlmRequest::RequestIdType> mTimedOutRequesterIds;
    std::unordered_set<LlmRequest::RequestIdType> mCancelRequestedSenderIds;
    std::unordered_set<LlmRequest::RequestIdType> mCancelRequestedRequesterIds;
    std::unordered_set<LlmRequest::RequestIdType> mCompletedSenderRequestIds;
    std::unordered_set<LlmRequest::RequestIdType> mFailedSenderRequestIds;
    std::unordered_map<LlmRequest::RequestIdType, std::shared_ptr<LlmRequest>> mSenderRequestsAwaitingConsensus;
    std::unordered_set<LlmRequest::RequestIdType> mCompletedRequesterRequestIds;
    std::unordered_set<LlmRequest::RequestIdType> mFailedRequesterRequestIds;
    std::unordered_map<LlmRequest::RequestIdType, std::shared_ptr<LlmRequest>> mRequesterRequestsAwaitingConsensus;
    std::atomic_size_t mSyncRequesterActive{0};
    // Live transfer containers are owned by the executor worker thread. Synchronous receive threads update only the
    // atomic count above. The executor publishes snapshots after state transitions, while the hang-detector thread only
    // copies the snapshot under this short lock.
    mutable std::mutex mStatusSnapshotMutex;
    StatusSnapshot mStatusSnapshot;
    mpi::MpiComm const* mMpiWorldComm{nullptr};

    std::shared_ptr<CacheTransceiverComm> mGroupComm;
    std::shared_ptr<CacheTransceiverComm> mGroupTensorParaComm, mGroupPipeParaComm, mGroupDataComm, mGroupTPInDPComm;
    std::unique_ptr<ContextTransferCoordinator> mContextTransferCoordinator;

    executor::kv_cache::CommState const* mCommState;
    std::unique_ptr<executor::kv_cache::CacheState> mCacheState;
    std::unique_ptr<executor::kv_cache::ConnectionManager> mManager;
    std::optional<executor::CacheTransceiverConfig> mCacheTransceiverConfig;
    std::vector<std::unique_ptr<kv_cache_manager::CacheTransBufferManager>> mCacheTransBufferManagers;
    std::vector<BaseTransBufferManager*> mCacheTransBufferManagerPtrs;

    // TODO(shreyasm): update this to use same container as kv by using base trans buffers instead
    std::unique_ptr<rnn_state_manager::RnnCacheTransBufferManager> mRnnCacheTransBufferManager{nullptr};

    // Unique instance identifier for CSV file naming (avoids collisions across gen instances)
    std::string mInstanceId;

    // Gen-side transfer summary CSV (written after timing sync)
    std::ofstream mGenTransferSummaryFile;
    std::mutex mGenTransferSummaryMutex;

    // library handle to the communicator related features,
    // this is used to defer dependency resolution until needed.
    static std::mutex mDllMutex;
    void* mWrapperLibHandle{nullptr};
};

} // namespace tensorrt_llm::batch_manager
