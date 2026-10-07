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

#include "tensorrt_llm/batch_manager/blockKey.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"

#include <iosfwd>

namespace tensorrt_llm::batch_manager
{

using BlockKey = kv_cache_manager::BlockKey;

// Used to store the information that needs to be sent to the context executor to ensure the generation
// executor smoothly receives the data.
class RequestInfo
{
public:
    /// @brief Constructor.
    /// @param requestId The ID used in the context phase of the current request.
    /// @param transState The state of the data transceiver.
    RequestInfo(RequestIdType requestId, executor::DataTransceiverState transState);

    RequestInfo(RequestIdType requestId, executor::DataTransceiverState transState, int32_t indexFromEnd,
        BlockKey const& lastBlockKey);
    RequestInfo() = default;

    /// @brief Equality comparison operator.
    /// @param rhs The right operand of the operator.
    [[nodiscard]] bool operator==(RequestInfo const& rhs) const;

    /// @brief Return the ID used in the context phase of the current request.
    /// @return The request ID.
    [[nodiscard]] RequestIdType getRequestId() const noexcept;

    [[nodiscard]] int32_t getIndexFromEnd() const noexcept
    {
        return mIndexFromEnd;
    }

    /// @brief Return the state of the data transceiver.
    /// @return The state of the data transceiver.
    [[nodiscard]] executor::DataTransceiverState const& getTransState() const noexcept;

    [[nodiscard]] BlockKey const& getLastBlockKey() const noexcept
    {
        return mLastBlockKey;
    }

    /// @brief Arbitrary (llmRequest-agnostic) transfer served from the sender's reuse tree.
    [[nodiscard]] bool isArbitraryTransfer() const noexcept
    {
        return mIsArbitraryTransfer;
    }

    void setIsArbitraryTransfer(bool isArbitraryTransfer) noexcept
    {
        mIsArbitraryTransfer = isArbitraryTransfer;
    }

    /// @brief Serialization.
    /// @param requestInfo Request information to be serialized.
    /// @param os The output stream to which the serialization result points.
    static void serialize(RequestInfo const& requestInfo, std::ostream& os);

    /// @brief Deserialization.
    /// @return The request information obtained from deserialization.
    [[nodiscard]] static RequestInfo deserialize(std::istream& is);

    /// @brief The number of bytes occupied by the serialized data structure.
    /// @param requestInfo Request information to be serialized.
    /// @return The number of bytes.
    [[nodiscard]] static std::size_t serializedSize(RequestInfo const& requestInfo);

private:
    // The ID used in the context phase of the current request.
    RequestIdType mRequestId;
    // Index from end indicating how many trailing blocks to transfer (index+1)
    int32_t mIndexFromEnd{0};

    // Last block key, used to derive other block keys on receiver
    BlockKey mLastBlockKey{};

    // True for arbitrary (llmRequest-agnostic) transfers served from the sender's reuse tree.
    bool mIsArbitraryTransfer{false};

    // The state of the data transceiver.
    executor::DataTransceiverState mTransState;
};

} // namespace tensorrt_llm::batch_manager
