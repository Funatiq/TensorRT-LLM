/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/eagleBuffers.h"
#include "tensorrt_llm/runtime/explicitDraftTokensBuffers.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/lookaheadBuffers.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <optional>
#include <vector>

namespace tensorrt_llm::batch_manager
{

class DecoderInputBuffers
{
public:
    using SizeType32 = runtime::SizeType32;
    using TensorPtr = runtime::ITensor::SharedPtr;

    explicit DecoderInputBuffers(
        SizeType32 maxBatchSize, SizeType32 maxDecoderSteps, runtime::BufferManager const& manager);

    // buffers for setup
    TensorPtr setupBatchSlots;
    TensorPtr inputsIds;

    // buffers for forward
    TensorPtr forwardBatchSlotsRequestOrder;
    TensorPtr forwardBatchSlotsRequestOrderDevice;
    TensorPtr fillValues;
    TensorPtr fillValuesDevice;
    std::vector<TensorPtr> forwardBatchSlots;
};

class DecoderBuffers
{
public:
    using SizeType32 = runtime::SizeType32;
    using TensorPtr = runtime::ITensor::SharedPtr;

    std::vector<TensorPtr> logits;
    TensorPtr cacheIndirectionInput;
    TensorPtr cacheIndirectionOutput;
    TensorPtr sequenceLengthsHost; // [mMaxNumRequests, beamWidth], pinned host tensor
    TensorPtr newOutputTokensHost; // [maxTokensPerStep, mMaxNumRequests, beamWidth]
    TensorPtr cumLogProbsHost;     // [mMaxNumRequests, beamWidth]
    TensorPtr logProbsHost;        // [mMaxNumRequests, beamWidth, maxSeqLen]
    TensorPtr finishedSumHost;     // [mMaxNumRequests], pinned host tensor
    TensorPtr finishReasonsHost;   // [mMaxNumRequests, beamWidth], pinned host tensor

    class DraftBuffers
    {
    public:
        TensorPtr nextDraftTokensDevice;        // [mMaxNumRequests, maxTokensPerStep-1]
        TensorPtr nextDraftTokensHost;          // [mMaxNumRequests, maxTokensPerStep-1]
        TensorPtr prevDraftTokensLengthsDevice; // [mMaxNumRequests]
        TensorPtr prevDraftTokensLengthsHost;   // [mMaxNumRequests]
        TensorPtr nextDraftTokensLengthsDevice; // [mMaxNumRequests]
        TensorPtr nextDraftTokensLengthsHost;   // [mMaxNumRequests]
        TensorPtr acceptedLengthsCumSumDevice;  // [mMaxNumRequests+1]
        TensorPtr acceptedPackedPathsDevice;    // [mMaxNumRequests * maxAcceptedTokens]
        std::vector<std::vector<runtime::ITensor::SharedPtr>>
            predictedDraftLogits;               // [mMaxNumRequests][mMaxNumHeads][maxDraftTokens + 1, vocabSize]

        void create(SizeType32 maxNumSequences, SizeType32 maxTokensPerStep, runtime::BufferManager const& manager,
            runtime::ModelConfig const& modelConfig);
    };

    DraftBuffers draftBuffers;
    runtime::ExplicitDraftTokensBuffers::Inputs explicitDraftTokensBuffers;
    runtime::EagleBuffers::Inputs eagleBuffers;
    std::optional<runtime::LookaheadDecodingBuffers> lookaheadBuffers;

    DecoderBuffers(SizeType32 maxNumSequences, SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow,
        SizeType32 maxSeqLen, SizeType32 maxTokensPerStep, runtime::BufferManager const& manager,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);

    void enableLookaheadDecoding(SizeType32 maxNumSequences, SizeType32 maxTokensPerStep);
    void disableLookaheadDecoding(SizeType32 maxNumSequences);
};

class DecoderStepAsyncSend
{
public:
    using SizeType32 = runtime::SizeType32;
    using BufferPtr = runtime::IBuffer::SharedPtr;

    DecoderStepAsyncSend(DecoderBuffers const& decoderBuffers, bool returnLogProbs, SizeType32 maxBeamWidth,
        bool useMedusa, std::shared_ptr<mpi::MpiComm> const& commSession, int peer);

    ~DecoderStepAsyncSend();

    static void recv(DecoderBuffers const& decoderBuffers, bool returnLogProbs, SizeType32 maxBeamWidth, bool useMedusa,
        std::shared_ptr<mpi::MpiComm> const& commSession, int peer);

    static void bcast(DecoderBuffers const& decoderBuffers, bool returnLogProbs, SizeType32 maxBeamWidth,
        bool useMedusa, std::shared_ptr<mpi::MpiComm> const& commSession, int root);

    static auto constexpr kMpiTagOffset = 0;
    static auto constexpr kMpiTagUpperBound = kMpiTagOffset + 9;

private:
    std::shared_ptr<mpi::MpiRequest> mRequest1;
    std::shared_ptr<mpi::MpiRequest> mRequest2;
    std::shared_ptr<mpi::MpiRequest> mRequest3;
    std::shared_ptr<mpi::MpiRequest> mRequest4;
    std::shared_ptr<mpi::MpiRequest> mRequest5;
    std::shared_ptr<mpi::MpiRequest> mRequest6;
    std::shared_ptr<mpi::MpiRequest> mRequest7;
    std::shared_ptr<mpi::MpiRequest> mRequest8;
    std::shared_ptr<mpi::MpiRequest> mRequest9;
};

class SlotDecoderBuffers
{
public:
    using SizeType32 = runtime::SizeType32;
    using TensorPtr = runtime::ITensor::SharedPtr;

    TensorPtr outputIds;           // [beamWidth, maxSeqLen], outputIds of single batch slot
    TensorPtr outputIdsHost;       // [beamWidth, maxSeqLen], outputIds of single batch slot
    TensorPtr sequenceLengths;     // [beamWidth]
    TensorPtr sequenceLengthsHost; // [beamWidth]
    TensorPtr cumLogProbs;         // [beamWidth]
    TensorPtr cumLogProbsHost;     // [beamWidth]
    TensorPtr logProbs;            // [beamWidth, maxSeqLen]
    TensorPtr logProbsHost;        // [beamWidth, maxSeqLen]
    TensorPtr finishReasonsHost;   // [beamWidth]

    SlotDecoderBuffers(SizeType32 maxBeamWidth, SizeType32 maxSeqLen, runtime::BufferManager const& manager);
};

class DecoderSlotAsyncSend
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;

    DecoderSlotAsyncSend(TensorPtr const& outputIds, TensorPtr const& sequenceLengths, TensorPtr const& cumLogProbs,
        TensorPtr const& logProbs, bool returnLogProbs, std::shared_ptr<mpi::MpiComm> const& commSession, int peer);

    DecoderSlotAsyncSend(SlotDecoderBuffers const& slotDecoderBuffers, bool returnLogProbs,
        std::shared_ptr<mpi::MpiComm> const& commSession, int peer);

    ~DecoderSlotAsyncSend();

    static void recv(SlotDecoderBuffers const& slotDecoderBuffers, bool returnLogProbs,
        std::shared_ptr<mpi::MpiComm> const& commSession, int peer);

    static auto constexpr kMpiTagOffset = 9;
    static auto constexpr kMpiTagUpperBound = kMpiTagOffset + 4;
    static_assert(kMpiTagOffset >= DecoderStepAsyncSend::kMpiTagUpperBound);

private:
    std::shared_ptr<mpi::MpiRequest> mRequest1;
    std::shared_ptr<mpi::MpiRequest> mRequest2;
    std::shared_ptr<mpi::MpiRequest> mRequest3;
    std::shared_ptr<mpi::MpiRequest> mRequest4;
};

} // namespace tensorrt_llm::batch_manager
