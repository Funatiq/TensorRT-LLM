/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/runtime/common.h"

namespace tensorrt_llm::layers
{

template <typename T>
class BeamSearchLayer : public BaseLayer
{
    using Base = BaseLayer;

public:
    BeamSearchLayer(DecoderDomain const& decoderDomain, std::shared_ptr<runtime::BufferManager> bufferManager);

    // Functions called after runtime data atrrives
    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

    // Functions called after runtime data atrrives
    void setup(runtime::SizeType32 const batchSize, runtime::SizeType32 const beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;
    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

private:
    // Functions called before runtime data atrrives
    void allocateBuffer();
    void configureBeamSearchLayer();

private:
    using Base::mDecoderDomain;

    size_t mByteMaxSharedMemoryPerBlock{0};   // Device information
    size_t mByteSharedMemoryStage1{0};        // Max dynamic shashared memoryn stage 1 kernel, useless in V2
    size_t mByteSharedMemoryStage3{0};        // Max static shared memory in stage 3 kernel
    size_t mVPart{0};                         // Count of parts the beamed-logProbs will be divided into, useless in V2
    size_t mWorkspaceSize{0};                 // Total workspace size for Beam Search kernels
    bool mV2{false};                          // Whether to use V2 Beam Search kernels

    TensorPtr mBeamSearchDiversityRateDevice; // [batchSize], in device memory.
    TensorPtr mLengthPenaltyDevice;           // [batchSize], in device memory.
    TensorPtr mEarlyStoppingDevice;           // [batchSize], in device memory.
    TensorPtr mBeamSearchDiversityRateHost;   // [batchSize], in pinned host memory.
    TensorPtr mLengthPenaltyHost;             // [batchSize], in pinned host memory.
    TensorPtr mEarlyStoppingHost;             // [batchSize], in pinned host memory.
};

} // namespace tensorrt_llm::layers
