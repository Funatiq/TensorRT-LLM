# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

add_tllm_kernel_library(
  tllm_kernel_tensor_operations SOURCES convertReqIndexToGlobal.cu
  cumsumLastDim.cu lookupKernels.cu)

add_tllm_kernel_library(
  tllm_kernel_cache_indexing SOURCES deepseekV4BlockTable.cu
  indexerKCacheGather.cu indexerKCacheScatter.cu nvfp4MlaKvCacheGather.cu)

add_tllm_kernel_library(
  tllm_kernel_indexer
  SOURCES
  indexerTopK.cu
  minimaxM3Fp8IndexerKernel.cu
  minimaxM3SelectBlocks.cu
  LINK_LIBRARIES
  tllm::common_environment)

add_tllm_kernel_library(
  tllm_kernel_normalization
  SOURCES
  layernormKernels.cu
  deepseekV4QNormKernel.cu
  rmsNormFp4QuantKernels.cu
  LINK_LIBRARIES
  tllm::common_environment)

add_tllm_kernel_library(
  tllm_kernel_activation_quantization
  SOURCES
  arcquantFP4.cu
  fp8PerTokenQuant/fp8_per_token_quant.cu
  fusedActivationQuant.cu
  fusedCatFp4.cu
  fusedCatFp8.cu
  LINK_LIBRARIES
  tllm::common_environment)

add_tllm_kernel_library(
  tllm_kernel_dit
  SOURCES
  fusedAdaptiveLayerNormKernel.cu
  fusedDiTGateResidNormShiftScaleKernel.cu
  fusedDiTQKNormRopeKernel.cu
  fusedDiTSplitNormKernel.cu
  fusedDiTSplitQKNormRopeKernel.cu
  ulyssesPermuteScatterKernel.cu
  ulyssesPostUnscatterKernel.cu
  LINK_LIBRARIES
  tllm::common_environment)

add_tllm_kernel_library(
  tllm_kernel_attention_utilities
  SOURCES
  buildRelativeAttentionBiasKernel.cu
  fusedQKNormRopeKernel.cu
  inverseRopeFp8QuantKernel.cu
  sageAttentionKernels.cu
  recoverFromRingAtten.cu
  LINK_LIBRARIES
  tllm::common_environment)

add_tllm_kernel_library(
  tllm_kernel_communication
  SOURCES
  communicationKernels/MiniMaxReduceRMSKernel.cu
  communicationKernels/customLowPrecisionAllReduceKernels.cu
  communicationKernels/mnnvlAllreduceKernels.cu
  helixAllToAll.cu
  helixKernels.cu
  LINK_LIBRARIES
  tllm::common_environment)

add_tllm_kernel_library(
  tllm_kernel_moe_prepare
  SOURCES
  moe/communication/megaMoePrepareKernel.cu
  moe/communication/moeAllReduceFusionKernels.cu
  moe/communication/moeAlltoAllKernels.cu
  moe/communication/moePrepareKernels.cu
  moe/utils/moeAlignKernels.cu
  LINK_LIBRARIES
  tllm::common_environment)

add_tllm_kernel_library(
  tllm_kernel_moe_custom_routing SOURCES moe/routing/customMoeRoutingKernels.cu
  noAuxTcKernels.cu LINK_LIBRARIES tllm::common_environment)

add_tllm_kernel_library(
  tllm_kernel_speculative_decoding
  SOURCES
  speculativeDecoding/dynamicTreeKernels.cu
  speculativeDecoding/kvCacheUpdateKernels.cu
  speculativeDecoding/mtpKernels.cu
  speculativeDecoding/suffixAutomaton/suffixAutomatonKernels.cu)

add_tllm_kernel_library(tllm_kernel_recurrent SOURCES mambaConv1dKernels.cu
                        lruKernel.cu)

add_tllm_kernel_library(tllm_kernel_stream SOURCES delayStream.cu
                        globalTimerKernel.cu)

add_tllm_kernel_library(tllm_kernel_qserve_gemm SOURCES qserveGemmPerChannel.cu
                        qserveGemmPerGroup.cu)

add_tllm_kernel_library(
  tllm_kernel_cuda_core_gemm_nvfp4 SOURCES
  weightOnlyBatchedGemv/cudaCoreGemmNVFP4.cu LINK_LIBRARIES
  tllm::kernel_cutlass_headers tllm::common_environment)

add_tllm_kernel_library(
  tllm_kernel_grouped_gemm
  SOURCES
  cuda_graph_grouped_gemm.cu
  PUBLIC_LINK_LIBRARIES
  tllm::kernel_cutlass_headers
  LINK_LIBRARIES
  tllm::kernel_lora
  ${TORCH_LIBRARIES})

add_tllm_kernel_library(
  tllm_kernel_tiny_gemm
  SOURCES
  tinygemm2/tinygemm2_cuda.cu
  LINK_LIBRARIES
  ${TORCH_LIBRARIES}
  Python3::Python
  CUDA::cuda_driver)
