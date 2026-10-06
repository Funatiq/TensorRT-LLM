#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#
if(NOT WIN32)
  # additional warnings
  #
  # Ignore overloaded-virtual warning. We intentionally change parameters of
  # some methods in derived class.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  if(WARNING_IS_ERROR)
    message(STATUS "Treating warnings as errors in GCC compilation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
else() # Windows
  # warning level 4
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

add_library(tllm_common_headers INTERFACE)
add_library(tllm::common_headers ALIAS tllm_common_headers)
target_include_directories(tllm_common_headers INTERFACE ${API_INCLUDE_DIR}
                                                         ${PROJECT_SOURCE_DIR})

add_library(tllm_common_error STATIC assert.cpp stringUtils.cpp
                                     tllmException.cpp)
add_library(tllm::common_error ALIAS tllm_common_error)
target_link_libraries(
  tllm_common_error
  PUBLIC tllm::common_headers
  PRIVATE ${CMAKE_DL_LIBS})
# stringUtils.h exposes CUDA half types, but needs no CUDA runtime linkage.
target_include_directories(tllm_common_error SYSTEM
                           PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(tllm_common_error PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tllm_common_logging STATIC logger.cpp timestampUtils.cpp)
add_library(tllm::common_logging ALIAS tllm_common_logging)
target_link_libraries(
  tllm_common_logging
  PUBLIC tllm::common_error
  PRIVATE Threads::Threads CUDA::cudart)
set_target_properties(tllm_common_logging PROPERTIES POSITION_INDEPENDENT_CODE
                                                     ON)

add_library(tllm_common_ip STATIC ipUtils.cpp)
add_library(tllm::common_ip ALIAS tllm_common_ip)
target_link_libraries(
  tllm_common_ip
  PUBLIC tllm::common_headers
  PRIVATE tllm::common_logging)
target_compile_features(tllm_common_ip PUBLIC cxx_std_20)
set_target_properties(tllm_common_ip PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tllm_common_profiler STATIC cudaProfilerUtils.cpp)
add_library(tllm::common_profiler ALIAS tllm_common_profiler)
target_link_libraries(tllm_common_profiler PUBLIC tllm::common_logging)
target_compile_features(tllm_common_profiler PUBLIC cxx_std_20)
set_target_properties(tllm_common_profiler PROPERTIES POSITION_INDEPENDENT_CODE
                                                      ON)

add_library(tllm_common_cuda STATIC cudaDriverWrapper.cpp memoryUtils.cu
                                    nvmlWrapper.cpp)
add_library(tllm::common_cuda ALIAS tllm_common_cuda)
target_link_libraries(
  tllm_common_cuda
  PUBLIC tllm::common_logging CUDA::cudart
  PRIVATE ${CMAKE_DL_LIBS})
target_compile_features(tllm_common_cuda PUBLIC cxx_std_20)
set_target_properties(
  tllm_common_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON
                              CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_cuda_architectures(tllm_common_cuda 89)

add_library(tllm_common_environment STATIC envUtils.cpp)
add_library(tllm::common_environment ALIAS tllm_common_environment)
target_link_libraries(tllm_common_environment PUBLIC tllm::common_cuda)
set_target_properties(tllm_common_environment
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tllm_common_cublas STATIC cublasMMWrapper.cpp)
add_library(tllm::common_cublas ALIAS tllm_common_cublas)
target_link_libraries(tllm_common_cublas PUBLIC tllm::common_cuda CUDA::cublas
                                                CUDA::cublasLt)
target_compile_features(tllm_common_cublas PUBLIC cxx_std_20)
set_target_properties(tllm_common_cublas PROPERTIES POSITION_INDEPENDENT_CODE
                                                    ON)
if(ENABLE_CUBLASLT_FP4_GEMM)
  target_compile_definitions(tllm_common_cublas
                             PRIVATE ENABLE_CUBLASLT_FP4_GEMM)
endif()

add_library(tllm_common_attention_workspace STATIC attentionWorkspace.cpp)
add_library(tllm::common_attention_workspace ALIAS
            tllm_common_attention_workspace)
target_link_libraries(tllm_common_attention_workspace PUBLIC tllm::common_error)
target_compile_features(tllm_common_attention_workspace PUBLIC cxx_std_20)
set_target_properties(tllm_common_attention_workspace
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tllm_common_nccl STATIC ncclUtils.cpp)
add_library(tllm::common_nccl ALIAS tllm_common_nccl)
target_link_libraries(tllm_common_nccl PUBLIC tllm::common_cuda)
target_compile_features(tllm_common_nccl PUBLIC cxx_std_20)
target_compile_definitions(
  tllm_common_nccl PUBLIC ENABLE_MULTI_DEVICE=$<BOOL:${ENABLE_MULTI_DEVICE}>)
if(ENABLE_MULTI_DEVICE)
  target_link_libraries(tllm_common_nccl PUBLIC ${NCCL_LIB})
endif()
set_target_properties(tllm_common_nccl PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tllm_common_nccl_comm STATIC ncclCommUtils.cpp)
add_library(tllm::common_nccl_comm ALIAS tllm_common_nccl_comm)
target_link_libraries(
  tllm_common_nccl_comm
  PUBLIC tllm::common_nccl
  PRIVATE tllm::runtime_distributed tllm::runtime_ipc_nvls)
target_compile_features(tllm_common_nccl_comm PUBLIC cxx_std_20)
set_target_properties(tllm_common_nccl_comm PROPERTIES POSITION_INDEPENDENT_CODE
                                                       ON)

add_library(tllm_common_mcast_registry STATIC mcastDevMemUtils.cpp)
add_library(tllm::common_mcast_registry ALIAS tllm_common_mcast_registry)
target_link_libraries(tllm_common_mcast_registry PUBLIC tllm::common_headers)
target_compile_features(tllm_common_mcast_registry PUBLIC cxx_std_20)
set_target_properties(
  tllm_common_mcast_registry PROPERTIES CXX_EXTENSIONS OFF
                                        POSITION_INDEPENDENT_CODE ON)

add_library(tllm_common_safetensors STATIC safetensors.cpp)
add_library(tllm::common_safetensors ALIAS tllm_common_safetensors)
target_link_libraries(tllm_common_safetensors PUBLIC tllm::common_logging)
target_include_directories(tllm_common_safetensors SYSTEM
                           PRIVATE ${CMAKE_BINARY_DIR}/_deps/json-src/include)
target_compile_features(tllm_common_safetensors PUBLIC cxx_std_20)
set_target_properties(
  tllm_common_safetensors PROPERTIES CXX_EXTENSIONS OFF
                                     POSITION_INDEPENDENT_CODE ON)

add_library(tllm_common_op_utils STATIC opUtils.cpp)
add_library(tllm::common_op_utils ALIAS tllm_common_op_utils)
target_link_libraries(
  tllm_common_op_utils
  PUBLIC tllm::common_cublas CUDA::cuda_driver
  PRIVATE Threads::Threads)
target_compile_definitions(
  tllm_common_op_utils
  PUBLIC ENABLE_MULTI_DEVICE=$<BOOL:${ENABLE_MULTI_DEVICE}>)
if(ENABLE_MULTI_DEVICE)
  target_include_directories(tllm_common_op_utils SYSTEM
                             PUBLIC ${NCCL_INCLUDE_DIR})
endif()
if(ENABLE_CUBLASLT_FP4_GEMM)
  target_compile_definitions(tllm_common_op_utils
                             PRIVATE ENABLE_CUBLASLT_FP4_GEMM)
endif()
target_compile_features(tllm_common_op_utils PUBLIC cxx_std_20)
set_target_properties(
  tllm_common_op_utils PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE
                                                     ON)

add_library(tllm_common_fp8 STATIC cudaFp8Utils.cu)
add_library(tllm::common_fp8 ALIAS tllm_common_fp8)
target_link_libraries(
  tllm_common_fp8
  PUBLIC tllm::common_cuda
  PRIVATE tllm::common_environment)

add_library(tllm_common_sage_quant STATIC sageQuant.cu)
add_library(tllm::common_sage_quant ALIAS tllm_common_sage_quant)
target_link_libraries(
  tllm_common_sage_quant
  PUBLIC tllm::common_cuda
  PRIVATE tllm::kernel_cutlass_headers)

foreach(component tllm_common_fp8 tllm_common_sage_quant)
  target_compile_features(${component} PUBLIC cxx_std_20)
  set_target_properties(
    ${component}
    PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_cuda_architectures(${component} 89)
endforeach()

add_library(tllm_common_attention STATIC attentionOp.cpp)
add_library(tllm::common_attention ALIAS tllm_common_attention)
target_link_libraries(
  tllm_common_attention
  PUBLIC tllm::common_op_utils tllm::common_attention_workspace
         trtllm_gen_fmha_interface
  PRIVATE tllm_kernel_attention_dispatch
          tllm::common_nccl_comm
          tllm::common_environment
          tllm::common_sage_quant
          tllm::runtime_debug
          tllm::runtime_distributed
          tllm::kernel_attention
          tllm::kernel_attention_decode
          tllm::kernel_attention_mask
          tllm::kernel_cascade_attention
          tllm::kernel_rope
          tllm::kernel_mla
          tllm::kernel_mla_chunked_prefill
          tllm::kernel_sparse_attention
          tllm::kernel_fp8_block_scale_gemm)
# Object backends are forwarded to final consumers; the orchestration archive
# owns the orchestration and dispatcher implementations.
target_link_libraries(
  tllm_common_attention
  INTERFACE context_attention_src
            decoder_attention_src
            flash_mla_src
            $<TARGET_OBJECTS:context_attention_src>
            $<TARGET_OBJECTS:decoder_attention_src>
            $<TARGET_OBJECTS:flash_mla_src>)
target_compile_features(tllm_common_attention PUBLIC cxx_std_20)
set_target_properties(
  tllm_common_attention PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE
                                                      ON)
if(ENABLE_CUBLASLT_FP4_GEMM)
  target_compile_definitions(tllm_common_attention
                             PRIVATE ENABLE_CUBLASLT_FP4_GEMM)
endif()
