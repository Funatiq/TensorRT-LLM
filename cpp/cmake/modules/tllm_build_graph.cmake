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

cmake_file_api(
  QUERY
  API_VERSION
  1
  CODEMODEL
  2
  CACHE
  2
  TOOLCHAINS
  1)

function(tllm_add_build_graph_targets)
  get_property(test_manifest GLOBAL PROPERTY TLLM_BUILD_GRAPH_TEST_MANIFEST)
  file(
    GENERATE
    OUTPUT "${CMAKE_BINARY_DIR}/tllm_build_graph_tests.tsv"
    CONTENT "${test_manifest}")
  get_cmake_property(variable_names VARIABLES)
  list(SORT variable_names)
  set(configuration_manifest)
  foreach(name IN LISTS variable_names)
    if(name MATCHES "^(BUILD_|ENABLE_|USING_OSS_)"
       OR name
          MATCHES
          "^(CMAKE_SYSTEM_NAME|CMAKE_SYSTEM_PROCESSOR|CMAKE_(CXX|CUDA|C)_FLAGS(_[A-Z]+)?|CMAKE_(CXX|CUDA)_COMPILER_LAUNCHER|CUDAToolkit_VERSION|NIXL_ROOT|MOONCAKE_ROOT|INTERNAL_CUTLASS_KERNELS_PATH)$"
    )
      string(APPEND configuration_manifest "${name}\t${${name}}\n")
    endif()
  endforeach()
  file(
    GENERATE
    OUTPUT "${CMAKE_BINARY_DIR}/tllm_build_graph_config.tsv"
    CONTENT "${configuration_manifest}")
  foreach(mode check report)
    set(arguments)
    if(mode STREQUAL "report")
      list(APPEND arguments --report-only)
    endif()
    add_custom_target(
      ${mode}-build-graph
      COMMAND
        "${Python3_EXECUTABLE}"
        "${PROJECT_SOURCE_DIR}/../scripts/check_cpp_build_graph.py" --build-dir
        "${CMAKE_BINARY_DIR}" --config "$<CONFIG>" --report
        "${CMAKE_BINARY_DIR}/build-graph-$<CONFIG>.json" ${arguments}
      VERBATIM)
  endforeach()
endfunction()
