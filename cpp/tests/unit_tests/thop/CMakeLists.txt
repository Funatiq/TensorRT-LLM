# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION &
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

if(${BUILD_PYT})
  add_tllm_gtest(
    thUtilsTest
    OWNER
    runtime
    SOURCES
    thUtilsTest.cpp
    LINK_LIBRARIES
    tllm::torch_utils
    LABELS
    "unit;runtime;thop;host")
  foreach(family Sequence VisualGen Decoding)
    if(family STREQUAL "Sequence")
      set(family_target tllm::torch_sequence)
    elseif(family STREQUAL "VisualGen")
      set(family_target tllm::torch_visual_gen)
    else()
      set(family_target tllm::torch_decoding)
    endif()
    add_tllm_gtest(
      torch${family}RegistrationTest
      OWNER
      runtime
      SOURCES
      torch${family}RegistrationTest.cpp
      LINK_LIBRARIES
      ${family_target}
      ${TORCH_LIBRARIES}
      LABELS
      "unit;runtime;thop;host")
  endforeach()
endif()
