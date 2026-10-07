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

#include "tensorrt_llm/batch_manager/cacheTransceiverComm.h"

#include <gtest/gtest.h>
#include <pybind11/embed.h>

namespace bm = tensorrt_llm::batch_manager;
namespace py = pybind11;

TEST(CacheTransceiverCommTest, MpiSplitRetainsNativeCommunicator)
{
#if ENABLE_MULTI_DEVICE
    bm::CacheTransceiverComm comm{&tensorrt_llm::mpi::MpiComm::session()};
    auto subgroup = comm.split(0, comm.getRank());
    EXPECT_TRUE(subgroup.isMpi());
    EXPECT_EQ(subgroup.getRank(), comm.getRank());
    EXPECT_EQ(subgroup.getSize(), comm.getSize());
#else
    GTEST_SKIP() << "Multi-device support is disabled";
#endif
}

TEST(CacheTransceiverCommTest, ProcessGroupSplitRequiresPython)
{
    ASSERT_FALSE(Py_IsInitialized());
    auto group = c10::make_intrusive<c10d::ProcessGroup>(0, 1);
    bm::CacheTransceiverComm comm{group};
    EXPECT_FALSE(comm.isMpi());
    EXPECT_THROW(comm.split(3, 7), tensorrt_llm::common::TllmException);
}

TEST(CacheTransceiverCommTest, PythonSplitForwardsArgumentsAndWrapsReturnedGroup)
{
    py::scoped_interpreter interpreter{};
    py::module_::import("sys").attr("path").attr("insert")(0, TLLM_TORCH_PACKAGE_ROOT);
    py::module_::import("torch");
    py::exec(R"PY(
import sys
import types
for name in ("tensorrt_llm", "tensorrt_llm._torch", "tensorrt_llm._torch.distributed", "tensorrt_llm._torch.distributed.pg_utils"):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
bridge = sys.modules["tensorrt_llm._torch.distributed.pg_utils"]
def split(color, key, group):
    bridge.last_args = (color, key)
    return group
bridge.split = split
)PY");
    auto group = c10::make_intrusive<c10d::ProcessGroup>(0, 1);
    bm::CacheTransceiverComm comm{group};
    auto subgroup = comm.split(3, 7);
    EXPECT_FALSE(subgroup.isMpi());
    EXPECT_EQ(subgroup.getRank(), 0);
    EXPECT_EQ(subgroup.getSize(), 1);
    auto bridge = py::module_::import("tensorrt_llm._torch.distributed.pg_utils");
    auto arguments = bridge.attr("last_args").cast<std::pair<int, int>>();
    EXPECT_EQ(arguments, (std::pair<int, int>{3, 7}));
    py::exec(R"PY(
def fail_split(color, key, group):
    raise RuntimeError("split failed")
bridge.split = fail_split
)PY");
    EXPECT_THROW(comm.split(3, 7), tensorrt_llm::common::TllmException);
}
