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

#include <pybind11/pybind11.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/python.h>

namespace tensorrt_llm::batch_manager
{

CacheTransceiverComm CacheTransceiverComm::split(int color, int key)
{
    if (isMpi())
    {
        auto subgroup = mMpiComm->split(color, key);
        return CacheTransceiverComm(std::make_shared<mpi::MpiComm const>(std::move(subgroup)));
    }
    bool const initialized = Py_IsInitialized();
    TLLM_CHECK_WITH_INFO(initialized, "Trying to use ProcessGroup communicator but Python is not initialized");
    try
    {
        c10::intrusive_ptr<c10d::ProcessGroup> pgSub;
        {
            pybind11::gil_scoped_acquire gil;
            auto const m = pybind11::module::import("tensorrt_llm._torch.distributed.pg_utils");
            // Properly box the existing intrusive_ptr ProcessGroup into an IValue
            // and convert to a Python object without constructing a new instance.
            auto const py_pg = torch::jit::toPyObject(c10::IValue(mPgComm));

            auto const py_sub_pg = m.attr("split")(color, key, py_pg);
            pgSub = torch::jit::toCustomClass<c10d::ProcessGroup>(py_sub_pg);
        }
        return CacheTransceiverComm(pgSub);
    }
    catch (...)
    {
        TLLM_THROW("Failed to split process group");
    }
}

} // namespace tensorrt_llm::batch_manager
