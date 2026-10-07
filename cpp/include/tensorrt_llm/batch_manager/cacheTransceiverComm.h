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

#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/pgUtils.h"

#include <memory>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager
{

class CacheTransceiverComm
{
public:
    // Construct from a non-owning raw pointer, won't take ownership of the pointer
    explicit CacheTransceiverComm(mpi::MpiComm const* mpiComm)
        : mMpiComm(std::shared_ptr<mpi::MpiComm const>(nullptr), mpiComm)
    {
    }

    // Construct from a shared_ptr with shared ownership
    explicit CacheTransceiverComm(std::shared_ptr<mpi::MpiComm const> mpiComm)
        : mMpiComm(std::move(mpiComm))
    {
    }

    // Construct from a ProcessGroup communicator
    explicit CacheTransceiverComm(c10::intrusive_ptr<c10d::ProcessGroup> pgComm)
        : mPgComm(std::move(pgComm))
    {
    }

    ~CacheTransceiverComm() = default;

    bool isMpi() const noexcept
    {
        return mMpiComm != nullptr;
    }

    int getRank() const
    {
        if (isMpi())
        {
            return mMpiComm->getRank();
        }
        return mPgComm->getRank();
    }

    int getSize() const
    {
        if (isMpi())
        {
            return mMpiComm->getSize();
        }
        return mPgComm->getSize();
    }

    void allgather(void const* sendbuf, void* recvbuf, int count, mpi::MpiType dtype) const
    {
        if (isMpi())
        {
            mMpiComm->allgather(sendbuf, recvbuf, count, dtype);
            return;
        }
        TLLM_THROW("Input arguments only supported in mpi");
    }

    template <typename Input, typename Output>
    bool allgather(Input input, Output output, c10d::AllgatherOptions options = c10d::AllgatherOptions()) const
    {
        if (isMpi())
        {
            TLLM_THROW("Input arguments only supported in pg");
        }
        tensorrt_llm::pg_utils::PgHelper pgh{mPgComm};

        PGCHECK_THROW(pgh.allgather(input, output, options));
        return true;
    }

    template <typename Input, typename Output>
    bool allgatherv(Input input, Output output, std::vector<int> const& sizes,
        c10d::AllgatherOptions options = c10d::AllgatherOptions()) const
    {
        if (isMpi())
        {
            TLLM_THROW("Input arguments only supported in pg");
        }
        tensorrt_llm::pg_utils::PgHelper pgh{mPgComm};
        PGCHECK_THROW(pgh.allgatherv(input, output, sizes, options));
        return true;
    }

    bool allgatherv(void const* sendbuf, int sendcount, mpi::MpiType sendtype, void* recvbuf,
        std::vector<int> const& recvcounts, std::vector<int> const& displs, mpi::MpiType recvtype) const
    {
        if (isMpi())
        {
            mMpiComm->allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype);
            return true;
        }
        TLLM_THROW("Input arguments only supported in mpi");
    }

    [[nodiscard]] std::unique_ptr<mpi::MpiRequest> sendAsync(
        void const* buffer, std::size_t size, mpi::MpiType dtype, int dest, mpi::MpiTag tag) const
    {
        TLLM_CHECK_WITH_INFO(isMpi(), "Point-to-point cache-transceiver status messages require MPI.");
        return mMpiComm->sendAsync(buffer, size, dtype, dest, tag);
    }

    [[nodiscard]] bool iprobe(int source, mpi::MpiTag tag, MPI_Status* status) const
    {
        TLLM_CHECK_WITH_INFO(isMpi(), "Point-to-point cache-transceiver status messages require MPI.");
        return mMpiComm->iprobe(source, tag, status);
    }

    void recv(void* buffer, std::size_t size, mpi::MpiType dtype, int source, mpi::MpiTag tag) const
    {
        TLLM_CHECK_WITH_INFO(isMpi(), "Point-to-point cache-transceiver status messages require MPI.");
        static_cast<void>(mMpiComm->recv(buffer, size, dtype, source, tag));
    }

    CacheTransceiverComm split(int color, int key);

private:
    std::shared_ptr<mpi::MpiComm const> mMpiComm;
    c10::intrusive_ptr<c10d::ProcessGroup> mPgComm;
};

} // namespace tensorrt_llm::batch_manager
