/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/cudaVirtMem.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storage/core.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

TEST(KvCacheManagerV2VirtMemTest, ParkReleasesMappingAndKeepsAddress)
{
    if (cudaSetDevice(0) != cudaSuccess)
    {
        GTEST_SKIP() << "CUDA device is unavailable";
    }
    constexpr size_t kChunkBytes = 2ULL << 20;
    PooledPhysMemAllocator allocator(kChunkBytes);
    VirtMem memory(2 * kChunkBytes, allocator, 2);
    auto const address = memory.address();
    ASSERT_EQ(memory.numPhysMem(), 2);

    memory.park();
    EXPECT_TRUE(memory.isParked());
    EXPECT_EQ(memory.address(), address);
    EXPECT_EQ(memory.logicalNumPhysMem(), 2);
    EXPECT_EQ(memory.numPhysMem(), 0);
    EXPECT_EQ(memory.mappedBytes(), 0);
    EXPECT_THROW(memory.extend(1), std::exception);
    EXPECT_THROW(memory.realloc(kChunkBytes), std::exception);

    auto replacement = memory.prepareResume();
    EXPECT_EQ(replacement.size(), 2);
    memory.resume(std::move(replacement));
    EXPECT_FALSE(memory.isParked());
    EXPECT_EQ(memory.address(), address);
    EXPECT_EQ(memory.numPhysMem(), 2);

    memory.park();
    memory.destroy();
}

TEST(KvCacheManagerV2VirtMemTest, FullPoolSnapshotsRestoreBytesAndMemset)
{
    if (cudaSetDevice(0) != cudaSuccess)
    {
        GTEST_SKIP() << "CUDA device is unavailable";
    }
    constexpr size_t kChunkBytes = 2ULL << 20;
    constexpr unsigned char kPattern = 0x5a;
    PooledPhysMemAllocator allocator(kChunkBytes);
    GpuSlotPool pool(kChunkBytes, kChunkBytes, allocator, SlotCount{1});
    auto const address = std::get<MemAddress>(pool.slotAddress(SlotId{0}));
    for (auto const mode : {PoolRestoreMode::kCpu, PoolRestoreMode::kPinned, PoolRestoreMode::kMemset})
    {
        ASSERT_EQ(cuMemsetD8(address, kPattern, kChunkBytes), CUDA_SUCCESS);
        auto state = pool.prepareSleep(mode, nullptr);
        pool.commitSleep(state);
        EXPECT_TRUE(pool.isParked());
        EXPECT_EQ(pool.numSlots(), SlotCount{1});
        EXPECT_EQ(std::get<MemAddress>(pool.slotAddress(SlotId{0})), address);
        pool.prepareWakeup(state);
        pool.commitWakeup(state, mode, nullptr);
        ASSERT_EQ(cuCtxSynchronize(), CUDA_SUCCESS);
        EXPECT_FALSE(pool.isParked());
        std::vector<unsigned char> restored(kChunkBytes);
        ASSERT_EQ(cuMemcpyDtoH(restored.data(), address, restored.size()), CUDA_SUCCESS);
        EXPECT_TRUE(std::all_of(restored.begin(), restored.end(),
            [mode](unsigned char value) { return value == (mode == PoolRestoreMode::kMemset ? 0 : kPattern); }));
    }
}

} // namespace
