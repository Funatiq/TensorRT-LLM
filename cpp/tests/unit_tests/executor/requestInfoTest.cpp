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

#include "tensorrt_llm/batch_manager/requestInfo.h"
#include "tensorrt_llm/executor/serializeUtils.h"

#include <gtest/gtest.h>
#include <sstream>

namespace bm = tensorrt_llm::batch_manager;
namespace kv = bm::kv_cache_manager;
namespace su = tensorrt_llm::executor::serialize_utils;

TEST(RequestInfoTest, PreservesWireFieldsAndStreamAlignment)
{
    kv::BlockKey key{true, 17, {{11, 4}, {12, 5}}, {}, "tenant"};
    bm::RequestInfo original{42, {}, 3, key};
    original.setIsArbitraryTransfer(true);
    std::ostringstream expected;
    su::serialize(bm::RequestIdType{42}, expected);
    su::serialize(int32_t{3}, expected);
    su::serialize(key, expected);
    su::serialize(true, expected);
    su::serialize(original.getTransState(), expected);
    std::ostringstream stream;
    bm::RequestInfo::serialize(original, stream);
    EXPECT_EQ(stream.str(), expected.str());
    EXPECT_EQ(stream.str().size(), bm::RequestInfo::serializedSize(original));
    su::serialize(uint64_t{0x12345678}, stream);
    std::istringstream input{stream.str()};
    auto decoded = bm::RequestInfo::deserialize(input);
    EXPECT_EQ(decoded, original);
    EXPECT_TRUE(decoded.isArbitraryTransfer());
    EXPECT_EQ(su::deserialize<uint64_t>(input), 0x12345678);
}

TEST(RequestInfoTest, KeyIdentityAndPartialMatching)
{
    kv::BlockKey key{true, 17, {{11, 4}, {12, 5}}, {}, "tenant"};
    auto prefix = key.shorten(1);
    EXPECT_EQ(prefix.numMatchingTokens(key), 1);
    EXPECT_EQ(prefix.loraTaskId, key.loraTaskId);
    EXPECT_EQ(prefix.cacheSalt, key.cacheSalt);
    auto other = key;
    other.cacheSalt = "other";
    EXPECT_NE(key, other);
    EXPECT_EQ(key.numMatchingTokens(other), 0);
    EXPECT_NE(kv::BlockKeyHasher::hash(key), kv::BlockKeyHasher::hash(other));
    other = key;
    other.uniqueTokens[0].tokenExtraId++;
    EXPECT_NE(key, other);
    EXPECT_NE(kv::BlockKeyHasher::hash(key), kv::BlockKeyHasher::hash(other));
    EXPECT_THROW(key.shorten(3), std::exception);
}
