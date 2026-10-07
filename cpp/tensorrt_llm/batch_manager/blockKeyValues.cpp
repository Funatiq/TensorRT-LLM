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

#include "tensorrt_llm/batch_manager/blockKey.h"

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

bool BlockKey::operator==(BlockKey const& other) const noexcept
{
    return (usesExtraIds == other.usesExtraIds && loraTaskId == other.loraTaskId && uniqueTokens == other.uniqueTokens
        && extraKeys == other.extraKeys && cacheSalt == other.cacheSalt);
}

BlockKey BlockKey::shorten(int newNumTokens) const
{
    TLLM_CHECK_WITH_INFO(extraKeys.empty(),
        "shorten() cannot be called on a BlockKey with extraKeys (partial matching is disabled when multimodal data is "
        "present).");
    TLLM_CHECK_WITH_INFO(
        newNumTokens >= 0 && newNumTokens <= getNumTokens(), "newNumTokens must be >= 0 and <= getNumTokens()");
    BlockKey result(*this);
    result.uniqueTokens.resize(newNumTokens);
    return result;
}

size_t BlockKeyHasher::hash(BlockKey const& blockKey, std::size_t parentHash) noexcept
{
    // Hashing algorithm adapted from StackOverflow:
    // https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    // Constants provide very good distribution - each input bit affects each output bit with ~50% probability.
    size_t seed = blockKey.uniqueTokens.size() ^ parentHash * UINT64_C(0xbf58476d1ce4e5b9);

    if (parentHash == 0 && blockKey.cacheSalt)
    {
        // Only mix the cache salt into the hash for the first block in the sequence.
        uint64_t c = static_cast<uint64_t>(std::hash<std::string>{}(blockKey.cacheSalt.value()));
        seed = hash64Mix(c, seed);
    }

    for (auto const& uniqueToken : blockKey.uniqueTokens)
    {
        uint32_t a = static_cast<uint32_t>(uniqueToken.tokenId);
        seed = hash32Mix(a, seed);
        if (blockKey.usesExtraIds)
        {
            uint64_t b = uniqueToken.tokenExtraId;
            seed = hash64Mix(b, seed);
        }
    }

    if (blockKey.loraTaskId)
    {
        uint64_t c = blockKey.loraTaskId.value();
        seed = hash64Mix(c, seed);
    }

    // Add extra keys for multimodal data mixing in external multimodal item hash and token offset within this sequence
    // block
    if (!blockKey.extraKeys.empty())
    {
        for (auto const& mmKey : blockKey.extraKeys)
        {
            auto const& mmHash = mmKey.hash;
            auto const& startOffset = mmKey.startOffset;
            // Hash the multimodal hash array in 32-bit chunks (more efficient)
            for (size_t i = 0; i < 32; i += 4)
            {
                // Combine 4 bytes into a 32-bit word (construct as little endian order)
                uint32_t word = static_cast<uint32_t>(mmHash[i]) | (static_cast<uint32_t>(mmHash[i + 1]) << 8)
                    | (static_cast<uint32_t>(mmHash[i + 2]) << 16) | (static_cast<uint32_t>(mmHash[i + 3]) << 24);

                // Mix the word into the seed
                seed = hash32Mix(word, seed);
            }

            // Hash the start offset
            uint64_t e = static_cast<uint64_t>(startOffset);
            seed = hash64Mix(e, seed);
        }
    }

    return seed;
}
} // namespace tensorrt_llm::batch_manager::kv_cache_manager
