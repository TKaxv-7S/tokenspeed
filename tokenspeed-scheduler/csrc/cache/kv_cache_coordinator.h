// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_group.h"
#include "cache/cache_types.h"

namespace tokenspeed {

// num_common_blocks is the fixpoint coverage every group can validly claim; per_group[i] is group i's
// PrefixMatch at exactly that length.
struct CoordinatorMatch {
    std::int32_t num_common_blocks{0};
    std::vector<PrefixMatch> per_group;
};

// Stateless multi-group fan-out over the per-attention managers, all sharing one BlockPool; the caller
// owns the per-request BlockTables (index-aligned to groups).
class KvCacheCoordinator {
public:
    KvCacheCoordinator(std::vector<CacheGroup> groups, BlockPool& pool)
        : groups_{std::move(groups)}, pool_{pool} {}

    std::int32_t NumGroups() const { return static_cast<std::int32_t>(groups_.size()); }

    CoordinatorMatch MatchPrefix(std::span<const std::string> content_hashes) const;

    // Pure claim into fresh tables, never fails; a non-empty per_group must be sized to the group count.
    void ClaimCommonPrefix(std::span<BlockTable> tables, const CoordinatorMatch& hit);

    // Free-list blocks the claim will consume (TouchBlock pulls ref-0 cached hits); gates charge these too.
    std::int32_t BlocksConsumedByClaim(const CoordinatorMatch& hit) const;

    // All-or-nothing across all groups: on shortfall allocates NOTHING and returns false (no rollback needed).
    bool Acquire(std::span<BlockTable> tables, std::int32_t num_tokens);

    // Single home of the gate-side page math; Acquire's check and the flat admission gates both build on it.
    std::int32_t BlocksNeededFor(std::span<const BlockTable> tables, std::int32_t num_tokens) const;
    // Fresh-table overload for a not-yet-allocated request (no tail credit).
    std::int32_t BlocksNeededFor(std::int32_t num_tokens) const;

    void CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                         std::int32_t first_slot = 0);
    void Free(std::span<BlockTable> tables);

    struct StoreCandidate {
        std::string key;    // group-wrapped (MakeKeyWithGroupId), the host-tier index key
        CacheBlock* block;  // pinned (TouchBlock) until the consumer unpins
    };
    void EnableStoreCandidateCollection() { collect_store_candidates_ = true; }
    std::vector<StoreCandidate> TakePendingStores() { return std::exchange(pending_stores_, {}); }

    // Managers without a retention window inherit the no-op default.
    void AdvanceWindow(std::span<BlockTable> tables, std::int32_t num_computed_tokens);

    // Pure twin of AdvanceWindow: pages a pending slide would free, so gates can credit it.
    std::int32_t BlocksFreedByAdvance(std::span<const BlockTable> tables, std::int32_t num_computed_tokens) const;

private:
    std::vector<std::string> keysForGroup(std::span<const std::string> content_hashes,
                                          std::uint32_t group_id) const;
    std::vector<CacheGroup> groups_;
    BlockPool& pool_;
    bool collect_store_candidates_ = false;
    std::vector<StoreCandidate> pending_stores_;
};

// One CacheGroup per spec (group_id = index); asserts every spec shares the same page_size.
KvCacheCoordinator MakeCoordinator(std::span<const KvCacheSpec> specs, BlockPool& pool);

}  // namespace tokenspeed
