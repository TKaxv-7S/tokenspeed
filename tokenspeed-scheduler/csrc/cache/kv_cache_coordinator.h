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
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_group.h"
#include "cache/cache_types.h"

namespace tokenspeed {

// Common-prefix match across all groups: num_common_blocks is the fixpoint
// coverage every group can validly claim (MatchPrefix computes it), and
// per_group[i] is group i's PrefixMatch at exactly that length.
struct CoordinatorMatch {
    std::int32_t num_common_blocks{0};
    std::vector<PrefixMatch> per_group;
};

// Stateless multi-group fan-out over the per-attention managers, all sharing
// one BlockPool. Per-request BlockTables are passed in by the caller, aligned
// by group index; the single content-hash stream is re-keyed per group.
class KvCacheCoordinator {
public:
    KvCacheCoordinator(std::vector<CacheGroup> groups, BlockPool& pool)
        : groups_{std::move(groups)}, pool_{pool} {}

    std::int32_t NumGroups() const { return static_cast<std::int32_t>(groups_.size()); }

    CoordinatorMatch MatchPrefix(std::span<const std::string> content_hashes) const;

    // Claim the hit blocks into each group's fresh table. Pure claim, never
    // fails. A default-constructed CoordinatorMatch is the canonical zero hit;
    // a non-empty per_group must be sized to the group count.
    void ClaimCommonPrefix(std::span<BlockTable> tables, const CoordinatorMatch& hit);

    // Pure query, gate-side twin of ClaimCommonPrefix: free-list blocks the
    // claim will consume (TouchBlock pulls ref-0 cached hit blocks from the free
    // list), which admission gates must charge on top of what Acquire takes.
    // Group-scoped hash keys keep a block in at most one group's match.
    std::int32_t BlocksConsumedByClaim(const CoordinatorMatch& hit) const;

    // Token-driven incremental allocation across all groups. Check-then-act
    // against the shared pool: on shortfall allocates NOTHING and returns false,
    // so no partial/unaligned state ever exists and no rollback is needed.
    bool Acquire(std::span<BlockTable> tables, std::int32_t num_tokens);

    // Pure pre-check: fresh pages the shared pool must supply for every group to
    // absorb num_tokens. Single home of the gate-side page math -- Acquire's
    // check and the scheduler's flat admission gates both build on it.
    std::int32_t BlocksNeededFor(std::span<const BlockTable> tables, std::int32_t num_tokens) const;
    // For a not-yet-allocated request: every group starts from a fresh, empty
    // table (no tail credit).
    std::int32_t BlocksNeededFor(std::int32_t num_tokens) const;

    // Register content_hashes[j] under every group's table slot first_slot + j.
    void CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                         std::int32_t first_slot = 0);
    void Free(std::span<BlockTable> tables);

    // Fan out window eviction to every group; managers without a retention
    // window inherit the no-op default.
    void AdvanceWindow(std::span<BlockTable> tables, std::int32_t num_computed_tokens);

    // Pure query, twin of AdvanceWindow: pages a pending
    // AdvanceWindow(tables, num_computed_tokens) would return to the pool. Lets
    // admission gates credit a coming slide without a second copy of the window math.
    std::int32_t BlocksFreedByAdvance(std::span<const BlockTable> tables, std::int32_t num_computed_tokens) const;

private:
    std::vector<std::string> keysForGroup(std::span<const std::string> content_hashes,
                                          std::uint32_t group_id) const;
    std::vector<CacheGroup> groups_;
    BlockPool& pool_;
};

// Factory: build one CacheGroup per spec (group_id = index), all sharing `pool`.
// Asserts every spec has the same page_size (single shared page geometry).
KvCacheCoordinator MakeCoordinator(std::span<const KvCacheSpec> specs, BlockPool& pool);

}  // namespace tokenspeed
