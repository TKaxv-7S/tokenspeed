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

#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_types.h"
#include "utils.h"

namespace tokenspeed {

// Abstract base for the per-attention-type KV managers. MatchPrefix and the
// window hooks are type-specific; the rest is shared and stateless over
// (pool_, page_size_) plus the BlockTable handed in. Holds no per-request state.
class KvCacheManager {
public:
    KvCacheManager(BlockPool& pool, std::int32_t page_size) : pool_{pool}, page_size_{page_size} {
        _assert(page_size > 0, "page_size must be > 0");
    }
    virtual ~KvCacheManager() = default;

    KvCacheManager(const KvCacheManager&) = delete;
    KvCacheManager& operator=(const KvCacheManager&) = delete;

    // Per-attention-type prefix match. Read-only: must NOT change ref counts.
    virtual PrefixMatch MatchPrefix(std::span<const std::string> block_hashes) const = 0;

    // Bounded match: matches only the first max_blocks hashes, so each manager's
    // validity invariant (e.g. SwaManager's trailing run) is enforced against the
    // BOUNDED end -- never computed unbounded and then chopped. May come back shorter.
    PrefixMatch MatchPrefix(std::span<const std::string> block_hashes, std::int32_t max_blocks) const {
        std::size_t bound = std::min(block_hashes.size(), static_cast<std::size_t>(std::max(max_blocks, 0)));
        return MatchPrefix(block_hashes.first(bound));
    }

    // Claim each real hit block into a fresh table (before any Acquire).
    // null_block holes are appended as-is to keep logical-page alignment but
    // never touched -- the null block is not ref counted.
    void ClaimHitBlocks(BlockTable& table, const PrefixMatch& hit) {
        _assert(table.blocks_.empty(), "ClaimHitBlocks requires a fresh (empty) table");
        for (CacheBlock* block : hit.blocks) {
            if (!block->IsNull()) {
                pool_.TouchBlock(block);
            }
            table.blocks_.push_back(block);
        }
    }

    // Token-driven incremental allocation, mirroring LocalKVAllocator::Acquire:
    // tail-page room first, then ceil(overflow / page_size) fresh pages.
    // All-or-nothing: on shortfall the table is left unchanged, returns false.
    bool Acquire(BlockTable& table, std::int32_t num_tokens) {
        if (num_tokens <= 0) {
            return true;
        }
        if (num_tokens <= table.tail_avail_) {
            table.tail_avail_ -= num_tokens;
            return true;
        }
        std::int32_t over = num_tokens - table.tail_avail_;
        std::int32_t num_pages = (over + page_size_ - 1) / page_size_;
        std::vector<CacheBlock*> new_blocks = pool_.AllocateBlocks(num_pages);
        if (static_cast<std::int32_t>(new_blocks.size()) < num_pages) {
            return false;  // AllocateBlocks is all-or-nothing; nothing claimed.
        }
        for (CacheBlock* block : new_blocks) {
            table.blocks_.push_back(block);
        }
        std::int32_t used_in_tail = over % page_size_;
        table.tail_avail_ = (used_in_tail == 0) ? 0 : page_size_ - used_in_tail;
        return true;
    }

    // Pure query mirroring Acquire's page math exactly; allocates and mutates
    // nothing. The coordinator sums it across groups before committing.
    std::int32_t BlocksNeededFor(const BlockTable& table, std::int32_t num_tokens) const {
        if (num_tokens <= table.tail_avail_) {
            return 0;
        }
        std::int32_t over = num_tokens - table.tail_avail_;
        return (over + page_size_ - 1) / page_size_;
    }

    // Register block_hashes[j] under table slot first_slot + j (partial tail
    // excluded by the caller) so later requests can prefix-hit those pages;
    // pages already carrying a hash are skipped.
    void CacheFullBlocks(BlockTable& table, std::span<const std::string> block_hashes,
                         std::int32_t first_slot = 0) {
        _assert(first_slot >= 0, "first_slot must be >= 0");
        _assert(static_cast<std::int64_t>(first_slot) + static_cast<std::int64_t>(block_hashes.size()) <=
                    table.NumBlocks(),
                "hash range exceeds table size");
        for (std::size_t j = 0; j < block_hashes.size(); ++j) {
            CacheBlock* block = table.blocks_[static_cast<std::size_t>(first_slot) + j];
            if (block->IsCached()) {
                continue;  // already registered (prefix hit or earlier call)
            }
            pool_.CacheFullBlocks(block, block_hashes[j]);
        }
    }

    // Advance the retention window to num_computed_tokens, releasing whatever
    // fell out. Full-history managers retain everything mid-sequence (no-op
    // default); window-evicting managers override.
    virtual void AdvanceWindow(BlockTable& /*table*/, std::int32_t /*num_computed_tokens*/) {}

    // Pure query: pages AdvanceWindow(table, num_computed_tokens) would return
    // to the pool. Overridden in lockstep with AdvanceWindow; admission gates
    // use it to credit a pending slide.
    virtual std::int32_t BlocksFreedByAdvanceWindow(const BlockTable& /*table*/,
                                                    std::int32_t /*num_computed_tokens*/) const {
        return 0;
    }

    // Release every page the table holds and reset it. Cached pages keep their
    // hash on free, so they remain prefix-reusable until evicted.
    void Free(BlockTable& table) {
        pool_.FreeBlocks(table.blocks_);
        table.blocks_.clear();
        table.tail_avail_ = 0;
    }

protected:
    BlockPool& pool_;
    std::int32_t page_size_;
};

}  // namespace tokenspeed
