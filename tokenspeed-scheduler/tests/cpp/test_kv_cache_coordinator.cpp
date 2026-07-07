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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_group.h"
#include "cache/kv_cache_coordinator.h"
#include "cache/cache_types.h"
#include "cache/full_attn_manager.h"
#include "cache/swa_manager.h"
#include "scheduler/page_hasher.h"

namespace tokenspeed::test {
namespace {

using token_span = std::span<const std::int32_t>;

std::vector<std::string> ContentHashes(const std::vector<std::vector<std::int32_t>>& pages) {
    std::vector<token_span> spans;
    spans.reserve(pages.size());
    for (const auto& p : pages) {
        spans.emplace_back(p.data(), p.size());
    }
    return ComputePagedHashes(spans, "");
}

// Cache then free, so the block is prefix-hittable via MatchPrefix.
CacheBlock* CacheForGroup(BlockPool& pool, const std::string& content_hash, std::uint32_t group_id) {
    std::string key = MakeKeyWithGroupId(content_hash, group_id);
    std::vector<CacheBlock*> got = pool.AllocateBlocks(1);
    pool.CacheFullBlocks(got.front(), key);
    pool.FreeBlocks(got);
    return got.front();
}

// Asserts no null hole inside the last min(len, contiguous_needed) blocks.
void ExpectSwaWindowIntact(const PrefixMatch& m, std::int32_t window, std::int32_t page_size) {
    std::int32_t len = static_cast<std::int32_t>(m.blocks.size());
    std::int32_t contiguous_needed = (window - 1 + page_size - 1) / page_size;
    std::int32_t need = std::min(len, contiguous_needed);
    for (std::int32_t i = len - need; i < len; ++i) {
        EXPECT_FALSE(m.blocks[static_cast<std::size_t>(i)]->IsNull())
            << "null hole inside the last window at slot " << i << " of " << len;
    }
}

TEST(CacheGroupTest, HoldsSpecGroupIdManager) {
    BlockPool pool(8);
    auto mgr = std::make_unique<FullAttnManager>(pool, 4);
    CacheGroup g(KvCacheSpec{AttnKind::kFull, 4, 0}, /*group_id=*/0, std::move(mgr));
    EXPECT_EQ(g.GroupId(), 0u);
    EXPECT_EQ(g.Spec().page_size, 4);
    EXPECT_EQ(g.Spec().kind, AttnKind::kFull);
}

TEST(MakeCoordinatorTest, BuildsOneGroupPerSpec) {
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 4, 10},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    EXPECT_EQ(coord.NumGroups(), 2);
}

TEST(MakeCoordinatorTest, RejectsMismatchedPageSize) {
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 8, 10},  // different page_size
    };
    EXPECT_THROW(MakeCoordinator(specs, pool), std::runtime_error);
}

TEST(CoordinatorMatchTest, BothGroupsAllMiss) {
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 0);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_TRUE(m.per_group[0].blocks.empty());
    EXPECT_TRUE(m.per_group[1].blocks.empty());
}

TEST(CoordinatorMatchTest, CommonIsMinCoverageFullDeeperThanSwa) {
    // full caches 4 contiguous pages; swa (window 10 -> contiguous_needed 3)
    // caches only the last 3. Common = min(4, 3) = 3.
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    // swa front 3-run (a TAIL run would null-pad back to index 0 -> coverage 4).
    CacheForGroup(pool, ch[0], 1);
    CacheForGroup(pool, ch[1], 1);
    CacheForGroup(pool, ch[2], 1);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 3);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 3u);
    EXPECT_EQ(m.per_group[1].blocks.size(), 3u);
    // Full had 4 real hits, truncated to 3 -> num_hit recomputed to 3.
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 3);
}

TEST(CoordinatorMatchTest, SwaMissForcesZeroCommon) {
    // full caches 2 pages, swa caches nothing -> common = min(2, 0) = 0.
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[1], 0);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 0);
    EXPECT_EQ(m.per_group[0].blocks.size(), 0u);
    EXPECT_EQ(m.per_group[1].blocks.size(), 0u);
}

TEST(CoordinatorAllocTest, ColdStartAllocatesAlignedPages) {
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    CoordinatorMatch hit = coord.MatchPrefix(ch);
    EXPECT_EQ(hit.num_common_blocks, 0);

    std::vector<BlockTable> tables(2);
    coord.ClaimCommonPrefix(tables, hit);              // no hits -> no-op
    ASSERT_TRUE(coord.Acquire(tables, /*num_tokens=*/8));
    // 8 tokens / page 4 = 2 pages in EACH group; tables aligned.
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(CoordinatorAllocTest, ClaimsCommonPrefixThenAllocatesRemainder) {
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    // swa window 4 -> contiguous_needed 1, so a single cached front page is a hit.
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[0], 1);

    CoordinatorMatch hit = coord.MatchPrefix(ch);
    ASSERT_EQ(hit.num_common_blocks, 1);

    std::vector<BlockTable> tables(2);
    // 8 tokens total, 1 page (4 tokens) common -> 4 uncached tokens -> +1 page each.
    coord.ClaimCommonPrefix(tables, hit);              // claim the 1 cached page each
    ASSERT_TRUE(coord.Acquire(tables, 8 - hit.num_common_blocks * 4));
    EXPECT_EQ(tables[0].NumBlocks(), 2);  // 1 claimed + 1 allocated
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(CoordinatorAllocTest, CrossGroupShortfallAllocatesNothing) {
    BlockPool pool(5);  // 4 usable after null reservation
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    CoordinatorMatch hit = coord.MatchPrefix(ch);  // all miss, common 0
    ASSERT_EQ(hit.num_common_blocks, 0);

    std::vector<BlockTable> tables(2);
    std::int32_t free_before = pool.NumFreeBlocks();
    coord.ClaimCommonPrefix(tables, hit);          // no hits -> no-op
    // 12 tokens -> 3 pages per group = 6 needed, only 4 free -> fail, nothing taken.
    EXPECT_FALSE(coord.Acquire(tables, 12));
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);   // untouched, not rolled back
}

TEST(CoordinatorStepTest, AcquireKeepsGroupsAligned) {
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(coord.Acquire(tables, 4));   // 1 page each
    EXPECT_EQ(tables[0].NumBlocks(), 1);
    EXPECT_EQ(tables[1].NumBlocks(), 1);
    ASSERT_TRUE(coord.Acquire(tables, 4));   // 1 more each
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(CoordinatorStepTest, AcquireShortfallAllocatesNothing) {
    BlockPool pool(3);  // 2 usable
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<BlockTable> tables(2);
    std::int32_t free_before = pool.NumFreeBlocks();
    // 2 pages per group (8 tokens) = 4 blocks, only 2 free -> fail, nothing taken.
    EXPECT_FALSE(coord.Acquire(tables, 8));
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
}

TEST(CoordinatorStepTest, CacheFullBlocksThenMatchHits) {
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(coord.Acquire(tables, 4));   // 1 page each
    coord.CacheFullBlocks(tables, ch);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 1);
}

TEST(CoordinatorStepTest, FreeReturnsAllGroups) {
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(coord.Acquire(tables, 8));   // 2 pages each = 4 blocks
    std::int32_t free_mid = pool.NumFreeBlocks();
    coord.Free(tables);
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), free_mid + 4);
}

TEST(CoordinatorStepTest, EndToEndTwoRequestsSharePrefix) {
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});

    // Request A: cold, allocate 2 pages each, cache both, free.
    {
        CoordinatorMatch m = coord.MatchPrefix(ch);
        EXPECT_EQ(m.num_common_blocks, 0);
        std::vector<BlockTable> a(2);
        coord.ClaimCommonPrefix(a, m);
        ASSERT_TRUE(coord.Acquire(a, 8));
        coord.CacheFullBlocks(a, ch);
        coord.Free(a);
    }
    // Request B: shares the prefix -> common 2 pages in both groups.
    {
        CoordinatorMatch m = coord.MatchPrefix(ch);
        EXPECT_EQ(m.num_common_blocks, 2);
        std::vector<BlockTable> b(2);
        coord.ClaimCommonPrefix(b, m);
        ASSERT_TRUE(coord.Acquire(b, 8 - m.num_common_blocks * 4));
        EXPECT_EQ(b[0].NumBlocks(), 2);
        EXPECT_EQ(b[1].NumBlocks(), 2);
        coord.Free(b);
    }
}

TEST(CoordinatorStepTest, CacheFullBlocksAtSlotOffsetExtendsPrefix) {
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes(
        {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}, {5, 5, 5, 5}});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(coord.Acquire(tables, 24));   // 6 pages each
    coord.CacheFullBlocks(tables, std::span(ch).first(4));   // prefill path: slots 0..3
    coord.CacheFullBlocks(tables, std::span(ch).subspan(4), /*first_slot=*/4);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 6);
    ASSERT_EQ(m.per_group.size(), 2u);
    ASSERT_EQ(m.per_group[0].blocks.size(), 6u);
    for (std::size_t s = 0; s < 6; ++s) {
        EXPECT_EQ(m.per_group[0].blocks[s], tables[0].Blocks()[s]) << "slot " << s;
    }
    // swa window 4 -> contiguous_needed 1: tail run maps to the offset-registered slot-5 block.
    ASSERT_EQ(m.per_group[1].blocks.size(), 6u);
    EXPECT_EQ(m.per_group[1].blocks[5], tables[1].Blocks()[5]);
}

TEST(CoordinatorStepTest, CacheFullBlocksAtOffsetSkipsSwaHoles) {
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes(
        {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}, {5, 5, 5, 5}});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(coord.Acquire(tables, 24));   // 6 pages each
    // num_computed=20 -> swa skipped = 20-4+1 = 17 -> 17/4 = 4 pages punched:
    // swa slots 0..3 are null holes.
    coord.AdvanceWindow(tables, /*num_computed_tokens=*/20);
    ASSERT_TRUE(tables[1].Blocks()[3]->IsNull());
    ASSERT_FALSE(tables[1].Blocks()[4]->IsNull());

    coord.CacheFullBlocks(tables, std::span(ch).subspan(2), /*first_slot=*/2);
    for (std::size_t s = 2; s < 6; ++s) {
        EXPECT_NE(pool.GetCachedBlock(MakeKeyWithGroupId(ch[s], 0)), nullptr) << "full slot " << s;
    }
    EXPECT_EQ(pool.GetCachedBlock(MakeKeyWithGroupId(ch[2], 1)), nullptr);
    EXPECT_EQ(pool.GetCachedBlock(MakeKeyWithGroupId(ch[3], 1)), nullptr);
    EXPECT_NE(pool.GetCachedBlock(MakeKeyWithGroupId(ch[4], 1)), nullptr);
    EXPECT_NE(pool.GetCachedBlock(MakeKeyWithGroupId(ch[5], 1)), nullptr);
}

TEST(CoordinatorStepTest, CacheFullBlocksRejectsOutOfRangeFirstSlot) {
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{7, 7, 7, 7}});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(coord.Acquire(tables, 8));   // 2 pages each
    EXPECT_THROW(coord.CacheFullBlocks(tables, ch, /*first_slot=*/2), std::runtime_error);
    EXPECT_THROW(coord.CacheFullBlocks(tables, ch, /*first_slot=*/-1), std::runtime_error);
}

TEST(CoordinatorMatchTest, SwaRunCutByFullBoundDropsToNoValidMatch) {
    // full covers 4; swa's tail run {2,3,4} bounded to 4 leaves run {2,3} <
    // contiguous_needed 3 with holes at 0,1 -> no valid swa match, common = 0.
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[1], 0);
    CacheForGroup(pool, ch[2], 0);
    CacheForGroup(pool, ch[3], 0);
    CacheForGroup(pool, ch[2], 1);
    CacheForGroup(pool, ch[3], 1);
    CacheForGroup(pool, ch[4], 1);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 0);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_TRUE(m.per_group[0].blocks.empty());
    EXPECT_TRUE(m.per_group[1].blocks.empty());
    ExpectSwaWindowIntact(m.per_group[1], /*window=*/10, /*page_size=*/4);
}

TEST(CoordinatorMatchTest, FullShorterThanSwaBoundsSwaWithRunIntact) {
    // full covers 4; swa caches 1..4. Bounded to 4 the run {1,2,3} still reaches
    // contiguous_needed 3, so common stays 4 -- hole only OUTSIDE the last window.
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[1], 0);
    CacheForGroup(pool, ch[2], 0);
    CacheForGroup(pool, ch[3], 0);
    CacheForGroup(pool, ch[1], 1);
    CacheForGroup(pool, ch[2], 1);
    CacheForGroup(pool, ch[3], 1);
    CacheForGroup(pool, ch[4], 1);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 4);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 4u);
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 4);
    ASSERT_EQ(m.per_group[1].blocks.size(), 4u);
    EXPECT_TRUE(m.per_group[1].blocks[0]->IsNull());
    EXPECT_FALSE(m.per_group[1].blocks[1]->IsNull());
    EXPECT_FALSE(m.per_group[1].blocks[2]->IsNull());
    EXPECT_FALSE(m.per_group[1].blocks[3]->IsNull());
    EXPECT_EQ(m.per_group[1].num_hit_blocks, 3);
    ExpectSwaWindowIntact(m.per_group[1], /*window=*/10, /*page_size=*/4);
}

TEST(CoordinatorMatchTest, SwaShorterThanFullTruncatesFull) {
    // swa's best valid match is 4 blocks [null, b1, b2, b3]; full truncates 5 -> 4.
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    CacheForGroup(pool, ch[1], 1);
    CacheForGroup(pool, ch[2], 1);
    CacheForGroup(pool, ch[3], 1);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 4);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 4u);
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 4);
    ASSERT_EQ(m.per_group[1].blocks.size(), 4u);
    EXPECT_TRUE(m.per_group[1].blocks[0]->IsNull());
    EXPECT_EQ(m.per_group[1].num_hit_blocks, 3);
    ExpectSwaWindowIntact(m.per_group[1], /*window=*/10, /*page_size=*/4);
}

TEST(CoordinatorMatchTest, TwoSwaGroupsIterateToFixpoint) {
    // contiguous_needed 3. full: 6 -> swaA {0,2,3,4}: run {2,3,4} keeps 5 ->
    // swaB {0,1,2,3}: run {1,2,3} keeps 4 -> swaA re-match: run {2,3} too short,
    // left run {0} keeps 1 -> swaB re-match at 1: fixpoint common = 1
    // (at length 1 the window clamps to the sequence start).
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 4, 10},
        {AttnKind::kSlidingWindow, 4, 10},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes(
        {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}, {5, 5, 5, 5}});
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    CacheForGroup(pool, ch[0], 1);
    CacheForGroup(pool, ch[2], 1);
    CacheForGroup(pool, ch[3], 1);
    CacheForGroup(pool, ch[4], 1);
    CacheForGroup(pool, ch[0], 2);
    CacheForGroup(pool, ch[1], 2);
    CacheForGroup(pool, ch[2], 2);
    CacheForGroup(pool, ch[3], 2);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 1);
    ASSERT_EQ(m.per_group.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(m.per_group[i].blocks.size(), 1u) << "group " << i;
        EXPECT_FALSE(m.per_group[i].blocks[0]->IsNull()) << "group " << i;
        EXPECT_EQ(m.per_group[i].num_hit_blocks, 1) << "group " << i;
    }
    ExpectSwaWindowIntact(m.per_group[1], /*window=*/10, /*page_size=*/4);
    ExpectSwaWindowIntact(m.per_group[2], /*window=*/10, /*page_size=*/4);
}

TEST(CoordinatorMatchTest, AllFullGroupsMinTruncationUnchanged) {
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kFull, 4, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    CacheForGroup(pool, ch[0], 1);
    CacheForGroup(pool, ch[1], 1);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 2);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 2);
    EXPECT_EQ(m.per_group[1].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[1].num_hit_blocks, 2);
}

TEST(CoordinatorMatchTest, SingleFullGroupUnchanged) {
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[1], 0);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 2);
    ASSERT_EQ(m.per_group.size(), 1u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 2);
}

TEST(CoordinatorMatchTest, SwaOnlyConfigKeepsTailRunWithLeadingHoles) {
    // No full bound: tail run {2,3,4} covers the window; leading holes null-pad to page 0.
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    CacheForGroup(pool, ch[2], 0);
    CacheForGroup(pool, ch[3], 0);
    CacheForGroup(pool, ch[4], 0);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 5);
    ASSERT_EQ(m.per_group.size(), 1u);
    ASSERT_EQ(m.per_group[0].blocks.size(), 5u);
    EXPECT_TRUE(m.per_group[0].blocks[0]->IsNull());
    EXPECT_TRUE(m.per_group[0].blocks[1]->IsNull());
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 3);
    ExpectSwaWindowIntact(m.per_group[0], /*window=*/10, /*page_size=*/4);
}

TEST(CoordinatorAllocTest, AcquireShortfallLeavesClaimedPrefixForCallerToFree) {
    BlockPool pool(6);  // 5 usable
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[0], 1);
    std::int32_t free_before = pool.NumFreeBlocks();  // after caching, before claim

    CoordinatorMatch hit = coord.MatchPrefix(ch);
    ASSERT_EQ(hit.num_common_blocks, 1);

    std::vector<BlockTable> tables(2);
    coord.ClaimCommonPrefix(tables, hit);   // claim 1 cached page each (2 blocks)
    EXPECT_EQ(tables[0].NumBlocks(), 1);
    EXPECT_EQ(tables[1].NumBlocks(), 1);

    // Uncached 8 tokens -> 2 pages/group = 4 needed; 5 usable - 2 claimed = 3 free -> fail.
    EXPECT_FALSE(coord.Acquire(tables, 8));
    EXPECT_EQ(tables[0].NumBlocks(), 1);    // claimed prefix still there
    EXPECT_EQ(tables[1].NumBlocks(), 1);

    coord.Free(tables);
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
}

TEST(KvCacheCoordinatorAdvanceWindow, OnlySlidingWindowGroupEvicts) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kFull, /*page_size=*/2, /*sliding_window=*/0},
        KvCacheSpec{AttnKind::kSlidingWindow, /*page_size=*/2, /*sliding_window=*/4},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    // 6 tokens -> 3 pages per group.
    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/6));
    ASSERT_EQ(tables[0].NumBlocks(), 3);
    ASSERT_EQ(tables[1].NumBlocks(), 3);

    auto full_before = tables[0].Blocks();
    std::vector<CacheBlock*> full_snapshot(full_before.begin(), full_before.end());

    // num_computed_tokens=5 -> swa skipped=5-4+1=2 -> skipped_blocks=2/2=1 -> page 0 evicted.
    coordinator.AdvanceWindow(tables, /*num_computed_tokens=*/5);

    ASSERT_EQ(tables[0].NumBlocks(), 3);
    auto full_after = tables[0].Blocks();
    for (std::int32_t i = 0; i < tables[0].NumBlocks(); ++i) {
        EXPECT_EQ(full_after[i], full_snapshot[i]) << "full group block " << i << " changed";
        EXPECT_NE(full_after[i], pool.NullBlock()) << "full group got a null hole at " << i;
    }

    ASSERT_EQ(tables[1].NumBlocks(), 3);
    EXPECT_EQ(tables[1].Blocks()[0], pool.NullBlock());
    EXPECT_NE(tables[1].Blocks()[1], pool.NullBlock());
    EXPECT_NE(tables[1].Blocks()[2], pool.NullBlock());
}

TEST(CoordinatorMatchTest, ThreeGroupsCommonIsMinCoverageAcrossAll) {
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 4, 40},
        {AttnKind::kSlidingWindow, 4, 40},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch =
        ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    CacheForGroup(pool, ch[0], 1);
    CacheForGroup(pool, ch[1], 1);
    CacheForGroup(pool, ch[2], 1);
    CacheForGroup(pool, ch[0], 2);
    CacheForGroup(pool, ch[1], 2);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 2) << "common = min(4, 3, 2)";
    ASSERT_EQ(m.per_group.size(), 3u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[1].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[2].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 2);
    EXPECT_EQ(m.per_group[1].num_hit_blocks, 2);
    EXPECT_EQ(m.per_group[2].num_hit_blocks, 2);
}

TEST(CoordinatorMatchTest, ThreeGroupsOneAllMissForcesZeroCommon) {
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 4, 40},
        {AttnKind::kSlidingWindow, 4, 40},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    // groups 0 and 1 fully cache both pages; group 2 caches nothing.
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    for (const std::string& h : ch) CacheForGroup(pool, h, 1);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 0) << "one group all-miss -> common 0";
}

TEST(KvCacheCoordinatorStoreCandidates, CollectsPinnedNewRegistrations) {
    BlockPool pool(/*total_num_blocks=*/16, /*enable_caching=*/true);
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kFull, /*page_size=*/2, /*sliding_window=*/0},
        KvCacheSpec{AttnKind::kSlidingWindow, /*page_size=*/2, /*sliding_window=*/4},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    coordinator.EnableStoreCandidateCollection();
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/4));
    std::vector<std::string> hashes = ContentHashes({{1, 2}, {3, 4}});
    const std::int32_t free_before = pool.NumFreeBlocks();

    coordinator.CacheFullBlocks(tables, hashes, /*first_slot=*/0);
    std::vector<KvCacheCoordinator::StoreCandidate> pending = coordinator.TakePendingStores();

    ASSERT_EQ(pending.size(), 4u);  // 2 pages x 2 groups, group-wrapped keys
    EXPECT_EQ(pool.NumFreeBlocks(), free_before) << "pinning ref'd blocks must not touch the free list";
    for (const auto& c : pending) {
        EXPECT_GE(c.block->RefCount(), 2) << "pinned on top of the table ref";
    }
    // Keys are group-wrapped and distinct across groups for the same content hash.
    std::set<std::string> keys;
    for (const auto& c : pending) keys.insert(c.key);
    EXPECT_EQ(keys.size(), 4u);

    // Re-registering the same hashes yields nothing new (IsCached skip).
    coordinator.CacheFullBlocks(tables, hashes, 0);
    EXPECT_TRUE(coordinator.TakePendingStores().empty());

    // Unpin restores balance.
    for (const auto& c : pending) {
        pool.FreeBlocks({c.block});
    }
    coordinator.Free(tables);
    EXPECT_EQ(pool.NumFreeBlocks(), 15);  // all but null block 0
}

TEST(KvCacheCoordinatorStoreCandidates, DisabledByDefaultCollectsNothing) {
    BlockPool pool(16, true);
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kFull, /*page_size=*/2, /*sliding_window=*/0},
        KvCacheSpec{AttnKind::kSlidingWindow, /*page_size=*/2, /*sliding_window=*/4},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(coordinator.Acquire(tables, 4));
    std::vector<std::string> hashes = ContentHashes({{1, 2}, {3, 4}});
    coordinator.CacheFullBlocks(tables, hashes, 0);
    EXPECT_TRUE(coordinator.TakePendingStores().empty());
    coordinator.Free(tables);
}

// The exact slide-credit rule: collection-on credits a slide-out block only when it is CACHED
// (an uncached one is this op's own registration and gets pinned before the slide); collection-off
// keeps the RefCount()==1 rule so uncached same-round registrations still free.
TEST(KvCacheCoordinatorStoreCandidates, SlideCreditExcludesUncachedOnlyWhenCollecting) {
    BlockPool pool(16, true);
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kSlidingWindow, /*page_size=*/2, /*sliding_window=*/4},
    };
    KvCacheCoordinator off = MakeCoordinator(specs, pool);
    std::vector<BlockTable> tables(off.NumGroups());
    ASSERT_TRUE(off.Acquire(tables, /*num_tokens=*/8));  // pages 0..3; N=8 slides out pages 0,1
    EXPECT_EQ(off.BlocksFreedByAdvance(tables, 8), 2) << "collection-off counts uncached ref-1 blocks";
    off.Free(tables);

    KvCacheCoordinator on = MakeCoordinator(specs, pool);
    on.EnableStoreCandidateCollection();
    std::vector<BlockTable> tables2(on.NumGroups());
    ASSERT_TRUE(on.Acquire(tables2, 8));
    EXPECT_EQ(on.BlocksFreedByAdvance(tables2, 8), 0) << "collection-on excludes uncached slide-out blocks";

    std::vector<std::string> hashes = ContentHashes({{1, 2}, {3, 4}});
    on.CacheFullBlocks(tables2, hashes, 0);  // registers + pins pages 0,1
    EXPECT_EQ(on.BlocksFreedByAdvance(tables2, 8), 0) << "cached but still pinned: ref 2";
    for (const auto& c : on.TakePendingStores()) {
        pool.FreeBlocks({c.block});  // unpin, as WriteBackDone would
    }
    EXPECT_EQ(on.BlocksFreedByAdvance(tables2, 8), 2) << "cached and unpinned: exact credit restored";
    on.Free(tables2);
}

}  // namespace
}  // namespace tokenspeed::test
