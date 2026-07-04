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
#include <string>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_types.h"
#include "cache/forward_cache_ops.h"
#include "cache/kv_cache_coordinator.h"
#include "resource/allocator/paged_cache_group.h"
#include "scheduler/page_hasher.h"
#include "scheduler/types.h"

namespace tokenspeed::test {
namespace {

KvCacheCoordinator MakeTwoGroup(BlockPool& pool) {
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kFull, /*page_size=*/2, /*sliding_window=*/0},
        KvCacheSpec{AttnKind::kSlidingWindow, /*page_size=*/2, /*sliding_window=*/4},
    };
    return MakeCoordinator(specs, pool);
}

TEST(ForwardCacheOpsFree, ReturnsAllPagesToPool) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);
    const std::int32_t free_before = pool.NumFreeBlocks();

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/6));
    ASSERT_LT(pool.NumFreeBlocks(), free_before);

    FreeRequest(coordinator, tables);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
}

TEST(ForwardCacheOpsPrefill, FirstChunkAcquiresPagesForTokens) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    // No prefix cached -> MatchPrefix empty -> Acquire 4 tokens -> 2 pages/group.
    std::vector<std::string> hashes;  // empty: miss path
    ASSERT_TRUE(PrefillFirstChunk(coordinator, tables, hashes, /*num_tokens=*/4));
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(ForwardCacheOpsPrefill, ChunkAcquiresAndCachesFullBlocks) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    std::vector<std::string> hashes0;
    ASSERT_TRUE(PrefillFirstChunk(coordinator, tables, hashes0, /*num_tokens=*/4));

    // Second chunk: 4 more tokens -> +2 pages/group; cache the 2 now-full pages.
    // num_computed = 4 (chunk 0's tokens) -> skipped = 4-4+1 = 1 -> 1/2 = 0
    // fully-slid-out pages: the slide punches nothing yet.
    std::vector<std::string> hashes2{std::string(64, 'a'), std::string(64, 'b')};
    ASSERT_TRUE(PrefillChunk(coordinator, tables, hashes2, /*num_tokens=*/4, /*num_full_blocks=*/2,
                             /*num_computed_tokens=*/4));
    EXPECT_EQ(tables[0].NumBlocks(), 4);
    EXPECT_EQ(tables[1].NumBlocks(), 4);
    for (CacheBlock* b : tables[1].Blocks()) {
        EXPECT_FALSE(b->IsNull()) << "num_computed=4, W=4: no page is fully out of window yet";
    }
}

// Mid-prefill slide: a chunk whose prior tokens already exceed the window must
// punch the fully-slid-out SWA pages BEFORE acquiring its own pages, and the
// punched pages' hashes must have been registered first (register -> punch
// ordering; CacheFullBlocks skips holes, so the reverse order would lose them).
TEST(ForwardCacheOpsPrefill, ChunkSlidesSwaWindowAndKeepsPunchedPageHashes) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);  // page=2, W=4
    std::vector<BlockTable> tables(coordinator.NumGroups());

    // Chunk 0: 8 tokens -> 4 pages/group (chunk >> window is fine: the slide
    // happens on the NEXT op; see TODO(flat-swa-alloc)).
    std::vector<std::string> hashes0;
    ASSERT_TRUE(PrefillFirstChunk(coordinator, tables, hashes0, /*num_tokens=*/8));
    const std::int32_t free_before_chunk = pool.NumFreeBlocks();

    // Chunk 1: num_computed = 8 -> skipped = 8-4+1 = 5 -> 5/2 = 2 pages fully
    // out of window: SWA slots 0,1 punched, then 2 fresh pages acquired.
    std::vector<std::string> hashes{std::string(64, 'a'), std::string(64, 'b'), std::string(64, 'c'),
                                    std::string(64, 'd')};
    ASSERT_TRUE(PrefillChunk(coordinator, tables, hashes, /*num_tokens=*/4, /*num_full_blocks=*/4,
                             /*num_computed_tokens=*/8));

    // Full group keeps history; SWA group has holes exactly at slots 0,1.
    EXPECT_EQ(tables[0].NumBlocks(), 6);
    for (CacheBlock* b : tables[0].Blocks()) {
        EXPECT_FALSE(b->IsNull());
    }
    ASSERT_EQ(tables[1].NumBlocks(), 6);
    EXPECT_TRUE(tables[1].Blocks()[0]->IsNull());
    EXPECT_TRUE(tables[1].Blocks()[1]->IsNull());
    for (std::int32_t i = 2; i < 6; ++i) {
        EXPECT_FALSE(tables[1].Blocks()[i]->IsNull()) << "slot " << i;
    }

    // Pool: the slide freed 2 SWA pages, the acquire took 2/group = 4 -> net -2.
    EXPECT_EQ(pool.NumFreeBlocks(), free_before_chunk + 2 - 4);

    // Register-then-punch: the punched SWA pages went to the free list WITH
    // their hashes (cached-with-hash free blocks), so they stay prefix-hittable.
    for (const std::string& h : {hashes[0], hashes[1]}) {
        EXPECT_NE(pool.GetCachedBlock(MakeKeyWithGroupId(h, /*group_id=*/1)), nullptr)
            << "slid-out page must keep its registered hash";
    }
}

// Finalize slides too: the prefill->decode transition releases the pages the
// FIRST decode step no longer needs (query at position P reads keys back to
// P - W + 1), instead of retaining them until the second decode step.
TEST(ForwardCacheOpsPrefill, FinalizeSlidesSwaWindowBeforeReserveAcquire) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);  // page=2, W=4
    std::vector<BlockTable> tables(coordinator.NumGroups());

    // 12-token prefill -> 6 pages/group, tails full.
    std::vector<std::string> hashes0;
    ASSERT_TRUE(PrefillFirstChunk(coordinator, tables, hashes0, /*num_tokens=*/12));
    const std::int32_t free_before = pool.NumFreeBlocks();

    // num_computed = 12 -> skipped = 12-4+1 = 9 -> 9/2 = 4 pages punched
    // (slots 0..3); reserve 1 token -> 1 fresh page/group.
    std::vector<std::string> hashes(6, "");
    for (std::size_t i = 0; i < hashes.size(); ++i) {
        hashes[i] = std::string(64, static_cast<char>('a' + i));
    }
    ASSERT_TRUE(FinalizePrefillAndReserveDecode(coordinator, tables, hashes, /*reserve_tokens=*/1,
                                                /*num_computed_tokens=*/12));

    ASSERT_EQ(tables[1].NumBlocks(), 7);
    for (std::int32_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(tables[1].Blocks()[i]->IsNull()) << "slot " << i;
    }
    for (std::int32_t i = 4; i < 7; ++i) {
        EXPECT_FALSE(tables[1].Blocks()[i]->IsNull()) << "slot " << i;
    }
    EXPECT_EQ(tables[0].NumBlocks(), 7);
    // Pool: slide freed 4, reserve acquire took 1/group = 2 -> net +2.
    EXPECT_EQ(pool.NumFreeBlocks(), free_before + 4 - 2);
}

TEST(ForwardCacheOpsDecode, StepAcquiresAndSlidesSwaWindow) {
    BlockPool pool(/*total_num_blocks=*/64, /*enable_caching=*/true);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);  // swa window=4, page_size=2
    std::vector<BlockTable> tables(coordinator.NumGroups());

    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/6));  // 3 pages/group

    for (std::int32_t computed = 7; computed <= 13; ++computed) {
        ASSERT_TRUE(DecodeStep(coordinator, tables, /*num_tokens=*/1, /*num_computed_tokens=*/computed));
    }
    // Full group: 13 tokens -> ceil(13/2)=7 pages, no nulls.
    EXPECT_EQ(tables[0].NumBlocks(), 7);
    std::int32_t full_nulls = 0;
    for (auto* b : tables[0].Blocks()) {
        if (b == pool.NullBlock()) ++full_nulls;
    }
    EXPECT_EQ(full_nulls, 0);
    // Swa group: active (non-null) pages bounded.
    std::int32_t swa_active = 0;
    for (auto* b : tables[1].Blocks()) {
        if (b != pool.NullBlock()) ++swa_active;
    }
    EXPECT_LE(swa_active, 3);
}

TEST(ForwardCacheOpsSpecs, TranslatesPagedCacheGroups) {
    SchedulerConfig config;
    config.page_size = 16;
    PagedCacheGroupConfig full_grp;
    full_grp.group_id = "full";
    full_grp.retention = PagedCacheGroupConfig::Retention::FullHistory;
    PagedCacheGroupConfig swa_grp;
    swa_grp.group_id = "swa";
    swa_grp.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
    swa_grp.sliding_window_tokens = 128;
    config.paged_cache_groups = {full_grp, swa_grp};

    std::vector<KvCacheSpec> specs = MakeSpecsFromConfig(config);
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].kind, AttnKind::kFull);
    EXPECT_EQ(specs[0].page_size, 16);
    EXPECT_EQ(specs[0].sliding_window, 0);
    EXPECT_EQ(specs[1].kind, AttnKind::kSlidingWindow);
    EXPECT_EQ(specs[1].page_size, 16);
    EXPECT_EQ(specs[1].sliding_window, 128);
}

TEST(ForwardCacheOpsBuildFlatBlockTables, TwoGroupsRowsAndIds) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    // 6 tokens, page_size 2 -> 3 pages per group.
    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/6));

    std::vector<std::string> group_ids{"full", "swa"};
    auto built = BuildFlatBlockTables(tables, group_ids);

    ASSERT_EQ(built.size(), 2u);
    ASSERT_TRUE(built.count("full"));
    ASSERT_TRUE(built.count("swa"));
    EXPECT_EQ(built.at("full").size(), 3u);
    EXPECT_EQ(built.at("swa").size(), 3u);
    // Full group has no null holes: every id is a real (non-zero) physical page.
    for (std::int32_t id : built.at("full")) {
        EXPECT_GT(id, 0);
    }
    // Rows must match the source span exactly: absolute logical-page order,
    // no compaction, null hole = 0 in its exact slot.
    std::vector<std::int32_t> expected_full;
    for (auto* b : tables[0].Blocks()) {
        expected_full.push_back(b->IsNull() ? 0 : b->BlockId());
    }
    std::vector<std::int32_t> expected_swa;
    for (auto* b : tables[1].Blocks()) {
        expected_swa.push_back(b->IsNull() ? 0 : b->BlockId());
    }
    EXPECT_EQ(built.at("full"), expected_full);
    EXPECT_EQ(built.at("swa"), expected_swa);
}

TEST(ForwardCacheOpsBuildFlatBlockTables, SwaRowGetsNullHoleAfterAdvance) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    // sliding_window=4, page_size=2. Acquire enough tokens that the window
    // (4 tokens = 2 pages) leaves an earlier page out of window.
    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/8));      // 4 pages/group
    coordinator.AdvanceWindow(tables, /*num_computed_tokens=*/8);

    std::vector<std::string> group_ids{"full", "swa"};
    auto built = BuildFlatBlockTables(tables, group_ids);
    // Full row never has a 0 hole.
    for (std::int32_t id : built.at("full")) {
        EXPECT_GT(id, 0);
    }
    // SWA row has at least one null hole (id 0) where a page slid out of window.
    const auto& swa = built.at("swa");
    EXPECT_NE(std::find(swa.begin(), swa.end(), 0), swa.end());
    // Pin the exact hole position (not just its existence): the row must equal
    // the source span verbatim, with null holes mapped to 0 in their own slot.
    std::vector<std::int32_t> expected_swa;
    for (auto* b : tables[1].Blocks()) {
        expected_swa.push_back(b->IsNull() ? 0 : b->BlockId());
    }
    EXPECT_EQ(swa, expected_swa);
}

TEST(ForwardCacheOpsBuildFlatBlockTables, FreshTablesProduceEmptyRows) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);
    // Never Acquire: both BlockTables have zero blocks.
    std::vector<BlockTable> tables(coordinator.NumGroups());

    std::vector<std::string> group_ids{"full", "swa"};
    auto built = BuildFlatBlockTables(tables, group_ids);

    // One key per group, each row empty (zero pages allocated).
    ASSERT_EQ(built.size(), 2u);
    EXPECT_TRUE(built.at("full").empty());
    EXPECT_TRUE(built.at("swa").empty());
}

TEST(ForwardCacheOpsBuildFlatBlockTables, SingleGroupRowMatchesSource) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kFull, /*page_size=*/2, /*sliding_window=*/0},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/4));  // 2 pages

    std::vector<std::string> group_ids{"only"};
    auto built = BuildFlatBlockTables(tables, group_ids);

    ASSERT_EQ(built.size(), 1u);
    std::vector<std::int32_t> expected;
    for (auto* b : tables[0].Blocks()) {
        expected.push_back(b->IsNull() ? 0 : b->BlockId());
    }
    EXPECT_EQ(built.at("only"), expected);
    // Sanity: keyed by the supplied group_id, not a bare index.
    EXPECT_EQ(built.count("0"), 0u);
}

TEST(ForwardCacheOpsBuildFlatBlockTables, KeyMatchesSuppliedGroupIdStrings) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/4));

    // Arbitrary (non-"full"/"swa") ids: keys must come straight from the input,
    // and group_ids[i] must pair with tables[i] by index.
    std::vector<std::string> group_ids{"alpha", "beta"};
    auto built = BuildFlatBlockTables(tables, group_ids);

    ASSERT_EQ(built.size(), 2u);
    EXPECT_TRUE(built.count("alpha"));
    EXPECT_TRUE(built.count("beta"));
    std::vector<std::int32_t> expected_alpha;
    for (auto* b : tables[0].Blocks()) {
        expected_alpha.push_back(b->IsNull() ? 0 : b->BlockId());
    }
    EXPECT_EQ(built.at("alpha"), expected_alpha);
}

}  // namespace
}  // namespace tokenspeed::test
