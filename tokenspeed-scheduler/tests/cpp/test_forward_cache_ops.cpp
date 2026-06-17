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

#include <cstdint>
#include <string>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_types.h"
#include "cache/forward_cache_ops.h"
#include "cache/kv_cache_coordinator.h"
#include "resource/allocator/paged_cache_group.h"
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
    std::vector<std::string> hashes2{std::string(64, 'a'), std::string(64, 'b')};
    ASSERT_TRUE(PrefillChunk(coordinator, tables, hashes2, /*num_tokens=*/4, /*num_full_blocks=*/2));
    EXPECT_EQ(tables[0].NumBlocks(), 4);
    EXPECT_EQ(tables[1].NumBlocks(), 4);
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

}  // namespace
}  // namespace tokenspeed::test
