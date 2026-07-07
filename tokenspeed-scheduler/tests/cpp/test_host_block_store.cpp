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

#include "cache/host_block_store.h"

namespace tokenspeed::test {

TEST(HostBlockStoreTest, BeginCommitLookupRoundTrip) {
    HostBlockStore store(/*num_pages=*/4);
    EXPECT_EQ(store.NumFreePages(), 4);
    auto page = store.BeginStore("k1");
    ASSERT_TRUE(page.has_value());
    EXPECT_FALSE(store.Lookup("k1").has_value());  // in-flight, not yet indexed
    store.CommitStore("k1", *page);
    EXPECT_EQ(store.Lookup("k1"), page);
    EXPECT_EQ(store.NumIndexed(), 1);
}

TEST(HostBlockStoreTest, DedupeCoversInFlightAndIndexed) {
    HostBlockStore store(4);
    auto p = store.BeginStore("k1");
    EXPECT_FALSE(store.BeginStore("k1").has_value());  // in-flight dup
    store.CommitStore("k1", *p);
    EXPECT_FALSE(store.BeginStore("k1").has_value());  // indexed dup
}

TEST(HostBlockStoreTest, AbortReleasesPage) {
    HostBlockStore store(1);
    auto p = store.BeginStore("k1");
    store.AbortStore("k1", *p);
    EXPECT_FALSE(store.Lookup("k1").has_value());
    EXPECT_TRUE(store.BeginStore("k2").has_value());  // page reusable
}

TEST(HostBlockStoreTest, LruEvictsOldestIndexedOnly) {
    HostBlockStore store(2);
    auto p1 = store.BeginStore("k1");
    store.CommitStore("k1", *p1);
    auto p2 = store.BeginStore("k2");  // in-flight, unevictable
    auto p3 = store.BeginStore("k3");  // must evict k1 (the only indexed entry)
    ASSERT_TRUE(p3.has_value());
    EXPECT_FALSE(store.Lookup("k1").has_value());
    store.CommitStore("k2", *p2);
    store.CommitStore("k3", *p3);
    EXPECT_TRUE(store.BeginStore("k4").has_value());  // k2/k3 evictable now
}

TEST(HostBlockStoreTest, FullWithAllPinnedReturnsNullopt) {
    HostBlockStore store(1);
    (void)store.BeginStore("k1");  // in-flight pins the only page
    EXPECT_FALSE(store.BeginStore("k2").has_value());
}

TEST(HostBlockStoreTest, TouchMovesToMru) {
    HostBlockStore store(2);
    auto p1 = store.BeginStore("k1");
    store.CommitStore("k1", *p1);
    auto p2 = store.BeginStore("k2");
    store.CommitStore("k2", *p2);
    store.Touch("k1");  // k2 becomes LRU
    auto p3 = store.BeginStore("k3");
    ASSERT_TRUE(p3.has_value());
    EXPECT_TRUE(store.Lookup("k1").has_value());
    EXPECT_FALSE(store.Lookup("k2").has_value());
}

TEST(HostBlockStoreTest, EvictedKeyCanBeStoredAgain) {
    HostBlockStore store(1);
    auto p1 = store.BeginStore("k1");
    store.CommitStore("k1", *p1);
    auto p2 = store.BeginStore("k2");  // evicts k1
    store.CommitStore("k2", *p2);
    auto p3 = store.BeginStore("k1");  // full lifecycle: evicted key re-enters
    ASSERT_TRUE(p3.has_value());
    store.CommitStore("k1", *p3);
    EXPECT_EQ(store.Lookup("k1"), p3);
    EXPECT_FALSE(store.Lookup("k2").has_value());
}

TEST(HostBlockStoreTest, MisuseThrows) {
    HostBlockStore store(1);
    EXPECT_ANY_THROW(store.CommitStore("never-began", 0));
}

}  // namespace tokenspeed::test
