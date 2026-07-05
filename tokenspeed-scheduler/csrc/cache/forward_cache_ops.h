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
#include <map>
#include <span>
#include <string>
#include <vector>

#include "cache/cache_types.h"
#include "cache/kv_cache_coordinator.h"

namespace tokenspeed {

struct SchedulerConfig;  // defined in scheduler/types.h; only used by-ref below

// Claim/slide safety while KV writes are still in flight (overlap scheduling):
// all forwards share one execution stream, so any reuse write of a freed page is
// enqueued after the in-flight kernels; slid-out pages also lie below every later
// batch's read window, and claimed pages are ref>1 so they cannot be freed and
// rewritten from outside the stream.
// TODO(flat-l2): out-of-stream writers (load-back H2D) must fence before joining the flat path.

// Prefill first chunk: claim the admission-layer prefix hit into fresh tables,
// then acquire pages for the NEW tokens only (claimed pages are full, no tail
// credit, so the page math is exact). On false (pool short) nothing is acquired
// but the claimed blocks REMAIN in the tables -- the caller must FreeRequest.
// TODO(flat-swa-alloc): SWA Acquire transiently allocates the full chunk (even chunk >> window); gates charge it.
bool PrefillFirstChunk(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                       const CoordinatorMatch& hit, std::int32_t num_new_tokens);

// Register the prior chunks' completed pages, slide the SWA window to
// num_computed_tokens (tokens of chunks 0..k-1), then acquire this chunk's
// pages. False = pool short: registration and slide already ran, nothing was
// allocated.
bool PrefillChunk(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                  std::span<const std::string> content_hashes, std::int32_t num_tokens,
                  std::int32_t num_computed_tokens);

// One decode step, same register->slide->acquire shape as PrefillChunk: hash j
// registers under slot first_page_slot + j. False = pool short, nothing allocated.
bool DecodeStep(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                std::span<const std::string> content_hashes, std::int32_t first_page_slot,
                std::int32_t num_tokens, std::int32_t num_computed_tokens);

// Prefill -> decode transition: register the remaining full prefill pages,
// slide the SWA window to num_computed_tokens (the full prefill length), then
// acquire the first decode step's reservation. False = pool short: registration
// and slide already ran, nothing was allocated.
bool FinalizePrefillAndReserveDecode(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                                     std::span<const std::string> content_hashes, std::int32_t reserve_tokens,
                                     std::int32_t num_computed_tokens);

// Translate the Python-provided per-group cache config into KvCacheSpecs (one
// per paged_cache_group, group_id = index). All groups share config.page_size.
std::vector<KvCacheSpec> MakeSpecsFromConfig(const SchedulerConfig& config);

// Finish / abort: return every page in every table to the pool.
void FreeRequest(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables);

// One row per group, keyed by config group_id (e.g. "full"/"swa") to match the
// Python-side assertions: the BlockId() sequence in absolute logical-page order,
// null-block holes written as 0 (no compaction).
std::map<std::string, std::vector<std::int32_t>> BuildFlatBlockTables(
    const std::vector<BlockTable>& tables, std::span<const std::string> group_ids);

}  // namespace tokenspeed
