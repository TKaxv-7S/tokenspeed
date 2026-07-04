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

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "resource/types.h"
#include "scheduler/types.h"
#include "scheduler/request.h"
#include "scheduler/execution_plan.h"
#include "scheduler/execution_event.h"
#include "scheduler/kv_cache_events.h"

#include "resource/allocator/page_allocator.h"
#include "resource/allocator/paged_cache_group.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/allocator/req_pool_allocator.h"
#include "resource/allocator/mamba_chunk_allocator.h"
#include "resource/allocator/mamba_host_allocator.h"
#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"

#include "fsm/forward_events.h"
#include "fsm/cache_events.h"
#include "fsm/pd_events.h"

#if TOKENSPEED_FLAT_KVCACHE
#include "cache/block_pool.h"
#include "cache/kv_cache_coordinator.h"
#include "cache/forward_cache_ops.h"
#endif
namespace tokenspeed {

class Scheduler {
public:
    explicit Scheduler(SchedulerConfig config);

    void SubmitRequests(const std::vector<RequestSpec>& request_specs);
    std::vector<std::string> CalcRollingHash(const std::vector<std::int32_t>& input_tokens, bool apply_match = false);

    ExecutionPlan NextExecutionPlan();

    void Advance(const ExecutionEvent& event);
    std::vector<KvCacheEvent> DrainKvEvents();

    std::size_t WaitingSize() const;
    std::size_t DecodingSize() const;
    std::size_t RetractedSize() const;
    std::size_t AvailableKvPages() const;
    std::size_t ActiveKvPages() const;
    std::size_t PrefillSize() const;
    std::int32_t GetRequestTokenSize(const std::string& id) const;
    std::vector<std::string> PagedCacheGroupIds() const;
    std::int32_t PagedCacheGroupTotalPages(const std::string& group_id) const;
    std::int32_t PagedCacheGroupAvailablePages(const std::string& group_id) const;
    std::int64_t PagedCacheGroupFailedAllocCount(const std::string& group_id) const;
    std::vector<std::int32_t> GetRequestPagedCachePageIds(const std::string& request_id,
                                                          const std::string& group_id) const;
    // Compact-view base logical-page offset; 0 for full-history / unseen.
    std::int32_t GetRequestPagedCacheBaseLogicalPage(const std::string& request_id, const std::string& group_id) const;
#if TOKENSPEED_FLAT_KVCACHE
    // Free physical pages in the flat shared BlockPool (flat path only).
    // AvailableKvPages() reports the same value on flat builds (the Python
    // binding redirects there; see scheduler.cpp); this int32 accessor stays
    // for C++ tests.
    std::int32_t FlatPoolFreeBlocks() const { return block_pool_.NumFreeBlocks(); }
#endif

private:
    // Second element is LoadBackOperation list (normal path) or WriteBackOperation list (retract triggered).
    std::tuple<std::vector<ForwardOperation>,
               std::variant<std::vector<LoadBackOperation>, std::vector<WriteBackOperation>>>
    newForwardOperation(std::vector<Request*> candidates);
    std::vector<WriteBackOperation> newWriteBackOperation(
        std::unordered_map<std::string, std::unique_ptr<Request>>& requests);
    std::optional<WriteBackOperation> newRetractOperation(Request* retract_request);

    PrefillOperation applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillFirstChunkEvent event);
    PrefillOperation applyEventAndGenerateOp(Request* request, fsm::SchedulePrefillEvent event);
    DecodeOperation applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeEvent event);
    DecodeOperation applyEventAndGenerateOp(Request* request, fsm::ScheduleDecodeFromRetractedEvent event);
    std::optional<WriteBackOperation> applyEventAndGenerateOp(Request* request, fsm::ScheduleRetractEvent event);
    PrefetchOperation applyEventAndGenerateOp(Request* request, fsm::SchedulePrefetchEvent event);

    std::optional<fsm::SchedulePrefetchEvent> schedulePrefetch(Request* request, const MatchResult& match);

    std::optional<fsm::SchedulePrefillFirstChunkEvent> schedulePrefillFirstChunk(
        Request* request, std::int32_t remaining, std::int32_t reserve_num_tokens_in_next_schedule_event,
        bool disable_l2_cache, std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::SchedulePrefillEvent> schedulePrefill(Request* request, std::int32_t remaining,
                                                             std::int32_t reserve_num_tokens_in_next_schedule_event,
                                                             std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::ScheduleDecodeEvent> scheduleDecode(Request* request,
                                                           std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::ScheduleDecodeFromRetractedEvent> scheduleDecodeFromRetracted(
        Request* request, std::map<std::string, std::int32_t>& simulated_free);
    std::optional<fsm::ScheduleRetractEvent> scheduleRetract(Request* request);

    void check_device_mem();

private:
    void handleEvent(const cache::PrefetchDone& event);
    void handleEvent(const cache::WriteBackDone& event);
    void handleEvent(const pd::BootstrappedEvent& event);
    void handleEvent(const pd::FailedEvent& event);
    void handleEvent(const pd::SucceededEvent& event);
    void handleEvent(const pd::RemotePrefillDoneEvent& event);
    void handleEvent(const forward::ExtendResult& event);
    void handleEvent(const forward::Abort& event);
    void handleEvent(const forward::Finish& event);
    void handleEvent(const forward::UpdateReserveNumTokens& event);

private:
    Request* find_request(std::string rid) {
        auto it = requests_.find(rid);
        return it != requests_.end() ? it->second.get() : nullptr;
    }

    // Group-id list for flat KV-cache ops; empty span on the radix path so call
    // sites stay #if-free.
    std::span<const std::string> FlatGroupIds() const {
#if TOKENSPEED_FLAT_KVCACHE
        return flat_group_ids_;
#else
        return {};
#endif
    }

private:
    SchedulerConfig config_;

private:
    PageAllocator device_allocator_;
    PageAllocator host_allocator_;
    std::optional<MambaChunkAllocator> mamba_allocator_{};
    std::optional<MambaHostAllocator> mamba_host_allocator_{};
    KVPrefixCache kv_prefix_cache_;
    ReqPoolAllocator req_pool_allocator_;
    std::optional<HybridPrefixCache> hybrid_prefix_cache_{};

#if TOKENSPEED_FLAT_KVCACHE
    BlockPool block_pool_;
    KvCacheCoordinator coordinator_;
    std::vector<std::string> flat_group_ids_;  // group_id per cache group, index-aligned to coordinator groups
    // Forward results the executor still owes us, per request: one ExtendResult
    // per emitted decode op and one per prefill-completing op (mid-prefill chunk
    // ops produce no event, so they are not counted). Incremented when
    // newForwardOperation emits such an op, decremented on ExtendResult, erased
    // on Finish/Abort/PD-success. Non-empty means a forward is in flight whose
    // completion can still finish a request and free pool pages -- the flat
    // starvation-deadlock check keys off this (see newForwardOperation).
    std::unordered_map<std::string, std::int32_t> pending_forward_results_;
    // Decode-reserve pages promised but not yet acquired, per request. A
    // prefill-completing admission (first or final chunk) charges its gate
    // chunk + decode reserve, but the reserve is only Acquired at the
    // PrefillDone->Decoding transition -- between those rounds the promised
    // pages still sit in the pool's free count. Entries record the exact page
    // need (BlocksNeededFor on the post-prefill table shape, computed at
    // admission time) so other candidates' gates can subtract them and not be
    // admitted into promised pages. Inserted when the prefill-completing
    // admission commits, erased when the PrefillDone->Decoding transition
    // acquires the reserve, and on Finish/Abort/PD-success (no phantom
    // reservation may outlive its request).
    std::unordered_map<std::string, std::int32_t> flat_reserved_pages_;
    // Consecutive fully-starved fused rounds (every candidate deferred, pool
    // pages held, nothing in flight). The deadlock assert in
    // newForwardOperation requires TWO such rounds: a single round can be a
    // false positive when a pool-freeing Finish is queued between a request's
    // ExtendResult (which empties pending_forward_results_) and its Finish
    // (which frees the pages). Reset on any round that schedules ops or fails
    // the starvation predicate. Residual risk (accepted for this fail-loud
    // stopgap until TODO(flat-retract)): a Finish arriving two or more rounds
    // after the last ExtendResult still trips the assert on a recoverable
    // state.
    std::int32_t flat_starved_rounds_{0};

    // Sum of flat_reserved_pages_ excluding request_id's own entry: a request
    // gating its own decode is the one consuming its reservation, so it must
    // not be blocked by it.
    std::int32_t flatReservedPagesExcept(const std::string& request_id) const {
        std::int32_t total = 0;
        for (const auto& [id, pages] : flat_reserved_pages_) {
            if (id != request_id) {
                total += pages;
            }
        }
        return total;
    }
#endif

private:
    std::unordered_map<std::string, std::unique_ptr<Request>> requests_;
    std::unordered_map<cache_op_id, CacheOpSpec> cache_op_tracker_;
    std::vector<KvCacheEvent> kv_events_;
    // Stats
    SchedulerStats stats_;
};

}  // namespace tokenspeed
