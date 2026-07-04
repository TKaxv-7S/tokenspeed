# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel import (
    attn_merge_state,
    mha_decode_scheduler_metadata,
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.layers.attention.utils import build_page_table
from tokenspeed.runtime.utils.common import ceil_div
from tokenspeed.runtime.utils.env import global_server_args_dict

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

_KERNEL_SOLUTION_BY_BACKEND = {
    "mha": None,
    "fa3": "fa3",
    "fa4": "fa4",
    "triton": "triton",
    "flashinfer": "flashinfer",
}


@dataclass(kw_only=True)
class MHAPrefillMetadata:
    # Device-side metadata:
    # - seq_lens: total length after this step
    # - extend_seq_lens: length of new tokens
    #   cu_extend_seq_lens: the cumsum version of extend_seq_lens
    # - extend_prefix_lens: length of the cached prefix tokens
    # seq_lens[i] = extend_prefix_lens[i] + extend_seq_lens[i]
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    cu_extend_seq_lens: torch.Tensor
    extend_prefix_lens: torch.Tensor
    # Host-side metadata:
    extend_seq_lens_cpu: list[int]
    cu_extend_seq_lens_cpu: list[int]
    max_extend_seq_len: int
    max_extend_prefix_len: int = 0
    # Flat KV-cache per-group page tables (group_id -> [num_reqs, max_pages]).
    # None on the single-table path. TODO(radix-removal): drop with the single
    # page_table field once flat is the only path.
    page_tables: dict[str, torch.Tensor] | None = None
    # Flat per-group KV WRITE locations (group_id -> [num_tokens] int32),
    # derived from page_tables[gid] so writes land in the pages the group
    # reads (M-W1). None on the single-table path. Built wherever
    # page_tables is built -- same groups, same lifecycle.
    out_cache_locs: dict[str, torch.Tensor] | None = None


@dataclass(kw_only=True)
class MHADecodeMetadata:
    # Device-side metadata.
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    scheduler_metadata: torch.Tensor | None = None
    page_tables: dict[str, torch.Tensor] | None = None
    # Flat per-group KV WRITE locations (group_id -> [num_tokens] int32),
    # derived from page_tables[gid] so writes land in the pages the group
    # reads (M-W1). None on the single-table path. Built wherever
    # page_tables is built -- same groups, same lifecycle.
    out_cache_locs: dict[str, torch.Tensor] | None = None


class MHAAttnBackend(AttentionBackend):
    """Standard MHA backend that routes through tokenspeed_kernel attention APIs."""

    # Consume flat per-group page tables when the scheduler delivers them. The
    # flag is unconditional; safety comes from upstream gating plus a replay
    # guard: (a) the pool publishes paged_cache_group_specs only when the
    # tokenspeed_scheduler ext is flat-built and spec decode is off (see the
    # publication rule in kv_cache/mha.py), so capture only allocates flat
    # buffers when the scheduler will actually deliver tables, and (b)
    # init_forward_metadata_replay_cuda_graph raises if a flat-captured graph
    # would replay without fresh tables. TODO(radix-removal): unconditional
    # once radix is gone.
    uses_flat_cache_groups: bool = True

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        return forward_mode is not None and forward_mode.is_decode()

    def __init__(self, config: MHAConfig):
        super().__init__(config)
        # Map the selected backend to the corresponding kernel solution string.
        backend_name = config.backend_name or "mha"
        self.kernel_solution = _KERNEL_SOLUTION_BY_BACKEND[backend_name]

        # Set the MHA extend mode:
        # - "paged": write kv to cache first and use a single kernel for prefill
        # - "ragged": split the cached prefix and the non-cached part into two
        #             kernels and merge their outputs.
        self.mha_extend_mode = global_server_args_dict.get("mha_extend_mode", "paged")

        # Static information needed for metadata construction and kernel dispatch
        self.max_context_len = config.context_len
        self.page_size = config.page_size
        self.max_num_pages = ceil_div(self.max_context_len, self.page_size)
        num_q_heads = config.num_attention_heads
        num_kv_heads = config.num_kv_heads
        self.tp_q_head_num = max(num_q_heads // config.attn_tp_size, 1)
        self.tp_kv_head_num = max(num_kv_heads // config.attn_tp_size, 1)
        self.head_dim = config.head_dim
        self.qkv_dtype = config.dtype

        # Forward metadata is initialized in the runner per forward call
        self.forward_decode_metadata: MHADecodeMetadata | None = None
        self.forward_prefill_metadata: MHAPrefillMetadata | None = None

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        forward_mode: ForwardMode,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        flat_block_tables: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ):
        assert not forward_mode.is_mixed(), "mha backend does not support mixed batch"

        seq_lens = seq_lens[:bs]
        page_table = build_page_table(
            req_pool_indices[:bs],
            req_to_page,
            self.page_size,
            self.max_context_len,
        )

        # Flat per-group tables (Phase 3 delivered them already bridged to GPU
        # tensors). Present only when uses_flat_cache_groups AND the op carried a
        # non-empty flat_block_tables (the wrapper gates on the flag). None -> the
        # single-table path below is unchanged.
        flat_page_tables = flat_block_tables or None

        # Per-group KV WRITE locations, derived from the same tables the
        # groups read (M-W1). Extend uses the raw extend_prefix_lens sliced
        # here (equivalent to the [:bs] slice inside the extend branch).
        flat_out_cache_locs = None
        if flat_page_tables:
            if forward_mode.is_extend_or_mixed():
                flat_out_cache_locs = self._compute_flat_extend_out_cache_locs(
                    flat_page_tables,
                    seq_lens,
                    extend_prefix_lens[:bs],
                    self.page_size,
                )
            else:
                flat_out_cache_locs = self._compute_flat_decode_out_cache_locs(
                    flat_page_tables,
                    seq_lens,
                    self.page_size,
                )
            self._maybe_check_flat_write_locs(
                flat_page_tables, flat_out_cache_locs, self.page_size
            )

        if forward_mode.is_extend_or_mixed():
            extend_seq_lens = extend_seq_lens[:bs]
            extend_seq_lens_cpu = [int(x) for x in extend_seq_lens_cpu[:bs].tolist()]
            cu_extend_seq_lens, cu_extend_seq_lens_cpu = self._make_cu_extend_seq_lens(
                extend_seq_lens,
                extend_seq_lens_cpu,
            )
            extend_prefix_lens = extend_prefix_lens[:bs]
            max_extend_seq_len = max(extend_seq_lens_cpu)
            max_extend_prefix_len = int(extend_prefix_lens_cpu[:bs].max().item())

            self.forward_prefill_metadata = MHAPrefillMetadata(
                page_table=page_table,
                seq_lens=seq_lens,
                extend_seq_lens=extend_seq_lens,
                cu_extend_seq_lens=cu_extend_seq_lens,
                extend_prefix_lens=extend_prefix_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                cu_extend_seq_lens_cpu=cu_extend_seq_lens_cpu,
                max_extend_seq_len=max_extend_seq_len,
                max_extend_prefix_len=max_extend_prefix_len,
                page_tables=flat_page_tables,
                out_cache_locs=flat_out_cache_locs,
            )

            # Drafter: also fill decode_metadata so step 1+ multi-step has
            # metadata under EXTEND/MIXED target. seq_lens is the drafter's
            # live alias buffer (wrapper pre-writes it before this call).
            if self.is_draft:
                self.forward_decode_metadata = MHADecodeMetadata(
                    page_table=page_table,
                    seq_lens=seq_lens,
                    page_tables=flat_page_tables,
                    out_cache_locs=flat_out_cache_locs,
                )
        else:
            if self.spec_num_tokens > 1:
                if self.is_draft:
                    self.forward_decode_metadata = MHADecodeMetadata(
                        page_table=page_table,
                        seq_lens=seq_lens,
                        page_tables=flat_page_tables,
                        out_cache_locs=flat_out_cache_locs,
                    )
                else:
                    expanded_page_table, expanded_seq_lens = (
                        self._make_spec_metadata_buffers(
                            bs,
                            page_table.device,
                        )
                    )
                    self._fill_spec_metadata(
                        expanded_page_table,
                        expanded_seq_lens,
                        page_table,
                        seq_lens,
                    )
                    self.forward_decode_metadata = MHADecodeMetadata(
                        page_table=expanded_page_table,
                        seq_lens=expanded_seq_lens,
                        page_tables=flat_page_tables,
                        out_cache_locs=flat_out_cache_locs,
                    )
            else:
                scheduler_metadata = self._maybe_compute_scheduler_metadata(
                    bs,
                    seq_lens,
                )
                self.forward_decode_metadata = MHADecodeMetadata(
                    page_table=page_table,
                    seq_lens=seq_lens,
                    scheduler_metadata=scheduler_metadata,
                    page_tables=flat_page_tables,
                    out_cache_locs=flat_out_cache_locs,
                )

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor):
        assert (
            seq_lens_buf.dtype == torch.int32
            and seq_lens_buf.dim() == 1
            and seq_lens_buf.shape[0] >= max_bs
        ), (
            f"seq_lens_buf must be int32 with shape[0] >= {max_bs}, "
            f"got {seq_lens_buf.dtype} {tuple(seq_lens_buf.shape)}"
        )

        self.cuda_graph_decode_metadata = {}
        if self.spec_num_tokens > 1 and not self.is_draft:
            page_table, seq_lens = self._make_spec_metadata_buffers(
                max_bs,
                self.device,
            )
            self.cuda_graph_page_table = page_table
            self.cuda_graph_seq_lens = seq_lens
            self.cuda_graph_page_table.zero_()
        else:
            # Alias controller's seq_lens_buf — backend never mutates it.
            self.cuda_graph_page_table = torch.zeros(
                (max_bs, self.max_num_pages), dtype=torch.int32, device=self.device
            )
            self.cuda_graph_seq_lens = seq_lens_buf

        # Flat per-group CUDA-graph buffers (gid -> [max_bs, max_num_pages]),
        # lazily allocated on first capture keyed by the flat dict's groups.
        # TODO(radix-removal): parallels cuda_graph_page_table for the flat path.
        self.cuda_graph_flat_page_tables: dict[str, torch.Tensor] = {}
        # Flat per-group write-loc buffers (gid -> [max_bs] int32), lazily
        # allocated with the page-table buffers at first capture. Replay
        # recomputes locs from the persistent tables and copy_'s here.
        self.cuda_graph_flat_out_cache_locs: dict[str, torch.Tensor] = {}
        self._cuda_graph_max_bs = max_bs

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        flat_cache_group_ids: tuple[str, ...] = (),
        **kwargs,
    ):
        assert not forward_mode.is_extend_or_mixed()

        # Flat per-group tables: lazily allocate persistent buffers for the
        # given group ids and hand metadata slices into them, so replay can
        # copy_ into the same addresses the graph recorded. Real tables only
        # arrive at replay, so capture needs nothing but the ids. Empty when
        # the flat path is off — single-table behavior is unchanged.
        page_tables = None
        out_cache_locs = None
        if flat_cache_group_ids:
            # Flat + spec-decode is out of scope for M10: the per-group views
            # below are bs rows while spec buffers are expanded to
            # bs * spec_num_tokens rows (gpt-oss does not use spec decode).
            assert not (self.spec_num_tokens > 1 and not self.is_draft), (
                "flat_cache_group_ids is unsupported with spec_num_tokens > 1"
            )
            page_tables = {}
            out_cache_locs = {}
            for gid in flat_cache_group_ids:
                buf = self.cuda_graph_flat_page_tables.get(gid)
                if buf is None:
                    buf = torch.zeros(
                        (self._cuda_graph_max_bs, self.max_num_pages),
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self.cuda_graph_flat_page_tables[gid] = buf
                # Write-loc buffer alongside the page-table buffer (same gid
                # source, same lifecycle): metadata gets a view so the graph
                # records the persistent address; replay copy_'s fresh locs.
                loc_buf = self.cuda_graph_flat_out_cache_locs.get(gid)
                if loc_buf is None:
                    loc_buf = torch.zeros(
                        (self._cuda_graph_max_bs,),
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self.cuda_graph_flat_out_cache_locs[gid] = loc_buf
                page_tables[gid] = buf[:bs, :]
                out_cache_locs[gid] = loc_buf[:bs]

        if self.spec_num_tokens > 1 and not self.is_draft:
            expanded_bs = bs * self.spec_num_tokens
            metadata = MHADecodeMetadata(
                page_table=self.cuda_graph_page_table[:expanded_bs, :],
                seq_lens=self.cuda_graph_seq_lens[:expanded_bs],
                page_tables=page_tables,
                out_cache_locs=out_cache_locs,
            )
            self._fill_spec_seq_lens(
                metadata.seq_lens,
                seq_lens[:bs].clamp_min(self.spec_num_tokens),
            )
            self.cuda_graph_decode_metadata[bs] = metadata
            self.forward_decode_metadata = metadata
        else:
            seq_lens = self.cuda_graph_seq_lens[:bs]
            metadata = MHADecodeMetadata(
                page_table=self.cuda_graph_page_table[:bs, :],
                seq_lens=seq_lens,
                page_tables=page_tables,
                out_cache_locs=out_cache_locs,
            )
            self.cuda_graph_decode_metadata[bs] = metadata
            self.forward_decode_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        forward_mode: ForwardMode,
        flat_block_tables: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ):
        assert not forward_mode.is_extend_or_mixed()

        # A flat-captured graph replayed without fresh tables would silently
        # compute over stale/zero page tables — fail loudly instead. bs == 0
        # keeps the documented skip: buffers hold valid col-0 entries (capture
        # zeros, real pages, or dummy-row pad 0 — never -1) and outputs are
        # discarded. The bs == 0 skip is only reachable from unit tests: the
        # wrapper always replays with padded_bs >= 1 and synthesizes idle
        # tables for the bs == 0 idle graph.
        if self.cuda_graph_flat_page_tables and bs > 0:
            if not flat_block_tables:
                raise RuntimeError(
                    "MHAAttnBackend replay: flat per-group CUDA-graph buffers "
                    f"exist for groups "
                    f"{sorted(self.cuda_graph_flat_page_tables)} "
                    f"but flat_block_tables is missing/empty at bs={bs}; the "
                    "captured graph would read stale page tables."
                )
            # Per-group hole: a group silently absent from a non-empty dict
            # (e.g. dropped upstream) would leave its captured buffer stale.
            missing = set(self.cuda_graph_flat_page_tables) - set(flat_block_tables)
            if missing:
                raise RuntimeError(
                    "MHAAttnBackend replay: flat_block_tables at bs="
                    f"{bs} is missing captured groups {sorted(missing)} "
                    f"(delivered: {sorted(flat_block_tables)}); the captured "
                    "graph would read stale page tables for those groups."
                )

        if self.spec_num_tokens > 1 and not self.is_draft:
            base_page_table = req_to_page[req_pool_indices[:bs], : self.max_num_pages]
            self._fill_spec_metadata(
                self.cuda_graph_page_table[: bs * self.spec_num_tokens, :],
                self.cuda_graph_seq_lens[: bs * self.spec_num_tokens],
                base_page_table,
                seq_lens[:bs],
            )
        else:
            self.cuda_graph_page_table[:bs, : self.max_num_pages].copy_(
                req_to_page[req_pool_indices[:bs], : self.max_num_pages]
            )

        # Flat per-group tables: copy_ into the persistent capture buffers so
        # the graph-recorded addresses see the new pages (absolute pages, no
        # base-offset math). The single-table update above still serves the
        # radix/single-table path.
        #
        # Padding contract (bs here is the padded bs): dummy ROWS in
        # [actual_bs, bs) arrive from the wrapper padded with 0 — they replay
        # with seq_lens=1, so the captured kernel dereferences exactly their
        # col-0 entry, which must be page 0 (the zero-initialized dummy page).
        # Column tails beyond a row's pages pad with -1 — past cache_seqlens,
        # never dereferenced. The fill_(-1) below also hits dummy rows'
        # columns 1.. (giving [0, -1, ...]), which is fine: with seq_lens=1
        # nothing past col 0 is read for those rows.
        if flat_block_tables:
            # The missing-groups guard above enforced that every captured
            # group is present in this dict (the bridge also raises at the
            # source when a group's row count disagrees with the batch, so a
            # hole can only come from a future upstream change). Iterating
            # the delivered dict, a group the graph never captured raises
            # KeyError below — also loud.
            for gid, src in flat_block_tables.items():
                # Buffers exist: capture allocated them for these groups.
                buf = self.cuda_graph_flat_page_tables[gid]
                cols = src.shape[1]
                # cols >= 1 keeps the col-0 contract self-enforcing: a
                # zero-width table would leave dummy rows' col 0 unwritten.
                assert 1 <= cols <= buf.shape[1], (
                    f"flat table for group {gid!r}: {cols} cols outside"
                    f" [1, {buf.shape[1]}] (CUDA-graph buffer width)"
                )
                assert src.shape[0] >= bs, (
                    f"flat table for group {gid!r} has {src.shape[0]} rows"
                    f" < padded bs {bs}"
                )
                buf[:bs, :cols].copy_(src[:bs, :])
                if cols < buf.shape[1]:
                    buf[:bs, cols:].fill_(-1)

            # Recompute per-group write locs from the freshly-updated
            # persistent tables and copy_ into the loc buffers the captured
            # graph reads. cuda_graph_seq_lens aliases the controller's
            # seq_lens_buf, which input prep fills with the CURRENT step's
            # lens (active rows) and 1s (padding tail) BEFORE this replay
            # call, so [:bs] is current here. Padded dummy rows: seq_len 1
            # + page-0 col 0 -> loc 0 in the zero-init dummy page (M10 pad
            # contract).
            locs = self._compute_flat_decode_out_cache_locs(
                {
                    gid: self.cuda_graph_flat_page_tables[gid][:bs, :]
                    for gid in flat_block_tables
                },
                self.cuda_graph_seq_lens[:bs],
                self.page_size,
            )
            for gid, val in locs.items():
                self.cuda_graph_flat_out_cache_locs[gid][:bs].copy_(val)

        if bs in self.cuda_graph_decode_metadata:
            self.forward_decode_metadata = self.cuda_graph_decode_metadata[bs]

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        assert layer.qk_head_dim == layer.v_head_dim
        assert (k is None) == (v is None)
        has_kv = k is not None

        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        if has_kv:
            k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
            v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        # Route the KV write to this layer's group pages on the flat path
        # (M-W1). Selected ONCE here; the inner set_kv_buffer sites reuse it.
        out_cache_loc = self._select_out_cache_loc(
            layer, self.forward_decode_metadata, out_cache_loc
        )

        return self._forward_decode(
            q,
            k,
            v,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            self.forward_decode_metadata,
            save_kv_cache=save_kv_cache,
            sinks=kwargs.get("sinks"),
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        assert layer.qk_head_dim == layer.v_head_dim
        assert (k is None) == (v is None)
        has_kv = k is not None
        assert has_kv

        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        metadata = self.forward_prefill_metadata
        # Route the KV write to this layer's group pages on the flat path
        # (M-W1). Selected ONCE here; the inner set_kv_buffer sites reuse it.
        out_cache_loc = self._select_out_cache_loc(layer, metadata, out_cache_loc)
        if metadata.max_extend_prefix_len > 0:
            if self.mha_extend_mode == "ragged":
                return self._forward_extend_split(
                    q,
                    k,
                    v,
                    layer,
                    out_cache_loc,
                    token_to_kv_pool,
                    metadata,
                    save_kv_cache,
                    kwargs.get("sinks"),
                )
            else:
                return self._forward_extend(
                    q,
                    k,
                    v,
                    layer,
                    out_cache_loc,
                    token_to_kv_pool,
                    metadata,
                    save_kv_cache,
                    kwargs.get("sinks"),
                )
        return self._forward_prefill(
            q,
            k,
            v,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            metadata,
            save_kv_cache,
            kwargs.get("sinks"),
        )

    def _select_page_table(self, layer, metadata):
        """Pick this layer's page table: its group's table on the flat path,
        else the single shared table.

        TODO(radix-removal): the empty-group_id / single-table fallbacks exist to
        keep non-group-aware layers and single-group models working while radix
        coexists. Once flat is the only path and every KV layer carries a
        group_id, this collapses to `metadata.page_tables[layer.group_id]`.
        """
        if metadata.page_tables is None:
            return metadata.page_table
        group_id = getattr(layer, "group_id", "")
        tables = metadata.page_tables
        if not group_id or group_id not in tables:
            if len(tables) == 1:
                return next(iter(tables.values()))
            raise KeyError(
                f"_select_page_table: layer group_id={group_id!r} not in "
                f"flat_block_tables keys {sorted(tables)}"
            )
        return tables[group_id]

    def _select_out_cache_loc(self, layer, metadata, out_cache_loc):
        """Pick this layer's KV WRITE locations: its group's derived locs on
        the flat path, else the caller's single-stream out_cache_loc.
        Mirror of _select_page_table (M-W1: writes must land in the pages
        the layer's group reads). Same fallback ladder: None -> single
        stream; empty/unknown group_id + single group -> that group.
        TODO(radix-removal): collapses to `locs[layer.group_id]` with
        _select_page_table once flat is the only path.
        """
        locs = metadata.out_cache_locs
        if locs is None:
            return out_cache_loc
        group_id = getattr(layer, "group_id", "")
        if not group_id or group_id not in locs:
            if len(locs) == 1:
                return next(iter(locs.values()))
            raise KeyError(
                f"_select_out_cache_loc: layer group_id={group_id!r} not in "
                f"flat write locs {sorted(locs)}"
            )
        return locs[group_id]

    def select_out_cache_loc(self, layer, out_cache_loc):
        """Per-group write locations for OUT-OF-BACKEND KV writers (the
        fused RoPE+prewrite path). Prewrite only runs on decode
        (support_kv_cache_prewrite), so this reads forward_decode_metadata.
        """
        metadata = self.forward_decode_metadata
        if metadata is None or metadata.out_cache_locs is None:
            return out_cache_loc
        return self._select_out_cache_loc(layer, metadata, out_cache_loc)

    @staticmethod
    def _compute_flat_decode_out_cache_locs(page_tables, seq_lens, page_size):
        """Per-group write locations for decode: one new token per request at
        position seq_len-1. Derived from the group's own read table, so
        writes land in the pages that group reads (M-W1). The tail page is
        never a hole (SWA holes only at the window front), so the gather is
        always a real page; padded dummy rows (table row 0-filled, seq_len 1)
        resolve to loc 0 inside the zero-init dummy page 0 -- same contract
        as the single-stream out_cache_loc padding.
        """
        pos = (seq_lens - 1).to(torch.int64)
        page_idx = pos // page_size
        off = (pos % page_size).to(torch.int32)
        out = {}
        for gid, table in page_tables.items():
            pages = table.gather(1, page_idx.unsqueeze(1)).squeeze(1)
            out[gid] = pages * page_size + off
        return out

    @staticmethod
    def _compute_flat_extend_out_cache_locs(
        page_tables, seq_lens, extend_prefix_lens, page_size
    ):
        """Per-group write locations for extend/prefill: positions
        [prefix_len, seq_len) per request, flattened in request order --
        matching the q/k/v token layout (cu_extend_seq_lens order). Prefill
        runs once per request; the per-request Python loop is acceptable
        bring-up cost (TODO(flat-perf): batch via repeat_interleave).
        """
        out = {gid: [] for gid in page_tables}
        for i in range(int(seq_lens.shape[0])):
            start = int(extend_prefix_lens[i])
            end = int(seq_lens[i])
            pos = torch.arange(
                start, end, dtype=torch.int64, device=seq_lens.device
            )
            page_idx = pos // page_size
            off = (pos % page_size).to(torch.int32)
            for gid, table in page_tables.items():
                pages = table[i].gather(0, page_idx)
                out[gid].append(pages * page_size + off)
        return {
            gid: torch.cat(chunks)
            if chunks
            else torch.empty(0, dtype=torch.int32, device=seq_lens.device)
            for gid, chunks in out.items()
        }

    @staticmethod
    def _maybe_check_flat_write_locs(page_tables, out_cache_locs, page_size):
        """TOKENSPEED_FLAT_DEBUG=1: write pages must be real (non-hole) and
        inside the group's table. Costs a GPU sync -- eager path only, off
        by default. Graph-path coverage comes from the GPU write probe.
        Do NOT reuse against graph-padded batches: dummy rows legitimately
        resolve to page 0 and would trip the non-hole assert.
        """
        import os

        if os.environ.get("TOKENSPEED_FLAT_DEBUG") != "1":
            return
        for gid, locs in out_cache_locs.items():
            pages = (locs // page_size).to(torch.int32)
            table = page_tables[gid]
            assert (pages != 0).all(), (
                f"flat write loc in null page 0 for group {gid!r}"
            )
            real = table[table > 0]
            assert torch.isin(pages, real).all(), (
                f"flat write pages escape group {gid!r}'s table"
            )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHAPrefillMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        result = mha_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens=metadata.cu_extend_seq_lens,
            cu_seqlens_cpu=metadata.cu_extend_seq_lens_cpu,
            max_seqlen=metadata.max_extend_seq_len,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            solution=self.kernel_solution,
        )
        output = self._unwrap_output(result)
        output = output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
        if save_kv_cache:
            token_to_kv_pool.set_kv_buffer(
                layer,
                out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )
        return output

    def _forward_extend_split(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHAPrefillMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        chunk_result = mha_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens=metadata.cu_extend_seq_lens,
            cu_seqlens_cpu=metadata.cu_extend_seq_lens_cpu,
            max_seqlen=metadata.max_extend_seq_len,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            return_lse=True,
            solution=self.kernel_solution,
        )
        chunk_out, chunk_lse = chunk_result

        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        prefix_result = mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=metadata.cu_extend_seq_lens,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=self._select_page_table(layer, metadata),
            cache_seqlens=metadata.extend_prefix_lens,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            return_lse=True,
            max_seqlen_q=metadata.max_extend_seq_len,
            max_seqlen_k=metadata.max_extend_prefix_len,
            solution=self.kernel_solution,
        )
        prefix_out, prefix_lse = prefix_result

        output, _ = attn_merge_state(
            chunk_out.contiguous(),
            chunk_lse.contiguous(),
            prefix_out.contiguous(),
            prefix_lse.contiguous(),
        )
        if save_kv_cache:
            token_to_kv_pool.set_kv_buffer(
                layer,
                out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHAPrefillMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        if save_kv_cache:
            token_to_kv_pool.set_kv_buffer(
                layer,
                out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        result = mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=metadata.cu_extend_seq_lens,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=self._select_page_table(layer, metadata),
            cache_seqlens=metadata.seq_lens,
            is_causal=True,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            max_seqlen_q=metadata.max_extend_seq_len,
            max_seqlen_k=self.max_context_len,
            solution=self.kernel_solution,
        )
        output = self._unwrap_output(result)
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHADecodeMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        if save_kv_cache:
            token_to_kv_pool.set_kv_buffer(
                layer,
                out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        result = mha_decode_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=self._select_page_table(layer, metadata),
            cache_seqlens=metadata.seq_lens,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            max_seqlen_k=self.max_context_len,
            scheduler_metadata=metadata.scheduler_metadata,
            solution=self.kernel_solution,
        )
        output = self._unwrap_output(result)
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _get_kv_cache(self, layer: PagedAttention, token_to_kv_pool):
        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1,
            self.page_size,
            layer.tp_k_head_num,
            layer.qk_head_dim,
        )
        v_cache = token_to_kv_pool.get_value_buffer(layer.layer_id).view(
            -1,
            self.page_size,
            layer.tp_v_head_num,
            layer.v_head_dim,
        )
        return k_cache, v_cache

    def _make_spec_metadata_buffers(
        self,
        bs: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded_bs = bs * self.spec_num_tokens
        cuda_graph_page_table = torch.empty(
            (expanded_bs, self.max_num_pages),
            dtype=torch.int32,
            device=device,
        )
        cuda_graph_seq_lens = torch.empty(
            (expanded_bs,),
            dtype=torch.int32,
            device=device,
        )
        return (cuda_graph_page_table, cuda_graph_seq_lens)

    def _fill_spec_metadata(
        self,
        expanded_page_table: torch.Tensor,
        expanded_seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        bs = seq_lens.shape[0]
        spec_num_tokens = self.spec_num_tokens
        expanded_page_table = expanded_page_table.view(
            bs, spec_num_tokens, self.max_num_pages
        )
        expanded_page_table.copy_(page_table[:, None, :])
        self._fill_spec_seq_lens(expanded_seq_lens, seq_lens)

    def _fill_spec_seq_lens(
        self,
        expanded_seq_lens: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        bs = seq_lens.shape[0]
        spec_num_tokens = self.spec_num_tokens
        spec_decode_offsets = torch.arange(
            spec_num_tokens - 1,
            -1,
            -1,
            dtype=torch.int32,
            device=seq_lens.device,
        )
        torch.sub(
            seq_lens[:, None],
            spec_decode_offsets,
            out=expanded_seq_lens.view(bs, spec_num_tokens),
        )

    def _make_cu_extend_seq_lens(
        self,
        lengths: torch.Tensor,
        extend_seq_lens_cpu: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        cu_extend_seq_lens = torch.nn.functional.pad(
            torch.cumsum(lengths, dim=0, dtype=torch.int32),
            (1, 0),
        )
        cu_extend_seq_lens_cpu = [0]
        for length in extend_seq_lens_cpu:
            cu_extend_seq_lens_cpu.append(cu_extend_seq_lens_cpu[-1] + length)
        return cu_extend_seq_lens, cu_extend_seq_lens_cpu

    def _unwrap_output(self, result):
        if isinstance(result, tuple):
            return result[0]
        return result

    def _maybe_compute_scheduler_metadata(
        self, bs: int, seq_lens: torch.Tensor
    ) -> torch.Tensor | None:
        """Pre-compute FA3 decode scheduler metadata once per step.

        Returns ``None`` when the active backend does not consume pre-computed
        scheduler metadata (only FA3 on Hopper does); the kernel then falls
        back to its internal prepare_varlen_num_blocks launch.
        """
        return mha_decode_scheduler_metadata(
            batch_size=bs,
            max_seqlen_q=1,
            max_seqlen_k=self.max_context_len,
            num_heads_q=self.tp_q_head_num,
            num_heads_kv=self.tp_kv_head_num,
            headdim=self.head_dim,
            cache_seqlens=seq_lens,
            qkv_dtype=self.qkv_dtype,
            page_size=self.page_size,
            causal=True,
        )


for _backend_name in _KERNEL_SOLUTION_BY_BACKEND:
    register_backend(_backend_name, {AttentionArch.MHA}, MHAAttnBackend)
