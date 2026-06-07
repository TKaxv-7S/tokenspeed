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

"""NCCL symmetric-memory (symk) *zero-copy* all-reduce backend.

Targets the prefill phase on NVLink/NVSwitch fabrics (GB200 / NVL72) where
NCCL >= 2.27 can dispatch the symmetric-memory kernel
``ncclSymkDevKernel_AllReduce_RSxLDMC_AGxSTMC`` (LDMC/STMC + in-network
NVSwitch reduction) instead of the default ``RING_LL`` kernel.

Zero-copy design (replaces the earlier copy-based MVP)
-----------------------------------------------------
The earlier MVP built a *private* PyNccl communicator, allocated a symmetric
workspace via ``ncclMemAlloc`` + ``ncclCommWindowRegister``, and copied the
activation in/out with ``cudaMemcpyAsync`` on every all-reduce. That double
memcpy cancels symk's "almost no SM" advantage and never beat RING_LL at small
scale; worse, a bare ``ncclCommWindowRegister`` on a private comm does not
reliably make NCCL pick the symk kernel.

This backend instead makes the all-reduce input land *directly* in a registered
symmetric window, so the reduction is a plain in-place
``torch.distributed.all_reduce`` with **no copy**:

  * Each TP group owns one ``torch.cuda.MemPool`` backed by the NCCL allocator
    (``ProcessGroupNCCL.mem_allocator`` -> ``ncclMemAlloc``).
  * On first use the pool is collectively registered as a symmetric window via
    ``ProcessGroupNCCL.register_mem_pool(pool, symm=True)``
    (-> ``ncclCommWindowRegister(NCCL_WIN_COLL_SYMMETRIC)``). The ``symm=True``
    flag is what actually triggers the symk kernel; the default ``symm=False``
    only does a plain ``ncclCommRegister`` (UB) and stays on RING/LL.
  * Callers (``RowParallelLinear.forward``) wrap the GEMM in :meth:`pool_context`
    so its output is allocated inside that pool. The subsequent
    ``all_reduce`` then runs over a window-resident buffer and NCCL selects the
    symk kernel automatically.

:meth:`all_reduce` itself therefore does **not** do anything special -- it just
delegates to the NCCL fallback. Zero-copy symk happens transparently for
pool-resident tensors; everything else runs RING_LL.

Adaptive threshold (TRT-LLM nRanks fit, overridable via ``TS_SYMK_MIN_BYTES``)
-----------------------------------------------------------------------------
Only outputs at least as large as a threshold are allocated in the symk pool
(and thus use the symk kernel); smaller ones use the normal caching allocator
and run RING_LL. By default the threshold follows TRT-LLM's empirical
nRanks-adaptive fit ``max(0, a * nRanks + b)`` elements (see
:data:`_SYMK_THRESHOLD_FIT_A`): small domains (e.g. TP4) keep small messages on
ring, while large domains (e.g. NVL72) clamp the threshold to 0 so every
message uses symk. ``TS_SYMK_MIN_BYTES`` overrides this with an absolute byte
lower bound (cf. TRT-LLM ``TLLM_NCCL_MIN_REGISTRATION``) -- the experiment knob:
  * set it very large  -> effectively all-ring baseline,
  * set it to 0        -> push every eligible message onto symk,
  * sweep it           -> find the size where symk starts to win over RING_LL.

SPMD consistency contract
-------------------------
``register_mem_pool`` is collective and would hang if only some ranks reached
it. Every branch decision here (enable flag, platform, ``NCCL_*`` env, world
size, dtype, byte size vs threshold/capacity) depends only on quantities that
are identical across ranks under tensor-parallel SPMD execution, so all ranks
enter :meth:`pool_context`, register, and all-reduce together. Per-rank layout
(e.g. contiguity) is never used for branching.
"""

import os
from contextlib import contextmanager

import torch
import torch.distributed

from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.comm_backend.base import CommBackend, Group
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


def _env_enabled() -> bool:
    return os.environ.get("TOKENSPEED_ENABLE_SYMK_ALLREDUCE", "0") == "1"


def _env_max_tokens() -> int:
    try:
        return int(os.environ.get("TOKENSPEED_SYMK_MAX_TOKENS", "8192"))
    except (TypeError, ValueError):
        return 8192


def _env_min_bytes() -> int | None:
    """Explicit byte-threshold override for routing an all-reduce through symk.

    Returns ``None`` when ``TS_SYMK_MIN_BYTES`` is unset -- the backend then
    falls back to TRT-LLM's nRanks-adaptive threshold (see
    :func:`SymkAllReduceBackend._threshold_bytes`). When set, the value is an
    absolute lower bound in bytes that overrides the adaptive formula,
    mirroring TRT-LLM's ``TLLM_NCCL_MIN_REGISTRATION`` escape hatch. Outputs
    below the resulting threshold stay on the normal allocator -> RING_LL.
    """
    raw = os.environ.get("TS_SYMK_MIN_BYTES")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


# TRT-LLM's empirical linear fit (GB200, reverse-engineered from
# allreduceOp.cpp runNCCLAllReduceSymmetric) for the symmetric-memory
# registration threshold: messages smaller than ``max(0, a * nRanks + b)``
# elements are not worth routing through symk -- at small scale the
# window-registration overhead outweighs the in-network (NVLS multicast)
# reduction benefit, so they stay on RING_LL. The fit decreases with nRanks:
# more ranks -> lower threshold -> symk kicks in earlier; at large domains
# (e.g. NVL72) it clamps to 0 so every message uses symk. Override the whole
# formula with TS_SYMK_MIN_BYTES.
_SYMK_THRESHOLD_FIT_A = -4986.43478503
_SYMK_THRESHOLD_FIT_B = 156716.52177552


class SymkAllReduceBackend(CommBackend):
    """Zero-copy NCCL symmetric-memory all-reduce backend for the prefill phase.

    Keyed per-group: each group owns a registered NCCL-allocator ``MemPool``.
    The backend does not perform the all-reduce itself -- it exposes
    :meth:`pool_context` so the upstream GEMM output lands in the symmetric
    window, and :meth:`all_reduce` simply delegates to *fallback* (plain NCCL,
    which auto-selects the symk kernel for window-resident buffers).
    """

    def __init__(self, fallback: CommBackend):
        self._fallback = fallback
        # group_tuple -> {pool, nccl_backend, device_group, registered,
        #                 dtype, hidden_dim, capacity}
        self._resources: dict = {}
        # groups that failed gating/setup -- never retried (avoids repeated
        # collective attempts that could diverge across ranks).
        self._failed: set = set()
        self._enabled = _env_enabled()
        self._max_token_num = _env_max_tokens()
        self._min_bytes_override = _env_min_bytes()
        if self._enabled:
            logger.info(
                "SymkAllReduceBackend (zero-copy) enabled: max_token_num=%d, "
                "min_bytes_override=%s (None=nRanks-adaptive), "
                "NCCL_CUMEM_ENABLE=%s, NCCL_NVLS_ENABLE=%s",
                self._max_token_num,
                self._min_bytes_override,
                os.environ.get("NCCL_CUMEM_ENABLE"),
                os.environ.get("NCCL_NVLS_ENABLE"),
            )

    # ------------------------------------------------------------------
    # Gating (all conditions are rank-consistent under SPMD)
    # ------------------------------------------------------------------

    def _gating_reasons(self, group: Group) -> list[str]:
        reasons = []
        if not self._enabled:
            reasons.append("feature disabled (TOKENSPEED_ENABLE_SYMK_ALLREDUCE != 1)")
        if not current_platform().is_nvidia:
            reasons.append("non-NVIDIA platform")
        if len(group) == 1:
            reasons.append("world_size == 1")
        if os.environ.get("NCCL_CUMEM_ENABLE") != "1":
            reasons.append(
                "NCCL_CUMEM_ENABLE=%r (need '1')" % os.environ.get("NCCL_CUMEM_ENABLE")
            )
        if os.environ.get("NCCL_NVLS_ENABLE") != "1":
            reasons.append(
                "NCCL_NVLS_ENABLE=%r (need '1')" % os.environ.get("NCCL_NVLS_ENABLE")
            )
        return reasons

    def should_handle(self, group: Group) -> bool:
        """Whether AutoBackend should route ``all_reduce`` of *group* here.

        True when the feature is enabled and the group has not permanently
        failed gating. (Routing here vs. NCCL is cosmetic for the zero-copy
        path -- both run a plain NCCL all-reduce -- but it keeps the public
        backend surface consistent with the previous implementation.)
        """
        return self._enabled and group not in self._failed

    def has_symk(self, group: Group) -> bool:
        res = self._resources.get(group)
        return res is not None and res["registered"]

    def _threshold_bytes(self, group: Group, dtype: torch.dtype) -> int:
        """Minimum all-reduce input size (bytes) worth routing through symk.

        An explicit ``TS_SYMK_MIN_BYTES`` override wins; otherwise use TRT-LLM's
        nRanks-adaptive linear fit ``max(0, a * nRanks + b)`` (in elements,
        scaled by ``dtype`` itemsize). ``nRanks`` is the group's world size,
        which is identical across ranks, so the threshold is SPMD-consistent.
        """
        if self._min_bytes_override is not None:
            return self._min_bytes_override
        n_ranks = len(group)
        elems = _SYMK_THRESHOLD_FIT_A * n_ranks + _SYMK_THRESHOLD_FIT_B
        if elems < 0:
            elems = 0.0
        return int(elems * dtype.itemsize)

    # ------------------------------------------------------------------
    # Pool lifecycle (lazy)
    # ------------------------------------------------------------------

    def _ensure_pool(
        self, group: Group, dtype: torch.dtype, hidden_dim: int
    ) -> dict | None:
        """Lazily create the (not-yet-registered) NCCL-allocator MemPool.

        Returns the resource dict, or None when symk is gated off / setup
        fails (the group is then marked failed so we never retry the
        collective registration with a diverging decision across ranks).
        """
        res = self._resources.get(group)
        if res is not None:
            return res
        if group in self._failed:
            return None

        reasons = self._gating_reasons(group)
        if reasons:
            logger.warning(
                "symk all-reduce disabled for group %s: %s", group, "; ".join(reasons)
            )
            self._failed.add(group)
            return None

        try:
            from tokenspeed.runtime.distributed.process_group_manager import (
                process_group_manager as pg_manager,
            )

            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            pg = pg_manager.get_process_group("nccl", group)

            # Obtain the ProcessGroupNCCL backend and its NCCL allocator.
            nccl_backend = pg._get_backend(device)
            mem_allocator = getattr(nccl_backend, "mem_allocator", None)
            if mem_allocator is None or not hasattr(nccl_backend, "register_mem_pool"):
                raise RuntimeError(
                    "ProcessGroupNCCL lacks mem_allocator/register_mem_pool "
                    "(need PyTorch >= 2.6 with NCCL symmetric-memory support)"
                )
            nccl_alloc = mem_allocator() if callable(mem_allocator) else mem_allocator

            pool = torch.cuda.MemPool(nccl_alloc)
            capacity = self._max_token_num * hidden_dim * dtype.itemsize

            res = {
                "pool": pool,
                "nccl_backend": nccl_backend,
                "device_group": pg,
                "registered": False,
                "dtype": dtype,
                "hidden_dim": hidden_dim,
                "capacity": capacity,
            }
            self._resources[group] = res
            return res
        except Exception as exc:
            logger.warning(
                "symk pool setup failed for group %s: %r -- falling back to "
                "default NCCL (RING_LL) all-reduce",
                group,
                exc,
            )
            self._failed.add(group)
            return None

    @contextmanager
    def pool_context(
        self, group: Group, nbytes: int, dtype: torch.dtype, hidden_dim: int
    ):
        """Allocate tensors created in this context inside the registered symk pool.

        Wrap the GEMM that produces an all-reduce input with this context so the
        output lands in the symmetric window (-> zero-copy symk all-reduce).

        Yields a no-op (normal allocator -> RING_LL) when symk is disabled,
        gated off, the message is below ``TS_SYMK_MIN_BYTES``, exceeds the
        symmetric workspace capacity, or any setup step fails. All of these
        conditions are rank-consistent under SPMD.
        """
        # Fast no-op paths -- all rank-consistent.
        if not self._enabled or group in self._failed:
            yield
            return
        # Never touch the symk pool while a CUDA graph is being captured:
        # register_mem_pool is a NCCL collective + window registration that
        # invalidates stream capture (cudaErrorStreamCaptureInvalidated), and
        # decode (which runs under CUDA graphs) uses the fused all-reduce path
        # anyway -- only the eager prefill all-reduce needs symk.
        # is_current_stream_capturing() is SPMD-consistent (all ranks capture
        # the same graphs), so every rank takes this no-op branch together.
        if torch.cuda.is_current_stream_capturing():
            yield
            return
        if nbytes < self._threshold_bytes(group, dtype):
            yield
            return

        res = self._ensure_pool(group, dtype, hidden_dim)
        if res is None or dtype != res["dtype"] or nbytes > res["capacity"]:
            yield
            return

        pool = res["pool"]
        # Enter the pool allocator and grow it to capacity. Only *setup* errors
        # fall back to RING_LL; the caller's GEMM (run at ``yield``) must NOT
        # have its errors swallowed, otherwise the caller would read an
        # unassigned output (UnboundLocalError).
        pool_cm = None
        try:
            pool_cm = torch.cuda.use_mem_pool(pool)
            pool_cm.__enter__()
            if not res["registered"]:
                # Grow the pool to full capacity once so every later (smaller)
                # shape is carved from the registered segment; the window then
                # covers the max prefill all-reduce size.
                pad = torch.empty(
                    res["capacity"], dtype=torch.uint8, device=pool_device(res)
                )
                del pad
        except Exception as exc:
            if pool_cm is not None:
                try:
                    pool_cm.__exit__(None, None, None)
                except Exception:
                    pass
            logger.warning(
                "symk pool setup failed for group %s: %r -- using RING_LL",
                group,
                exc,
            )
            self._failed.add(group)
            yield
            return

        # Setup OK: run the caller's GEMM inside the pool. Body errors propagate
        # (the finally still restores the default allocator).
        try:
            yield
        finally:
            pool_cm.__exit__(None, None, None)

        # Register the window once, after the first allocation, outside the
        # use_mem_pool block (collective: every rank reaches this together).
        if not res["registered"]:
            try:
                res["nccl_backend"].register_mem_pool(pool, symm=True)
                res["registered"] = True
                logger.info(
                    "symk pool registered (symm=True) for group %s: "
                    "hidden_dim=%d, dtype=%s, capacity=%d bytes (max_token_num=%d)",
                    group,
                    res["hidden_dim"],
                    res["dtype"],
                    res["capacity"],
                    self._max_token_num,
                )
            except Exception as exc:
                logger.warning(
                    "symk register_mem_pool(symm=True) failed for group %s: %r "
                    "-- using RING_LL",
                    group,
                    exc,
                )
                self._failed.add(group)

    # ------------------------------------------------------------------
    # CommBackend interface
    # ------------------------------------------------------------------

    def all_reduce(self, tensor: torch.Tensor, group: Group, op=None) -> torch.Tensor:
        """Plain in-place NCCL all-reduce.

        Zero-copy symk happens transparently when ``tensor`` was allocated in
        the registered pool via :meth:`pool_context`; NCCL selects the symk
        kernel for window-resident buffers and RING_LL otherwise. No copy and
        no private communicator are involved.
        """
        return self._fallback.all_reduce(tensor, group, op=op)

    # ---- Delegate everything else to fallback ----

    def all_gather(
        self, tensor: torch.Tensor, group: Group, dim: int = 0
    ) -> torch.Tensor:
        return self._fallback.all_gather(tensor, group, dim)

    def all_gather_into_tensor(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None:
        return self._fallback.all_gather_into_tensor(output, input, group)

    def reduce_scatter(self, tensor: torch.Tensor, group: Group) -> torch.Tensor:
        return self._fallback.reduce_scatter(tensor, group)

    def token_all_gather(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        raise NotImplementedError("Use AutoBackend for token-aware ops")

    def token_reduce_scatter(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        raise NotImplementedError("Use AutoBackend for token-aware ops")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Drop pool references (best-effort, on exit).

        ``register_mem_pool`` windows are torn down with the communicator at
        process shutdown; we just release our references here.
        """
        self._resources.clear()


def pool_device(res: dict) -> torch.device:
    """Device of the group's symk pool (current CUDA device)."""
    return torch.device(f"cuda:{torch.cuda.current_device()}")
