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

"""NCCL symmetric-memory (symk) all-reduce backend.

Targets the prefill phase on NVLink/NVSwitch fabrics (GB200 / NVL72) where
NCCL >= 2.27 can dispatch the symmetric-memory kernel
``ncclSymkDevKernel_AllReduce_RSxLDMC_AGxSTMC`` (LDMC/STMC + in-network
NVSwitch reduction) instead of the default ``RING_LL`` kernel.

Design (copy-based symmetric workspace, MVP):
  * Each TP group gets one fixed-size symmetric workspace, allocated via
    ``ncclMemAlloc`` and registered as a symmetric window via
    ``ncclCommWindowRegister`` (a collective call done once).
  * On every accelerated ``all_reduce`` the activation is copied into the
    workspace (``cudaMemcpyAsync``), reduced in place (NCCL detects the
    symmetric window and selects the symk kernel), then copied back -- all on
    the caller's current compute stream.

Group configuration is *lazy*: the backend is created without model info, so
the workspace is sized and registered on the first ``all_reduce`` for a group
using ``tensor.shape[-1]`` as ``hidden_dim``. Because tensor parallelism is
SPMD, every rank reaches the first all-reduce together with an identical
hidden size, making the collective registration safe.

Any gating failure (non-NVIDIA, NCCL < 2.27, ``NCCL_CUMEM_ENABLE`` /
``NCCL_NVLS_ENABLE`` unset, world_size == 1, capacity exceeded, or any
exception) silently delegates to the provided fallback backend, preserving the
default behavior.

SPMD consistency contract
-------------------------
The symk path runs on a *private* NCCL communicator while the fallback runs on
the default ``torch.distributed`` communicator. If, for a single logical
all-reduce, some ranks took the symk branch and others the fallback branch,
the two communicators would deadlock. To prevent this, every branch decision
is based only on quantities that are identical across ranks under tensor-
parallel SPMD execution (feature flag, platform, ``NCCL_*`` env, world size,
reduce op, dtype, and ``numel*itemsize`` vs capacity). Per-rank layout details
(e.g. contiguity) are intentionally NOT used for branching -- a non-contiguous
input is densified inside the symk path instead of falling back.
"""

import ctypes
import os

import torch
import torch.distributed

from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.comm_backend.base import CommBackend, Group


def _env_enabled() -> bool:
    return os.environ.get("TOKENSPEED_ENABLE_SYMK_ALLREDUCE", "0") == "1"


def _env_max_tokens() -> int:
    try:
        return int(os.environ.get("TOKENSPEED_SYMK_MAX_TOKENS", "8192"))
    except (TypeError, ValueError):
        return 8192


class SymkAllReduceBackend(CommBackend):
    """Backend using NCCL symmetric-memory all-reduce for the prefill phase.

    Keyed per-group: each group owns a private ``PyNcclCommunicator`` plus a
    registered symmetric workspace. Only ``all_reduce`` (SUM) is accelerated;
    every other op delegates to *fallback*.
    """

    def __init__(self, fallback: CommBackend):
        self._fallback = fallback
        # group_tuple -> {pynccl, ws_ptr, win, capacity, hidden_dim, dtype, world_size}
        self._resources: dict = {}
        # groups that failed gating/configure -- never retried (avoids
        # repeated collective attempts that could diverge across ranks).
        self._failed: set = set()
        self._enabled = _env_enabled()
        self._max_token_num = _env_max_tokens()

    def _load_comm(self) -> bool:
        return current_platform().is_nvidia

    # ------------------------------------------------------------------
    # Group configuration (lazy)
    # ------------------------------------------------------------------

    def configure_group(
        self,
        rank: int,
        group: Group,
        hidden_dim: int,
        dtype: torch.dtype,
    ) -> bool:
        """Allocate + register the symmetric workspace for *group*.

        Returns True on success. Idempotent per group. All gating failures are
        recorded so the (collective) setup is attempted exactly once per group.
        """
        if group in self._resources:
            return True
        if group in self._failed:
            return False

        # ---- gating ----
        # Every condition here is identical across ranks under SPMD, so all
        # ranks make the same configure / no-configure decision together (the
        # registration below is collective and would hang otherwise).
        if (
            not self._enabled
            or not self._load_comm()
            or len(group) == 1
            or os.environ.get("NCCL_CUMEM_ENABLE") != "1"
            or os.environ.get("NCCL_NVLS_ENABLE") != "1"
        ):
            self._failed.add(group)
            return False

        pynccl = None
        ws_ptr = None
        try:
            from tokenspeed.runtime.distributed.device_communicators.pynccl import (
                PyNcclCommunicator,
            )
            from tokenspeed.runtime.distributed.process_group_manager import (
                process_group_manager as pg_manager,
            )

            gloo_group = pg_manager.get_process_group("gloo", group)
            pynccl = PyNcclCommunicator(
                group=gloo_group,
                device=torch.device(f"cuda:{torch.cuda.current_device()}"),
            )

            if not pynccl.ensure_symk():
                self._failed.add(group)
                return False

            capacity = self._max_token_num * hidden_dim * dtype.itemsize
            ws_ptr = pynccl.mem_alloc(capacity)
            win = pynccl.register_symmetric_window(ws_ptr, capacity)

            self._resources[group] = {
                "pynccl": pynccl,
                "ws_ptr": ws_ptr,
                "win": win,
                "capacity": capacity,
                "hidden_dim": hidden_dim,
                "dtype": dtype,
                "world_size": len(group),
            }
            return True
        except Exception:
            # Best-effort cleanup of a partially-created workspace.
            try:
                if pynccl is not None and ws_ptr is not None:
                    pynccl.mem_free(ws_ptr)
            except Exception:
                pass
            self._failed.add(group)
            return False

    def has_symk(self, group: Group) -> bool:
        return group in self._resources

    def should_handle(self, group: Group) -> bool:
        """Whether AutoBackend should route all_reduce of *group* here.

        True when the feature is enabled and the group has not permanently
        failed gating. Returns True on the very first call (before lazy
        configuration) so the workspace can be set up; once a group is marked
        failed, returns False so AutoBackend goes straight to NCCL.
        """
        return self._enabled and group not in self._failed

    # ------------------------------------------------------------------
    # CommBackend interface
    # ------------------------------------------------------------------

    def all_reduce(self, tensor: torch.Tensor, group: Group, op=None) -> torch.Tensor:
        if op is None:
            op = torch.distributed.ReduceOp.SUM

        res = self._resources.get(group)

        # Lazy first-touch configuration: size the workspace from the very
        # first tensor's hidden dim. Safe because TP all-reduce is SPMD.
        if (
            res is None
            and self._enabled
            and group not in self._failed
            and tensor.dim() >= 1
            and op == torch.distributed.ReduceOp.SUM
        ):
            local_rank = group.index(torch.distributed.get_rank())
            self.configure_group(
                rank=local_rank,
                group=group,
                hidden_dim=tensor.shape[-1],
                dtype=tensor.dtype,
            )
            res = self._resources.get(group)

        # Branch only on rank-consistent quantities (op, dtype, byte size vs
        # capacity). Contiguity is deliberately excluded -- a non-contiguous
        # tensor is densified inside ``_symk_allreduce`` so every rank stays on
        # the symk communicator (see the SPMD consistency contract above).
        if (
            res is not None
            and op == torch.distributed.ReduceOp.SUM
            and tensor.dtype == res["dtype"]
        ):
            nbytes = tensor.numel() * tensor.element_size()
            if nbytes <= res["capacity"]:
                result = self._symk_allreduce(tensor, res, nbytes)
                if result is not None:
                    return result

        return self._fallback.all_reduce(tensor, group, op=op)

    def _symk_allreduce(
        self, tensor: torch.Tensor, res: dict, nbytes: int
    ) -> torch.Tensor | None:
        """Copy-in -> in-place symk all-reduce -> copy-out. None on failure."""
        try:
            from tokenspeed_kernel.thirdparty.cuda.cuda_ipc import cudart

            pynccl = res["pynccl"]
            ws_ptr = res["ws_ptr"]
            stream = torch.cuda.current_stream().cuda_stream

            # cudaMemcpy needs a dense source. Densify per-rank if necessary --
            # done unconditionally (not as a fallback) so a non-contiguous
            # input never diverts this rank off the symk communicator.
            src = tensor if tensor.is_contiguous() else tensor.contiguous()

            # copy activation into the registered symmetric workspace
            cudart.cudaMemcpyAsync(
                ctypes.c_void_p(ws_ptr),
                ctypes.c_void_p(src.data_ptr()),
                nbytes,
                ctypes.c_void_p(stream),
            )
            # in-place all-reduce on the symmetric buffer (selects symk kernel)
            pynccl.symk_all_reduce(ws_ptr, src.numel(), src.dtype, stream)
            # copy the reduced result back
            if src is tensor:
                cudart.cudaMemcpyAsync(
                    ctypes.c_void_p(tensor.data_ptr()),
                    ctypes.c_void_p(ws_ptr),
                    nbytes,
                    ctypes.c_void_p(stream),
                )
            else:
                # restore the reduced data into the dense temporary, then write
                # back through a layout-aware copy on the same stream.
                cudart.cudaMemcpyAsync(
                    ctypes.c_void_p(src.data_ptr()),
                    ctypes.c_void_p(ws_ptr),
                    nbytes,
                    ctypes.c_void_p(stream),
                )
                tensor.copy_(src)
            return tensor
        except Exception:
            return None

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
        """Deregister windows and free workspaces (best-effort, on exit)."""
        for res in self._resources.values():
            pynccl = res.get("pynccl")
            try:
                if pynccl is not None and res.get("win") is not None:
                    pynccl.deregister_symmetric_window(res["win"])
            except Exception:
                pass
            try:
                if pynccl is not None and res.get("ws_ptr") is not None:
                    pynccl.mem_free(res["ws_ptr"])
            except Exception:
                pass
        self._resources.clear()
