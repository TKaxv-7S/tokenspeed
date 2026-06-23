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

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.moe.triton.mxfp4 import (
    _sum_topk_rows,
    triton_mxfp4_moe_apply,
    triton_mxfp4_moe_process_weights,
)
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_amd,
    reason="dynamic MXFP4 activation MoE epilogue fusion is AMD-only",
)

HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 128
NUM_EXPERTS = 8
TOPK = 2
TOKENS = 4
MXFP4_BLOCK = 32


class _DynamicMxfp4Config:
    use_dynamic_mxfp4_activations = True


class _Weights(torch.nn.Module):
    pass


def _make_weights(
    device: str,
    *,
    swiglu: bool = True,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
) -> _Weights:
    generator = torch.Generator(device=device).manual_seed(20260623)
    weights = _Weights()
    weights.top_k = TOPK
    weights.num_experts = NUM_EXPERTS
    weights.num_local_experts = NUM_EXPERTS
    weights.ep_size = 1
    weights.ep_rank = 0
    weights.quant_config = _DynamicMxfp4Config()
    if swiglu:
        weights.swiglu_arg = SimpleNamespace(alpha=swiglu_alpha, limit=None)
        weights.swiglu_beta = swiglu_beta
    weights.register_parameter(
        "w13_weight",
        torch.nn.Parameter(
            torch.randint(
                0,
                256,
                (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 2),
                device=device,
                dtype=torch.uint8,
                generator=generator,
            ),
            requires_grad=False,
        ),
    )
    weights.register_parameter(
        "w2_weight",
        torch.nn.Parameter(
            torch.randint(
                0,
                256,
                (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE // 2),
                device=device,
                dtype=torch.uint8,
                generator=generator,
            ),
            requires_grad=False,
        ),
    )
    weights.register_parameter(
        "w13_weight_scale",
        torch.nn.Parameter(
            torch.full(
                (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // MXFP4_BLOCK),
                127,
                device=device,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        ),
    )
    weights.register_parameter(
        "w2_weight_scale",
        torch.nn.Parameter(
            torch.full(
                (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE // MXFP4_BLOCK),
                127,
                device=device,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        ),
    )
    weights.register_parameter(
        "w13_weight_bias",
        torch.nn.Parameter(
            torch.zeros(
                NUM_EXPERTS,
                2 * INTERMEDIATE_SIZE,
                device=device,
                dtype=torch.float32,
            ),
            requires_grad=False,
        ),
    )
    weights.register_parameter(
        "w2_weight_bias",
        torch.nn.Parameter(
            torch.zeros(NUM_EXPERTS, HIDDEN_SIZE, device=device, dtype=torch.float32),
            requires_grad=False,
        ),
    )
    return weights


def test_dynamic_mxfp4_swiglu_quant_epilogue_matches_unfused(device: str) -> None:
    generator = torch.Generator(device=device).manual_seed(3001)
    hidden = (
        torch.randn(
            (TOKENS, HIDDEN_SIZE),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        / 20
    ).contiguous()
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 4]],
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [0.55, 0.45], [0.8, 0.2]],
        device=device,
        dtype=torch.float32,
    )
    router_logits = torch.empty((TOKENS, NUM_EXPERTS), device=device, dtype=torch.float32)

    fused_weights = _make_weights(device)
    reference_weights = _make_weights(device)
    triton_mxfp4_moe_process_weights({}, fused_weights)
    triton_mxfp4_moe_process_weights({}, reference_weights)
    reference_weights._mxfp4_intermediate_size = None

    fused = triton_mxfp4_moe_apply(
        {},
        hidden,
        fused_weights,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    reference = triton_mxfp4_moe_apply(
        {},
        hidden,
        reference_weights,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(fused, reference, rtol=1e-2, atol=1e-2)


def test_dynamic_mxfp4_standard_swiglu_matches_silu_gate_up(device: str) -> None:
    generator = torch.Generator(device=device).manual_seed(3002)
    hidden = (
        torch.randn(
            (TOKENS, HIDDEN_SIZE),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        / 20
    ).contiguous()
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 4]],
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [0.55, 0.45], [0.8, 0.2]],
        device=device,
        dtype=torch.float32,
    )
    router_logits = torch.empty((TOKENS, NUM_EXPERTS), device=device, dtype=torch.float32)

    fused_weights = _make_weights(device, swiglu=True, swiglu_beta=None)
    reference_weights = _make_weights(device, swiglu=False)
    triton_mxfp4_moe_process_weights({}, fused_weights)
    triton_mxfp4_moe_process_weights({}, reference_weights)
    fused_weights._mxfp4_intermediate_size = None

    fused = triton_mxfp4_moe_apply(
        {},
        hidden,
        fused_weights,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    reference = triton_mxfp4_moe_apply(
        {},
        hidden,
        reference_weights,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    torch.cuda.synchronize()

    # Fused activation consumes FP32 accumulators, while the unfused reference
    # rounds GEMM1 to BF16 before SiLU. This still catches gate/up layout errors.
    torch.testing.assert_close(fused, reference, rtol=5e-1, atol=20)


def test_topk_sum_reduce_matches_torch(device: str) -> None:
    generator = torch.Generator(device=device).manual_seed(20260624)
    output = torch.randn(
        (TOKENS * TOPK, HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )

    reduced = _sum_topk_rows(output, TOKENS, TOPK)
    expected = output.view(TOKENS, TOPK, HIDDEN_SIZE).sum(dim=1)
    torch.cuda.synchronize()

    torch.testing.assert_close(reduced, expected, rtol=1e-2, atol=1e-2)
