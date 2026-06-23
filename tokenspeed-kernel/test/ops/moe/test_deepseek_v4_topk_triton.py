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

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.thirdparty.triton import deepseek_v4_softplus_topk


def _reference(
    logits: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = F.softplus(logits.float()).sqrt()
    choice_scores = scores
    if correction_bias is not None:
        choice_scores = choice_scores + correction_bias.unsqueeze(0)
    topk_ids = torch.topk(choice_scores, k=topk, dim=-1, sorted=True)[1]
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


@pytest.mark.parametrize("num_tokens", [0, 1, 2, 8, 16])
@pytest.mark.parametrize("num_experts", [64, 384])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("renormalize", [False, True])
def test_deepseek_v4_softplus_topk_matches_reference(
    device: str,
    num_tokens: int,
    num_experts: int,
    dtype: torch.dtype,
    renormalize: bool,
) -> None:
    torch.manual_seed(5)
    base = torch.linspace(-2.5, 2.5, num_experts, device=device, dtype=torch.float32)
    row_offsets = torch.arange(num_tokens, device=device, dtype=torch.float32)[:, None]
    logits = base.unsqueeze(0) + row_offsets * 0.01
    logits = logits + torch.randn_like(logits) * 0.01
    logits = logits.to(dtype)
    correction_bias = torch.linspace(
        0.25,
        -0.25,
        num_experts,
        device=device,
        dtype=torch.float32,
    )

    topk_weights, topk_ids = deepseek_v4_softplus_topk(
        logits,
        correction_bias=correction_bias,
        topk=8,
        renormalize=renormalize,
    )
    torch.cuda.synchronize()

    expected_weights, expected_ids = _reference(
        logits,
        correction_bias,
        topk=8,
        renormalize=renormalize,
    )
    assert torch.equal(topk_ids, expected_ids)
    torch.testing.assert_close(topk_weights, expected_weights, rtol=2e-4, atol=2e-4)


def test_deepseek_v4_softplus_topk_supports_unbiased_choice(device: str) -> None:
    torch.manual_seed(11)
    logits = torch.randn(4, 384, device=device, dtype=torch.float32)

    topk_weights, topk_ids = deepseek_v4_softplus_topk(
        logits,
        correction_bias=None,
        topk=8,
        renormalize=True,
    )
    torch.cuda.synchronize()

    expected_weights, expected_ids = _reference(
        logits,
        correction_bias=None,
        topk=8,
        renormalize=True,
    )
    assert torch.equal(topk_ids, expected_ids)
    torch.testing.assert_close(topk_weights, expected_weights, rtol=2e-4, atol=2e-4)
