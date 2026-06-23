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

import torch

from tokenspeed.runtime.layers.moe.topk import (
    BackendRoutingConfig,
    TopK,
    TopKOutputFormat,
)


def _backend_routing_config() -> BackendRoutingConfig:
    return BackendRoutingConfig(
        kernel_feature="backend_routing:deepseek_v4",
        routing_method_type=1,
        top_k=2,
        renormalize=True,
        score_function="sqrtsoftplus",
        choice_score_function="sqrtsoftplus_plus_correction_bias",
    )


def test_router_logits_output_preserves_raw_inputs() -> None:
    backend_routing_config = _backend_routing_config()
    topk = TopK(
        top_k=2,
        output_format=TopKOutputFormat.ROUTER_LOGITS,
        backend_routing_config=backend_routing_config,
    )
    hidden_states = torch.randn(3, 4)
    router_logits = torch.randn(3, 8)

    output = topk(hidden_states, router_logits)

    assert output.format.is_router_logits()
    assert output.hidden_states is hidden_states
    assert output.router_logits is router_logits
    assert output.backend_routing_config is backend_routing_config


def test_bypassed_output_still_preserves_raw_inputs() -> None:
    topk = TopK(top_k=2, output_format=TopKOutputFormat.BYPASSED)
    hidden_states = torch.randn(3, 4)
    router_logits = torch.randn(3, 8)

    output = topk(hidden_states, router_logits)

    assert output.format.is_bypassed()
    assert output.hidden_states is hidden_states
    assert output.router_logits is router_logits


def test_standard_output_still_materializes_topk() -> None:
    topk = TopK(top_k=2, output_format=TopKOutputFormat.STANDARD)
    hidden_states = torch.randn(3, 4)
    router_logits = torch.randn(3, 8)

    output = topk(hidden_states, router_logits)

    assert output.format.is_standard()
    assert output.router_logits is router_logits
    assert output.topk_weights.shape == (3, 2)
    assert output.topk_ids.shape == (3, 2)
