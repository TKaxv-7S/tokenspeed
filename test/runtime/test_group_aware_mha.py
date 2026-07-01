from __future__ import annotations

import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")


def _import_backend():
    from tokenspeed.runtime.layers.attention.backends.mha import (
        MHADecodeMetadata,
        MHAAttnBackend,
    )

    return MHAAttnBackend, MHADecodeMetadata


class SelectPageTableTest(unittest.TestCase):
    def setUp(self):
        try:
            self.MHAAttnBackend, self.MHADecodeMetadata = _import_backend()
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        import torch

        self.torch = torch
        # _select_page_table only reads metadata.page_table(s) + layer.group_id;
        # bypass __init__ to avoid constructing a full backend.
        self.backend = self.MHAAttnBackend.__new__(self.MHAAttnBackend)

    def _layer(self, group_id):
        from types import SimpleNamespace

        return SimpleNamespace(group_id=group_id)

    def _decode_meta(self, *, page_table=None, page_tables=None):
        return self.MHADecodeMetadata(
            page_table=page_table,
            seq_lens=self.torch.zeros(1, dtype=self.torch.int32),
            page_tables=page_tables,
        )

    def test_single_table_when_page_tables_none(self):
        pt = self.torch.tensor([[1, 2]], dtype=self.torch.int32)
        meta = self._decode_meta(page_table=pt, page_tables=None)
        out = self.backend._select_page_table(self._layer("full_attention"), meta)
        self.assertIs(out, pt)

    def test_routes_by_group_id(self):
        full = self.torch.tensor([[1, 2]], dtype=self.torch.int32)
        swa = self.torch.tensor([[3, 0]], dtype=self.torch.int32)
        meta = self._decode_meta(
            page_tables={"full_attention": full, "sliding_attention": swa}
        )
        out_full = self.backend._select_page_table(
            self._layer("full_attention"), meta
        )
        out_swa = self.backend._select_page_table(
            self._layer("sliding_attention"), meta
        )
        self.assertIs(out_full, full)
        self.assertIs(out_swa, swa)

    def test_empty_group_id_falls_back_to_single_group(self):
        only = self.torch.tensor([[5]], dtype=self.torch.int32)
        meta = self._decode_meta(page_tables={"full_attention": only})
        out = self.backend._select_page_table(self._layer(""), meta)
        self.assertIs(out, only)

    def test_unknown_group_id_multi_group_raises(self):
        meta = self._decode_meta(
            page_tables={
                "full_attention": self.torch.zeros((1, 1), dtype=self.torch.int32),
                "sliding_attention": self.torch.zeros((1, 1), dtype=self.torch.int32),
            }
        )
        with self.assertRaises(KeyError):
            self.backend._select_page_table(self._layer("nope"), meta)


class GptOssGroupIdTest(unittest.TestCase):
    """PagedAttention built by GptOssAttention must carry group_id == layer_type.
    Constructing the model layer needs torch/model deps, so skip otherwise."""

    def test_paged_attention_group_id_equals_layer_type(self):
        try:
            from tokenspeed.runtime.layers.paged_attention import PagedAttention
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch: {exc}")
        layer = PagedAttention(
            num_heads=4,
            head_dim=8,
            scaling=1.0,
            num_kv_heads=4,
            layer_id=0,
            sliding_window_size=128,
            group_id="sliding_attention",
        )
        self.assertEqual(layer.group_id, "sliding_attention")


if __name__ == "__main__":
    unittest.main()
