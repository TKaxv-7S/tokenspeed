from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

_CONFIGS_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "configs"
)


def _load(mod_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(
        mod_name, _CONFIGS_DIR / file_name
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: on py3.9 @dataclass + `from __future__ import
    # annotations` resolves field types via sys.modules[cls.__module__].
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_pcs = _load("paged_cache_spec_under_test", "paged_cache_spec.py")
group_specs_from_layer_types = _pcs.group_specs_from_layer_types
PagedCacheGroupSpec = _pcs.PagedCacheGroupSpec


class GroupSpecsFromLayerTypesTest(unittest.TestCase):
    def test_gpt_oss_mixed_shape_yields_two_groups(self):
        # GPT-OSS 交替: full / sliding。归组函数应产两条 spec。
        layer_types = [
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ]
        specs = group_specs_from_layer_types(
            layer_types=layer_types,
            sliding_window_tokens=128,
            page_size=16,
        )
        self.assertEqual(len(specs), 2)
        by_id = {s.group_id: s for s in specs}
        self.assertIn("full_attention", by_id)
        self.assertIn("sliding_attention", by_id)

        full = by_id["full_attention"]
        self.assertEqual(full.retention, "full_history")
        self.assertIsNone(full.sliding_window_tokens)
        self.assertEqual(full.rows_per_page, 16)
        self.assertEqual(full.entry_stride_tokens, 1)
        self.assertEqual(full.family, "history")

        swa = by_id["sliding_attention"]
        self.assertEqual(swa.retention, "sliding_window")
        self.assertEqual(swa.sliding_window_tokens, 128)
        self.assertEqual(swa.rows_per_page, 16)
        self.assertEqual(swa.family, "history")

    def test_all_full_yields_single_group(self):
        # 纯 full-attention model (llama/qwen) must yield exactly one spec ->
        # equivalent to today's single group, no regression.
        specs = group_specs_from_layer_types(
            layer_types=["full_attention"] * 8,
            sliding_window_tokens=None,
            page_size=16,
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].group_id, "full_attention")
        self.assertEqual(specs[0].retention, "full_history")
        self.assertIsNone(specs[0].sliding_window_tokens)

    def test_group_order_is_first_appearance(self):
        # sliding appears first -> sliding group is ordered first. Order is
        # deterministic and stable across runs.
        specs = group_specs_from_layer_types(
            layer_types=["sliding_attention", "full_attention", "full_attention"],
            sliding_window_tokens=64,
            page_size=8,
        )
        self.assertEqual(
            [s.group_id for s in specs],
            ["sliding_attention", "full_attention"],
        )

    def test_unknown_layer_type_raises(self):
        with self.assertRaises(ValueError):
            group_specs_from_layer_types(
                layer_types=["full_attention", "banana_attention"],
                sliding_window_tokens=None,
                page_size=16,
            )

    def test_sliding_without_window_raises(self):
        with self.assertRaises(ValueError):
            group_specs_from_layer_types(
                layer_types=["sliding_attention"],
                sliding_window_tokens=None,
                page_size=16,
            )

    def test_sliding_with_nonpositive_window_raises(self):
        with self.assertRaises(ValueError):
            group_specs_from_layer_types(
                layer_types=["sliding_attention"],
                sliding_window_tokens=0,
                page_size=16,
            )


class PoolToPagedCacheGroupsIntegrationTest(unittest.TestCase):
    """Integration: specs a pool publishes convert to a multi-group scheduler
    config via pool_to_paged_cache_groups. Requires torch + the compiled
    tokenspeed_scheduler C++ extension, so it SKIPS where those are absent
    (e.g. local dev boxes) and runs for real in the GPU container."""

    def _import_converter(self):
        try:
            from tokenspeed.runtime.engine.scheduler_utils import (
                pool_to_paged_cache_groups,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(
                f"pool_to_paged_cache_groups unavailable (needs torch + "
                f"tokenspeed_scheduler ext): {exc}"
            )
        return pool_to_paged_cache_groups

    def test_two_group_specs_convert_to_two_scheduler_groups(self):
        from types import SimpleNamespace

        pool_to_paged_cache_groups = self._import_converter()

        specs = group_specs_from_layer_types(
            layer_types=["full_attention", "sliding_attention"],
            sliding_window_tokens=128,
            page_size=16,
        )
        # Duck-typed stand-in: only the two attributes the converter reads.
        fake_pool = SimpleNamespace(
            paged_cache_group_specs=specs,
            paged_cache_group_page_counts={s.group_id: 1024 for s in specs},
        )

        groups = pool_to_paged_cache_groups(fake_pool)

        self.assertEqual(len(groups), 2)
        group_ids = {g.group_id for g in groups}
        self.assertEqual(group_ids, {"full_attention", "sliding_attention"})

    def test_empty_specs_convert_to_no_groups(self):
        pool_to_paged_cache_groups = self._import_converter()

        from types import SimpleNamespace

        fake_pool = SimpleNamespace(
            paged_cache_group_specs=(),
            paged_cache_group_page_counts={},
        )
        self.assertEqual(pool_to_paged_cache_groups(fake_pool), [])


if __name__ == "__main__":
    unittest.main()
