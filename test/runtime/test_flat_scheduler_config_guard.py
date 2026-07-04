"""Flat-scheduler config guard at scheduler-config assembly (R2-4).

Rule under test (configs/paged_cache_spec.validate_flat_scheduler_config,
called from engine/event_loop before the C++ Scheduler ctor): on a flat-built
(TOKENSPEED_FLAT_KVCACHE) scheduler ext,

* a backend that consumes paged-cache groups through the radix populate path
  but is not flat-group capable (uses_paged_cache_groups without
  uses_flat_cache_groups, i.e. DeepSeek V4/MLA-style) must be rejected loudly
  — the flat build compiles that path out, so the run would otherwise replay
  CUDA graphs against stale capture placeholders and produce silent garbage;
* zero published groups must be rejected with the actual knob named (spec
  decode gates MHA publication off; state-only pools like mamba publish none)
  instead of the cryptic MakeCoordinator abort inside the C++ ctor;
* a flat-capable backend with >=1 published group passes;
* a radix-built ext makes the guard a no-op regardless of backend/groups.

The validator is pure and torch-free; paged_cache_spec is loaded by file path
when the full runtime deps (torch/transformers, pulled in by the configs
package __init__) are unavailable, so this file runs on a bare interpreter.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def _load_paged_cache_spec():
    try:
        from tokenspeed.runtime.configs import paged_cache_spec

        return paged_cache_spec
    except (ImportError, ModuleNotFoundError):
        # The configs package __init__ pulls transformers-backed model
        # configs; the module itself is torch-free, so load it directly.
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        path = os.path.join(
            repo_root,
            "python",
            "tokenspeed",
            "runtime",
            "configs",
            "paged_cache_spec.py",
        )
        spec = importlib.util.spec_from_file_location("_paged_cache_spec_guard", path)
        module = importlib.util.module_from_spec(spec)
        # dataclass processing resolves cls.__module__ through sys.modules, so
        # the module must be registered before exec.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


_pcs = _load_paged_cache_spec()


class FakeV4StyleBackend:
    """DeepSeek V4/MLA shape: radix-populate consumer, not flat-capable."""

    uses_paged_cache_groups = True
    uses_flat_cache_groups = False


class FakeFlatMHABackend:
    """backends/mha.py shape: flat-group capable."""

    uses_paged_cache_groups = False
    uses_flat_cache_groups = True


class FakeV4Pool:
    pass


class FakeMHAPool:
    pass


class FakeMambaPool:
    pass


class FakeGroup:
    group_id = "full_attention"


class ValidateFlatSchedulerConfigTest(unittest.TestCase):
    def test_flat_ext_v4_style_backend_raises(self):
        # V4 pool publishes specs unconditionally, so groups are non-empty;
        # the backend flags alone must trip the guard.
        with self.assertRaises(RuntimeError) as ctx:
            _pcs.validate_flat_scheduler_config(
                flat_kvcache_ext=True,
                paged_cache_groups=[FakeGroup()],
                attn_backend=FakeV4StyleBackend(),
                kv_pool=FakeV4Pool(),
                speculative_enabled=False,
            )
        msg = str(ctx.exception)
        self.assertIn("does not support this model's cache layout", msg)
        self.assertIn("FakeV4StyleBackend", msg)
        self.assertIn("FakeV4Pool", msg)
        self.assertIn("radix-built", msg)

    def test_flat_ext_zero_groups_spec_decode_names_the_knob(self):
        # MHA publication correctly gated off under spec decode -> empty
        # groups. The error must say spec decode is the cause, not surface
        # the C++ MakeCoordinator abort.
        with self.assertRaises(RuntimeError) as ctx:
            _pcs.validate_flat_scheduler_config(
                flat_kvcache_ext=True,
                paged_cache_groups=[],
                attn_backend=FakeFlatMHABackend(),
                kv_pool=FakeMHAPool(),
                speculative_enabled=True,
            )
        msg = str(ctx.exception)
        self.assertIn("at least one paged-cache group", msg)
        self.assertIn("speculative decoding is enabled", msg)
        self.assertIn("Disable speculative decoding", msg)

    def test_flat_ext_zero_groups_groupless_pool_names_the_pool(self):
        # Group-less pools (e.g. mamba) publish nothing without spec decode.
        with self.assertRaises(RuntimeError) as ctx:
            _pcs.validate_flat_scheduler_config(
                flat_kvcache_ext=True,
                paged_cache_groups=[],
                attn_backend=FakeFlatMHABackend(),
                kv_pool=FakeMambaPool(),
                speculative_enabled=False,
            )
        msg = str(ctx.exception)
        self.assertIn("at least one paged-cache group", msg)
        self.assertIn("FakeMambaPool", msg)
        self.assertIn("mamba", msg)
        self.assertIn("radix-built", msg)

    def test_flat_ext_mha_groups_passes(self):
        _pcs.validate_flat_scheduler_config(
            flat_kvcache_ext=True,
            paged_cache_groups=[FakeGroup()],
            attn_backend=FakeFlatMHABackend(),
            kv_pool=FakeMHAPool(),
            speculative_enabled=False,
        )

    def test_radix_ext_is_a_noop_regardless(self):
        # Radix build: groups are transport-only; every combination that the
        # flat arms reject must pass untouched.
        for backend, pool, groups, spec in (
            (FakeV4StyleBackend(), FakeV4Pool(), [FakeGroup()], False),
            (FakeFlatMHABackend(), FakeMHAPool(), [], True),
            (FakeFlatMHABackend(), FakeMambaPool(), [], False),
        ):
            _pcs.validate_flat_scheduler_config(
                flat_kvcache_ext=False,
                paged_cache_groups=groups,
                attn_backend=backend,
                kv_pool=pool,
                speculative_enabled=spec,
            )

    def test_backend_without_flags_defaults_safe(self):
        # Backends predating the class flags (getattr defaults False) must
        # not trip the layout arm; the zero-group arm still applies.
        class LegacyBackend:
            pass

        _pcs.validate_flat_scheduler_config(
            flat_kvcache_ext=True,
            paged_cache_groups=[FakeGroup()],
            attn_backend=LegacyBackend(),
            kv_pool=FakeMHAPool(),
            speculative_enabled=False,
        )


if __name__ == "__main__":
    unittest.main()
