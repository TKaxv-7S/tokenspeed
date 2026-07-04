"""E2E: the M12 hybrid-slab KV layout's capacity gain shows up as prefix-cache hits.

Mechanism under test: on a hybrid full/sliding model (gpt-oss), the M12 slab
layout (``hybrid_slab_group_size``) shares one K/V slab between one layer from
each attention group, so the KV byte budget is divided by layers-per-group
instead of total layers -- the page pool doubles in tokens versus the legacy
per-layer-buffer layout. This test boots the SAME commit twice under the same
byte budget (slab: natural; legacy: predicate nulled via an env-injected
sitecustomize, see ``_LEGACY_SITECUSTOMIZE``) and asserts the gain twice over:

1. directly: ``scheduler_info["max_total_num_tokens"]`` doubles (measured
   halving when the predicate is nulled -- e.g. 168832 -> 84416 on b200-77);
2. behaviorally: K distinct ~2074-token prompts are generated twice in the
   same order, with K sized so one round's page-allocation footprint is
   ~0.72x the slab pool (so ~1.44x the legacy pool). Slab: cached prefixes
   survive round 1; round 2 hits ~99% of prompt tokens (bounded only by the
   M9 host-match cap, 32 pages = 2048/prompt). Legacy: the round's
   allocations exceed the pool, so cached prefixes are recycled in insertion
   order, and the same-order revisit is the worst case -- every prompt is
   evicted before reuse, hits collapse to ~0.

A prompt's allocation footprint is ~2x its token count, not 1x: the
full-attention group's ~33 pages are retained as cached prefix, but the
sliding-window group ALSO allocates ~33 pages during prefill (every position
is written before the window advances) which return to the pool afterwards.
The shared BlockPool hands out never-used pages first and then recycles the
OLDEST freed pages -- which are the earliest prompts' cached prefixes, not
the just-freed sliding transients. Hence the cliff is at cumulative
round allocations = pool size (verified on b200-77: K=20 -> 1320 pages of
2638 -> 97% hits; K=59 -> 3894 of 2638 -> 0% hits on the SAME slab pool).

K is computed from the MEASURED slab capacity, not fixed: the profiled pool
depends on the GPU's free memory at boot (an overnight run on a busier
b200-77 got the 97%/0% contrast with K=20 at util 0.165; on an idle GPU the
same budget yields a pool where K=20 fits both layouts, hiding the
contrast). Both arms use the same K, and the direct capacity assertion (1)
keeps the test honest if the slab sizing ever regresses to legacy (the
workload would then fit both arms and (2) alone could not tell).

``cached_tokens`` provenance (verified, so this metric is valid for FLAT
hits, not a radix-only counter): ``Engine.generate()`` meta_info
``cached_tokens`` is accumulated per request in
``engine/generation_output_processor.py::add_cached_tokens`` as
``max(0, extend_prefix_len - computed_length)`` from every prefill chunk's
``forward_op.extend_prefix_lens``. ``extend_prefix_len`` is the C++
scheduler's ``already_scheduled_len``
(``tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp``,
``applyPrefillEvent``), which the prefill-scheduling transition advances past
the prefix hit on both builds (flat: ``ClaimCommonPrefix``; radix: device
tree match). It therefore counts exactly the prompt tokens served from cache
under the flat scheduler.

Requires a flat-built (TOKENSPEED_FLAT_KVCACHE) tokenspeed_scheduler ext;
skips cleanly on a radix build.

Usage:
    cd test/runtime
    python3 -m unittest test_slab_capacity_prefix_hits -v
"""

import math
import os
import shutil
import sys
import tempfile
import unittest

import torch

# Repository root on sys.path so ``test.runners`` resolves.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from test.runners import get_dtype_str  # noqa: E402

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=900, suite="runtime-prefix-cache-e2e")

from tokenspeed.runtime.entrypoints.engine import Engine  # noqa: E402

_MODEL = "openai/gpt-oss-20b"
# 130 numbered sentences tokenize to ~2074 tokens per prompt with the
# gpt-oss tokenizer (33 pages at page_size 64). Guarded at runtime by the
# band below so tokenizer drift cannot silently change the regime: staying
# under 2270 keeps the slab-arm hit-ratio floor above 0.9 despite the M9
# host-match cap (32 pages = 2048 hit tokens per prompt).
_SENTENCES_PER_PROMPT = 130
_APPROX_PROMPT_TOKENS = 2074
_PROMPT_TOKENS_MIN = 1900
_PROMPT_TOKENS_MAX = 2270

# A prompt's page-allocation footprint per round: full-history retention
# (~prompt tokens) + the sliding group's prefill transient (~prompt tokens
# again; see module docstring). Small terms (retained sliding tail, decode
# pages) are absorbed by the fill margins.
_APPROX_ALLOC_TOKENS = 2 * _APPROX_PROMPT_TOKENS
# Round allocation footprint as a fraction of the measured slab pool:
# comfortably under the recycling cliff on the slab arm, and (at the
# expected 2x gain) ~1.44x the legacy pool -- deep enough past the cliff
# that the same-order revisit recycles every cached prefix.
_TARGET_POOL_FILL = 0.72
_NUM_PROMPTS_MIN = 8
_NUM_PROMPTS_MAX = 120

_SAMPLING = {"max_new_tokens": 4, "temperature": 0}

_WORDS = [
    "amber",
    "birch",
    "cobalt",
    "damson",
    "ember",
    "fennel",
    "garnet",
    "hazel",
    "indigo",
    "juniper",
    "kestrel",
    "larch",
    "mallow",
    "nutmeg",
]

# Injected into engine subprocesses via PYTHONPATH for the legacy arm only.
#
# Why a sitecustomize: the layout decision is made inside the SPAWNED
# scheduler/executor subprocesses (registry KV-memory profile +
# MHATokenToKVPool._create_buffers), which a patch in this test process
# cannot reach; a sitecustomize on PYTHONPATH runs at every child
# interpreter's startup. ``hybrid_slab_group_size`` is the single activation
# source for the slab layout (sizing divisor and buffer layout both consume
# it), and both consumers bind it by from-import, so nulling it in the three
# namespaces below reproduces the pre-M12 layout exactly -- a same-commit
# A/B. The patch is lazy (a sys.meta_path find_spec hook fires on every
# uncached import and patches once the module lands in sys.modules) because
# sitecustomize runs before any tokenspeed module exists.
#
# Chaining: Python imports only the FIRST sitecustomize on sys.path. Test
# environments may already carry one (e.g. a front-loaded flat scheduler
# ext); since this tempdir is prepended, ours must locate and exec the next
# sitecustomize.py on sys.path or that setup would be silently dropped.
_LEGACY_SITECUSTOMIZE = '''\
"""Force the legacy (pre-M12) per-layer KV buffer layout in this interpreter.

Injected by test_slab_capacity_prefix_hits.py. Active only when
TOKENSPEED_FORCE_LEGACY_KV_LAYOUT=1; always chains to the next sitecustomize
on sys.path (Python only imports the first one).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _chain_next_sitecustomize():
    for entry in sys.path:
        if not entry or os.path.abspath(entry) == _HERE:
            continue
        cand = os.path.join(entry, "sitecustomize.py")
        if os.path.isfile(cand):
            with open(cand) as f:
                src = f.read()
            g = {"__name__": "sitecustomize", "__file__": cand}
            exec(compile(src, cand, "exec"), g)
            return


_chain_next_sitecustomize()

if os.environ.get("TOKENSPEED_FORCE_LEGACY_KV_LAYOUT") == "1":
    # Both consumers bind hybrid_slab_group_size by from-import at module
    # top, so the defining module alone is not enough: patch every consumer
    # namespace once it appears in sys.modules. Re-assert on EVERY hook
    # fire rather than mark-and-skip: a module is registered in sys.modules
    # while its body is still executing, so an early patch (fired by one of
    # the module's own imports) is silently overwritten when its
    # from-import line runs -- only a later re-assert survives.
    _TARGETS = (
        "tokenspeed.runtime.configs.paged_cache_spec",
        "tokenspeed.runtime.layers.attention.registry",
        "tokenspeed.runtime.layers.attention.kv_cache.mha",
    )

    def _null_predicate(*a, **k):
        return None

    def _null_slab_predicate():
        for name in _TARGETS:
            mod = sys.modules.get(name)
            if mod is None:
                continue
            if getattr(mod, "hybrid_slab_group_size", None) is not _null_predicate:
                mod.hybrid_slab_group_size = _null_predicate

    class _LegacyLayoutHook:
        """meta_path hook: patch targets lazily on every uncached import."""

        def find_spec(self, name, path=None, target=None):
            _null_slab_predicate()
            return None  # never handles the import itself

    sys.meta_path.insert(0, _LegacyLayoutHook())
    _null_slab_predicate()
'''


def _build_prompt(i: int) -> str:
    """A ~2074-token prompt, distinct per ``i`` from the first sentence on.

    Numbered head plus a varied, non-repetitive body (word cycling + per-
    sentence numbers). Deliberately NOT a repeated identical filler:
    pathologically repetitive text produces logit ties that make greedy
    outputs unstable across layouts.
    """
    parts = [f"Ledger {i} opens with a fresh manifest of arrivals."]
    for j in range(_SENTENCES_PER_PROMPT):
        word = _WORDS[j % len(_WORDS)]
        parts.append(
            f"Entry {i}-{j}: the {word} shipment number "
            f"{i * 7 + j * 3 + 3} arrived intact."
        )
    parts.append(f"Ledger {i} summary: report the last entry number only.")
    return " ".join(parts)


def _make_engine() -> Engine:
    return Engine(
        model=_MODEL,
        dtype=get_dtype_str(torch.bfloat16),
        seed=42,
        enable_prefix_caching=True,
        # The slab layout is incompatible with kvstore L2 host copies
        # (per-layer copies would alias shared slabs); resolve_cache would
        # auto-enable kvstore with prefix caching on, tripping the slab
        # guard in MHATokenToKVPool._check_slab_guards.
        disable_kvstore=True,
        max_model_len=8192,
        max_num_seqs=2,
        # The byte budget both arms share. The profiled pool additionally
        # depends on free GPU memory at boot, which is why the working set
        # is sized from the measured capacity instead of hard-coded.
        gpu_memory_utilization=0.165,
        moe_backend="flashinfer_mxfp4",
        disable_prefill_graph=True,
    )


def _run_round(engine: Engine, prompts: list) -> tuple:
    """Generate every prompt once; return (cached_tokens, prompt_tokens) sums."""
    total_cached = 0
    total_prompt = 0
    for prompt in prompts:
        resp = engine.generate(
            prompt=prompt,
            sampling_params=_SAMPLING,
            return_logprob=False,
            stream=False,
        )
        meta = resp["meta_info"]
        prompt_tokens = int(meta["prompt_tokens"])
        if not _PROMPT_TOKENS_MIN <= prompt_tokens <= _PROMPT_TOKENS_MAX:
            # Not a bare assert: must survive python -O.
            raise AssertionError(
                f"prompt tokenized to {prompt_tokens} tokens, outside the "
                f"proven regime [{_PROMPT_TOKENS_MIN}, {_PROMPT_TOKENS_MAX}];"
                " retune _SENTENCES_PER_PROMPT"
            )
        total_cached += int(meta.get("cached_tokens", 0))
        total_prompt += prompt_tokens
    return total_cached, total_prompt


def _measure_arm(engine: Engine, num_prompts: int, tag: str) -> tuple:
    """Two sequential rounds over the same prompts; return (r1_ratio, r2_ratio).

    Round 1 is the cold pass that populates the cache; round 2 revisits the
    prompts in the same order (the LRU worst case when the working set
    overflows the pool).
    """
    prompts = [_build_prompt(i) for i in range(num_prompts)]
    cached1, prompt1 = _run_round(engine, prompts)
    cached2, prompt2 = _run_round(engine, prompts)
    print(
        f"[{tag}] K={num_prompts} round-1 cached/prompt: {cached1}/{prompt1} "
        f"round-2 cached/prompt: {cached2}/{prompt2} = {cached2 / prompt2:.3f}"
    )
    return cached1 / prompt1, cached2 / prompt2


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestSlabCapacityPrefixHits(unittest.TestCase):
    """Same commit, same byte budget, same workload: the slab arm keeps the
    working set cached; the legacy arm's halved pool LRU-cycles it. One
    self-contained method: the two arms must share K and be compared on
    measured capacity, which per-arm tests could not do without cross-test
    state."""

    def setUp(self):
        try:
            import tokenspeed_scheduler
        except ImportError:
            self.skipTest("tokenspeed_scheduler ext is not installed")
        if not getattr(tokenspeed_scheduler, "FLAT_KVCACHE", False):
            self.skipTest(
                "requires a flat-built (TOKENSPEED_FLAT_KVCACHE) "
                "tokenspeed_scheduler ext; radix builds have no slab layout"
            )
        if os.environ.get("TOKENSPEED_CI_SMALL_KV_SIZE"):
            self.skipTest(
                "TOKENSPEED_CI_SMALL_KV_SIZE pins the token pool for both "
                "layouts, erasing the capacity contrast under test"
            )

    def test_slab_layout_doubles_capacity_and_prefix_hits(self):
        # --- Arm 1: natural slab layout ---
        engine = _make_engine()
        try:
            slab_capacity = int(engine.scheduler_info["max_total_num_tokens"])
            num_prompts = math.ceil(
                _TARGET_POOL_FILL * slab_capacity / _APPROX_ALLOC_TOKENS
            )
            if not _NUM_PROMPTS_MIN <= num_prompts <= _NUM_PROMPTS_MAX:
                self.skipTest(
                    f"measured slab pool ({slab_capacity} tokens) needs "
                    f"K={num_prompts} prompts, outside "
                    f"[{_NUM_PROMPTS_MIN}, {_NUM_PROMPTS_MAX}]; free GPU "
                    "memory is too far from the proven regime"
                )
            print(f"[slab layout] max_total_num_tokens={slab_capacity}")
            r1_slab, r2_slab = _measure_arm(engine, num_prompts, "slab layout")
        finally:
            engine.shutdown()
        # Prompts are distinct from the first sentence on, so no page-aligned
        # prefix can match cold: this pins cached_tokens to real cache reuse.
        self.assertEqual(
            r1_slab, 0, f"slab round 1 must be cold, got ratio {r1_slab:.3f}"
        )
        # The round's allocations are ~0.72x the pool, under the recycling
        # cliff, so every cached prefix survives; per-prompt hits are bounded
        # by the M9 cap (2048 of ~2074 tokens -> ~0.99). 0.9 leaves margin
        # for tokenizer drift within the guarded band.
        self.assertGreaterEqual(
            r2_slab,
            0.9,
            f"slab layout: round-2 hit ratio {r2_slab:.3f} below 0.9 -- "
            "working set no longer resident, capacity gain regressed",
        )

        # --- Arm 2: legacy layout forced on the same commit ---
        tmpdir = tempfile.mkdtemp(prefix="ts_legacy_kv_site_")
        old_pythonpath = os.environ.get("PYTHONPATH")
        old_flag = os.environ.get("TOKENSPEED_FORCE_LEGACY_KV_LAYOUT")
        try:
            with open(os.path.join(tmpdir, "sitecustomize.py"), "w") as f:
                f.write(_LEGACY_SITECUSTOMIZE)
            os.environ["PYTHONPATH"] = tmpdir + (
                os.pathsep + old_pythonpath if old_pythonpath else ""
            )
            os.environ["TOKENSPEED_FORCE_LEGACY_KV_LAYOUT"] = "1"
            engine = _make_engine()
            try:
                legacy_capacity = int(engine.scheduler_info["max_total_num_tokens"])
                print(f"[legacy layout] max_total_num_tokens={legacy_capacity}")
                cap_ratio = slab_capacity / legacy_capacity
                # The M12 claim measured directly: gpt-oss pairs 12 sliding
                # with 12 full layers, so the sizing divisor halves and the
                # token pool exactly doubles. The band tolerates free-memory
                # drift between the two boots.
                self.assertTrue(
                    1.9 <= cap_ratio <= 2.1,
                    f"slab/legacy capacity ratio {cap_ratio:.3f} "
                    f"({slab_capacity}/{legacy_capacity}) outside [1.9, 2.1] "
                    "-- either the slab sizing regressed or the legacy "
                    "forcing did not reach the engine subprocesses",
                )
                r1_leg, r2_leg = _measure_arm(engine, num_prompts, "legacy layout")
            finally:
                engine.shutdown()
        finally:
            if old_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = old_pythonpath
            if old_flag is None:
                os.environ.pop("TOKENSPEED_FORCE_LEGACY_KV_LAYOUT", None)
            else:
                os.environ["TOKENSPEED_FORCE_LEGACY_KV_LAYOUT"] = old_flag
            shutil.rmtree(tmpdir, ignore_errors=True)
        self.assertEqual(
            r1_leg, 0, f"legacy round 1 must be cold, got ratio {r1_leg:.3f}"
        )
        # Round allocations ~1.44x the halved pool + same-order revisit =
        # every cached prefix recycled before reuse; proven 0%. 0.2 is the
        # collapse bound; with the slab arm's 0.9 floor it guarantees a
        # >= 4.5x hit-ratio contrast.
        self.assertLessEqual(
            r2_leg,
            0.2,
            f"legacy layout: round-2 hit ratio {r2_leg:.3f} above 0.2 -- "
            "the halved pool unexpectedly held the working set",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
