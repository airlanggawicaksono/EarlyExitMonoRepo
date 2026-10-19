"""Tests for exp2/serve/fam_llama.py.

Part 1 needs no torch: checkpoints, argument checks, advance/begin/finish orchestration (a stub adapter
replaces the two torch methods `_view` and `_step`), import hygiene.
Part 2 builds a tiny random Llama on CPU and is skipped when torch or transformers is missing.
"""
import ast
import contextlib
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from exp2 import contracts
from exp2.serve import families, fam_llama
from exp2.serve.fam_llama import LlamaAdapter, LlamaState

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENS = 32
PROMPT = 128


class StubAdapter(LlamaAdapter):
    """No torch: records view enter/exit and decode steps, can fail on a chosen token."""

    def __init__(self, fail_at=None):
        super().__init__()
        self.events = []
        self.fail_at = fail_at

    @contextlib.contextmanager
    def _view(self, exit):
        self.events.append(("enter", exit))
        try:
            yield
        finally:
            self.events.append(("exit", exit))

    def _step(self, state):
        token = state.n_done + 1
        if self.fail_at == token:
            self.events.append(("boom", token))
            raise RuntimeError("boom")
        self.events.append(("step", token))
        state.n_done = token


class FakeIds:
    def __init__(self):
        self.moved_to = []

    def to(self, device):
        self.moved_to.append(device)
        return ("moved", device)


def stub_with_inputs(preprocess, n_layers=16):
    a = StubAdapter()
    a._inputs = [FakeIds(), FakeIds()]
    a._device = "dev0"
    a._preprocess = preprocess
    a._n_layers = n_layers
    return a


def run_poisoned(code):
    prelude = "import sys\nfor m in ('torch', 'transformers', 'cv2', 'datasets'):\n    sys.modules[m] = None\n"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, "-c", prelude + code], cwd=str(REPO_ROOT), env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


# ================================================================ part 1: no torch

def test_family_and_checkpoints_match_contract_and_spec():
    a = LlamaAdapter()
    assert a.family == "llama"
    for exit_ in (0, 7, 15):
        ck = a.checkpoints(exit_)
        assert ck == contracts.default_checkpoints("llama", exit_)
        assert ck == list(range(1, TOKENS + 1))


@pytest.mark.parametrize("bad", [-1, 16, 1.5, None, True, "3"])
def test_checkpoints_reject_exit_outside_contract_range(bad):
    with pytest.raises(ValueError):
        LlamaAdapter().checkpoints(bad)


def test_adapter_satisfies_family_adapter_protocol():
    assert isinstance(LlamaAdapter(), contracts.FamilyAdapter)
    assert isinstance(families.get_adapter("llama", {}), contracts.FamilyAdapter)


def test_constants_follow_contract():
    assert fam_llama.LLAMA_TOKENS == contracts.LLAMA_TOKENS == TOKENS
    assert fam_llama.LLAMA_PROMPT_TOKENS == contracts.LLAMA_PROMPT_TOKENS == PROMPT


def test_module_and_adapter_work_when_torch_cannot_be_imported():
    out = run_poisoned(
        "from exp2.serve import families\n"
        "a = families.get_adapter('llama', {})\n"
        "assert type(a).__name__ == 'LlamaAdapter'\n"
        "assert a.checkpoints(3) == list(range(1, 33))\n"
        "print('ok')\n")
    assert out == "ok"


# ---- advance: one step per token, inside the exit view

def test_advance_steps_each_token_inside_one_exit_view_per_call():
    a = StubAdapter()
    s = LlamaState(exit=3, ids=None)
    a.advance(s, 1)
    a.advance(s, 4)
    assert a.events == [("enter", 3), ("step", 1), ("exit", 3),
                        ("enter", 3), ("step", 2), ("step", 3), ("step", 4), ("exit", 3)]
    assert s.n_done == 4


def test_advance_to_last_token_in_one_call_for_no_checkpoints():
    a = StubAdapter()
    s = LlamaState(exit=0, ids=None)
    a.advance(s, TOKENS)
    assert [e for e in a.events if e[0] == "step"] == [("step", i) for i in range(1, TOKENS + 1)]
    assert a.events[0] == ("enter", 0) and a.events[-1] == ("exit", 0)
    assert len(a.events) == TOKENS + 2


@pytest.mark.parametrize("ckpt", [0, -1, TOKENS + 1, 100, 2.0, True, None, "3"])
def test_advance_rejects_bad_ckpt_without_running_anything(ckpt):
    a = StubAdapter()
    s = LlamaState(exit=2, ids=None)
    with pytest.raises(ValueError):
        a.advance(s, ckpt)
    assert a.events == [] and s.n_done == 0


@pytest.mark.parametrize("ckpt", [5, 4, 1])
def test_advance_rejects_repeat_or_backward_ckpt(ckpt):
    a = StubAdapter()
    s = LlamaState(exit=2, ids=None)
    a.advance(s, 5)
    a.events.clear()
    with pytest.raises(ValueError):
        a.advance(s, ckpt)
    assert a.events == [] and s.n_done == 5


def test_decode_error_propagates_once_and_view_is_left():
    a = StubAdapter(fail_at=3)
    s = LlamaState(exit=1, ids=None)
    with pytest.raises(RuntimeError, match="boom"):
        a.advance(s, 5)
    assert a.events == [("enter", 1), ("step", 1), ("step", 2), ("boom", 3), ("exit", 1)]
    assert s.n_done == 2


def test_decode_path_has_no_try_block_so_no_silent_fallback():
    for fn in (LlamaAdapter.advance, LlamaAdapter._step):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.Try)], fn.__name__


# ---- begin: host-side preprocessing only for per_request

def test_begin_per_request_moves_ids_to_device():
    a = stub_with_inputs("per_request")
    s = a.begin(1, 5)
    assert s.ids == ("moved", "dev0")
    assert a._inputs[1].moved_to == ["dev0"] and a._inputs[0].moved_to == []
    assert (s.exit, s.n_done, s.tokens) == (5, 0, [])


def test_begin_preload_uses_stored_ids_without_copy():
    a = stub_with_inputs("preload")
    s = a.begin(0, 2)
    assert s.ids is a._inputs[0]
    assert a._inputs[0].moved_to == []


def test_begin_before_make_inputs_raises_runtime_error():
    with pytest.raises(RuntimeError):
        LlamaAdapter().begin(0, 0)


@pytest.mark.parametrize("idx", [-1, 2, 99])
def test_begin_rejects_input_index_outside_inputs(idx):
    with pytest.raises(ValueError):
        stub_with_inputs("preload").begin(idx, 0)


@pytest.mark.parametrize("exit_", [-1, 16, 99])
def test_begin_rejects_exit_outside_model(exit_):
    with pytest.raises(ValueError):
        stub_with_inputs("preload").begin(0, exit_)


def test_begin_rejects_exit_beyond_loaded_layer_count():
    a = stub_with_inputs("preload", n_layers=4)
    a.begin(0, 3)
    with pytest.raises(ValueError):
        a.begin(0, 4)


# ---- finish / sync / load / make_inputs argument checks

def test_finish_none_is_a_noop_without_torch():
    out = run_poisoned(
        "from exp2.serve.fam_llama import LlamaAdapter, LlamaState\n"
        "assert LlamaAdapter().finish(LlamaState(exit=0, ids=None), None) is None\n"
        "print('ok')\n")
    assert out == "ok"


@pytest.mark.parametrize("reached", [0, -1, 4, 33, True, 1.0])
def test_finish_rejects_reached_outside_tokens_done(reached):
    s = LlamaState(exit=0, ids=None)
    s.n_done = 3
    with pytest.raises(ValueError):
        LlamaAdapter().finish(s, reached)


def test_sync_before_load_raises_and_after_load_calls_sync_fn():
    a = LlamaAdapter()
    with pytest.raises(RuntimeError):
        a.sync()
    calls = []
    a._sync = lambda: calls.append(1)
    a.sync()
    assert calls == [1]


@pytest.mark.parametrize("exits", [[], [16], [-1], [3, 16]])
def test_load_validates_exits_before_building_the_model(exits):
    a = LlamaAdapter()

    def no_build(cfg):
        raise AssertionError("model must not be built for invalid exits")

    a._build_model = no_build
    with pytest.raises(ValueError):
        a.load(exits, {})


def test_load_rejects_unknown_preprocess_before_building_the_model():
    a = LlamaAdapter()
    a._build_model = lambda cfg: (_ for _ in ()).throw(AssertionError("must not build"))
    with pytest.raises(ValueError):
        a.load([3], {"preprocess": "bogus"})


@pytest.mark.parametrize("n,mode", [(0, "synthetic"), (-2, "synthetic"), (2, "bogus"), (2, None), (1.5, "synthetic")])
def test_make_inputs_rejects_bad_arguments(n, mode):
    with pytest.raises(ValueError):
        LlamaAdapter().make_inputs(n, 0, mode)


def test_make_inputs_before_load_raises_runtime_error():
    with pytest.raises(RuntimeError):
        LlamaAdapter().make_inputs(2, 0, "synthetic")


def test_mem_helpers_delegate_to_families(monkeypatch):
    monkeypatch.setattr(families, "mem_stats", lambda: {"sentinel": 1})
    seen = []
    monkeypatch.setattr(families, "reset_peak", lambda: seen.append(1))
    a = LlamaAdapter()
    assert a.mem_stats() == {"sentinel": 1}
    a.reset_peak()
    assert seen == [1]


# ================================================================ part 2: tiny random Llama (torch)

TINY = dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)


@pytest.fixture
def torch_mod():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return torch


class TinyAdapter(LlamaAdapter):
    """Real adapter logic on a random 4-layer Llama (CPU, float32) instead of the 1B checkpoint."""

    def _build_model(self, cfg):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(**TINY)).eval()
        for p in model.parameters():
            p.requires_grad = False
        return model


@pytest.fixture
def tiny(torch_mod):
    a = TinyAdapter()
    a.load([0, 1, 3], {"preprocess": "per_request", "compile": False, "sync": "event"})
    a.make_inputs(2, 7, "synthetic")
    return a


def reference_logits(adapter, state, t):
    """Logits after token t from a full forward over prompt + first t-1 generated tokens, no cache."""
    import torch

    base = adapter._base
    seq = torch.cat([state.ids, *state.tokens[: t - 1]], dim=1)
    with torch.no_grad(), fam_llama._exit_at(base, state.exit):
        hidden = base.model(input_ids=seq).last_hidden_state[:, -1:, :]
    return base.lm_head(hidden)[:, -1, :]


def test_exit_at_exposes_first_layers_and_restores(tiny):
    model = tiny._base
    with fam_llama._exit_at(model, 1):
        assert len(model.model.layers) == 2
        assert model.config.num_hidden_layers == 2
    assert len(model.model.layers) == 4 and model.config.num_hidden_layers == 4
    with pytest.raises(RuntimeError):
        with fam_llama._exit_at(model, 0):
            raise RuntimeError("inside")
    assert len(model.model.layers) == 4 and model.config.num_hidden_layers == 4


def test_load_reports_layers_and_no_compile(torch_mod):
    a = TinyAdapter()
    info = a.load([1], {"compile": False})
    assert info["n_layers"] == 4 and info["compiled_layers"] == 0 and info["load_s"] >= 0.0


def test_prompt_is_fixed_128_token_long_ids(tiny, torch_mod):
    s = tiny.begin(0, 1)
    assert tuple(s.ids.shape) == (1, PROMPT)
    assert s.ids.dtype == torch_mod.int64
    assert 0 <= int(s.ids.min()) and int(s.ids.max()) < TINY["vocab_size"]


def test_inputs_are_deterministic_per_seed_and_distinct(tiny, torch_mod):
    first = [tiny.begin(i, 0).ids.clone() for i in (0, 1)]
    tiny.make_inputs(2, 7, "synthetic")
    again = [tiny.begin(i, 0).ids for i in (0, 1)]
    assert all(torch_mod.equal(x, y) for x, y in zip(first, again))
    assert not torch_mod.equal(first[0], first[1])
    tiny.make_inputs(2, 8, "synthetic")
    assert not torch_mod.equal(first[0], tiny.begin(0, 0).ids)


def test_dataset_mode_uses_dataset_rows_and_checks_prompt_length(tiny, torch_mod):
    rows = torch_mod.arange(2 * PROMPT).reshape(2, PROMPT) % TINY["vocab_size"]
    tiny._dataset_ids = lambda n, seed: rows[:n]
    tiny.make_inputs(2, 0, "dataset")
    assert torch_mod.equal(tiny.begin(1, 0).ids, rows[1:2])
    tiny._dataset_ids = lambda n, seed: rows[:, : PROMPT - 1]
    with pytest.raises(ValueError):
        tiny.make_inputs(2, 0, "dataset")


@pytest.mark.parametrize("exit_", [0, 1, 3])
def test_every_token_logits_match_full_recompute_reference(tiny, torch_mod, exit_):
    s = tiny.begin(0, exit_)
    for t in range(1, TOKENS + 1):
        tiny.advance(s, t)
        ref = reference_logits(tiny, s, t)
        assert torch_mod.allclose(s.logits, ref, rtol=1e-4, atol=1e-5), (exit_, t)
    assert s.n_done == TOKENS


def test_one_call_equals_token_by_token(tiny, torch_mod):
    one = tiny.begin(0, 3)
    tiny.advance(one, TOKENS)
    many = tiny.begin(0, 3)
    for t in range(1, TOKENS + 1):
        tiny.advance(many, t)
    assert tiny.finish(one, TOKENS) == tiny.finish(many, TOKENS)
    assert torch_mod.equal(one.logits, many.logits)


def test_interleaved_states_do_not_share_cache(tiny, torch_mod):
    a = tiny.begin(0, 3)
    b = tiny.begin(1, 1)
    tiny.advance(a, 3)
    tiny.advance(b, 3)
    tiny.advance(a, 6)
    alone = tiny.begin(0, 3)
    tiny.advance(alone, 6)
    assert tiny.finish(a, 6) == tiny.finish(alone, 6)
    assert torch_mod.equal(a.logits, alone.logits)


def test_finish_returns_generated_token_prefix(tiny):
    s = tiny.begin(0, 2)
    tiny.advance(s, 10)
    full = tiny.finish(s, 10)
    assert len(full) == 10 and all(isinstance(x, int) for x in full)
    assert tiny.finish(s, 6) == full[:6]
    assert tiny.finish(s, None) is None


def test_eos_is_ignored_and_all_32_tokens_decode(tiny, torch_mod):
    class ConstHead(torch_mod.nn.Module):
        def forward(self, hidden):
            out = torch_mod.zeros(*hidden.shape[:-1], TINY["vocab_size"])
            out[..., 7] = 1.0
            return out

    tiny._base.lm_head = ConstHead()
    tiny._base.config.eos_token_id = 7
    s = tiny.begin(0, 1)
    tiny.advance(s, TOKENS)
    assert s.n_done == TOKENS
    assert tiny.finish(s, TOKENS) == [7] * TOKENS


def test_prefill_then_single_token_decode_with_kv_cache(tiny, monkeypatch):
    s = tiny.begin(0, 3)
    orig = tiny._base.model.forward
    shapes = []

    def spy(*args, **kwargs):
        shapes.append(tuple(kwargs["input_ids"].shape))
        return orig(*args, **kwargs)

    monkeypatch.setattr(tiny._base.model, "forward", spy)
    tiny.advance(s, 4)
    assert shapes == [(1, PROMPT), (1, 1), (1, 1), (1, 1)]
    assert s.past.get_seq_length() == PROMPT + 3


def test_model_failure_propagates_without_recompute_and_restores_layers(tiny, monkeypatch):
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("cache api drift")

    monkeypatch.setattr(tiny._base.model, "forward", boom)
    s = tiny.begin(0, 2)
    with pytest.raises(RuntimeError, match="cache api drift"):
        tiny.advance(s, 3)
    assert calls == [1]
    assert s.n_done == 0
    assert len(tiny._base.model.layers) == 4


def test_begin_rejects_exit_beyond_tiny_model(tiny):
    with pytest.raises(ValueError):
        tiny.begin(0, 4)


def test_sync_and_memory_helpers_run_on_cpu(tiny):
    tiny.sync()
    assert set(tiny.mem_stats()) == {"torch_reserved_mb", "torch_peak_alloc_mb", "rss_mb"}
    tiny.reset_peak()
