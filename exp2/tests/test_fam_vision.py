"""Tests for exp2/serve/fam_vision.py (ViT-large adapter, K6).

Part 1 needs no torch: checkpoints, argument checks, advance/begin/finish orchestration (a stub adapter replaces
the three torch methods `_embed`, `_block`, `_head`), the block-list and block-output helpers, import hygiene.
Part 2 builds a tiny random ViT on CPU and is skipped when torch or transformers is missing. It compares the segmented
forward with the unmodified backbone forward and with the legacy truncated forward at every checkpoint.
"""
import ast
import contextlib
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from exp2 import contracts
from exp2.serve import families, fam_vision
from exp2.serve.fam_vision import VisionAdapter, VisionState

REPO_ROOT = Path(__file__).resolve().parents[2]


class StubAdapter(VisionAdapter):
    """No torch: records the three model calls, can fail on a chosen block. Hidden state "hN" = N blocks done."""

    def __init__(self, fail_at_block=None):
        super().__init__()
        self.events = []
        self.fail_at_block = fail_at_block

    def _embed(self, state):
        self.events.append("embed")
        return "emb"

    def _block(self, state, i):
        if self.fail_at_block == i:
            self.events.append(("boom", i))
            raise RuntimeError("boom")
        self.events.append(("block", i))
        return "h%d" % (i + 1)

    def _head(self, state, reached):
        self.events.append(("head", reached, state.hiddens[reached]))
        return "class@%d" % reached


def new_state(exit_):
    return VisionState(exit_, "px", contracts.default_checkpoints("vision", exit_))


class FakeTensor:
    def __init__(self, name):
        self.name = name
        self.moved_to = []

    def to(self, device):
        self.moved_to.append(device)
        return ("moved", self.name, device)


def stub_with_inputs(preprocess, n_layers=24, pixelize=None, raw=False):
    a = StubAdapter()
    a._inputs = ["img0", "img1"] if raw else [FakeTensor("px0"), FakeTensor("px1")]
    a._pixelize = pixelize
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


def has_try(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return any(isinstance(n, ast.Try) for n in ast.walk(tree))


# ================================================================ part 1: no torch

def test_family_and_checkpoints_match_contract():
    a = VisionAdapter()
    assert a.family == "vision"
    for exit_ in range(1, 25):
        ck = a.checkpoints(exit_)
        assert ck == contracts.default_checkpoints("vision", exit_)
        assert ck[-1] == exit_ and ck == sorted(set(ck))


def test_checkpoints_are_every_4th_layer_plus_the_target():
    a = VisionAdapter()
    assert a.checkpoints(24) == [4, 8, 12, 16, 20, 24]
    assert a.checkpoints(11) == [4, 8, 11]
    assert a.checkpoints(9) == [4, 8, 9]
    assert a.checkpoints(6) == [4, 6]
    assert a.checkpoints(4) == [4]
    assert a.checkpoints(1) == [1]


@pytest.mark.parametrize("bad", [0, -1, 25, 1.5, None, True, "3"])
def test_checkpoints_reject_exit_outside_contract_range(bad):
    with pytest.raises(ValueError):
        VisionAdapter().checkpoints(bad)


def test_adapter_satisfies_family_adapter_protocol():
    assert isinstance(VisionAdapter(), contracts.FamilyAdapter)
    assert isinstance(families.get_adapter("vision", {}), contracts.FamilyAdapter)


def test_constants_name_the_pretrained_backbone_and_cifar10_heads():
    assert fam_vision.MODEL_ID == "google/vit-large-patch16-224"
    assert fam_vision.NUM_LABELS == 10
    assert fam_vision.PREPROCESS_MODES == ("per_request", "preload")
    assert fam_vision.INPUT_MODES == ("synthetic", "dataset")


def test_module_and_adapter_work_when_torch_cannot_be_imported():
    out = run_poisoned(
        "from exp2.serve import families\n"
        "a = families.get_adapter('vision', {})\n"
        "assert type(a).__name__ == 'VisionAdapter'\n"
        "assert a.checkpoints(11) == [4, 8, 11]\n"
        "print('ok')\n")
    assert out == "ok"


def test_module_level_imports_are_light_and_not_benchmark_modules():
    tree = ast.parse(Path(fam_vision.__file__).read_text(encoding="utf-8"))
    top = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            top.append(node.module)
    assert top, "expected module level imports"
    assert not [m for m in top if m.split(".")[0] in ("torch", "transformers", "cv2", "datasets", "PIL")], top
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                assert name.split(".")[0] not in ("shared", "AnyTimeVisionenc"), name
                assert "benchmark" not in name, name


def test_only_the_backbone_is_loaded_never_a_second_classification_model():
    """The ported loader skips `_broadcast_or_random_head`, which loads a second ViT-large (about 1.2 GB)."""
    tree = ast.parse(Path(fam_vision.__file__).read_text(encoding="utf-8"))
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    used |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not used & {"ViTForImageClassification", "AutoModelForImageClassification", "_broadcast_or_random_head"}
    loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_model")
    loads = [n for n in ast.walk(loader) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "from_pretrained"]
    assert len(loads) == 1


# ---- block list and block output helpers (duck typed, no torch)

def test_block_list_finds_the_blocks_in_both_transformers_layouts():
    v5_blocks, v4_blocks, v4_plural = ["a", "b"], ["c", "d"], ["e", "f"]
    assert fam_vision._block_list(SimpleNamespace(layers=v5_blocks)) is v5_blocks
    assert fam_vision._block_list(SimpleNamespace(encoder=SimpleNamespace(layer=v4_blocks))) is v4_blocks
    assert fam_vision._block_list(SimpleNamespace(encoder=SimpleNamespace(layers=v4_plural))) is v4_plural


def test_block_list_raises_a_clear_error_when_no_layout_matches():
    with pytest.raises(AttributeError, match="block"):
        fam_vision._block_list(SimpleNamespace(encoder=SimpleNamespace()))
    with pytest.raises(AttributeError, match="block"):
        fam_vision._block_list(SimpleNamespace())


def test_block_output_unwraps_a_tuple_and_passes_a_tensor_through():
    marker = object()
    assert fam_vision._block_output((marker,)) is marker
    assert fam_vision._block_output((marker, None)) is marker
    assert fam_vision._block_output([marker, None]) is marker
    assert fam_vision._block_output(marker) is marker


# ---- advance: embeddings once, then blocks up to the checkpoint

def test_advance_embeds_once_then_runs_blocks_up_to_each_checkpoint():
    a = StubAdapter()
    s = new_state(11)
    a.advance(s, 4)
    assert a.events == ["embed", ("block", 0), ("block", 1), ("block", 2), ("block", 3)]
    a.events.clear()
    a.advance(s, 8)
    assert a.events == [("block", 4), ("block", 5), ("block", 6), ("block", 7)]
    assert (s.n_done, s.hidden) == (8, "h8")


def test_advance_keeps_the_hidden_state_of_every_checkpoint_it_reached():
    a = StubAdapter()
    s = new_state(11)
    a.advance(s, 4)
    a.advance(s, 8)
    a.advance(s, 11)
    assert s.hiddens == {4: "h4", 8: "h8", 11: "h11"}
    assert s.n_done == 11


def test_advance_straight_to_the_target_runs_every_block_once_and_keeps_only_that_checkpoint():
    a = StubAdapter()
    s = new_state(24)
    a.advance(s, 24)
    assert a.events == ["embed"] + [("block", i) for i in range(24)]
    assert list(s.hiddens) == [24]


@pytest.mark.parametrize("ckpt", [0, 1, 3, 5, 9, 12, 24, -1, 100, 8.0, True, None, "8"])
def test_advance_rejects_a_value_that_is_not_a_checkpoint_of_the_exit_without_running_anything(ckpt):
    a = StubAdapter()
    s = new_state(11)
    with pytest.raises(ValueError):
        a.advance(s, ckpt)
    assert a.events == [] and s.n_done == 0 and s.hiddens == {}


@pytest.mark.parametrize("ckpt", [4, 8])
def test_advance_rejects_repeat_or_backward_checkpoint(ckpt):
    a = StubAdapter()
    s = new_state(11)
    a.advance(s, 8)
    a.events.clear()
    with pytest.raises(ValueError):
        a.advance(s, ckpt)
    assert a.events == [] and s.n_done == 8 and list(s.hiddens) == [8]


def test_advance_rejects_checkpoint_beyond_the_target_of_the_exit():
    a = StubAdapter()
    s = new_state(6)
    with pytest.raises(ValueError):
        a.advance(s, 8)
    assert a.events == []


def test_block_error_propagates_once_and_state_stays_at_the_last_finished_block():
    a = StubAdapter(fail_at_block=5)
    s = new_state(11)
    a.advance(s, 4)
    a.events.clear()
    with pytest.raises(RuntimeError, match="boom"):
        a.advance(s, 8)
    assert a.events == [("block", 4), ("boom", 5)]
    assert (s.n_done, s.hidden) == (5, "h5")
    assert list(s.hiddens) == [4]


def test_a_failed_segment_can_be_resumed_from_the_last_finished_block():
    a = StubAdapter(fail_at_block=5)
    s = new_state(11)
    with pytest.raises(RuntimeError):
        a.advance(s, 8)
    a.fail_at_block = None
    a.events.clear()
    a.advance(s, 8)
    assert a.events == [("block", 5), ("block", 6), ("block", 7)]
    assert s.hiddens == {8: "h8"}


@pytest.mark.parametrize("fn", [VisionAdapter.advance, VisionAdapter._embed, VisionAdapter._block,
                                VisionAdapter._head, VisionAdapter.finish, fam_vision._compile_per_block],
                         ids=lambda f: f.__name__)
def test_model_path_has_no_try_block_so_no_silent_fallback(fn):
    assert not has_try(fn)


# ---- begin: host-side work only for per_request

def test_begin_per_request_moves_the_pixels_to_the_device():
    a = stub_with_inputs("per_request")
    s = a.begin(1, 5)
    assert s.pixels == ("moved", "px1", "dev0")
    assert a._inputs[0].moved_to == []
    assert (s.exit, s.n_done, s.hidden, s.hiddens) == (5, 0, None, {})
    assert s.ckpts == [4, 5]


def test_begin_per_request_runs_the_host_preprocessing_then_the_copy():
    calls = []

    def pixelize(raw):
        calls.append(raw)
        return FakeTensor("pix(%s)" % raw)

    a = stub_with_inputs("per_request", pixelize=pixelize, raw=True)
    assert calls == []
    s = a.begin(1, 8)
    assert calls == ["img1"]
    assert s.pixels == ("moved", "pix(img1)", "dev0")
    a.begin(1, 8)
    assert calls == ["img1", "img1"], "preprocessing happens in every begin, not once"


def test_begin_preload_uses_stored_tensors_without_copy():
    a = stub_with_inputs("preload")
    s = a.begin(0, 11)
    assert s.pixels is a._inputs[0]
    assert all(t.moved_to == [] for t in a._inputs)
    assert s.ckpts == [4, 8, 11]


def test_begin_before_make_inputs_raises_runtime_error():
    with pytest.raises(RuntimeError):
        VisionAdapter().begin(0, 1)


@pytest.mark.parametrize("idx", [-1, 2, 99, 1.0, True, None])
def test_begin_rejects_input_index_outside_inputs(idx):
    with pytest.raises(ValueError):
        stub_with_inputs("preload").begin(idx, 1)


@pytest.mark.parametrize("exit_", [0, -1, 25, 99, None, True])
def test_begin_rejects_exit_outside_contract(exit_):
    with pytest.raises(ValueError):
        stub_with_inputs("preload").begin(0, exit_)


def test_begin_rejects_exit_beyond_loaded_block_count():
    a = stub_with_inputs("preload", n_layers=12)
    a.begin(0, 12)
    with pytest.raises(ValueError):
        a.begin(0, 13)


# ---- finish

def test_finish_none_is_a_noop_without_torch():
    out = run_poisoned(
        "from exp2.serve.fam_vision import VisionAdapter, VisionState\n"
        "assert VisionAdapter().finish(VisionState(5, None, [4, 5]), None) is None\n"
        "print('ok')\n")
    assert out == "ok"


def test_finish_runs_the_head_on_the_hidden_state_of_reached_only():
    a = StubAdapter()
    s = new_state(11)
    a.advance(s, 4)
    a.advance(s, 8)
    a.events.clear()
    assert a.finish(s, 4) == "class@4"
    assert a.events == [("head", 4, "h4")]
    assert a.finish(s, 8) == "class@8"
    assert a.finish(s, None) is None
    assert a.events == [("head", 4, "h4"), ("head", 8, "h8")]


@pytest.mark.parametrize("reached", [0, 3, 4, 11, 12, -1, True, 8.0, "8"])
def test_finish_rejects_reached_that_was_not_advanced_to(reached):
    a = StubAdapter()
    s = new_state(11)
    a.advance(s, 8)
    a.events.clear()
    with pytest.raises(ValueError):
        a.finish(s, reached)
    assert a.events == []


# ---- sync / load / make_inputs argument checks

def test_sync_before_load_raises_and_after_load_calls_sync_fn():
    a = VisionAdapter()
    with pytest.raises(RuntimeError):
        a.sync()
    calls = []
    a._sync = lambda: calls.append(1)
    a.sync()
    assert calls == [1]


@pytest.mark.parametrize("exits", [[], [0], [25], [-1], [4, 25]])
def test_load_validates_exits_before_building_the_model(exits):
    a = VisionAdapter()

    def no_build(cfg):
        raise AssertionError("model must not be built for invalid exits")

    a._build_model = no_build
    with pytest.raises(ValueError):
        a.load(exits, {})


def test_load_rejects_unknown_preprocess_before_building_the_model():
    a = VisionAdapter()
    a._build_model = lambda cfg: (_ for _ in ()).throw(AssertionError("must not build"))
    with pytest.raises(ValueError):
        a.load([4], {"preprocess": "bogus"})


@pytest.mark.parametrize("n,mode", [(0, "synthetic"), (-2, "synthetic"), (2, "bogus"), (2, None), (1.5, "synthetic"),
                                    (True, "synthetic")])
def test_make_inputs_rejects_bad_arguments(n, mode):
    with pytest.raises(ValueError):
        VisionAdapter().make_inputs(n, 0, mode)


def test_make_inputs_before_load_raises_runtime_error():
    with pytest.raises(RuntimeError):
        VisionAdapter().make_inputs(2, 0, "synthetic")


def test_mem_helpers_delegate_to_families(monkeypatch):
    monkeypatch.setattr(families, "mem_stats", lambda: {"sentinel": 1})
    seen = []
    monkeypatch.setattr(families, "reset_peak", lambda: seen.append(1))
    a = VisionAdapter()
    assert a.mem_stats() == {"sentinel": 1}
    a.reset_peak()
    assert seen == [1]


# ================================================================ part 2: tiny random ViT (torch)

TINY = dict(hidden_size=32, num_hidden_layers=8, num_attention_heads=4, intermediate_size=64, image_size=32,
            patch_size=8)
LABELS = 10
IMG = TINY["image_size"]
ATOL = 1e-5


@pytest.fixture
def torch_mod():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return torch


def tiny_model():
    """(backbone, heads) like fam_vision._load_model returns: eval, frozen, one head per block."""
    import torch
    from transformers import ViTConfig, ViTModel

    torch.manual_seed(0)
    backbone = ViTModel(ViTConfig(**TINY), add_pooling_layer=False).eval()
    heads = torch.nn.ModuleList([torch.nn.Linear(TINY["hidden_size"], LABELS) for _ in range(TINY["num_hidden_layers"])])
    heads.eval()
    for p in list(backbone.parameters()) + list(heads.parameters()):
        p.requires_grad = False
    return backbone, heads


class TinyAdapter(VisionAdapter):
    """Real adapter logic on a random 8-block ViT (CPU, float32) instead of the 304M-parameter checkpoint."""

    def _build_model(self, cfg):
        return tiny_model()


def make_tiny(torch_mod, preprocess="per_request", n=2):
    a = TinyAdapter()
    a.load([4, 6, 8], {"preprocess": preprocess, "compile": False, "sync": "event"})
    a.make_inputs(n, 7, "synthetic")
    return a


@pytest.fixture
def tiny(torch_mod):
    return make_tiny(torch_mod)


def reference_hidden_states(adapter, pixels):
    """hidden_states of the unmodified backbone: (embeddings, after block 1, ..., after block 8)."""
    import torch

    with torch.no_grad():
        return adapter._backbone(pixel_values=pixels, output_hidden_states=True).hidden_states


def reference_logits(adapter, pixels, n_blocks):
    """heads[n_blocks - 1] on the CLS token after n_blocks blocks of the full forward."""
    import torch

    hs = reference_hidden_states(adapter, pixels)
    with torch.no_grad():
        return adapter._heads[n_blocks - 1](hs[n_blocks][:, 0])


@contextlib.contextmanager
def truncated(backbone, n_blocks):
    """Port of benchmark_trained_vit._exit_at_trained: expose only the first n_blocks blocks."""
    import torch

    holder, attr = (backbone, "layers") if hasattr(backbone, "layers") else (backbone.encoder, "layer")
    full = getattr(holder, attr)
    setattr(holder, attr, torch.nn.ModuleList(list(full)[:n_blocks]))
    try:
        yield
    finally:
        setattr(holder, attr, full)


def legacy_forward_exit(adapter, pixels, n_blocks):
    """Port of benchmark_trained_vit._trained_forward_exit (dropout is the identity in eval)."""
    import torch

    with torch.no_grad(), truncated(adapter._backbone, n_blocks):
        hs = adapter._backbone(pixel_values=pixels, output_hidden_states=True).hidden_states
    assert len(hs) == n_blocks + 1
    with torch.no_grad():
        return adapter._heads[n_blocks - 1](hs[-1][:, 0])


class TupleBlock:
    """A block that returns a tuple, like ViTLayer in transformers 4.x."""

    def __init__(self, block, extra=()):
        self.block, self.extra = block, extra

    def __call__(self, hidden):
        return (self.block(hidden), *self.extra)


def test_load_reports_blocks_and_no_compile(torch_mod):
    a = TinyAdapter()
    info = a.load([4], {"compile": False})
    assert info["n_layers"] == 8 and info["compiled_layers"] == 0 and info["load_s"] >= 0.0


def test_load_rejects_exit_beyond_the_loaded_model(torch_mod):
    with pytest.raises(ValueError):
        TinyAdapter().load([9], {})


def test_load_model_loads_the_backbone_once_and_never_a_classification_model(torch_mod, monkeypatch):
    import transformers

    calls = []

    def fake_backbone(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return tiny_model()[0]

    def forbidden(*args, **kwargs):
        raise AssertionError("a second model must not be loaded")

    monkeypatch.setattr(transformers.ViTModel, "from_pretrained", staticmethod(fake_backbone))
    for name in ("ViTForImageClassification", "AutoModel", "AutoModelForImageClassification"):
        monkeypatch.setattr(getattr(transformers, name), "from_pretrained", staticmethod(forbidden))
    backbone, heads = fam_vision._load_model("some/vit", LABELS)
    assert calls == [("some/vit", {"add_pooling_layer": False})]
    assert len(heads) == 8
    assert (heads[0].in_features, heads[0].out_features) == (TINY["hidden_size"], LABELS)
    assert not backbone.training and not heads.training
    assert not any(p.requires_grad for p in list(backbone.parameters()) + list(heads.parameters()))


def test_synthetic_pixels_have_the_model_image_shape(tiny, torch_mod):
    s = tiny.begin(0, 4)
    assert tuple(s.pixels.shape) == (1, 3, IMG, IMG)
    assert s.pixels.dtype == torch_mod.float32


def test_inputs_are_deterministic_per_seed_and_distinct(tiny, torch_mod):
    first = [tiny.begin(i, 4).pixels.clone() for i in (0, 1)]
    tiny.make_inputs(2, 7, "synthetic")
    again = [tiny.begin(i, 4).pixels for i in (0, 1)]
    assert all(torch_mod.equal(x, y) for x, y in zip(first, again))
    assert not torch_mod.equal(first[0], first[1])
    tiny.make_inputs(2, 8, "synthetic")
    assert not torch_mod.equal(first[0], tiny.begin(0, 4).pixels)


@pytest.mark.parametrize("exit_", range(1, 9))
def test_segmented_forward_matches_full_forward_at_every_checkpoint(tiny, torch_mod, exit_):
    s = tiny.begin(0, exit_)
    ckpts = tiny.checkpoints(exit_)
    hs = reference_hidden_states(tiny, s.pixels)
    for c in ckpts:
        tiny.advance(s, c)
        assert torch_mod.allclose(s.hiddens[c], hs[c], atol=ATOL, rtol=ATOL), (exit_, c)
    for c in ckpts:
        cls = tiny.finish(s, c)
        assert torch_mod.allclose(s.logits, reference_logits(tiny, s.pixels, c), atol=ATOL, rtol=ATOL), (exit_, c)
        assert cls == int(reference_logits(tiny, s.pixels, c).argmax(dim=-1).item())
    assert s.n_done == exit_ and sorted(s.hiddens) == ckpts


@pytest.mark.parametrize("exit_", range(1, 9))
def test_segmented_forward_matches_the_legacy_truncated_forward(tiny, torch_mod, exit_):
    s = tiny.begin(1, exit_)
    for c in tiny.checkpoints(exit_):
        tiny.advance(s, c)
    tiny.finish(s, exit_)
    assert torch_mod.allclose(s.logits, legacy_forward_exit(tiny, s.pixels, exit_), atol=ATOL, rtol=ATOL)
    assert len(tiny._backbone.layers if hasattr(tiny._backbone, "layers") else tiny._backbone.encoder.layer) == 8


def test_finish_reads_the_checkpoint_reached_not_the_latest_segment(tiny, torch_mod):
    s = tiny.begin(0, 8)
    tiny.advance(s, 4)
    tiny.advance(s, 8)
    tiny.finish(s, 4)
    early = reference_logits(tiny, s.pixels, 4)
    late = reference_logits(tiny, s.pixels, 8)
    assert torch_mod.allclose(s.logits, early, atol=ATOL, rtol=ATOL)
    assert not torch_mod.allclose(early, late, atol=1e-3), "guard: the two heads must give different logits"
    assert tiny.finish(s, None) is None


def test_one_call_equals_checkpoint_by_checkpoint(tiny, torch_mod):
    one = tiny.begin(0, 8)
    tiny.advance(one, 8)
    many = tiny.begin(0, 8)
    for c in (4, 8):
        tiny.advance(many, c)
    assert tiny.finish(one, 8) == tiny.finish(many, 8)
    assert torch_mod.equal(one.hiddens[8], many.hiddens[8])


def test_interleaved_states_do_not_share_hidden_state(tiny, torch_mod):
    a = tiny.begin(0, 8)
    b = tiny.begin(1, 6)
    tiny.advance(a, 4)
    tiny.advance(b, 4)
    tiny.advance(a, 8)
    tiny.advance(b, 6)
    assert torch_mod.allclose(a.hiddens[8], reference_hidden_states(tiny, a.pixels)[8], atol=ATOL, rtol=ATOL)
    assert torch_mod.allclose(b.hiddens[6], reference_hidden_states(tiny, b.pixels)[6], atol=ATOL, rtol=ATOL)
    assert not torch_mod.equal(a.hiddens[4], b.hiddens[4])


@pytest.mark.parametrize("extra", [(), (None,)], ids=["1-tuple", "2-tuple"])
def test_tuple_returning_blocks_give_the_same_result_as_tensor_returning_blocks(torch_mod, extra):
    plain = make_tiny(torch_mod)
    wrapped = make_tiny(torch_mod)
    wrapped._blocks = [TupleBlock(b, extra) for b in wrapped._blocks]
    p, w = plain.begin(0, 8), wrapped.begin(0, 8)
    for c in (4, 8):
        plain.advance(p, c)
        wrapped.advance(w, c)
    assert torch_mod.equal(p.hiddens[8], w.hiddens[8])
    assert plain.finish(p, 8) == wrapped.finish(w, 8)
    assert torch_mod.equal(p.logits, w.logits)


def test_the_model_runs_under_no_grad_even_when_parameters_require_grad(tiny):
    for p in list(tiny._backbone.parameters()) + list(tiny._heads.parameters()):
        p.requires_grad = True
    s = tiny.begin(0, 6)
    tiny.advance(s, 4)
    tiny.advance(s, 6)
    tiny.finish(s, 6)
    assert not s.hiddens[6].requires_grad and not s.logits.requires_grad


def test_block_failure_propagates_and_the_segment_can_be_resumed(tiny, torch_mod):
    class Flaky:
        def __init__(self, block):
            self.block, self.fail = block, True

        def __call__(self, hidden):
            if self.fail:
                self.fail = False
                raise RuntimeError("kernel launch failed")
            return self.block(hidden)

    blocks = list(tiny._blocks)
    blocks[2] = Flaky(blocks[2])
    tiny._blocks = blocks
    s = tiny.begin(0, 8)
    with pytest.raises(RuntimeError, match="kernel launch failed"):
        tiny.advance(s, 4)
    assert s.n_done == 2 and s.hiddens == {}
    tiny.advance(s, 4)
    tiny.advance(s, 8)
    assert torch_mod.allclose(s.hiddens[8], reference_hidden_states(tiny, s.pixels)[8], atol=ATOL, rtol=ATOL)


def test_compile_flag_wraps_each_block_once_and_the_result_is_unchanged(torch_mod, monkeypatch):
    wrapped = []

    def fake_compile(module):
        wrapped.append(module)
        return module

    monkeypatch.setattr(torch_mod, "compile", fake_compile)
    a = TinyAdapter()
    info = a.load([4], {"compile": True})
    assert info["compiled_layers"] == 8 and len(wrapped) == 8
    assert fam_vision._compile_per_block(a._blocks, False) == 0 and len(wrapped) == 8


# ---- dataset mode: raw images, a processor stub

class FakeProcessor:
    """Stands in for AutoImageProcessor: an 'image' is a float, the pixel tensor is filled with it."""

    def __init__(self, torch, shape=(1, 3, IMG, IMG)):
        self.torch, self.shape, self.calls = torch, shape, []

    def __call__(self, image, return_tensors=None):
        assert return_tensors == "pt"
        self.calls.append(image)
        return {"pixel_values": self.torch.full(self.shape, float(image))}


def make_dataset_adapter(torch_mod, preprocess, images=(1.0, 2.0), shape=(1, 3, IMG, IMG)):
    proc = FakeProcessor(torch_mod, shape)
    a = TinyAdapter()
    a._dataset_images = lambda n, seed: list(images)[:n]
    a._processor = lambda: proc
    a.load([4, 8], {"preprocess": preprocess})
    return a, proc


def test_dataset_per_request_preprocesses_in_begin_not_in_make_inputs(torch_mod):
    a, proc = make_dataset_adapter(torch_mod, "per_request")
    a.make_inputs(2, 0, "dataset")
    assert proc.calls == []
    s = a.begin(1, 4)
    assert proc.calls == [2.0]
    assert torch_mod.equal(s.pixels, torch_mod.full((1, 3, IMG, IMG), 2.0))
    a.begin(1, 4)
    assert proc.calls == [2.0, 2.0]


def test_dataset_preload_preprocesses_everything_once_in_make_inputs(torch_mod):
    a, proc = make_dataset_adapter(torch_mod, "preload")
    a.make_inputs(2, 0, "dataset")
    assert proc.calls == [1.0, 2.0]
    s = a.begin(0, 4)
    a.begin(1, 4)
    assert proc.calls == [1.0, 2.0]
    assert torch_mod.equal(s.pixels, torch_mod.full((1, 3, IMG, IMG), 1.0))


def test_dataset_preload_rejects_pixels_of_the_wrong_shape(torch_mod):
    a, _ = make_dataset_adapter(torch_mod, "preload", shape=(1, 3, IMG - 8, IMG - 8))
    with pytest.raises(ValueError):
        a.make_inputs(2, 0, "dataset")


def test_dataset_rejects_a_wrong_number_of_images(torch_mod):
    a, _ = make_dataset_adapter(torch_mod, "per_request", images=(1.0,))
    with pytest.raises(ValueError):
        a.make_inputs(2, 0, "dataset")


def test_switching_back_to_synthetic_drops_the_dataset_preprocessing(torch_mod):
    a, proc = make_dataset_adapter(torch_mod, "per_request")
    a.make_inputs(2, 0, "dataset")
    a.make_inputs(2, 0, "synthetic")
    a.begin(0, 4)
    assert proc.calls == []


def test_begin_rejects_exit_beyond_tiny_model(tiny):
    with pytest.raises(ValueError):
        tiny.begin(0, 9)


def test_sync_and_memory_helpers_run_on_cpu(tiny):
    tiny.sync()
    assert set(tiny.mem_stats()) == {"torch_reserved_mb", "torch_peak_alloc_mb", "rss_mb"}
    tiny.reset_peak()
