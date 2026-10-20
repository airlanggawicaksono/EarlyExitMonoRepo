"""Tests for exp2/serve/fam_bert.py (ElasticBERT-large adapter, K6).

Part 1 needs no torch: checkpoints, argument checks, advance/begin/finish orchestration (a stub adapter replaces
the three torch methods `_embed`, `_layer`, `_head`), the ported `_broadcast_pooler`, SST-2 file helpers,
import hygiene.
Part 2 builds a tiny random ElasticBERT on CPU and is skipped when torch or transformers is missing.
"""
import ast
import contextlib
import inspect
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from exp2 import contracts
from exp2.serve import families, fam_bert
from exp2.serve.fam_bert import BertAdapter, BertState

REPO_ROOT = Path(__file__).resolve().parents[2]
SEQ = 128


class StubAdapter(BertAdapter):
    """No torch: records the three model calls, can fail on a chosen layer."""

    def __init__(self, fail_at_layer=None):
        super().__init__()
        self.events = []
        self.fail_at_layer = fail_at_layer

    def _embed(self, state):
        self.events.append("embed")
        return "emb", "ext"

    def _layer(self, state, i):
        if self.fail_at_layer == i:
            self.events.append(("boom", i))
            raise RuntimeError("boom")
        self.events.append(("layer", i))
        return "h%d" % i

    def _head(self, state, reached):
        self.events.append(("head", reached, state.hiddens[reached]))
        return "class@%d" % reached


def new_state(exit_):
    return BertState(exit_, "ids", "mask", "types", contracts.default_checkpoints("bert", exit_))


class FakeTensor:
    def __init__(self, name):
        self.name = name
        self.moved_to = []

    def to(self, device):
        self.moved_to.append(device)
        return ("moved", self.name, device)


def stub_with_inputs(preprocess, n_layers=24):
    a = StubAdapter()
    a._inputs = [tuple(FakeTensor("%s%d" % (k, j)) for k in ("ids", "mask", "types")) for j in range(2)]
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
    a = BertAdapter()
    assert a.family == "bert"
    for exit_ in range(0, 24):
        ck = a.checkpoints(exit_)
        assert ck == contracts.default_checkpoints("bert", exit_)
        assert ck[-1] == exit_ and ck == sorted(set(ck))


def test_checkpoints_are_every_4th_layer_plus_the_target():
    a = BertAdapter()
    assert a.checkpoints(23) == [3, 7, 11, 15, 19, 23]
    assert a.checkpoints(5) == [3, 5]
    assert a.checkpoints(3) == [3]
    assert a.checkpoints(0) == [0]


@pytest.mark.parametrize("bad", [-1, 24, 1.5, None, True, "3"])
def test_checkpoints_reject_exit_outside_contract_range(bad):
    with pytest.raises(ValueError):
        BertAdapter().checkpoints(bad)


def test_adapter_satisfies_family_adapter_protocol():
    assert isinstance(BertAdapter(), contracts.FamilyAdapter)
    assert isinstance(families.get_adapter("bert", {}), contracts.FamilyAdapter)


def test_module_and_adapter_work_when_torch_cannot_be_imported():
    out = run_poisoned(
        "from exp2.serve import families\n"
        "a = families.get_adapter('bert', {})\n"
        "assert type(a).__name__ == 'BertAdapter'\n"
        "assert a.checkpoints(11) == [3, 7, 11]\n"
        "print('ok')\n")
    assert out == "ok"


def test_module_level_imports_are_light_and_not_benchmark_modules():
    tree = ast.parse(Path(fam_bert.__file__).read_text(encoding="utf-8"))
    top = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            top.append(node.module)
    assert top, "expected module level imports"
    assert not [m for m in top if m.split(".")[0] in ("torch", "transformers", "cv2", "datasets")], top
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                assert name.split(".")[0] not in ("shared", "AnyTimeBert"), name
                assert "benchmark" not in name, name


def test_reference_dir_holds_the_elasticbert_model_definition():
    assert (fam_bert.REFERENCE_DIR / "models" / "modeling_elasticbert.py").is_file()
    assert (fam_bert.REFERENCE_DIR / "models" / "configuration_elasticbert.py").is_file()


def test_elasticbert_helper_puts_reference_dir_on_path_once_and_returns_the_classes(monkeypatch):
    cfg_mod = types.ModuleType("models.configuration_elasticbert")
    cfg_mod.ElasticBertConfig = type("ElasticBertConfig", (), {})
    model_mod = types.ModuleType("models.modeling_elasticbert")
    model_mod.ElasticBertForSequenceClassification = type("ElasticBertForSequenceClassification", (), {})
    pkg = types.ModuleType("models")
    pkg.__path__ = []
    for name, mod in (("models", pkg), ("models.configuration_elasticbert", cfg_mod),
                      ("models.modeling_elasticbert", model_mod)):
        monkeypatch.setitem(sys.modules, name, mod)
    ref = str(fam_bert.REFERENCE_DIR)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != ref])
    got = fam_bert._elasticbert()
    assert got == (cfg_mod.ElasticBertConfig, model_mod.ElasticBertForSequenceClassification)
    assert sys.path[0] == ref
    fam_bert._elasticbert()
    assert sys.path.count(ref) == 1


# ---- advance: embeddings once, then layers up to the checkpoint

def test_advance_embeds_once_then_runs_layers_up_to_each_checkpoint():
    a = StubAdapter()
    s = new_state(11)
    a.advance(s, 3)
    assert a.events == ["embed", ("layer", 0), ("layer", 1), ("layer", 2), ("layer", 3)]
    a.events.clear()
    a.advance(s, 7)
    assert a.events == [("layer", 4), ("layer", 5), ("layer", 6), ("layer", 7)]
    assert (s.n_done, s.hidden, s.ext_mask) == (8, "h7", "ext")


def test_advance_keeps_the_hidden_state_of_every_checkpoint_it_reached():
    a = StubAdapter()
    s = new_state(11)
    a.advance(s, 3)
    a.advance(s, 7)
    a.advance(s, 11)
    assert s.hiddens == {3: "h3", 7: "h7", 11: "h11"}


def test_advance_straight_to_the_target_runs_every_layer_once_and_keeps_only_that_checkpoint():
    a = StubAdapter()
    s = new_state(23)
    a.advance(s, 23)
    assert a.events == ["embed"] + [("layer", i) for i in range(24)]
    assert list(s.hiddens) == [23]


@pytest.mark.parametrize("ckpt", [0, 1, 4, 5, 12, 24, -1, 100, 7.0, True, None, "7"])
def test_advance_rejects_a_value_that_is_not_a_checkpoint_of_the_exit_without_running_anything(ckpt):
    a = StubAdapter()
    s = new_state(11)
    with pytest.raises(ValueError):
        a.advance(s, ckpt)
    assert a.events == [] and s.n_done == 0 and s.hiddens == {}


@pytest.mark.parametrize("ckpt", [3, 7])
def test_advance_rejects_repeat_or_backward_checkpoint(ckpt):
    a = StubAdapter()
    s = new_state(11)
    a.advance(s, 7)
    a.events.clear()
    with pytest.raises(ValueError):
        a.advance(s, ckpt)
    assert a.events == [] and s.n_done == 8 and list(s.hiddens) == [7]


def test_advance_rejects_checkpoint_beyond_the_target_of_the_exit():
    a = StubAdapter()
    s = new_state(5)
    with pytest.raises(ValueError):
        a.advance(s, 7)
    assert a.events == []


def test_layer_error_propagates_once_and_state_stays_at_the_last_finished_layer():
    a = StubAdapter(fail_at_layer=5)
    s = new_state(11)
    a.advance(s, 3)
    a.events.clear()
    with pytest.raises(RuntimeError, match="boom"):
        a.advance(s, 7)
    assert a.events == [("layer", 4), ("boom", 5)]
    assert (s.n_done, s.hidden) == (5, "h4")
    assert list(s.hiddens) == [3]


def test_a_failed_segment_can_be_resumed_from_the_last_finished_layer():
    a = StubAdapter(fail_at_layer=5)
    s = new_state(11)
    with pytest.raises(RuntimeError):
        a.advance(s, 7)
    a.fail_at_layer = None
    a.events.clear()
    a.advance(s, 7)
    assert a.events == [("layer", 5), ("layer", 6), ("layer", 7)]
    assert s.hiddens == {7: "h7"}


@pytest.mark.parametrize("fn", [BertAdapter.advance, BertAdapter._embed, BertAdapter._layer, BertAdapter._head,
                                BertAdapter.finish], ids=lambda f: f.__name__)
def test_model_path_has_no_try_block_so_no_silent_fallback(fn):
    assert not has_try(fn)


# ---- begin: host-side work only for per_request

def test_begin_per_request_moves_ids_mask_and_types_to_the_device():
    a = stub_with_inputs("per_request")
    s = a.begin(1, 5)
    assert (s.ids, s.mask, s.types) == (("moved", "ids1", "dev0"), ("moved", "mask1", "dev0"),
                                        ("moved", "types1", "dev0"))
    assert a._inputs[0][0].moved_to == []
    assert (s.exit, s.n_done, s.hidden, s.hiddens) == (5, 0, None, {})
    assert s.ckpts == [3, 5]


def test_begin_preload_uses_stored_tensors_without_copy():
    a = stub_with_inputs("preload")
    s = a.begin(0, 11)
    assert (s.ids, s.mask, s.types) == a._inputs[0]
    assert all(t.moved_to == [] for t in a._inputs[0])
    assert s.ckpts == [3, 7, 11]


def test_begin_before_make_inputs_raises_runtime_error():
    with pytest.raises(RuntimeError):
        BertAdapter().begin(0, 0)


@pytest.mark.parametrize("idx", [-1, 2, 99, 1.0, True, None])
def test_begin_rejects_input_index_outside_inputs(idx):
    with pytest.raises(ValueError):
        stub_with_inputs("preload").begin(idx, 0)


@pytest.mark.parametrize("exit_", [-1, 24, 99, None, True])
def test_begin_rejects_exit_outside_contract(exit_):
    with pytest.raises(ValueError):
        stub_with_inputs("preload").begin(0, exit_)


def test_begin_rejects_exit_beyond_loaded_layer_count():
    a = stub_with_inputs("preload", n_layers=12)
    a.begin(0, 11)
    with pytest.raises(ValueError):
        a.begin(0, 12)


# ---- finish

def test_finish_none_is_a_noop_without_torch():
    out = run_poisoned(
        "from exp2.serve.fam_bert import BertAdapter, BertState\n"
        "assert BertAdapter().finish(BertState(5, None, None, None, [3, 5]), None) is None\n"
        "print('ok')\n")
    assert out == "ok"


def test_finish_runs_the_head_on_the_hidden_state_of_reached_only():
    a = StubAdapter()
    s = new_state(11)
    a.advance(s, 3)
    a.advance(s, 7)
    a.events.clear()
    assert a.finish(s, 3) == "class@3"
    assert a.events == [("head", 3, "h3")]
    assert a.finish(s, 7) == "class@7"
    assert a.finish(s, None) is None
    assert a.events == [("head", 3, "h3"), ("head", 7, "h7")]


@pytest.mark.parametrize("reached", [0, 3, 4, 11, 12, -1, True, 3.0, "7"])
def test_finish_rejects_reached_that_was_not_advanced_to(reached):
    a = StubAdapter()
    s = new_state(11)
    a.advance(s, 7)
    a.events.clear()
    with pytest.raises(ValueError):
        a.finish(s, reached)
    assert a.events == []


def test_finish_before_any_advance_raises():
    with pytest.raises(ValueError):
        StubAdapter().finish(new_state(5), 3)


# ---- sync / load / make_inputs argument checks

def test_sync_before_load_raises_and_after_load_calls_sync_fn():
    a = BertAdapter()
    with pytest.raises(RuntimeError):
        a.sync()
    calls = []
    a._sync = lambda: calls.append(1)
    a.sync()
    assert calls == [1]


@pytest.mark.parametrize("exits", [[], [24], [-1], [3, 24]])
def test_load_validates_exits_before_building_the_model(exits):
    a = BertAdapter()

    def no_build(cfg):
        raise AssertionError("model must not be built for invalid exits")

    a._build_model = no_build
    with pytest.raises(ValueError):
        a.load(exits, {})


def test_load_rejects_unknown_preprocess_before_building_the_model():
    a = BertAdapter()
    a._build_model = lambda cfg: (_ for _ in ()).throw(AssertionError("must not build"))
    with pytest.raises(ValueError):
        a.load([5], {"preprocess": "bogus"})


@pytest.mark.parametrize("n,mode", [(0, "synthetic"), (-2, "synthetic"), (2, "bogus"), (2, None), (1.5, "synthetic"),
                                    (True, "synthetic")])
def test_make_inputs_rejects_bad_arguments(n, mode):
    with pytest.raises(ValueError):
        BertAdapter().make_inputs(n, 0, mode)


def test_make_inputs_before_load_raises_runtime_error():
    with pytest.raises(RuntimeError):
        BertAdapter().make_inputs(2, 0, "synthetic")


class Batch:
    """Stands in for a (n, 128) tensor: records the device, slices to one row."""

    def __init__(self, shape, device=None):
        self.shape, self.device = shape, device

    def to(self, device):
        return Batch(self.shape, device)

    def __getitem__(self, sl):
        return (self.device, sl.start, sl.stop)


class BatchStub(StubAdapter):
    def __init__(self, preprocess, dataset_width=SEQ):
        super().__init__()
        self._model, self._device, self._preprocess = object(), "dev0", preprocess
        self.dataset_width = dataset_width
        self.asked = []

    def _synthetic_batch(self, n, seed):
        self.asked.append(("synthetic", n, seed))
        return Batch((n, SEQ)), Batch((n, SEQ)), Batch((n, SEQ))

    def _dataset_batch(self, n, seed):
        self.asked.append(("dataset", n, seed))
        return Batch((n, self.dataset_width)), Batch((n, SEQ)), Batch((n, SEQ))


def test_make_inputs_per_request_keeps_host_tensors_and_slices_one_row_each():
    a = BatchStub("per_request")
    a.make_inputs(3, 5, "synthetic")
    assert a.asked == [("synthetic", 3, 5)]
    assert a._inputs == [((None, i, i + 1),) * 3 for i in range(3)]


def test_make_inputs_preload_moves_every_tensor_to_the_device_once():
    a = BatchStub("preload")
    a.make_inputs(2, 0, "synthetic")
    assert a._inputs == [(("dev0", i, i + 1),) * 3 for i in range(2)]


def test_make_inputs_dataset_mode_uses_the_dataset_batch_and_checks_the_shape():
    a = BatchStub("per_request")
    a.make_inputs(2, 9, "dataset")
    assert a.asked == [("dataset", 2, 9)]
    bad = BatchStub("per_request", dataset_width=SEQ - 1)
    with pytest.raises(ValueError):
        bad.make_inputs(2, 9, "dataset")
    assert bad._inputs is None


def test_mem_helpers_delegate_to_families(monkeypatch):
    monkeypatch.setattr(families, "mem_stats", lambda: {"sentinel": 1})
    seen = []
    monkeypatch.setattr(families, "reset_peak", lambda: seen.append(1))
    a = BertAdapter()
    assert a.mem_stats() == {"sentinel": 1}
    a.reset_peak()
    assert seen == [1]


# ---- ported _broadcast_pooler (stub poolers, no torch)

class FakePooler:
    def __init__(self, config):
        self.config = config
        self.sd = {}

    def state_dict(self):
        return dict(self.sd)

    def load_state_dict(self, sd):
        self.sd = dict(sd)


def fake_model(n=4, src=3):
    src_pooler = FakePooler("cfg")
    src_pooler.sd = {"w": 1.5}
    pooler = [None] * n
    pooler[src] = src_pooler
    enc = SimpleNamespace(pooler=pooler, current_pooler_num=src)
    return SimpleNamespace(config="cfg", elasticbert=SimpleNamespace(encoder=enc)), enc


def test_broadcast_pooler_clones_the_trained_pooler_into_every_empty_slot():
    model, enc = fake_model()
    trained = enc.pooler[3]
    assert fam_bert._broadcast_pooler(model) == 3
    assert enc.pooler[3] is trained
    for i in (0, 1, 2):
        assert type(enc.pooler[i]) is FakePooler and enc.pooler[i] is not trained
        assert enc.pooler[i].config == "cfg" and enc.pooler[i].sd == {"w": 1.5}
    assert len({id(p) for p in enc.pooler}) == 4


def test_broadcast_pooler_keeps_slots_that_already_hold_a_pooler_and_is_idempotent():
    model, enc = fake_model()
    other = FakePooler("cfg")
    other.sd = {"w": 9.0}
    enc.pooler[0] = other
    assert fam_bert._broadcast_pooler(model) == 2
    assert enc.pooler[0] is other and other.sd == {"w": 9.0}
    assert fam_bert._broadcast_pooler(model) == 0


def test_broadcast_pooler_is_a_noop_without_a_trained_source():
    no_pooler = SimpleNamespace(config="cfg", elasticbert=SimpleNamespace(encoder=SimpleNamespace()))
    assert fam_bert._broadcast_pooler(no_pooler) == 0
    model, enc = fake_model()
    enc.current_pooler_num = None
    assert fam_bert._broadcast_pooler(model) == 0 and enc.pooler.count(None) == 3
    model, enc = fake_model()
    enc.pooler = None
    assert fam_bert._broadcast_pooler(model) == 0
    model, enc = fake_model()
    enc.pooler[3] = None
    assert fam_bert._broadcast_pooler(model) == 0


def test_check_poolers_requires_a_module_in_every_slot_that_a_loaded_exit_uses():
    model, enc = fake_model(n=4, src=3)
    with pytest.raises(RuntimeError):
        fam_bert._check_poolers(model, [3, 1])
    fam_bert._check_poolers(model, [3])
    fam_bert._broadcast_pooler(model)
    fam_bert._check_poolers(model, [0, 1, 2, 3])
    with pytest.raises(RuntimeError):
        fam_bert._check_poolers(model, [4])
    enc.pooler = None
    with pytest.raises(RuntimeError):
        fam_bert._check_poolers(model, [3])


# ---- per-layer compile (fake torch)

def test_compile_per_layer_wraps_every_layer_and_returns_the_count(monkeypatch):
    fake = types.ModuleType("torch")
    fake.compile = lambda m: ("compiled", m)
    monkeypatch.setitem(sys.modules, "torch", fake)
    enc = SimpleNamespace(layer=["l0", "l1", "l2"])
    assert fam_bert._compile_per_layer(enc, True) == 3
    assert enc.layer == [("compiled", "l0"), ("compiled", "l1"), ("compiled", "l2")]


def test_compile_per_layer_does_not_swallow_a_compile_error(monkeypatch):
    fake = types.ModuleType("torch")

    def boom(module):
        raise RuntimeError("triton mismatch")

    fake.compile = boom
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(RuntimeError, match="triton mismatch"):
        fam_bert._compile_per_layer(SimpleNamespace(layer=["l0"]), True)


def test_compile_per_layer_disabled_never_touches_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    enc = SimpleNamespace(layer=["l0"])
    assert fam_bert._compile_per_layer(enc, False) == 0
    assert enc.layer == ["l0"]


# ---- SST-2 file helpers (stdlib only)

def write_tsv(tmp_path, rows, header="sentence\tlabel"):
    p = tmp_path / "dev.tsv"
    p.write_text("\n".join([header] + rows) + "\n", encoding="utf-8")
    return p


def test_read_sst2_skips_header_and_keeps_sentences_with_quotes_intact(tmp_path):
    p = write_tsv(tmp_path, ['it \'s a "charming" film . \t1', "a sometimes tedious film . \t0", "", '" quoted \t0'])
    assert fam_bert.read_sst2_sentences(p) == ['it \'s a "charming" film . ', "a sometimes tedious film . ",
                                               '" quoted ']


def test_read_sst2_rejects_a_file_without_the_sentence_header_or_without_rows(tmp_path):
    with pytest.raises(ValueError):
        fam_bert.read_sst2_sentences(write_tsv(tmp_path, ["a\t1"], header="text\tlabel"))
    with pytest.raises(ValueError):
        fam_bert.read_sst2_sentences(write_tsv(tmp_path, []))


def test_read_sst2_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        fam_bert.read_sst2_sentences(tmp_path / "nope.tsv")


def test_default_sst2_path_points_into_the_repo_glue_data():
    assert fam_bert.SST2_PATH == REPO_ROOT / "AnyTimeBert" / "glue_data" / "SST-2" / "dev.tsv"


def test_pick_indices_is_deterministic_distinct_and_wraps():
    a = fam_bert.pick_indices(100, 10, 3)
    assert a == fam_bert.pick_indices(100, 10, 3)
    assert len(set(a)) == 10 and all(0 <= i < 100 for i in a)
    assert a != fam_bert.pick_indices(100, 10, 4)
    wrapped = fam_bert.pick_indices(4, 10, 0)
    assert len(wrapped) == 10 and set(wrapped) == {0, 1, 2, 3}
    assert wrapped[:4] == wrapped[4:8]


def test_pick_indices_rejects_an_empty_pool():
    with pytest.raises(ValueError):
        fam_bert.pick_indices(0, 3, 0)


# ================================================================ part 2: tiny random ElasticBERT (torch)

TINY = dict(vocab_size=64, hidden_size=16, num_hidden_layers=12, num_attention_heads=2, intermediate_size=32,
            max_output_layers=12, num_output_layers=1, max_position_embeddings=128, initializer_range=0.3)


@pytest.fixture
def torch_mod():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return torch


def tiny_model(seed=0):
    import torch

    config_cls, model_cls = fam_bert._elasticbert()
    torch.manual_seed(seed)
    return model_cls(config_cls(**TINY))


class TinyAdapter(BertAdapter):
    """Real adapter logic on a random 12-layer ElasticBERT (CPU, float32) instead of the large checkpoint."""

    def _build_model(self, cfg):
        import torch

        model = tiny_model()
        fam_bert._broadcast_pooler(model)
        torch.manual_seed(1)
        for pooler in model.elasticbert.encoder.pooler:  # distinct weights per slot: a wrong slot changes logits
            pooler.dense.weight.data.normal_(mean=0.0, std=0.2)
        return fam_bert._prepare(model)


@pytest.fixture
def tiny(torch_mod):
    a = TinyAdapter()
    a.load([5, 11], {"preprocess": "per_request", "compile": False, "sync": "event"})
    a.make_inputs(3, 7, "synthetic")
    return a


@contextlib.contextmanager
def exit_at(model, force_exit):
    """Copy of AnyTimeBert/src/benchmark.py:_exit_at, the full-forward reference for a truncated model."""
    from torch import nn

    enc = model.elasticbert.encoder
    full_layers, full_n, full_pooler_num = enc.layer, enc.num_hidden_layers, enc.current_pooler_num
    enc.layer = nn.ModuleList(list(full_layers)[: force_exit + 1])
    enc.num_hidden_layers = force_exit + 1
    enc.current_pooler_num = force_exit
    try:
        yield
    finally:
        enc.layer, enc.num_hidden_layers, enc.current_pooler_num = full_layers, full_n, full_pooler_num


def reference_logits(model, state, ckpt):
    import torch

    with torch.no_grad(), exit_at(model, ckpt):
        _, logits = model(input_ids=state.ids, attention_mask=state.mask, token_type_ids=state.types)
    return logits


def close(torch, a, b):
    return torch.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_load_reports_layers_and_no_compile(torch_mod):
    a = TinyAdapter()
    info = a.load([5], {"compile": False})
    assert info["n_layers"] == 12 and info["compiled_layers"] == 0 and info["load_s"] >= 0.0


def test_load_freezes_weights_and_puts_the_model_in_eval_mode(tiny):
    assert not tiny._model.training
    assert not any(p.requires_grad for p in tiny._model.parameters())


@pytest.mark.parametrize("exit_", [5, 11])
def test_every_checkpoint_matches_the_full_forward_reference(tiny, torch_mod, exit_):
    s = tiny.begin(0, exit_)
    seen = []
    for ckpt in tiny.checkpoints(exit_):
        tiny.advance(s, ckpt)
        cls = tiny.finish(s, ckpt)
        ref = reference_logits(tiny._model, s, ckpt)
        assert close(torch_mod, s.logits, ref), (exit_, ckpt)
        assert cls == int(ref.argmax(dim=-1).item())
        seen.append(ref)
    assert not close(torch_mod, seen[0], seen[-1]), "reference logits do not depend on the exit: vacuous test"


def test_jumping_straight_to_the_target_equals_stepping_through_the_checkpoints(tiny, torch_mod):
    one = tiny.begin(0, 11)
    tiny.advance(one, 11)
    tiny.finish(one, 11)
    many = tiny.begin(0, 11)
    for ckpt in (3, 7, 11):
        tiny.advance(many, ckpt)
    tiny.finish(many, 11)
    assert close(torch_mod, one.logits, many.logits)
    assert close(torch_mod, one.logits, reference_logits(tiny._model, one, 11))


def test_finish_after_a_late_segment_uses_the_last_on_time_checkpoint(tiny, torch_mod):
    s = tiny.begin(1, 11)
    tiny.advance(s, 3)
    tiny.advance(s, 7)  # this segment finished after the deadline: it still ran, the request reached 3
    tiny.finish(s, 3)
    assert close(torch_mod, s.logits, reference_logits(tiny._model, s, 3))
    assert not close(torch_mod, s.logits, reference_logits(tiny._model, s, 7))
    tiny.finish(s, 7)
    assert close(torch_mod, s.logits, reference_logits(tiny._model, s, 7))


def test_padding_mask_is_applied_like_the_model_does(tiny, torch_mod):
    s = tiny.begin(0, 11)
    s.mask = s.mask.clone()
    s.mask[:, 100:] = 0
    unmasked = reference_logits(tiny._model, tiny.begin(0, 11), 11)
    for ckpt in (3, 7, 11):
        tiny.advance(s, ckpt)
        tiny.finish(s, ckpt)
        assert close(torch_mod, s.logits, reference_logits(tiny._model, s, ckpt)), ckpt
    assert not close(torch_mod, s.logits, unmasked), "mask has no effect: vacuous test"


def test_interleaved_requests_do_not_share_state(tiny, torch_mod):
    a = tiny.begin(0, 11)
    b = tiny.begin(1, 5)
    tiny.advance(a, 3)
    tiny.advance(b, 3)
    tiny.advance(a, 7)
    tiny.advance(b, 5)
    tiny.advance(a, 11)
    tiny.finish(a, 11)
    tiny.finish(b, 5)
    assert close(torch_mod, a.logits, reference_logits(tiny._model, a, 11))
    assert close(torch_mod, b.logits, reference_logits(tiny._model, b, 5))


def test_no_autograd_graph_is_built(tiny):
    s = tiny.begin(0, 5)
    tiny.advance(s, 5)
    tiny.finish(s, 5)
    assert not s.hiddens[5].requires_grad and not s.logits.requires_grad


def test_layer_failure_propagates_and_state_stays_at_the_last_finished_layer(tiny, monkeypatch):
    s = tiny.begin(0, 11)
    tiny.advance(s, 3)

    def boom(*args, **kwargs):
        raise RuntimeError("kernel exploded")

    monkeypatch.setattr(tiny._model.elasticbert.encoder.layer[5], "forward", boom)
    with pytest.raises(RuntimeError, match="kernel exploded"):
        tiny.advance(s, 7)
    assert s.n_done == 5 and list(s.hiddens) == [3]


def test_begin_rejects_exit_beyond_tiny_model(tiny):
    with pytest.raises(ValueError):
        tiny.begin(0, 12)


def test_synthetic_inputs_have_the_benchmark_shape_dtype_and_value_ranges(tiny, torch_mod):
    for i in range(3):
        s = tiny.begin(i, 5)
        for t in (s.ids, s.mask, s.types):
            assert tuple(t.shape) == (1, SEQ) and t.dtype == torch_mod.int64
        assert 1 <= int(s.ids.min()) and int(s.ids.max()) < TINY["vocab_size"]
        assert bool((s.mask == 1).all()) and bool((s.types == 0).all())


def test_synthetic_inputs_are_deterministic_per_seed_and_distinct(tiny, torch_mod):
    first = [tiny.begin(i, 0).ids.clone() for i in range(3)]
    tiny.make_inputs(3, 7, "synthetic")
    assert all(torch_mod.equal(x, tiny.begin(i, 0).ids) for i, x in enumerate(first))
    assert not torch_mod.equal(first[0], first[1])
    tiny.make_inputs(3, 8, "synthetic")
    assert not torch_mod.equal(first[0], tiny.begin(0, 0).ids)


class FakeTokenizer:
    def __init__(self, torch_mod, width=SEQ):
        self.torch, self.width, self.calls = torch_mod, width, []

    def __call__(self, sentences, **kwargs):
        self.calls.append((list(sentences), kwargs))
        n = len(sentences)
        ids = (self.torch.arange(n * self.width).reshape(n, self.width) % 60) + 1
        return {"input_ids": ids, "attention_mask": self.torch.ones(n, self.width, dtype=self.torch.int64),
                "token_type_ids": self.torch.zeros(n, self.width, dtype=self.torch.int64)}


def test_dataset_mode_tokenizes_seeded_sst2_sentences_to_the_fixed_length(tiny, torch_mod, tmp_path):
    sentences = ["sentence %d ." % i for i in range(5)]
    tiny._sst2_path = write_tsv(tmp_path, ["%s \t1" % s for s in sentences])
    tok = FakeTokenizer(torch_mod)
    tiny._tokenizer = lambda: tok
    tiny.make_inputs(3, 4, "dataset")
    (got, kwargs), = tok.calls
    assert got == ["%s " % sentences[i] for i in fam_bert.pick_indices(5, 3, 4)]
    assert kwargs == {"max_length": SEQ, "padding": "max_length", "truncation": True, "return_tensors": "pt"}
    ids = tok(["a", "b", "c"])["input_ids"]
    assert torch_mod.equal(tiny.begin(1, 5).ids.cpu(), ids[1:2])   # begin() moves inputs to cuda on the board
    assert tuple(tiny.begin(2, 5).mask.shape) == (1, SEQ)


def test_dataset_mode_rejects_tokenizer_output_of_the_wrong_length(tiny, torch_mod, tmp_path):
    tiny._sst2_path = write_tsv(tmp_path, ["a sentence . \t1", "another one . \t0"])
    tiny._tokenizer = lambda: FakeTokenizer(torch_mod, width=SEQ - 1)
    with pytest.raises(ValueError):
        tiny.make_inputs(2, 0, "dataset")


def test_load_model_from_a_saved_checkpoint_fills_every_pooler_slot(torch_mod, tmp_path):
    model = tiny_model()
    assert model.elasticbert.encoder.pooler[0] is None  # one trained pooler at the last layer only
    model.save_pretrained(str(tmp_path))
    loaded = fam_bert._load_model(str(tmp_path), 2)
    enc = loaded.elasticbert.encoder
    assert len(enc.layer) == 12 and len(enc.pooler) == 12
    assert all(p is not None for p in enc.pooler)
    trained = enc.pooler[11].state_dict()
    for pooler in enc.pooler:
        for key, value in pooler.state_dict().items():
            assert torch_mod.equal(value, trained[key])
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    assert loaded.classifier.out_features == 2


def test_sync_and_memory_helpers_run_on_cpu(tiny):
    tiny.sync()
    assert set(tiny.mem_stats()) == {"torch_reserved_mb", "torch_peak_alloc_mb", "rss_mb"}
    tiny.reset_peak()
