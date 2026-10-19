"""Tests for exp2/serve/fam_yolo.py.

Part 1 needs no torch. A FakeNet (28 plain-Python modules wired like gelan-m-ee.yaml) stands in for the model, so the
resumable module loop is checked against a port of the legacy restart-from-zero `_forward_to_exit`:
  - every backbone module runs once, in order, and never restarts;
  - the head of `reached` sees exactly the saved maps of its own exit, even after later checkpoints ran;
  - checkpoint, argument, completion and import-hygiene checks.
Part 2 builds a tiny conv network with the same wiring on CPU (skipped without torch).
Part 3 runs the real EarlyExitModel from the yaml at 128 px (skipped without torch or the yolov9 packages).
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
from exp2.serve import families, fam_yolo
from exp2.serve.fam_yolo import YoloAdapter, YoloState

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_DEPTH = {0: 7, 1: 8, 2: 9, 3: 15, 4: 18, 5: 21}  # gelan-m-ee.yaml, written out so a contract drift is caught
HEAD = 22
N_MODULES = 28
HEAD_F = [[4, 6, 7], [4, 6, 8], [4, 6, 9], [15, 12, 9], [15, 18, 9], [15, 18, 21]]
CAT_F = {11: [-1, 6], 14: [-1, 4], 17: [-1, 12], 20: [-1, 9]}


# ================================================================ fakes shared by part 1 and the references

class FakeMod:
    """Backbone: output = (i, input), so equal outputs mean equal data flow. Head: mimics DDetect."""

    def __init__(self, net, i, f):
        self.net, self.i, self.f = net, i, f

    def __call__(self, x):
        self.net.log.append(self.i)
        if self.net.fail_at == self.i:
            raise RuntimeError("boom")
        if self.i < HEAD:
            return (self.i, tuple(x) if isinstance(x, list) else x)
        inputs = tuple(x)
        for k in range(len(x)):  # DDetect replaces the entries of the list it is given
            x[k] = ("mutated", self.i, k)
        return (("pred", self.i, inputs), x)


class FakeNet:
    def __init__(self, n_modules=N_MODULES):
        self.log = []
        self.fail_at = None
        f = [CAT_F.get(i, -1) for i in range(HEAD)] + HEAD_F
        self.model = [FakeMod(self, i, f[i]) for i in range(min(n_modules, len(f)))]
        self.save = sorted({j for fi in f for j in ([fi] if isinstance(fi, int) else fi) if j != -1})

    def _resolve_input(self, m, x, y):  # same as early_exit.model.EarlyExitModel._resolve_input
        if m.f == -1:
            return x
        if isinstance(m.f, int):
            return y[m.f]
        return [x if j == -1 else y[j] for j in m.f]

    def parameters(self):
        yield SimpleNamespace(device="dev0")


def reference_forward(net, x, force_exit):
    """Port of the legacy `_forward_to_exit` (AnyTimeYolo/src/benchmark.py:154-179): restarts at module 0."""
    max_d = MAX_DEPTH[force_exit]
    head_i = HEAD + force_exit
    y, out = [], x
    for m in net.model:
        if m.i > max_d and m.i != head_i:
            y.append(None)
            continue
        x_in = net._resolve_input(m, out, y)
        out = m(x_in)
        y.append(out if m.i in net.save else None)
    return out


class StubAdapter(YoloAdapter):
    """No torch: grad context and the host/device helpers are replaced, the module loop is the real code."""

    def __init__(self, net=None):
        super().__init__()
        self._model = net if net is not None else FakeNet()
        self._save = frozenset(self._model.save)
        self.converted = []

    def _no_grad(self):
        return contextlib.nullcontext()

    def _summarize(self, out):
        return out

    def _to_input(self, raw):
        self.converted.append(raw)
        return ("input", raw)


def ready(preprocess="per_request", net=None):
    a = StubAdapter(net)
    a._inputs = ["img0", "img1"]
    a._preprocess = preprocess
    a._device = "dev0"
    return a


X0 = ("input", "img0")


def run_poisoned(code):
    prelude = "import sys\nfor m in ('torch', 'transformers', 'cv2', 'datasets'):\n    sys.modules[m] = None\n"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, "-c", prelude + code], cwd=str(REPO_ROOT), env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def method_tree(fn):
    return ast.parse(textwrap.dedent(inspect.getsource(fn)))


# ================================================================ part 1: no torch

def test_family_and_checkpoints_match_contract_and_spec():
    a = YoloAdapter()
    assert a.family == "yolo"
    for exit_ in range(6):
        ck = a.checkpoints(exit_)
        assert ck == contracts.default_checkpoints("yolo", exit_)
        assert ck == list(range(0, exit_ + 1))


@pytest.mark.parametrize("bad", [-1, 6, 1.5, None, True, "3"])
def test_checkpoints_reject_exit_outside_contract_range(bad):
    with pytest.raises(ValueError):
        YoloAdapter().checkpoints(bad)


def test_adapter_satisfies_family_adapter_protocol():
    assert isinstance(YoloAdapter(), contracts.FamilyAdapter)
    assert isinstance(families.get_adapter("yolo", {}), contracts.FamilyAdapter)
    assert type(families.get_adapter("yolo", {})) is YoloAdapter


def test_constants_follow_contract_and_yaml():
    assert fam_yolo.YOLO_EXIT_MAX_DEPTH == contracts.YOLO_EXIT_MAX_DEPTH == MAX_DEPTH
    assert fam_yolo.HEAD_OFFSET == HEAD
    assert fam_yolo.IMG_SIZE == 640
    assert fam_yolo.EE_YAML.name == "gelan-m-ee.yaml" and fam_yolo.EE_YAML.is_file()


def test_module_and_adapter_work_when_torch_cannot_be_imported():
    out = run_poisoned(
        "from exp2.serve import families\n"
        "from exp2.serve.fam_yolo import YoloState\n"
        "a = families.get_adapter('yolo', {})\n"
        "assert type(a).__name__ == 'YoloAdapter'\n"
        "assert a.checkpoints(3) == [0, 1, 2, 3]\n"
        "assert a.finish(YoloState(exit=0, out=None), None) is None\n"
        "print('ok')\n")
    assert out == "ok"


def test_importing_the_module_does_not_load_heavy_packages():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    code = ("import sys\nimport exp2.serve.fam_yolo\n"
            "print(sorted(m for m in ('torch', 'cv2', 'transformers') if m in sys.modules))\n")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT), env=env, capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]"


def test_no_module_level_heavy_import_and_no_ported_module_import():
    tree = ast.parse(Path(fam_yolo.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            assert not [n for n in names if n and n.split(".")[0] in ("torch", "cv2", "transformers")], names
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not [n for n in names if n.split(".")[0] == "shared" or "benchmark" in n], names


def test_hot_path_has_no_try_block_so_no_silent_fallback():
    for fn in (YoloAdapter.advance, YoloAdapter.finish, YoloAdapter.begin, YoloAdapter._to_input,
               YoloAdapter._summarize, fam_yolo._compile_backbone):
        tree = method_tree(fn)
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.Try)], fn.__name__


def test_advance_and_finish_import_nothing_themselves():
    for fn in (YoloAdapter.advance, YoloAdapter.finish, YoloAdapter.begin):
        tree = method_tree(fn)
        assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))], fn.__name__


# ---- advance: resumable module loop

def test_every_backbone_module_runs_once_in_order_and_never_restarts():
    a = ready()
    s = a.begin(0, 5)
    for ck in range(6):
        a.advance(s, ck)
    assert a._model.log == list(range(22))


@pytest.mark.parametrize("exit_", range(6))
def test_advance_through_exit_checkpoints_runs_exactly_up_to_max_depth_and_no_head(exit_):
    a = ready()
    s = a.begin(0, exit_)
    for ck in a.checkpoints(exit_):
        a.advance(s, ck)
        assert a._model.log == list(range(MAX_DEPTH[ck] + 1)), (exit_, ck)
        assert s.ckpt == ck
    assert all(i < HEAD for i in a._model.log)


@pytest.mark.parametrize("ck", range(6))
def test_modules_run_per_checkpoint_equal_contract_depth(ck):
    a = ready()
    s = a.begin(0, 5)
    a.advance(s, ck)
    assert len(s.y) == contracts.depth("yolo", ck)
    assert contracts.completion("yolo", 5, ck) == len(s.y) / 22


def test_one_call_to_the_last_checkpoint_equals_the_no_checkpoints_mode():
    a = ready()
    s = a.begin(0, 5)
    a.advance(s, 5)
    assert a._model.log == list(range(22)) and s.ckpt == 5
    assert a.finish(s, 5) == reference_forward(FakeNet(), X0, 5)


def test_saved_maps_are_only_the_save_list_and_aligned_with_module_index():
    a = ready()
    s = a.begin(0, 5)
    a.advance(s, 5)
    net = a._model
    assert len(s.y) == 22
    for i, v in enumerate(s.y):
        assert (v is not None) == (i in net.save), i
        if v is not None:
            assert v[0] == i


SCHEDULES = [[0, 1, 2, 3, 4, 5], [2, 5], [5], [0, 5], [1, 3, 4], [3]]


@pytest.mark.parametrize("schedule", SCHEDULES, ids=str)
def test_finish_equals_reference_forward_for_every_reached_after_every_advance(schedule):
    a = ready()
    s = a.begin(0, 5)
    for ck in schedule:
        a.advance(s, ck)
        for reached in range(ck + 1):
            assert a.finish(s, reached) == reference_forward(FakeNet(), X0, reached), (schedule, ck, reached)


def test_state_at_last_ontime_checkpoint_survives_later_checkpoints():
    a = ready()
    s = a.begin(0, 5)
    a.advance(s, 2)
    snap = list(s.y)
    before = a.finish(s, 2)
    a.advance(s, 3)
    a.advance(s, 5)
    assert all(s.y[i] is snap[i] for i in range(len(snap)))
    assert a.finish(s, 2) == before == reference_forward(FakeNet(), X0, 2)


@pytest.mark.parametrize("reached", range(6))
def test_finish_runs_only_the_head_of_reached_on_its_three_inputs(reached):
    a = ready()
    s = a.begin(0, 5)
    a.advance(s, 5)
    a._model.log.clear()
    pred, _feats = a.finish(s, reached)
    assert a._model.log == [HEAD + reached]
    assert pred[:2] == ("pred", HEAD + reached)
    assert len(pred[2]) == 3
    assert pred[2] == tuple(s.y[j] for j in HEAD_F[reached])


def test_finish_does_not_touch_saved_maps_and_is_repeatable():
    a = ready()
    s = a.begin(0, 5)
    a.advance(s, 5)
    snap, out = list(s.y), s.out
    first = a.finish(s, 5)
    second = a.finish(s, 5)
    assert first == second
    assert all(s.y[i] is snap[i] for i in range(len(snap))) and s.out is out


def test_finish_none_is_a_noop_that_runs_nothing():
    a = ready()
    s = a.begin(0, 2)
    assert a.finish(s, None) is None
    a.advance(s, 2)
    a._model.log.clear()
    assert a.finish(s, None) is None
    assert a._model.log == []


@pytest.mark.parametrize("reached", [-1, 3, 6, True, 1.0, "2"])
def test_finish_rejects_reached_beyond_advanced_or_not_an_int(reached):
    a = ready()
    s = a.begin(0, 5)
    a.advance(s, 2)
    a._model.log.clear()
    with pytest.raises(ValueError):
        a.finish(s, reached)
    assert a._model.log == []


def test_finish_on_a_state_that_never_advanced_rejects_every_reached():
    a = ready()
    s = a.begin(0, 5)
    for reached in range(6):
        with pytest.raises(ValueError):
            a.finish(s, reached)


@pytest.mark.parametrize("ckpt", [-1, 6, 100, 2.0, True, None, "3"])
def test_advance_rejects_bad_ckpt_without_running_anything(ckpt):
    a = ready()
    s = a.begin(0, 5)
    with pytest.raises(ValueError):
        a.advance(s, ckpt)
    assert a._model.log == [] and s.y == [] and s.ckpt == -1


def test_advance_rejects_ckpt_beyond_the_exit_of_the_request():
    a = ready()
    s = a.begin(0, 2)
    with pytest.raises(ValueError):
        a.advance(s, 3)
    assert a._model.log == []


@pytest.mark.parametrize("ckpt", [3, 2, 0])
def test_advance_rejects_repeated_or_backward_ckpt(ckpt):
    a = ready()
    s = a.begin(0, 5)
    a.advance(s, 3)
    a._model.log.clear()
    with pytest.raises(ValueError):
        a.advance(s, ckpt)
    assert a._model.log == [] and s.ckpt == 3 and len(s.y) == 16


def test_error_mid_checkpoint_propagates_and_resume_continues_without_restart():
    a = ready()
    net = a._model
    s = a.begin(0, 3)
    a.advance(s, 2)
    net.fail_at = 13
    net.log.clear()
    with pytest.raises(RuntimeError, match="boom"):
        a.advance(s, 3)
    assert net.log == [10, 11, 12, 13]
    assert len(s.y) == 13 and s.ckpt == 2
    assert a.finish(s, 2) == reference_forward(FakeNet(), X0, 2)
    net.fail_at = None
    net.log.clear()
    a.advance(s, 3)
    assert net.log == [13, 14, 15]
    assert a.finish(s, 3) == reference_forward(FakeNet(), X0, 3)


def test_advance_before_load_raises_runtime_error():
    a = YoloAdapter()
    with pytest.raises(RuntimeError):
        a.advance(YoloState(exit=2, out=None), 0)


def test_interleaved_requests_do_not_share_state():
    a = ready()
    s1 = a.begin(0, 5)
    s2 = a.begin(1, 3)
    a.advance(s1, 1)
    a.advance(s2, 3)
    a.advance(s1, 5)
    assert a.finish(s1, 5) == reference_forward(FakeNet(), X0, 5)
    assert a.finish(s2, 3) == reference_forward(FakeNet(), ("input", "img1"), 3)


# ---- begin

def test_begin_per_request_converts_the_stored_image_on_every_call():
    a = ready("per_request")
    s = a.begin(1, 4)
    a.begin(1, 4)
    assert a.converted == ["img1", "img1"]
    assert s.out == ("input", "img1")
    assert (s.exit, s.ckpt, s.y) == (4, -1, [])


def test_begin_preload_uses_the_stored_input_without_conversion():
    a = ready("preload")
    s = a.begin(0, 2)
    assert s.out is a._inputs[0]
    assert a.converted == []


def test_begin_before_make_inputs_raises_runtime_error():
    with pytest.raises(RuntimeError):
        YoloAdapter().begin(0, 0)


@pytest.mark.parametrize("idx", [-1, 2, 99, 1.0, True, None])
def test_begin_rejects_input_index_outside_inputs(idx):
    with pytest.raises(ValueError):
        ready("preload").begin(idx, 0)


@pytest.mark.parametrize("exit_", [-1, 6, 99, 1.5, None])
def test_begin_rejects_exit_outside_contract(exit_):
    with pytest.raises(ValueError):
        ready("preload").begin(0, exit_)


# ---- load / make_inputs argument checks

class BuildStub(YoloAdapter):
    """load() with a FakeNet instead of the checkpoint; records that no torch is needed."""

    def __init__(self, net=None):
        super().__init__()
        self.net = net if net is not None else FakeNet()
        self.built_with = []

    def _build_model(self, cfg):
        self.built_with.append(cfg)
        self._load_report = {"missing_keys": 3, "unexpected_keys": 1}
        return self.net


def test_load_stores_model_and_reports(monkeypatch):
    modes = []
    monkeypatch.setattr(families, "make_sync", lambda mode: (lambda: modes.append(mode)))
    a = BuildStub()
    info = a.load([1, 5], {"preprocess": "preload", "sync": "spin", "img_size": 64, "compile": False})
    assert a._model is a.net and a._save == frozenset(a.net.save)
    assert (a._preprocess, a._img_size, a._device) == ("preload", 64, "dev0")
    assert info["n_modules"] == N_MODULES and info["compiled_modules"] == 0 and info["load_s"] >= 0.0
    assert (info["missing_keys"], info["unexpected_keys"]) == (3, 1)
    a.sync()
    assert modes == ["spin"]  # load() asked families.make_sync for the configured mode and sync() calls its result


def test_load_defaults_to_per_request_and_640(monkeypatch):
    monkeypatch.setattr(families, "make_sync", lambda mode: (lambda: None))
    a = BuildStub()
    a.load([0], {})
    assert (a._preprocess, a._img_size) == ("per_request", 640)


@pytest.mark.parametrize("exits", [[], [6], [-1], [3, 6], [1.5], [True]])
def test_load_validates_exits_before_building_the_model(exits):
    a = BuildStub()
    with pytest.raises(ValueError):
        a.load(exits, {})
    assert a.built_with == []


def test_load_rejects_unknown_preprocess_and_bad_img_size_before_building():
    a = BuildStub()
    for cfg in ({"preprocess": "bogus"}, {"img_size": 0}, {"img_size": 31}, {"img_size": 64.0}, {"img_size": True}):
        with pytest.raises(ValueError):
            a.load([3], cfg)
    assert a.built_with == []


def test_load_rejects_a_model_without_the_exit_heads_and_stays_unloaded():
    a = BuildStub(FakeNet(n_modules=25))  # heads 22..24 only: exit 3 needs module 25
    with pytest.raises(ValueError):
        a.load([3], {})
    assert a._model is None


def test_load_accepts_a_model_that_has_exactly_the_heads_needed(monkeypatch):
    monkeypatch.setattr(families, "make_sync", lambda mode: (lambda: None))
    BuildStub(FakeNet(n_modules=25)).load([2], {})


class FakeBatch:
    """Stand-in for an (n, 3, S, S) uint8 tensor: has .shape and slicing."""

    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, sl):
        return ("view", sl.start, sl.stop)


class InputStub(StubAdapter):
    def __init__(self, shape=None):
        super().__init__()
        self._img_size = 64
        self.shape = shape
        self.calls = []

    def _synthetic_images(self, n, seed):
        self.calls.append(("synthetic", n, seed))
        return FakeBatch(self.shape or (n, 3, 64, 64))

    def _dataset_images(self, n, seed):
        self.calls.append(("dataset", n, seed))
        return FakeBatch(self.shape or (n, 3, 64, 64))


def test_make_inputs_per_request_keeps_raw_views_and_does_not_convert():
    a = InputStub()
    a._preprocess = "per_request"
    a.make_inputs(2, 5, "synthetic")
    assert a.calls == [("synthetic", 2, 5)]
    assert a._inputs == [("view", 0, 1), ("view", 1, 2)]
    assert a.converted == []


def test_make_inputs_preload_converts_every_input_once_up_front():
    a = InputStub()
    a._preprocess = "preload"
    a.make_inputs(2, 9, "dataset")
    assert a.calls == [("dataset", 2, 9)]
    assert a.converted == [("view", 0, 1), ("view", 1, 2)]
    assert a._inputs == [("input", ("view", 0, 1)), ("input", ("view", 1, 2))]


@pytest.mark.parametrize("shape", [(2, 3, 64, 32), (2, 1, 64, 64), (3, 3, 64, 64), (2, 3, 640, 640)])
def test_make_inputs_rejects_wrong_image_batch_shape(shape):
    a = InputStub(shape)
    with pytest.raises(ValueError):
        a.make_inputs(2, 0, "synthetic")


@pytest.mark.parametrize("n,mode", [(0, "synthetic"), (-2, "synthetic"), (2, "bogus"), (2, None), (1.5, "synthetic"),
                                    (True, "synthetic")])
def test_make_inputs_rejects_bad_arguments(n, mode):
    with pytest.raises(ValueError):
        InputStub().make_inputs(n, 0, mode)


def test_make_inputs_before_load_raises_runtime_error():
    with pytest.raises(RuntimeError):
        YoloAdapter().make_inputs(2, 0, "synthetic")


def test_sync_before_load_raises_and_after_load_calls_sync_fn():
    a = YoloAdapter()
    with pytest.raises(RuntimeError):
        a.sync()
    calls = []
    a._sync = lambda: calls.append(1)
    a.sync()
    assert calls == [1]


def test_mem_helpers_delegate_to_families(monkeypatch):
    monkeypatch.setattr(families, "mem_stats", lambda: {"sentinel": 1})
    seen = []
    monkeypatch.setattr(families, "reset_peak", lambda: seen.append(1))
    a = YoloAdapter()
    assert a.mem_stats() == {"sentinel": 1}
    a.reset_peak()
    assert seen == [1]


def test_compile_disabled_touches_nothing_and_needs_no_torch():
    net = FakeNet()
    before = list(net.model)
    assert fam_yolo._compile_backbone(net, False) == 0
    assert net.model == before


# ---- import paths and dataset folder lookup

def test_ensure_import_paths_adds_src_and_yolov9_with_src_first(monkeypatch):
    monkeypatch.setattr(sys, "path", [p for p in sys.path if "AnyTimeYolo" not in str(p)])
    fam_yolo._ensure_import_paths()
    src = str(REPO_ROOT / "AnyTimeYolo" / "src")
    v9 = str(REPO_ROOT / "AnyTimeYolo" / "model" / "yolov9")
    assert sys.path.index(src) < sys.path.index(v9)
    n = len(sys.path)
    fam_yolo._ensure_import_paths()
    assert len(sys.path) == n


def _mk(base, rel):
    (base / rel).mkdir(parents=True, exist_ok=True)
    return base / rel


def test_val_path_prefers_val_then_known_names(tmp_path):
    _mk(tmp_path, "images/val2017")
    assert fam_yolo._val_path(tmp_path) == tmp_path / "images/val2017"
    _mk(tmp_path, "val")
    assert fam_yolo._val_path(tmp_path) == tmp_path / "val"


def test_val_path_finds_nested_roboflow_folder_and_falls_back_to_val(tmp_path):
    nested = _mk(tmp_path, "proj-3/valid/images")
    assert fam_yolo._val_path(tmp_path) == nested
    empty = tmp_path / "empty"
    empty.mkdir()
    assert fam_yolo._val_path(empty) == empty / "val"


# ================================================================ part 2: tiny network (torch)

@pytest.fixture
def torch_mod():
    return pytest.importorskip("torch")


SPEC = [("conv", -1, 2), ("conv", -1, 2), ("conv", -1, 1), ("conv", -1, 2), ("conv", -1, 1),   # 0-4  (4 = P3)
        ("conv", -1, 2), ("conv", -1, 1),                                                        # 5-6  (6 = P4)
        ("conv", -1, 2), ("conv", -1, 1), ("conv", -1, 1),                                       # 7-9  (8 = P5)
        ("up", -1, 0), ("cat", [-1, 6], 0), ("conv", -1, 1),                                     # 10-12
        ("up", -1, 0), ("cat", [-1, 4], 0), ("conv", -1, 1),                                     # 13-15
        ("conv", -1, 2), ("cat", [-1, 12], 0), ("conv", -1, 1),                                  # 16-18
        ("conv", -1, 2), ("cat", [-1, 9], 0), ("conv", -1, 1)]                                   # 19-21


def make_tiny_net(torch):
    """22 backbone modules + 6 heads wired like gelan-m-ee.yaml, 4 channels, input 64 px (maps 8, 4, 2)."""
    nn = torch.nn
    fn = torch.nn.functional

    class Conv(nn.Module):
        def __init__(self, i, cin, stride):
            super().__init__()
            self.i, self.f = i, -1
            self.conv = nn.Conv2d(cin, 4, 3, stride, 1)

        def forward(self, x):
            return fn.silu(self.conv(x))

    class Up(nn.Module):
        def __init__(self, i):
            super().__init__()
            self.i, self.f = i, -1

        def forward(self, x):
            return fn.interpolate(x, scale_factor=2, mode="nearest")

    class Cat(nn.Module):
        def __init__(self, i, f):
            super().__init__()
            self.i, self.f = i, f

        def forward(self, xs):
            return torch.cat(xs, 1)

    class Head(nn.Module):
        def __init__(self, i, f):
            super().__init__()
            self.i, self.f = i, f
            self.cv = nn.ModuleList(nn.Conv2d(4, 8, 1) for _ in range(3))

        def forward(self, x):  # like DDetect in eval mode: overwrites the list entries, returns (pred, feats)
            for k in range(3):
                x[k] = self.cv[k](x[k])
            return torch.cat([xi.flatten(2) for xi in x], 2), x

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            layers, cin = [], 3
            for i, (kind, f, stride) in enumerate(SPEC):
                if kind == "conv":
                    layers.append(Conv(i, cin, stride))
                    cin = 4
                elif kind == "up":
                    layers.append(Up(i))
                else:
                    layers.append(Cat(i, f))
                    cin += 4
            for k, f in enumerate(HEAD_F):
                layers.append(Head(HEAD + k, f))
            self.model = nn.Sequential(*layers)
            self.save = sorted({j for m in layers for j in ([m.f] if isinstance(m.f, int) else m.f) if j != -1})

        def _resolve_input(self, m, x, y):
            if m.f == -1:
                return x
            if isinstance(m.f, int):
                return y[m.f]
            return [x if j == -1 else y[j] for j in m.f]

    torch.manual_seed(0)
    return Net().eval()


class TinyAdapter(YoloAdapter):
    def _build_model(self, cfg):
        import torch

        return make_tiny_net(torch)


def tiny_adapter(preprocess="per_request", **cfg):
    a = TinyAdapter()
    a.load([0, 1, 3, 5], {"preprocess": preprocess, "compile": False, "sync": "event", "img_size": 64, **cfg})
    a.make_inputs(2, 7, "synthetic")
    return a


def close(torch, got, want):
    return torch.allclose(got, want, rtol=1e-5, atol=1e-6, equal_nan=True)


def check_head(torch, got, want, where):
    assert close(torch, got[0], want[0]), where
    assert len(got[1]) == len(want[1]) == 3
    assert all(close(torch, g, w) for g, w in zip(got[1], want[1])), where


def test_tiny_net_shapes_are_consistent(torch_mod):
    net = make_tiny_net(torch_mod)
    with torch_mod.no_grad():
        pred, feats = reference_forward(net, torch_mod.zeros(1, 3, 64, 64), 5)
    assert tuple(pred.shape) == (1, 8, 64 + 16 + 4) and len(feats) == 3


def test_load_reports_modules_and_no_compile(torch_mod):
    a = TinyAdapter()
    info = a.load([1], {"compile": False, "img_size": 64})
    assert info["n_modules"] == N_MODULES and info["compiled_modules"] == 0 and info["load_s"] >= 0.0
    assert str(a._device) == "cpu"


def test_compile_backbone_wraps_only_the_22_backbone_modules(torch_mod, monkeypatch):
    net = make_tiny_net(torch_mod)
    wrapped = []

    def fake_compile(m):
        wrapped.append(m.i)
        return m

    monkeypatch.setattr(torch_mod, "compile", fake_compile)
    assert fam_yolo._compile_backbone(net, True) == 22
    assert wrapped == list(range(22))


def test_synthetic_inputs_are_uint8_square_deterministic_and_distinct(torch_mod):
    a = tiny_adapter()
    first = [t.clone() for t in a._inputs]
    assert all(tuple(t.shape) == (1, 3, 64, 64) and t.dtype == torch_mod.uint8 for t in first)
    a.make_inputs(2, 7, "synthetic")
    assert all(torch_mod.equal(x, y) for x, y in zip(first, a._inputs))
    assert not torch_mod.equal(first[0], first[1])
    a.make_inputs(2, 8, "synthetic")
    assert not torch_mod.equal(first[0], a._inputs[0])


def test_default_synthetic_size_is_640(torch_mod):
    a = TinyAdapter()
    a.load([0], {"compile": False})
    a.make_inputs(1, 0, "synthetic")
    assert tuple(a._inputs[0].shape) == (1, 3, 640, 640)


def test_begin_per_request_gives_float_unit_range_input_on_device(torch_mod):
    a = tiny_adapter("per_request")
    raw = a._inputs[1]
    s = a.begin(1, 3)
    assert s.out.dtype == torch_mod.float32 and tuple(s.out.shape) == (1, 3, 64, 64)
    assert 0.0 <= float(s.out.min()) and float(s.out.max()) <= 1.0
    assert torch_mod.equal(s.out, raw.float() / 255.0)
    assert a._inputs[1] is raw and raw.dtype == torch_mod.uint8


def test_preload_stores_converted_inputs_and_begin_does_not_copy(torch_mod):
    a = tiny_adapter("preload")
    assert all(t.dtype == torch_mod.float32 for t in a._inputs)
    s = a.begin(0, 2)
    assert s.out is a._inputs[0]


def dataset_stub(torch, size):
    class Fake:
        def __len__(self):
            return 10

        def __getitem__(self, i):
            return torch.full((3, size, size), i, dtype=torch.uint8), None, f"im{i}", None

    return Fake()


def test_dataset_mode_samples_distinct_images_by_seed(torch_mod):
    a = tiny_adapter()
    a._open_dataset = lambda: dataset_stub(torch_mod, 64)
    a.make_inputs(4, 3, "dataset")
    ids = [int(t[0, 0, 0, 0]) for t in a._inputs]
    assert len(set(ids)) == 4 and all(0 <= i < 10 for i in ids)
    a.make_inputs(4, 3, "dataset")
    assert [int(t[0, 0, 0, 0]) for t in a._inputs] == ids
    a.make_inputs(4, 4, "dataset")
    assert [int(t[0, 0, 0, 0]) for t in a._inputs] != ids


def test_dataset_mode_fails_loudly_on_too_few_images_or_wrong_size(torch_mod):
    a = tiny_adapter()
    a._open_dataset = lambda: dataset_stub(torch_mod, 64)
    with pytest.raises(RuntimeError):
        a.make_inputs(11, 0, "dataset")
    a._open_dataset = lambda: dataset_stub(torch_mod, 32)
    with pytest.raises(ValueError):
        a.make_inputs(2, 0, "dataset")


@pytest.mark.parametrize("schedule", SCHEDULES, ids=str)
def test_segmented_heads_equal_reference_forward_per_checkpoint(torch_mod, schedule):
    a = tiny_adapter()
    a._summarize = lambda out: out
    s = a.begin(0, 5)
    x0 = s.out
    for ck in schedule:
        a.advance(s, ck)
        for reached in range(ck + 1):
            with torch_mod.no_grad():
                want = reference_forward(a._model, x0, reached)
            check_head(torch_mod, a.finish(s, reached), want, (schedule, ck, reached))


def test_default_finish_returns_one_python_float_of_the_best_class_score(torch_mod):
    a = tiny_adapter()
    s = a.begin(0, 5)
    x0 = s.out
    a.advance(s, 2)
    got = a.finish(s, 2)
    with torch_mod.no_grad():
        pred = reference_forward(a._model, x0, 2)[0]
    assert type(got) is float
    assert got == pytest.approx(float(pred[:, 4:, :].max()), rel=1e-5, abs=1e-6)
    assert a.finish(s, None) is None


def test_modules_run_once_and_heads_only_inside_finish(torch_mod):
    a = tiny_adapter()
    calls = []
    for m in a._model.model:
        m.register_forward_hook(lambda mod, inp, out: calls.append(mod.i))
    s = a.begin(0, 5)
    for ck in range(6):
        a.advance(s, ck)
    assert calls == list(range(22))
    a.finish(s, 3)
    a.finish(s, 0)
    assert calls == list(range(22)) + [HEAD + 3, HEAD + 0]


def test_saved_maps_are_not_modified_by_later_modules_or_heads(torch_mod):
    a = tiny_adapter()
    s = a.begin(0, 5)
    a.advance(s, 2)
    snap = {j: t.clone() for j, t in enumerate(s.y) if t is not None}
    ids = {j: id(s.y[j]) for j in snap}
    a.advance(s, 3)
    a.advance(s, 5)
    for r in range(6):
        a.finish(s, r)
    assert all(id(s.y[j]) == ids[j] and torch_mod.equal(s.y[j], snap[j]) for j in snap)


def test_interleaved_states_do_not_share_feature_maps(torch_mod):
    a = tiny_adapter()
    a._summarize = lambda out: out
    s1, s2 = a.begin(0, 5), a.begin(1, 3)
    x1, x2 = s1.out, s2.out
    a.advance(s1, 2)
    a.advance(s2, 3)
    a.advance(s1, 5)
    with torch_mod.no_grad():
        check_head(torch_mod, a.finish(s1, 5), reference_forward(a._model, x1, 5), "s1")
        check_head(torch_mod, a.finish(s2, 3), reference_forward(a._model, x2, 3), "s2")


def test_model_failure_propagates_without_fallback_and_state_stays_resumable(torch_mod, monkeypatch):
    a = tiny_adapter()
    a._summarize = lambda out: out
    s = a.begin(0, 5)
    x0 = s.out
    a.advance(s, 2)
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("module 12 failed")

    monkeypatch.setattr(a._model.model[12], "forward", boom)
    with pytest.raises(RuntimeError, match="module 12 failed"):
        a.advance(s, 3)
    assert calls == [1] and len(s.y) == 12 and s.ckpt == 2
    monkeypatch.undo()
    a.advance(s, 3)
    with torch_mod.no_grad():
        check_head(torch_mod, a.finish(s, 3), reference_forward(a._model, x0, 3), "resumed")


def test_sync_and_memory_helpers_run_on_cpu(torch_mod):
    a = tiny_adapter()
    a.sync()
    assert set(a.mem_stats()) == {"torch_reserved_mb", "torch_peak_alloc_mb", "rss_mb"}
    a.reset_peak()


# ================================================================ part 3: the real EarlyExitModel (torch + yolov9)

SHADOWED = ("models", "utils", "early_exit")  # top-level names the yolov9 sources import; other code may own them


@pytest.fixture
def real_net(torch_mod):
    before_path = list(sys.path)
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.split(".")[0] in SHADOWED}
    try:
        try:
            fam_yolo._ensure_import_paths()
            from early_exit.model import EarlyExitModel
        except ImportError as exc:
            pytest.skip(f"yolov9 sources need packages that are not installed: {exc}")
        torch_mod.manual_seed(0)
        yield EarlyExitModel(str(fam_yolo.EE_YAML), ch=3).eval()
    finally:
        sys.path[:] = before_path
        for k in [k for k in sys.modules if k.split(".")[0] in SHADOWED]:
            del sys.modules[k]
        sys.modules.update(saved)


def test_real_model_has_the_wiring_the_adapter_assumes(real_net):
    assert len(real_net.model) == N_MODULES
    assert [m.i for m in real_net.model] == list(range(N_MODULES))
    assert [list(real_net.model[HEAD + k].f) for k in range(6)] == HEAD_F
    assert set(j for k in range(6) for j in HEAD_F[k]) <= set(real_net.save)


def test_real_model_segmented_equals_reference_forward(torch_mod, real_net):
    a = YoloAdapter()
    a._model, a._save, a._device = real_net, frozenset(real_net.save), torch_mod.device("cpu")
    a._summarize = lambda out: out
    x0 = torch_mod.rand(1, 3, 128, 128, generator=torch_mod.Generator().manual_seed(1))
    s = YoloState(5, x0)
    for ck in range(6):
        a.advance(s, ck)
        with torch_mod.no_grad():
            want = reference_forward(real_net, x0, ck)
        got = a.finish(s, ck)
        assert close(torch_mod, got[0], want[0]), ck
    for reached in range(6):  # the state of every earlier checkpoint is still intact after the last one
        with torch_mod.no_grad():
            want = reference_forward(real_net, x0, reached)
        assert close(torch_mod, a.finish(s, reached)[0], want[0]), reached


def test_real_model_default_finish_is_a_python_float(torch_mod, real_net):
    a = YoloAdapter()
    a._model, a._save, a._device = real_net, frozenset(real_net.save), torch_mod.device("cpu")
    s = YoloState(2, torch_mod.rand(1, 3, 128, 128, generator=torch_mod.Generator().manual_seed(2)))
    a.advance(s, 2)
    got = a.finish(s, 2)
    assert type(got) is float and 0.0 <= got <= 1.0
