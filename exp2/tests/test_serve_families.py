"""Tests for exp2/serve/families.py: adapter factory, compile setup, sync factory, memory helpers.

Torch is replaced by a recording fake in sys.modules, so none of these tests need a real torch.
A test with a real torch uses pytest.importorskip("torch") and is skipped when torch is absent.
"""
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from exp2.serve import families

REPO_ROOT = Path(__file__).resolve().parents[2]
THREADS_ENV = "TORCHINDUCTOR_COMPILE_THREADS"
FAMILY_CLASSES = [("bert", "BertAdapter"), ("vision", "VisionAdapter"), ("llama", "LlamaAdapter"),
                  ("yolo", "YoloAdapter")]


def make_fake_torch(monkeypatch, *, cuda=True, compile_error=None):
    """Install a recording fake torch (and torch._inductor.config) into sys.modules."""
    calls = {"events": [], "sync": 0, "compiled": 0, "reset_peak": 0}
    torch = types.ModuleType("torch")
    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.is_available = lambda: cuda

    def synchronize():
        calls["sync"] += 1

    class Event:
        def __init__(self, enable_timing=False, blocking=False, interprocess=False):
            self.blocking = blocking
            self.records = 0
            self.syncs = 0
            calls["events"].append(self)

        def record(self, stream=None):
            self.records += 1

        def synchronize(self):
            self.syncs += 1

    def reset_peak():
        calls["reset_peak"] += 1

    def compile_(fn=None, **kwargs):
        if compile_error is not None:
            raise compile_error
        calls["compiled"] += 1
        return fn if fn is not None else (lambda f: f)

    cuda_mod.synchronize = synchronize
    cuda_mod.Event = Event
    cuda_mod.memory_reserved = lambda: 3 * 2**20
    cuda_mod.max_memory_allocated = lambda: 2 * 2**20
    cuda_mod.reset_peak_memory_stats = reset_peak
    torch.cuda = cuda_mod
    torch.compile = compile_
    torch.randn = lambda *a, **k: 1.0
    inductor = types.ModuleType("torch._inductor")
    inductor_cfg = types.ModuleType("torch._inductor.config")
    inductor_cfg.compile_threads = 8
    inductor.config = inductor_cfg
    torch._inductor = inductor
    for name, mod in (("torch", torch), ("torch.cuda", cuda_mod), ("torch._inductor", inductor),
                      ("torch._inductor.config", inductor_cfg)):
        monkeypatch.setitem(sys.modules, name, mod)
    return calls, inductor_cfg


def run_poisoned(code):
    """Run code in a fresh interpreter where importing torch/transformers/cv2 raises ImportError."""
    prelude = "import sys\nfor m in ('torch', 'transformers', 'cv2', 'datasets'):\n    sys.modules[m] = None\n"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, "-c", prelude + code], cwd=str(REPO_ROOT), env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


# ---------------------------------------------------------------- get_adapter

def test_registry_matches_contract_families():
    from exp2 import contracts

    assert set(families.ADAPTERS) == set(contracts.FAMILIES)
    for fam, cls_name in FAMILY_CLASSES:
        assert families.ADAPTERS[fam] == (f"exp2.serve.fam_{fam}", cls_name)


@pytest.mark.parametrize("bad", ["gpt", "", "BERT", None, 3])
def test_get_adapter_rejects_unknown_family(bad):
    with pytest.raises(ValueError):
        families.get_adapter(bad, {})


@pytest.mark.parametrize("fam,cls_name", FAMILY_CLASSES)
def test_get_adapter_builds_class_from_family_module(monkeypatch, fam, cls_name):
    module = types.ModuleType(f"exp2.serve.fam_{fam}")
    marker = type(cls_name, (), {"family": fam})
    setattr(module, cls_name, marker)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    adapter = families.get_adapter(fam, {"compile": False})
    assert type(adapter) is marker


def test_get_adapter_accepts_missing_cfg(monkeypatch):
    module = types.ModuleType("exp2.serve.fam_bert")
    module.BertAdapter = type("BertAdapter", (), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert type(families.get_adapter("bert")).__name__ == "BertAdapter"


def test_get_adapter_imports_lazily_and_without_torch():
    out = run_poisoned(
        "import exp2.serve.families as f\n"
        "assert 'exp2.serve.fam_llama' not in sys.modules\n"
        "a = f.get_adapter('llama', {})\n"
        "assert a.family == 'llama', a.family\n"
        "for other in ('bert', 'vision', 'yolo'):\n"
        "    assert 'exp2.serve.fam_' + other not in sys.modules, other\n"
        "print('ok')\n")
    assert out == "ok"


# ---------------------------------------------------------------- compile smoke

def test_compile_smoke_true_when_compile_runs(monkeypatch):
    calls, _ = make_fake_torch(monkeypatch)
    assert families._compile_smoke_ok() is True
    assert calls["compiled"] == 1


def test_compile_smoke_false_and_logged_when_compile_raises(monkeypatch, capsys):
    make_fake_torch(monkeypatch, compile_error=RuntimeError("triton mismatch"))
    assert families._compile_smoke_ok() is False
    assert "RuntimeError" in capsys.readouterr().err


def test_compile_smoke_false_when_torch_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert families._compile_smoke_ok() is False


# ---------------------------------------------------------------- setup_compile

def test_setup_compile_off_skips_everything(monkeypatch):
    monkeypatch.setenv(THREADS_ENV, "8")

    def boom():
        raise AssertionError("smoke test must not run when compile is disabled")

    monkeypatch.setattr(families, "_compile_smoke_ok", boom)
    assert families.setup_compile(False) == "off"
    assert os.environ[THREADS_ENV] == "8"


def test_setup_compile_on_when_smoke_passes(monkeypatch):
    monkeypatch.setenv(THREADS_ENV, "8")
    monkeypatch.setattr(families, "_compile_smoke_ok", lambda: True)
    assert families.setup_compile(True) == "on"
    assert os.environ[THREADS_ENV] == "1"


def test_setup_compile_eager_fallback_when_smoke_fails(monkeypatch):
    monkeypatch.setenv(THREADS_ENV, "8")
    monkeypatch.setattr(families, "_compile_smoke_ok", lambda: False)
    assert families.setup_compile(True) == "eager_fallback"
    assert os.environ[THREADS_ENV] == "1"


def test_setup_compile_single_thread_inductor_before_smoke(monkeypatch):
    calls, inductor_cfg = make_fake_torch(monkeypatch)
    seen = {}
    real_smoke = families._compile_smoke_ok

    def spy():
        seen["threads_at_smoke"] = inductor_cfg.compile_threads
        return real_smoke()

    monkeypatch.setattr(families, "_compile_smoke_ok", spy)
    assert families.setup_compile(True) == "on"
    assert seen["threads_at_smoke"] == 1
    assert inductor_cfg.compile_threads == 1
    assert calls["compiled"] == 1


def test_setup_compile_without_torch_is_eager_fallback_not_crash(monkeypatch):
    monkeypatch.setenv(THREADS_ENV, "8")
    monkeypatch.setitem(sys.modules, "torch", None)
    assert families.setup_compile(True) == "eager_fallback"
    assert os.environ[THREADS_ENV] == "1"


# ---------------------------------------------------------------- make_sync

@pytest.mark.parametrize("bad", ["spinn", "", None, "EVENT"])
def test_make_sync_rejects_unknown_mode_before_touching_torch(monkeypatch, bad):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError):
        families.make_sync(bad)


def test_make_sync_event_uses_one_blocking_event(monkeypatch):
    calls, _ = make_fake_torch(monkeypatch)
    sync = families.make_sync("event")
    for _ in range(3):
        sync()
    assert len(calls["events"]) == 1
    event = calls["events"][0]
    assert event.blocking is True
    assert (event.records, event.syncs) == (3, 3)
    assert calls["sync"] == 0


def test_make_sync_spin_uses_device_synchronize_only(monkeypatch):
    calls, _ = make_fake_torch(monkeypatch)
    sync = families.make_sync("spin")
    for _ in range(3):
        sync()
    assert calls["sync"] == 3
    assert calls["events"] == []


@pytest.mark.parametrize("mode", ["event", "spin"])
def test_make_sync_is_noop_without_cuda(monkeypatch, mode):
    calls, _ = make_fake_torch(monkeypatch, cuda=False)
    sync = families.make_sync(mode)
    sync()
    assert calls["sync"] == 0
    assert calls["events"] == []


@pytest.mark.parametrize("mode", ["event", "spin"])
def test_make_sync_runs_with_real_torch(mode):
    pytest.importorskip("torch")
    families.make_sync(mode)()


# ---------------------------------------------------------------- memory helpers

def test_mem_stats_reports_mb_and_exact_keys(monkeypatch):
    make_fake_torch(monkeypatch)
    stats = families.mem_stats()
    assert set(stats) == {"torch_reserved_mb", "torch_peak_alloc_mb", "rss_mb"}
    assert stats["torch_reserved_mb"] == 3.0
    assert stats["torch_peak_alloc_mb"] == 2.0
    assert stats["rss_mb"] is None or stats["rss_mb"] > 0


def test_mem_stats_zero_torch_fields_without_cuda(monkeypatch):
    make_fake_torch(monkeypatch, cuda=False)
    stats = families.mem_stats()
    assert stats["torch_reserved_mb"] == 0.0
    assert stats["torch_peak_alloc_mb"] == 0.0


def test_reset_peak_calls_torch_only_with_cuda(monkeypatch):
    calls, _ = make_fake_torch(monkeypatch)
    families.reset_peak()
    assert calls["reset_peak"] == 1
    calls, _ = make_fake_torch(monkeypatch, cuda=False)
    families.reset_peak()
    assert calls["reset_peak"] == 0


def test_parse_vmrss_converts_kb_to_mb():
    assert families._parse_vmrss_mb("Name:\tpython\nVmRSS:\t 2097152 kB\nThreads:\t4\n") == 2048.0


def test_parse_vmrss_missing_returns_none():
    assert families._parse_vmrss_mb("Name:\tpython\nThreads:\t4\n") is None
