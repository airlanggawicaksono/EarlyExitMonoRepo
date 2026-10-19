"""Family adapter factory and the torch helpers every exp2.serve.fam_* module shares.

Adapter convention (all four families):
    module  exp2.serve.fam_<family>
    class   <Family>Adapter   (BertAdapter, VisionAdapter, LlamaAdapter, YoloAdapter)
    ctor    no arguments; the server passes cfg to adapter.load(exits, cfg)

torch is imported only inside functions, so importing this module never loads torch.
"""
import importlib
import os
import sys

THREADS_ENV = "TORCHINDUCTOR_COMPILE_THREADS"
SYNC_MODES = ("event", "spin")
_MB = float(2**20)

# family -> (module, class). Must equal contracts.FAMILIES (checked in test_serve_families).
ADAPTERS = {f: (f"exp2.serve.fam_{f}", f"{f.capitalize()}Adapter") for f in ("bert", "vision", "llama", "yolo")}


def get_adapter(family, cfg=None):
    """Return a fresh, unloaded adapter for `family`. Imports only that family's module.

    `cfg` is accepted so call sites can pass the same dict they later give to adapter.load();
    the adapter reads it in load(), not here.
    """
    if not isinstance(family, str) or family not in ADAPTERS:
        raise ValueError(f"unknown family {family!r}; expected one of {sorted(ADAPTERS)}")
    module_name, class_name = ADAPTERS[family]
    return getattr(importlib.import_module(module_name), class_name)()


# --- compile setup (ported from bench_jetson.py:135-182) ---------------------------------------

def _log(msg):
    print(f"[exp2.families] {msg}", file=sys.stderr)


def _force_single_thread_inductor():
    """Compile in-process so inductor never pickles its worker state.

    On Tegra the parallel async-compile pool raises `TypeError: cannot pickle '_thread.RLock'`
    when forking workers. One compile thread keeps torch.compile in-process.
    """
    os.environ[THREADS_ENV] = "1"
    try:
        import torch._inductor.config as ind

        ind.compile_threads = 1
    except Exception as e:  # torch missing or inductor absent: env var alone is still set
        _log(f"inductor single-thread setup skipped: {e}")


def _compile_smoke_ok():
    """Try to torch.compile a trivial kernel. True only if it runs end to end."""
    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"

        @torch.compile
        def _f(x):
            return x * 2 + 1

        _f(torch.randn(8, device=dev))
        return True
    except Exception as e:
        _log(f"compile smoke-test failed ({type(e).__name__}: {e})")
        return False


def setup_compile(enable):
    """Return "off" (disabled), "on" (smoke test passed) or "eager_fallback" (smoke test failed).

    The server writes the result to ready.versions.compile, so a hard fallback is never silent.
    """
    if not enable:
        return "off"
    _force_single_thread_inductor()
    if _compile_smoke_ok():
        _log("compile smoke-test OK -> compile ON (single-thread inductor)")
        return "on"
    _log("compile unavailable -> eager (fix: pin triton to the version matching torch)")
    return "eager_fallback"


# --- GPU sync ------------------------------------------------------------------------------------

def make_sync(mode):
    """Return a zero-argument callable that blocks until queued GPU work finishes.

    "event": torch.cuda.Event(blocking=True), the CPU thread sleeps while waiting.
    "spin":  torch.cuda.synchronize(), the CPU thread may spin.
    Without CUDA the callable does nothing.
    """
    if mode not in SYNC_MODES:
        raise ValueError(f"sync mode {mode!r} not in {SYNC_MODES}")
    import torch

    if not torch.cuda.is_available():
        return lambda: None
    if mode == "spin":
        return torch.cuda.synchronize
    event = torch.cuda.Event(blocking=True)

    def sync():
        event.record()
        event.synchronize()

    return sync


# --- memory (K6 mem_stats / reset_peak, identical for every family) ---------------------------------

def _parse_vmrss_mb(status_text):
    for line in status_text.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return None


def rss_mb():
    """Resident set size of this process in MB, or None where /proc is unavailable."""
    try:
        with open("/proc/self/status") as f:
            return _parse_vmrss_mb(f.read())
    except OSError:
        return None


def mem_stats():
    import torch

    cuda = torch.cuda.is_available()
    return {
        "torch_reserved_mb": torch.cuda.memory_reserved() / _MB if cuda else 0.0,
        "torch_peak_alloc_mb": torch.cuda.max_memory_allocated() / _MB if cuda else 0.0,
        "rss_mb": rss_mb(),
    }


def reset_peak():
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
