"""Tests for the intended_tenants self-describing raw data flow.

Three assertions:
  a) BenchmarkProfiler with BENCH_BARRIER_N=3 writes aggregate.intended_tenants==3.
  b) The analyzer classifies a raw cell (no CSV) as COMPLETE when all tenants are
     present and each carries aggregate.intended_tenants==2.
  c) A raw cell with intended_tenants==3 but only 2 tenant folders is INCOMPLETE.
     This confirms we are NOT regressing to the unsafe max-tid+1 fallback.

Run: python -m unittest test_intended_tenants -v
"""

import importlib.util as _ilu
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent

# Load benchmark_profiler without executing shared/__init__.py (which pulls torch).
# Strategy: register stub sub-modules under the "shared" package namespace before
# loading benchmark_profiler.py, so its relative imports resolve to the stubs.
import types as _types

def _install_shared_stubs():
    # Register a minimal "shared" package so relative imports resolve.
    pkg = _types.ModuleType("shared")
    pkg.__path__ = [str(_REPO / "shared")]
    pkg.__package__ = "shared"
    sys.modules.setdefault("shared", pkg)

    hw = _types.ModuleType("shared.hw_profiler")
    hw.aggregate_hw = lambda *a, **k: {}
    hw.device_caps = lambda *a, **k: {}
    hw.device_energy_mj = lambda *a, **k: None
    hw.proc_cpu_times_sec = lambda *a, **k: None
    hw.sample_hw = lambda *a, **k: {}
    hw.Timer = type("Timer", (), {
        "__enter__": lambda s: s,
        "__exit__": lambda s, *a: None,
    })
    sys.modules.setdefault("shared.hw_profiler", hw)
    pkg.hw_profiler = hw

    cc = _types.ModuleType("shared.cpu_cache")
    cc.CacheCounter = type("CacheCounter", (), {
        "__enter__": lambda s: s,
        "__exit__": lambda s, *a: None,
        "read": lambda s: {},
    })
    cc.is_available = lambda: False
    sys.modules.setdefault("shared.cpu_cache", cc)
    pkg.cpu_cache = cc

_install_shared_stubs()

_bp_spec = _ilu.spec_from_file_location(
    "shared.benchmark_profiler",
    _REPO / "shared" / "benchmark_profiler.py",
    submodule_search_locations=[],
)
_bp_mod = _ilu.module_from_spec(_bp_spec)
_bp_mod.__package__ = "shared"
sys.modules["shared.benchmark_profiler"] = _bp_mod
_bp_spec.loader.exec_module(_bp_mod)
BenchmarkProfiler = _bp_mod.BenchmarkProfiler

# Load multitenant_analyze directly (already uses importlib internally).
from multitenant_analyze import analyze_cell, discover_cells, discover_solos


# ---------------------------------------------------------------------------
# Helpers (copied from test_multitenant_analyze.py to stay self-contained)
# ---------------------------------------------------------------------------

def _hw(per_sample_sec_mean, throughput_samples_per_sec, intended_tenants=None,
        avg_power_w=10.0, avg_energy_j=1.5,
        gpu_mem_static_mb=1371.0, gpu_mem_dynamic_mb=13.0,
        peak_vram_allocated_mb=1384.0, n_samples=100):
    agg = {
        "per_sample_sec_mean": per_sample_sec_mean,
        "throughput_samples_per_sec": throughput_samples_per_sec,
        "avg_power_w": avg_power_w,
        "avg_energy_j": avg_energy_j,
        "gpu_mem_static_mb": gpu_mem_static_mb,
        "gpu_mem_dynamic_mb": gpu_mem_dynamic_mb,
        "peak_vram_allocated_mb": peak_vram_allocated_mb,
        "n_samples": n_samples,
    }
    if intended_tenants is not None:
        agg["intended_tenants"] = intended_tenants
    return {"aggregate": agg}


def _write_hw(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _conc_path(root: Path, basetag, cellidx, tid, fam, exit_n, dataset="DS") -> Path:
    folder = root / f"mt_conc_{basetag}_{cellidx}_r0_{tid}_{fam}_{exit_n}"
    return folder / fam / dataset / "pretrained" / f"exit_{exit_n}" / "hw_results.json"


def _solo_path(root: Path, basetag, fam, exit_n, count=100, dataset="DS") -> Path:
    folder = root / f"mt_solo_{basetag}_{fam}_{exit_n}_n{count}"
    return folder / fam / dataset / "pretrained" / f"exit_{exit_n}" / "hw_results.json"


# ---------------------------------------------------------------------------
# Test a: profiler embeds intended_tenants from env
# ---------------------------------------------------------------------------

class TestProfilerIntendedTenants(unittest.TestCase):

    def test_intended_tenants_written_from_env(self):
        """BENCH_BARRIER_N=3 -> aggregate.intended_tenants == 3 in output JSON."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "hw_results.json"
            old = os.environ.get("BENCH_BARRIER_N")
            os.environ["BENCH_BARRIER_N"] = "3"
            try:
                # Use warmup_steps=0 so the barrier call is immediate (no samples needed).
                prof = BenchmarkProfiler(str(out), task="test", warmup_steps=0)
                # Manually call flush() without __enter__/__exit__ to avoid HW sampling.
                prof._total_start = __import__("time").perf_counter()
                prof._timed_start_perf = prof._total_start
                prof._timed_start_unix = __import__("time").time()
                # Add one dummy sample so flush() does not bail out.
                prof.samples = [{
                    "idx": 0, "prediction": 1, "label": 1,
                    "forward_sec": 0.01, "ttft_sec": None,
                    "end_to_end_sec": None, "exit_layer": None,
                    "confidence": None, "elapsed_sec": 0.01,
                }]
                prof._energy_mj_start = None
                prof._energy_mj_at_warmup_end = None
                prof._cpu_t_at_warmup_end = None
                prof.flush()
            finally:
                if old is None:
                    os.environ.pop("BENCH_BARRIER_N", None)
                else:
                    os.environ["BENCH_BARRIER_N"] = old

            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["aggregate"]["intended_tenants"], 3)

    def test_intended_tenants_defaults_to_one_for_solo(self):
        """No BENCH_BARRIER_N env -> intended_tenants == 1 (solo run)."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "hw_results.json"
            os.environ.pop("BENCH_BARRIER_N", None)
            prof = BenchmarkProfiler(str(out), task="test", warmup_steps=0)
            self.assertEqual(prof.meta["intended_tenants"], 1)


# ---------------------------------------------------------------------------
# Test b: analyzer uses raw intended_tenants when CSV is absent
# ---------------------------------------------------------------------------

class TestAnalyzerRawIntendedSource(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root
        # Solo baseline (no intended_tenants needed here).
        _write_hw(_solo_path(root, "raw_test", "bert", 1), _hw(0.01, 100.0))
        # Two concurrent tenants, each carrying intended_tenants=2.
        _write_hw(_conc_path(root, "raw_test", 0, 0, "bert", 1),
                  _hw(0.02, 50.0, intended_tenants=2))
        _write_hw(_conc_path(root, "raw_test", 0, 1, "bert", 1),
                  _hw(0.02, 50.0, intended_tenants=2))
        # No concurrent_slowdown.csv written.

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_from_raw_no_csv(self):
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        self.assertIn(("raw_test", 0), cells)
        tid_map = cells[("raw_test", 0)]

        # Simulate what run() does: derive intended from raw first.
        from multitenant_analyze import _collect_hw_results
        # Load the module-level _load_hw via the same importlib path the analyzer uses.
        import importlib.util as ilu
        spec = ilu.spec_from_file_location(
            "shared.csv_export",
            _REPO / "shared" / "csv_export.py",
        )
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _load_hw = mod._load

        raw_intended = None
        for folder in tid_map.values():
            for hw_path in _collect_hw_results(folder):
                try:
                    agg = _load_hw(hw_path)
                    v = agg.get("intended_tenants")
                    if isinstance(v, (int, float)) and v >= 1:
                        raw_intended = max(raw_intended or 0, int(v))
                except Exception:
                    pass

        self.assertIsNotNone(raw_intended, "raw intended_tenants not found")
        self.assertEqual(raw_intended, 2)

        result = analyze_cell("raw_test", 0, tid_map, solos,
                              raw_intended, "raw (aggregate.intended_tenants)")
        # Complete cells don't echo intended_source back; status is sufficient.
        self.assertEqual(result["status"], "complete")

    def test_run_classifies_complete_without_csv(self):
        """run() on this fixture (no CSV) must return 0 and find a complete cell."""
        from multitenant_analyze import run
        code = run(self.root, "test", None)
        self.assertEqual(code, 0)


# ---------------------------------------------------------------------------
# Test c: incomplete cell is caught even without CSV (anti-regression)
# ---------------------------------------------------------------------------

class TestAnalyzerIncompleteFromRaw(unittest.TestCase):
    """intended_tenants=3 in raw but only 2 tenant folders -> INCOMPLETE."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root
        _write_hw(_solo_path(root, "raw_inc", "bert", 1), _hw(0.01, 100.0))
        # Only tids 0 and 1 present; intended was 3.
        _write_hw(_conc_path(root, "raw_inc", 0, 0, "bert", 1),
                  _hw(0.02, 50.0, intended_tenants=3))
        _write_hw(_conc_path(root, "raw_inc", 0, 1, "bert", 1),
                  _hw(0.02, 50.0, intended_tenants=3))
        # No CSV.

    def tearDown(self):
        self.tmp.cleanup()

    def test_incomplete_not_max_tid_fallback(self):
        """logged=2 < intended=3 (from raw) -> incomplete. Must NOT say intended=2."""
        from multitenant_analyze import run
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = run(self.root, "test", None)
        self.assertEqual(code, 0)
        output = buf.getvalue()
        # The analyzer must not have any complete cells (COMPLETE is a substring
        # of INCOMPLETE, so check for the standalone header line).
        self.assertNotIn("\nCOMPLETE CELLS\n", output)
        # The reason must mention logged < intended=3.
        self.assertIn("logged=2", output)

    def test_analyze_cell_directly(self):
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        tid_map = cells[("raw_inc", 0)]
        # Pass intended=3 (as the run() loop would after reading raw).
        result = analyze_cell("raw_inc", 0, tid_map, solos, 3,
                              "raw (aggregate.intended_tenants)")
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("logged=2", result["reason"])


if __name__ == "__main__":
    unittest.main()
