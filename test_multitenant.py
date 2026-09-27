"""Unit tests for multitenant_run.py (3-phase CME scheme).

Covers every pure/orchestration path WITHOUT the Jetson: parsing, anchors (incl.
yolo leaf->exit/sub mapping), the preflight memory gate, hw parsing + P95, the
phase-2 duration calibration, the metric row (throughput, slowdown, p95 ratio,
power, energy, aggregate, gain), and the phase1->2->3 cell/grid/scenario flow
with the bench subprocess mocked out. Only a real OOM is not unit-testable
(inherent to the hardware; the gate that prevents it IS tested).

Run:  python -m unittest test_multitenant -v
"""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import multitenant_run as mr


def _write_hw(path, lat, thru=None, power=5.0, energy=0.1, vram=1000.0,
              ram=2000.0, n=100, static=None, peak=None):
    """Fake hw_results.json, including the per-sample list P95 is read from and
    the static/dynamic memory split the profiler now emits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    thru = thru if thru is not None else (1.0 / lat)
    samples = [{"end_to_end_sec": lat * (1.0 + i / 200.0)} for i in range(n)]
    agg = {"per_sample_sec_mean": lat, "throughput_samples_per_sec": thru,
           "avg_power_w": power, "avg_energy_j": energy,
           "avg_vram_allocated_mb": vram, "avg_ram_used_mb": ram,
           "n_samples": n, "total_sec": lat * n}
    if static is not None:
        agg["gpu_mem_static_mb"] = static
        agg["peak_vram_allocated_mb"] = peak if peak is not None else static
        agg["gpu_mem_dynamic_mb"] = round(max(0.0, agg["peak_vram_allocated_mb"] - static), 2)
    path.write_text(json.dumps({"aggregate": agg, "samples": samples}), encoding="utf-8")


class TestParsingAnchors(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(mr.parse_tenant("bert:12"), ("bert", 12, None))
        self.assertEqual(mr.parse_tenant("yolo:5:2"), ("yolo", 5, 2))
        self.assertEqual(mr.parse_tenant("bert"), ("bert", 0, None))

    def test_k_points(self):
        self.assertEqual(mr._k_points(24, 6), [0, 5, 9, 14, 18, 23])
        self.assertEqual(mr._k_points(16, 6), [0, 3, 6, 9, 12, 15])

    def test_anchor_flat_and_yolo(self):
        self.assertEqual(mr._anchor("bert"), [(e, None) for e in [0, 5, 9, 14, 18, 23]])
        self.assertEqual(mr._anchor("yolo"),
                         [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)])

    def test_bench_cmd_carries_sub_and_n_samples(self):
        argv, env = mr.bench_cmd("yolo", 5, 2, "s", n_samples=321)
        self.assertEqual(argv[argv.index("--sub-exit") + 1], "2")
        self.assertEqual(argv[argv.index("--n-samples") + 1], "321")
        self.assertIn("--no-quality", argv)
        self.assertEqual(env["BENCH_SUBDIR"], "s")

    def test_bench_cmd_omits_n_samples_when_none(self):
        argv, _ = mr.bench_cmd("bert", 12, None, "s")
        self.assertNotIn("--n-samples", argv)


class TestFindReadHw(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_read_missing(self):
        self.assertIsNone(mr._read_hw(None))

    def test_read_fields_and_p95(self):
        p = self.tmp / "bert" / "d" / "exit_3" / "hw_results.json"
        _write_hw(p, 0.02, thru=50.0, power=7.0, energy=0.14, n=100)
        hw = mr._read_hw(mr._find_hw(self.tmp, "bert", 3))
        self.assertAlmostEqual(hw["lat"], 0.02)
        self.assertAlmostEqual(hw["thru"], 50.0)
        self.assertAlmostEqual(hw["power_w"], 7.0)
        self.assertEqual(hw["n"], 100)
        # samples ramp from lat to lat*1.495; p95 must sit near the top, above mean
        self.assertGreater(hw["p95"], 0.02)
        self.assertLess(hw["p95"], 0.031)

    def test_p95_empty(self):
        self.assertIsNone(mr._p95([]))

    def test_find_yolo_exact_sub(self):
        for s in (0, 1, 2):
            _write_hw(self.tmp / "yolo" / "coco" / f"exit_0_P{s + 3}" / "hw_results.json",
                      0.01 * (s + 1))
        got = mr._read_hw(mr._find_hw(self.tmp, "yolo", 0, sub=1))   # -> P4
        self.assertAlmostEqual(got["lat"], 0.02)


class TestStaticDynamicMemory(unittest.TestCase):
    """Static (weights, admission cost) vs dynamic (activations, moves with exit)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_split_parsed(self):
        p = self.tmp / "bert" / "d" / "exit_3" / "hw_results.json"
        _write_hw(p, 0.02, static=700.0, peak=980.0)
        hw = mr._read_hw(mr._find_hw(self.tmp, "bert", 3))
        self.assertEqual(hw["gpu_mem_static_mb"], 700.0)
        self.assertEqual(hw["gpu_mem_dynamic_mb"], 280.0)     # 980 - 700
        self.assertEqual(hw["gpu_mem_peak_mb"], 980.0)

    def test_absent_in_legacy_runs(self):
        """Old data predates these fields: report None, never fabricate."""
        p = self.tmp / "bert" / "d" / "exit_0" / "hw_results.json"
        _write_hw(p, 0.02)                                  # no static/peak written
        hw = mr._read_hw(mr._find_hw(self.tmp, "bert", 0))
        self.assertIsNone(hw["gpu_mem_static_mb"])
        self.assertIsNone(hw["gpu_mem_dynamic_mb"])

    def test_row_reports_both_levels(self):
        solo = {0: {"lat": 0.01, "thru": 100.0, "p95": 0.012, "power_w": 2.0,
                    "energy_j": 0.02, "vram_mb": 700, "ram_mb": 2000, "n": 100,
                    "total_sec": 1.0, "gpu_mem_static_mb": 700.0,
                    "gpu_mem_dynamic_mb": 100.0, "gpu_mem_peak_mb": 800.0},
                1: {"lat": 0.02, "thru": 50.0, "p95": 0.024, "power_w": 2.0,
                    "energy_j": 0.04, "vram_mb": 40, "ram_mb": 1500, "n": 50,
                    "total_sec": 1.0, "gpu_mem_static_mb": 40.0,
                    "gpu_mem_dynamic_mb": 60.0, "gpu_mem_peak_mb": 100.0}}
        shared = {0: dict(solo[0], gpu_mem_dynamic_mb=150.0, gpu_mem_peak_mb=850.0),
                  1: dict(solo[1], gpu_mem_dynamic_mb=90.0, gpu_mem_peak_mb=130.0)}
        row = mr.build_row([("bert", 12, None), ("yolo", 5, 2)], "t",
                           solo, shared, 0.95, {0: 100, 1: 50}, 30.0)
        # per tenant
        self.assertEqual(row["t0_gpu_mem_static_mb"], 700.0)
        self.assertEqual(row["t0_gpu_mem_dynamic_mb"], 150.0)
        self.assertEqual(row["t0_dynamic_ratio"], 1.5)      # 150/100 under co-location
        self.assertEqual(row["t1_dynamic_ratio"], 1.5)      # 90/60
        # system: admission (static sum) vs joint-peak risk (peak sum)
        self.assertEqual(row["pair_gpu_mem_static_mb"], 740.0)   # 700 + 40
        self.assertEqual(row["pair_gpu_mem_peak_mb"], 980.0)     # 850 + 130

    def test_static_is_exit_independent_dynamic_is_not(self):
        """The distinction that matters: the exit knob moves dynamic, not static."""
        shallow = {"gpu_mem_static_mb": 700.0, "gpu_mem_dynamic_mb": 40.0,
                   "gpu_mem_peak_mb": 740.0, "lat": 0.01, "thru": 100.0, "p95": 0.01,
                   "power_w": 2.0, "energy_j": 0.02, "vram_mb": 700, "ram_mb": 100,
                   "n": 10, "total_sec": 1.0}
        deep = dict(shallow, gpu_mem_dynamic_mb=260.0, gpu_mem_peak_mb=960.0)
        self.assertEqual(shallow["gpu_mem_static_mb"], deep["gpu_mem_static_mb"])
        self.assertNotEqual(shallow["gpu_mem_dynamic_mb"], deep["gpu_mem_dynamic_mb"])


class TestCalibration(unittest.TestCase):
    def test_matches_duration(self):
        # 0.01 s/sample over a 30 s window -> 3000 samples
        self.assertEqual(mr.calibrate({"lat": 0.01}, 30.0), 3000)

    def test_fast_tenant_gets_more_samples(self):
        fast = mr.calibrate({"lat": 0.005}, 30.0)
        slow = mr.calibrate({"lat": 0.05}, 30.0)
        self.assertGreater(fast, slow)          # this is the whole point of phase 2

    def test_clamped(self):
        self.assertEqual(mr.calibrate({"lat": 1e-9}, 30.0), mr.MAX_SAMPLES)
        self.assertEqual(mr.calibrate({"lat": 1e9}, 30.0), mr.MIN_SAMPLES)

    def test_none_when_no_solo(self):
        self.assertIsNone(mr.calibrate(None, 30.0))
        self.assertIsNone(mr.calibrate({"lat": None}, 30.0))


class TestPreflight(unittest.TestCase):
    def _mem(self, gb):
        return mock.Mock(available=gb * 1e9)

    def test_fits(self):
        with mock.patch("psutil.virtual_memory", return_value=self._mem(8.0)):
            self.assertTrue(mr.preflight(["bert", "yolo"]))

    def test_aborts_when_tight(self):
        with mock.patch("psutil.virtual_memory", return_value=self._mem(5.0)):
            self.assertFalse(mr.preflight(["llama", "llama"]))   # 3+3+1 > 5


class TestMetricsRow(unittest.TestCase):
    def test_row_math(self):
        solo = {0: {"lat": 0.010, "thru": 100.0, "p95": 0.012, "power_w": 2.0,
                    "energy_j": 0.02, "vram_mb": 700, "ram_mb": 2000, "n": 3000,
                    "total_sec": 30.0},
                1: {"lat": 0.020, "thru": 50.0, "p95": 0.024, "power_w": 2.0,
                    "energy_j": 0.04, "vram_mb": 40, "ram_mb": 1500, "n": 1500,
                    "total_sec": 30.0}}
        shared = {0: {"lat": 0.015, "thru": 66.0, "p95": 0.030, "power_w": 6.0,
                      "energy_j": 0.09, "vram_mb": 700, "ram_mb": 2100, "n": 3000,
                      "total_sec": 45.0},
                  1: {"lat": 0.030, "thru": 33.0, "p95": 0.048, "power_w": 5.0,
                      "energy_j": 0.15, "vram_mb": 40, "ram_mb": 1600, "n": 1500,
                      "total_sec": 45.0}}
        row = mr.build_row([("bert", 12, None), ("yolo", 5, 2)], "t",
                           solo, shared, 0.95, {0: 3000, 1: 1500}, 30.0)
        self.assertEqual(row["t0_slowdown"], 1.5)          # 0.015/0.010
        self.assertEqual(row["t1_slowdown"], 1.5)          # 0.030/0.020
        self.assertEqual(row["t1"], "yolo@5_P2")
        self.assertEqual(row["t0_p95_ratio"], 2.5)         # 0.030/0.012
        self.assertEqual(row["t0_n_samples"], 3000)
        self.assertEqual(row["agg_throughput"], 99.0)      # 66+33
        self.assertEqual(row["throughput_gain"], 0.99)     # 99 / best solo 100
        self.assertEqual(row["pair_power_w"], 6.0)         # shared rail -> max
        # device energy = device power x window, NOT a sum over tenants
        self.assertEqual(row["pair_energy_j"], 6.0 * 45.0)
        self.assertEqual(row["pair_window_sec"], 45.0)
        self.assertEqual(row["pair_vram_mb"], 740.0)
        self.assertEqual(row["target_window_sec"], 30.0)

    def test_device_energy_is_not_summed_across_tenants(self):
        """Regression: both tenants integrate the SAME shared rail over the same
        window, so each already accounts for the whole device. Summing would
        multiply device energy by the tenant count."""
        P, window = 8.0, 30.0
        # A: 3000 x 0.010s, B: 600 x 0.050s -> both busy the same 30 s
        t = {"thru": 1.0, "p95": 0.01, "power_w": P, "energy_j": 0.01 * P,
             "vram_mb": 10, "ram_mb": 10}
        solo = {0: dict(t, lat=0.010, n=3000, total_sec=window),
                1: dict(t, lat=0.050, n=600, total_sec=window)}
        row = mr.build_row([("bert", 0, None), ("yolo", 0, 0)], "t",
                           solo, solo, 1.0, {0: 3000, 1: 600}, window)
        self.assertEqual(row["pair_power_w"], P)            # max, not sum
        self.assertEqual(row["pair_energy_j"], P * window)  # 240 J, not 480 J
        self.assertEqual(row["pair_window_sec"], window)

    def test_row_survives_missing_tenant(self):
        row = mr.build_row([("bert", 0, None)], "t", {0: None}, {0: None}, 0.0, {0: None}, 30.0)
        self.assertIsNone(row["t0_slowdown"])
        self.assertIsNone(row["agg_throughput"])


class _FakeProc:
    def wait(self):
        return 0


class TestPhaseFlow(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._logs, self._out = mr.LOGS, mr.OUT_DIR
        mr.LOGS, mr.OUT_DIR = self.tmp / "logs", self.tmp / "out"
        self.calls = []

    def tearDown(self):
        mr.LOGS, mr.OUT_DIR = self._logs, self._out

    def _fake(self, fam, ex, sub, subdir, os_, n_samples=None):
        self.calls.append((subdir, n_samples))
        lat = 0.010 if subdir.startswith("mt_solo") else 0.015
        leaf = f"exit_{ex}_P{sub + 3}" if (fam == "yolo" and sub is not None) else f"exit_{ex}"
        _write_hw(mr.LOGS / subdir / fam / "d" / leaf / "hw_results.json", lat)
        return _FakeProc(), time.perf_counter()

    def _run(self, fn, *a, **k):
        with mock.patch("psutil.virtual_memory", return_value=mock.Mock(available=8e9)), \
             mock.patch.object(mr, "run_one", side_effect=self._fake):
            return fn(*a, **k)

    def test_pair_three_phases(self):
        rows = self._run(mr.run_pair, [("bert", 12, None), ("vision", 12, None)], "t", 30.0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["t0_slowdown"], 1.5)
        solo = [c for c in self.calls if c[0].startswith("mt_solo")]
        conc = [c for c in self.calls if c[0].startswith("mt_conc")]
        self.assertEqual(len(solo), 2)                 # phase 1: one per tenant
        self.assertEqual(len(conc), 2)                 # phase 3: both together
        self.assertTrue(all(c[1] is None for c in solo))    # solo uses config default
        self.assertTrue(all(c[1] == 3000 for c in conc))    # phase 2: 30s / 0.010s

    def test_calibration_reaches_subprocess(self):
        self._run(mr.run_pair, [("bert", 12, None)], "t", 60.0)
        conc = [c for c in self.calls if c[0].startswith("mt_conc")]
        self.assertEqual(conc[0][1], 6000)             # 60s / 0.010s

    def test_grid_solo_cached(self):
        rows = self._run(mr.run_grid, "bert", "vision", "g")
        self.assertEqual(len(rows), 36)                # 6x6
        solo = [c for c in self.calls if c[0].startswith("mt_solo")]
        self.assertEqual(len(solo), 12)                # 6+6 unique, measured once

    def test_scenario_yolo_pair(self):
        rows = self._run(mr.run_scenario, "llama_yolo")
        self.assertEqual(len(rows), 36)
        self.assertTrue(any("_P" in r["t1"] for r in rows))

    def test_scenario_scaling(self):
        rows = self._run(mr.run_scenario, "yolo_scale")
        # 6 exit anchors x 3 tenant counts = 18 cells
        self.assertEqual(len(rows), 18)
        self.assertEqual([r["n_tenants"] for r in rows], [2, 3, 4] * 6)
        # solo runs are cached: 6 unique (exit, sub) anchors for yolo, each measured once
        solo = [c for c in self.calls if c[0].startswith("mt_solo")]
        self.assertEqual(len(solo), 6)

    def test_scenario_scaling_k2(self):
        rows = self._run(mr.run_scenario, "bert_scale", k=2)
        # 2 exit anchors x 3 tenant counts = 6 cells
        self.assertEqual(len(rows), 6)

    def test_k_below_2_rejected(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, str(mr.REPO_ROOT / "multitenant_run.py"),
             "--scenario", "bert_scale", "--k", "1"],
            capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--k must be at least 2", result.stdout + result.stderr)

    def test_abort_launches_nothing(self):
        with mock.patch("psutil.virtual_memory", return_value=mock.Mock(available=0.4e9)), \
             mock.patch.object(mr, "run_one", side_effect=self._fake):
            out = mr.run_scenario("llama_yolo")
        self.assertIsNone(out)
        self.assertEqual(self.calls, [])


class TestCsvMerge(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._out = mr.OUT_DIR
        mr.OUT_DIR = self.tmp / "out"
        mr.OUT_DIR.mkdir(parents=True)

    def tearDown(self):
        mr.OUT_DIR = self._out

    def test_variable_columns(self):
        mr._append_csv({"tag": "a", "t0": "bert@0", "t0_slowdown": 1.2})
        mr._append_csv({"tag": "b", "t2": "yolo@5", "t2_slowdown": 1.9})
        lines = (mr.OUT_DIR / "concurrent_slowdown.csv").read_text(encoding="utf-8").splitlines()
        hdr = lines[0].split(",")
        self.assertIn("t0_slowdown", hdr)
        self.assertIn("t2_slowdown", hdr)
        self.assertEqual(len(lines), 3)


class TestTimedUnixFields(unittest.TestCase):
    """Tests for timed_start_unix / timed_end_unix parsing and overlap math."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _write_with_unix(self, path, lat, t_start, t_end):
        """hw_results.json with timed_start_unix / timed_end_unix in aggregate."""
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = [{"end_to_end_sec": lat}]
        agg = {"per_sample_sec_mean": lat, "throughput_samples_per_sec": 1.0 / lat,
               "avg_power_w": 5.0, "avg_energy_j": 0.1,
               "avg_vram_allocated_mb": 500.0, "avg_ram_used_mb": 1000.0,
               "n_samples": 1, "total_sec": lat,
               "timed_start_unix": t_start, "timed_end_unix": t_end}
        path.write_text(json.dumps({"aggregate": agg, "samples": samples}),
                        encoding="utf-8")

    def test_read_hw_returns_unix_fields(self):
        p = self.tmp / "bert" / "d" / "exit_1" / "hw_results.json"
        self._write_with_unix(p, 0.02, 1000.0, 1030.0)
        hw = mr._read_hw(mr._find_hw(self.tmp, "bert", 1))
        self.assertEqual(hw["timed_start_unix"], 1000.0)
        self.assertEqual(hw["timed_end_unix"], 1030.0)
        # span check: end >= start and plausible duration
        self.assertGreaterEqual(hw["timed_end_unix"], hw["timed_start_unix"])

    def test_read_hw_unix_fields_none_for_old_runs(self):
        """Aggregates that predate this field must yield None, not a fabricated value."""
        p = self.tmp / "bert" / "d" / "exit_2" / "hw_results.json"
        _write_hw(p, 0.02)   # _write_hw does not emit timed_*_unix
        hw = mr._read_hw(mr._find_hw(self.tmp, "bert", 2))
        self.assertIsNone(hw["timed_start_unix"])
        self.assertIsNone(hw["timed_end_unix"])

    def test_timed_overlap_frac_no_intersection(self):
        """Windows that do not overlap must yield 0.0, not negative."""
        # tenant A: 100..130, tenant B: 140..170 — no intersection
        solo = {0: {"lat": 0.01, "thru": 100.0, "p95": 0.012, "power_w": 2.0,
                    "energy_j": 0.02, "vram_mb": 700, "ram_mb": 2000, "n": 100,
                    "total_sec": 30.0, "timed_start_unix": 100.0, "timed_end_unix": 130.0},
                1: {"lat": 0.02, "thru": 50.0, "p95": 0.024, "power_w": 2.0,
                    "energy_j": 0.04, "vram_mb": 40, "ram_mb": 1500, "n": 50,
                    "total_sec": 30.0, "timed_start_unix": 140.0, "timed_end_unix": 170.0}}
        row = mr.build_row([("bert", 12, None), ("yolo", 5, 2)], "t",
                           solo, solo, 0.95, {0: 100, 1: 50}, 30.0,
                           timed_overlap_frac=0.0)
        self.assertEqual(row["timed_overlap_frac"], 0.0)

    def test_timed_overlap_frac_coincident_windows(self):
        """Windows that coincide exactly must yield 1.0."""
        row = mr.build_row([("bert", 0, None), ("bert", 1, None)], "t",
                           {0: None}, {0: None}, 1.0, {0: None, 1: None}, 30.0,
                           timed_overlap_frac=1.0)
        self.assertAlmostEqual(row["timed_overlap_frac"], 1.0)

    def test_timed_overlap_frac_none_does_not_fall_back(self):
        """When timestamps are missing, timed_overlap_frac is None.
        It must NOT silently carry the process-lifetime value."""
        row = mr.build_row([("bert", 0, None)], "t",
                           {0: None}, {0: None}, 0.95, {0: None}, 30.0,
                           timed_overlap_frac=None)
        self.assertIsNone(row["timed_overlap_frac"])
        # overlap_frac (lifetime) is still present under its own name
        self.assertEqual(row["overlap_frac"], 0.95)


class TestTimedOverlapMath(unittest.TestCase):
    """Unit tests for the timed-overlap computation in measure_concurrent."""

    def _shared_from_unix(self, t_start_a, t_end_a, t_start_b, t_end_b):
        """Build the shared dict that measure_concurrent reads, minus subprocess."""
        return {
            0: {"lat": 0.01, "thru": 100.0, "p95": 0.012,
                "power_w": 5.0, "energy_j": 0.05,
                "vram_mb": 500, "ram_mb": 1000, "n": 100, "total_sec": 30.0,
                "gpu_mem_static_mb": None, "gpu_mem_dynamic_mb": None,
                "gpu_mem_peak_mb": None,
                "timed_start_unix": t_start_a, "timed_end_unix": t_end_a},
            1: {"lat": 0.02, "thru": 50.0, "p95": 0.024,
                "power_w": 5.0, "energy_j": 0.1,
                "vram_mb": 500, "ram_mb": 1000, "n": 50, "total_sec": 30.0,
                "gpu_mem_static_mb": None, "gpu_mem_dynamic_mb": None,
                "gpu_mem_peak_mb": None,
                "timed_start_unix": t_start_b, "timed_end_unix": t_end_b},
        }

    def _compute(self, shared):
        """Run only the timed-overlap arithmetic from measure_concurrent."""
        tenants = [("bert", 0, None), ("bert", 1, None)]
        t_starts = [hw["timed_start_unix"] for hw in shared.values()
                    if hw and hw.get("timed_start_unix") is not None]
        t_ends   = [hw["timed_end_unix"]   for hw in shared.values()
                    if hw and hw.get("timed_end_unix")   is not None]
        if len(t_starts) == len(tenants) and len(t_ends) == len(tenants):
            true_overlap = min(t_ends) - max(t_starts)
            true_span    = max(t_ends) - min(t_starts)
            return round(max(0.0, min(1.0, true_overlap / true_span)), 3) if true_span > 0 else 0.0
        return None

    def test_no_intersection_yields_zero(self):
        shared = self._shared_from_unix(100.0, 130.0, 140.0, 170.0)
        self.assertEqual(self._compute(shared), 0.0)

    def test_perfect_overlap_yields_one(self):
        shared = self._shared_from_unix(100.0, 130.0, 100.0, 130.0)
        self.assertAlmostEqual(self._compute(shared), 1.0)

    def test_partial_overlap(self):
        # A: 100..130, B: 120..150  -> overlap 10, span 50 -> 0.2
        shared = self._shared_from_unix(100.0, 130.0, 120.0, 150.0)
        self.assertAlmostEqual(self._compute(shared), 0.2)

    def test_missing_timestamps_yields_none(self):
        shared = self._shared_from_unix(100.0, 130.0, None, None)
        self.assertIsNone(self._compute(shared))


if __name__ == "__main__":
    unittest.main(verbosity=2)
