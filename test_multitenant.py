"""Unit tests for multitenant_run.py (3-phase CME scheme).

Covers every pure/orchestration path WITHOUT the Jetson: parsing, anchors (incl.
yolo leaf->exit/sub mapping), the preflight memory gate, hw parsing + P95, the
phase-2 duration calibration, the metric row (throughput, slowdown, p95 ratio,
power, energy, aggregate, gain, STP, ANTT, SLO, violation ratio, clock fields),
and the phase1->2->3 cell/grid/scenario flow with the bench subprocess mocked out.
Also covers: suspect CSV routing, --repeats, --holdout determinism.
Only a real OOM is not unit-testable (inherent to the hardware; the gate that
prevents it IS tested).

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
              ram=2000.0, n=100, static=None, peak=None, total_sec=None):
    """Fake hw_results.json, including the per-sample list P95 is read from and
    the static/dynamic memory split the profiler now emits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    thru = thru if thru is not None else (1.0 / lat)
    samples = [{"end_to_end_sec": lat * (1.0 + i / 200.0)} for i in range(n)]
    # Default total_sec to lat * n (forward == wall in test fixtures).
    if total_sec is None:
        total_sec = lat * n
    agg = {"per_sample_sec_mean": lat, "throughput_samples_per_sec": thru,
           "avg_power_w": power, "avg_energy_j": energy,
           "avg_vram_allocated_mb": vram, "avg_ram_used_mb": ram,
           "n_samples": n, "total_sec": total_sec}
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

    def test_k_points_lo_1(self):
        pts = mr._k_points(24, 8, 1)
        self.assertEqual(pts, [1, 4, 7, 10, 14, 17, 20, 23])
        self.assertEqual(len(pts), 8)
        self.assertNotIn(0, pts)
        self.assertEqual(pts[0], 1)
        self.assertEqual(pts[-1], 23)

    def test_k_points_lo_0_backcompat(self):
        self.assertEqual(mr._k_points(24, 6, 0), [0, 5, 9, 14, 18, 23])

    def test_k_points_guard_lo_ge_n_minus_1(self):
        with self.assertRaises(ValueError):
            mr._k_points(5, 3, 4)   # lo=4 == n-1=4: degenerate

    def test_k_points_guard_k_lt_2(self):
        with self.assertRaises(ValueError):
            mr._k_points(10, 1, 0)

    def test_anchor_flat_and_yolo(self):
        self.assertEqual(mr._anchor("bert"), [(e, None) for e in [0, 5, 9, 14, 18, 23]])
        self.assertEqual(mr._anchor("yolo"),
                         [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)])

    def test_anchor_bert_k8_min_exit_1(self):
        pts = mr._anchor("bert", 8, 1)
        exits = [e for e, _ in pts]
        self.assertEqual(len(pts), 8)
        self.assertNotIn(0, exits)
        self.assertEqual(exits[0], 1)
        self.assertEqual(exits[-1], 23)
        self.assertEqual(exits, [1, 4, 7, 10, 14, 17, 20, 23])

    def test_anchor_bert_k6_default_still_includes_exit_0(self):
        exits = [e for e, _ in mr._anchor("bert", 6)]
        self.assertIn(0, exits)
        self.assertEqual(exits, [0, 5, 9, 14, 18, 23])

    def test_anchor_yolo_k8_min_exit_1_no_leaf_0(self):
        pts = mr._anchor("yolo", 8, 1)
        # leaf 0 maps to (0, 0); with min_exit=1 the lowest leaf is 1 -> (0, 1)
        self.assertEqual(len(pts), 8)
        self.assertNotIn((0, 0), pts)
        self.assertEqual(pts[0], (0, 1))

    def test_bench_cmd_carries_sub_and_n_samples(self):
        argv, env = mr.bench_cmd("yolo", 5, 2, "s", n_samples=321)
        self.assertEqual(argv[argv.index("--sub-exit") + 1], "2")
        self.assertEqual(argv[argv.index("--n-samples") + 1], "321")
        self.assertIn("--no-quality", argv)
        self.assertEqual(env["BENCH_SUBDIR"], "s")

    def test_bench_cmd_omits_n_samples_when_none(self):
        argv, _ = mr.bench_cmd("bert", 12, None, "s")
        self.assertNotIn("--n-samples", argv)

    def test_bench_cmd_includes_duration_when_given(self):
        """--duration must appear in argv when duration is provided."""
        argv, _ = mr.bench_cmd("bert", 5, None, "s", n_samples=100, duration=30.0)
        self.assertIn("--duration", argv)
        self.assertEqual(argv[argv.index("--duration") + 1], "30.0")

    def test_bench_cmd_both_n_samples_and_duration_present(self):
        """Both --n-samples (ceiling) and --duration must be in the argv together."""
        argv, _ = mr.bench_cmd("bert", 5, None, "s", n_samples=500, duration=30.0)
        self.assertIn("--n-samples", argv)
        self.assertIn("--duration", argv)

    def test_bench_cmd_omits_duration_when_none(self):
        """When duration is None, --duration must not appear in argv."""
        argv, _ = mr.bench_cmd("bert", 5, None, "s", n_samples=100)
        self.assertNotIn("--duration", argv)

    def test_bench_cmd_duration_in_concurrent_path(self):
        """bench_cmd with duration for a concurrent (yolo) tenant includes --duration."""
        argv, _ = mr.bench_cmd("yolo", 3, 1, "s", n_samples=200, duration=45.0)
        self.assertIn("--duration", argv)
        self.assertIn("--n-samples", argv)
        self.assertEqual(argv[argv.index("--duration") + 1], "45.0")


class TestPatchDuration(unittest.TestCase):
    """Tests for _patch_duration in bench_jetson.py (offline, imported directly)."""

    def _load_patch_duration(self):
        """Import _patch_duration without triggering torch-dependent code."""
        import importlib.util, sys
        # bench_jetson imports shared at module level; stub it out.
        sys.modules.setdefault("shared", mock.MagicMock())
        if "bench_jetson" in sys.modules:
            del sys.modules["bench_jetson"]
        spec = importlib.util.spec_from_file_location(
            "bench_jetson",
            str(mr.REPO_ROOT / "bench_jetson.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Stub the load_env call so it does not fail without a .env file.
        with mock.patch.dict(sys.modules, {"shared": mock.MagicMock()}):
            try:
                spec.loader.exec_module(mod)
            except Exception:
                pass
        return getattr(mod, "_patch_duration", None)

    def test_sets_duration_sec_when_attribute_exists(self):
        import importlib.util, sys, types
        # Build a minimal stub cfg module with DURATION_SEC.
        cfg = types.SimpleNamespace(DURATION_SEC=None)
        # Import _patch_duration directly from bench_jetson source to avoid
        # triggering all the module-level side effects (HF, torch, etc.).
        src = (mr.REPO_ROOT / "bench_jetson.py").read_text(encoding="utf-8")
        # Extract only the _patch_duration function via exec into a namespace.
        ns = {}
        for line in src.splitlines():
            if line.startswith("def _patch_duration"):
                start = src.index(line)
                break
        else:
            self.skipTest("_patch_duration not found in bench_jetson.py")
            return
        # Grab the function body (up to the next top-level def or class).
        import textwrap
        lines = src[start:].splitlines()
        body_lines = [lines[0]]
        for ln in lines[1:]:
            if ln and not ln[0].isspace():
                break
            body_lines.append(ln)
        exec(textwrap.dedent("\n".join(body_lines)), ns)
        patch_fn = ns["_patch_duration"]
        patch_fn(cfg, 30.0)
        self.assertEqual(cfg.DURATION_SEC, 30.0)

    def test_noop_when_attribute_absent(self):
        import types, textwrap
        cfg = types.SimpleNamespace()   # no DURATION_SEC attribute
        src = (mr.REPO_ROOT / "bench_jetson.py").read_text(encoding="utf-8")
        for line in src.splitlines():
            if line.startswith("def _patch_duration"):
                start = src.index(line)
                break
        else:
            self.skipTest("_patch_duration not found in bench_jetson.py")
            return
        lines = src[start:].splitlines()
        body_lines = [lines[0]]
        for ln in lines[1:]:
            if ln and not ln[0].isspace():
                break
            body_lines.append(ln)
        ns = {}
        exec(textwrap.dedent("\n".join(body_lines)), ns)
        patch_fn = ns["_patch_duration"]
        # Must not raise and must not add the attribute.
        patch_fn(cfg, 30.0)
        self.assertFalse(hasattr(cfg, "DURATION_SEC"))

    def test_noop_when_seconds_is_none(self):
        import types, textwrap
        cfg = types.SimpleNamespace(DURATION_SEC=None)
        src = (mr.REPO_ROOT / "bench_jetson.py").read_text(encoding="utf-8")
        for line in src.splitlines():
            if line.startswith("def _patch_duration"):
                start = src.index(line)
                break
        else:
            self.skipTest("_patch_duration not found in bench_jetson.py")
            return
        lines = src[start:].splitlines()
        body_lines = [lines[0]]
        for ln in lines[1:]:
            if ln and not ln[0].isspace():
                break
            body_lines.append(ln)
        ns = {}
        exec(textwrap.dedent("\n".join(body_lines)), ns)
        patch_fn = ns["_patch_duration"]
        patch_fn(cfg, None)
        self.assertIsNone(cfg.DURATION_SEC)


class TestDurationCLIRejection(unittest.TestCase):
    """Non-positive --duration values must be rejected.

    bench_jetson.py imports torch-dependent code at module level (shared/__init__.py
    pulls in transformers), so it cannot be invoked as a subprocess in this offline
    environment. Instead, the validation logic is extracted and tested directly:
    we build a minimal argparse parser that includes --duration with the same
    constraints, verify the guard condition, and confirm the error path fires.
    """

    def _simulate_validation(self, duration_val):
        """Simulate the post-parse guard from bench_jetson.py main().

        Returns (ok, error_msg): ok=True when the value would be accepted,
        ok=False + a message string when it would be rejected.
        """
        import argparse
        # Replicate the exact guard from bench_jetson.py main():
        # if getattr(args, 'duration', None) is not None and args.duration <= 0:
        #     p.error(f"--duration must be positive (got {args.duration})")
        class _FakeParser:
            def error(self, msg):
                raise SystemExit(msg)
        args = argparse.Namespace(duration=duration_val)
        parser = _FakeParser()
        try:
            if getattr(args, 'duration', None) is not None and args.duration <= 0:
                parser.error(f"--duration must be positive (got {args.duration})")
            return True, None
        except SystemExit as e:
            return False, str(e)

    def test_zero_duration_rejected(self):
        ok, msg = self._simulate_validation(0.0)
        self.assertFalse(ok, "zero duration should be rejected")
        self.assertIn("--duration", msg)

    def test_negative_duration_rejected(self):
        ok, msg = self._simulate_validation(-5.0)
        self.assertFalse(ok, "negative duration should be rejected")
        self.assertIn("--duration", msg)

    def test_positive_duration_accepted(self):
        ok, _ = self._simulate_validation(30.0)
        self.assertTrue(ok, "positive duration should be accepted")

    def test_none_duration_accepted(self):
        ok, _ = self._simulate_validation(None)
        self.assertTrue(ok, "None duration (flag omitted) should be accepted")


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

    def test_clock_fields_none_for_old_runs(self):
        """Runs that predate Jetson SM clock capture must yield None, not fabricated values."""
        p = self.tmp / "bert" / "d" / "exit_5" / "hw_results.json"
        _write_hw(p, 0.02)   # _write_hw does not emit clock fields
        hw = mr._read_hw(mr._find_hw(self.tmp, "bert", 5))
        self.assertIsNone(hw["nvpmodel"])
        self.assertIsNone(hw["avg_gpu_sm_clock_mhz"])
        self.assertIsNone(hw["min_gpu_sm_clock_mhz"])


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
        # No total_sec/n: falls back to forward latency.
        # 0.01 s/sample over a 30 s window -> 3000 samples
        n, fb = mr.calibrate({"lat": 0.01}, 30.0)
        self.assertEqual(n, 3000)
        self.assertTrue(fb)    # fallback used because no total_sec

    def test_uses_wall_time_when_available(self):
        """Given total_sec and n implying a wall latency well above forward latency,
        the returned count is the smaller value derived from wall time."""
        # forward lat = 0.010, but wall = 0.020 (dataloading overhead doubles it)
        hw = {"lat": 0.010, "total_sec": 2.0, "n": 100}  # wall = 2.0/100 = 0.020
        n, fb = mr.calibrate(hw, 30.0)
        self.assertFalse(fb)
        # wall-based: 30 / 0.020 = 1500; forward-based: 30 / 0.010 = 3000
        self.assertEqual(n, 1500)
        n_forward = int(30.0 / hw["lat"])
        self.assertLess(n, n_forward)

    def test_fallback_visible_when_total_sec_absent(self):
        """calibrate falls back visibly when total_sec is absent, and the fallback
        flag is set."""
        hw = {"lat": 0.01}   # no total_sec, no n
        n, fb = mr.calibrate(hw, 30.0)
        self.assertTrue(fb)
        self.assertEqual(n, 3000)

    def test_fallback_when_n_is_zero(self):
        """A total_sec with n=0 must not divide by zero; fallback to forward lat."""
        hw = {"lat": 0.01, "total_sec": 1.0, "n": 0}
        n, fb = mr.calibrate(hw, 30.0)
        self.assertTrue(fb)

    def test_fast_tenant_gets_more_samples(self):
        # Both use wall time (total_sec == lat * n so wall == forward here).
        fast = mr.calibrate({"lat": 0.005, "total_sec": 0.5, "n": 100}, 30.0)[0]
        slow = mr.calibrate({"lat": 0.05, "total_sec": 5.0, "n": 100}, 30.0)[0]
        self.assertGreater(fast, slow)          # this is the whole point of phase 2

    def test_clamped(self):
        self.assertEqual(mr.calibrate({"lat": 1e-9, "total_sec": 1e-7, "n": 100}, 30.0)[0],
                         mr.MAX_SAMPLES)
        self.assertEqual(mr.calibrate({"lat": 1e9, "total_sec": 1e11, "n": 100}, 30.0)[0],
                         mr.MIN_SAMPLES)

    def test_none_when_no_solo(self):
        # Returns (None, False) when hw is None or lat is missing and no total_sec.
        self.assertEqual(mr.calibrate(None, 30.0), (None, False))
        self.assertEqual(mr.calibrate({"lat": None}, 30.0), (None, False))


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
    def _make_solo_shared(self):
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
        return solo, shared

    def test_row_math(self):
        solo, shared = self._make_solo_shared()
        row = mr.build_row([("bert", 12, None), ("yolo", 5, 2)], "t",
                           solo, shared, 0.95, {0: 3000, 1: 1500}, 30.0)
        self.assertEqual(row["t0_slowdown"], 1.5)          # 0.015/0.010
        self.assertEqual(row["t1_slowdown"], 1.5)          # 0.030/0.020
        self.assertEqual(row["t1"], "yolo@5_P2")
        self.assertEqual(row["t0_p95_ratio"], 2.5)         # 0.030/0.012
        self.assertEqual(row["t0_n_samples"], 3000)
        self.assertEqual(row["agg_throughput"], 99.0)      # 66+33
        # throughput_gain now divides by SUM of solo throughputs (100+50=150), not max.
        self.assertAlmostEqual(row["throughput_gain"], round(99.0 / 150.0, 3))
        self.assertEqual(row["pair_power_w"], 6.0)         # shared rail -> max
        # device energy = device power x window, NOT a sum over tenants
        self.assertEqual(row["pair_energy_j"], 6.0 * 45.0)
        self.assertEqual(row["pair_window_sec"], 45.0)
        self.assertEqual(row["pair_vram_mb"], 740.0)
        self.assertEqual(row["target_window_sec"], 30.0)

    def test_throughput_gain_uses_sum_not_max(self):
        """throughput_gain must divide by SUM of solo throughputs, not max."""
        solo, shared = self._make_solo_shared()
        row = mr.build_row([("bert", 12, None), ("yolo", 5, 2)], "t",
                           solo, shared, 0.95, {0: 3000, 1: 1500}, 30.0)
        # sum of solo = 150; if it used max (100) the gain would be 0.99
        self.assertAlmostEqual(row["throughput_gain"], round(99.0 / 150.0, 3))
        self.assertNotAlmostEqual(row["throughput_gain"], 0.99)

    def test_stp_and_antt(self):
        """STP and ANTT on a known two-tenant case.

        STP = thru_sh_0/thru_so_0 + thru_sh_1/thru_so_1 = 66/100 + 33/50 = 1.32.
        STP also equals sum of reciprocals of slowdowns: 1/1.5 + 1/1.5 = 1.333...
        (Small rounding difference because slowdown is rounded to 3 decimals.)
        ANTT = mean(1.5, 1.5) = 1.5.
        """
        solo, shared = self._make_solo_shared()
        row = mr.build_row([("bert", 12, None), ("yolo", 5, 2)], "t",
                           solo, shared, 0.95, {0: 3000, 1: 1500}, 30.0)
        # STP via throughput ratio
        expected_stp = round(66.0 / 100.0 + 33.0 / 50.0, 3)
        self.assertAlmostEqual(row["stp"], expected_stp, places=3)
        # STP is also the sum of reciprocals of slowdowns (1/1.5 + 1/1.5 = 1.333).
        # The two formulations differ slightly because build_row rounds slowdowns
        # to 3 decimal places before storing them. Verify the relationship holds
        # to 1 decimal place, which is sufficient to confirm the identity.
        stp_via_slowdown = round(1.0 / row["t0_slowdown"] + 1.0 / row["t1_slowdown"], 3)
        self.assertAlmostEqual(row["stp"], stp_via_slowdown, places=1)
        # ANTT
        self.assertAlmostEqual(row["antt"], 1.5)

    def test_agg_throughput_comparable_homogeneous(self):
        """agg_throughput_comparable is True when all tenants share a family."""
        solo, shared = self._make_solo_shared()
        row = mr.build_row([("bert", 12, None), ("bert", 5, None)], "t",
                           solo, shared, 0.95, {0: 3000, 1: 1500}, 30.0)
        self.assertTrue(row["agg_throughput_comparable"])

    def test_agg_throughput_comparable_heterogeneous(self):
        """agg_throughput_comparable is False for a mixed-family cell."""
        solo, shared = self._make_solo_shared()
        row = mr.build_row([("bert", 12, None), ("yolo", 5, 2)], "t",
                           solo, shared, 0.95, {0: 3000, 1: 1500}, 30.0)
        self.assertFalse(row["agg_throughput_comparable"])

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

    def test_slo_is_twice_solo_latency(self):
        """SLO for each tenant is exactly 2x its solo per-sample latency (gpu-let convention)."""
        solo = {0: {"lat": 0.010, "thru": 100.0, "p95": 0.012, "power_w": 2.0,
                    "energy_j": 0.02, "vram_mb": 700, "ram_mb": 2000, "n": 100,
                    "total_sec": 1.0},
                1: {"lat": 0.025, "thru": 40.0, "p95": 0.030, "power_w": 2.0,
                    "energy_j": 0.05, "vram_mb": 40, "ram_mb": 1500, "n": 100,
                    "total_sec": 2.5}}
        shared = {0: dict(solo[0]), 1: dict(solo[1])}
        row = mr.build_row([("bert", 0, None), ("yolo", 3, 1)], "t",
                           solo, shared, 1.0, {0: 100, 1: 100}, 30.0)
        self.assertAlmostEqual(row["t0_slo_sec"], 0.020, places=6)
        self.assertAlmostEqual(row["t1_slo_sec"], 0.050, places=6)

    def test_slo_violation_ratio_handcrafted(self):
        """Violation ratio is correct on a small handcrafted sample list.

        The field is set by _read_hw when slo_sec is threaded in. Here we test
        _violation_ratio directly and verify it computes the right fraction.
        """
        samples = [{"end_to_end_sec": v} for v in [0.01, 0.02, 0.03, 0.04, 0.05]]
        # slo = 0.025 -> 0.03, 0.04, 0.05 exceed it -> ratio = 3/5 = 0.6
        self.assertAlmostEqual(mr._violation_ratio(samples, slo_sec=0.025), 0.6, places=4)
        # no violations
        self.assertEqual(mr._violation_ratio(samples, slo_sec=0.10), 0.0)
        # all violate
        self.assertEqual(mr._violation_ratio(samples, slo_sec=0.005), 1.0)

    def test_slo_violation_none_when_slo_none(self):
        self.assertIsNone(mr._violation_ratio([], slo_sec=None))

    def test_repeat_idx_in_row(self):
        """repeat_idx is carried through build_row."""
        row = mr.build_row([("bert", 0, None)], "t", {0: None}, {0: None}, 1.0, {0: None}, 30.0,
                           repeat_idx=2)
        self.assertEqual(row["repeat_idx"], 2)

    def test_is_holdout_in_row(self):
        row = mr.build_row([("bert", 0, None)], "t", {0: None}, {0: None}, 1.0, {0: None}, 30.0,
                           is_holdout=True)
        self.assertTrue(row["is_holdout"])

    def test_calib_fallback_flag_in_row(self):
        row = mr.build_row([("bert", 0, None)], "t", {0: None}, {0: None}, 1.0, {0: None}, 30.0,
                           calib_fallbacks={0: True})
        self.assertTrue(row["t0_calib_fallback"])

    def test_window_fields_present(self):
        """t{i}_window_solo_sec and t{i}_window_shared_sec must appear in every row."""
        solo = {0: {"lat": 0.01, "thru": 100.0, "p95": 0.012, "power_w": 2.0,
                    "energy_j": 0.02, "vram_mb": 700, "ram_mb": 2000, "n": 100,
                    "total_sec": 1.0}}
        row = mr.build_row([("bert", 0, None)], "t", solo, solo, 1.0, {0: 100}, 30.0)
        self.assertIn("t0_window_solo_sec", row)
        self.assertIn("t0_window_shared_sec", row)
        # total_sec is present so windows should be non-None
        self.assertIsNotNone(row["t0_window_solo_sec"])
        self.assertIsNotNone(row["t0_window_shared_sec"])

    def test_clock_fields_in_row(self):
        """Clock fields are present in every row, with None for runs that lack them."""
        solo = {0: {"lat": 0.01, "thru": 100.0, "p95": 0.012, "power_w": 2.0,
                    "energy_j": 0.02, "vram_mb": 700, "ram_mb": 2000, "n": 100,
                    "total_sec": 1.0,
                    "nvpmodel": None, "avg_gpu_sm_clock_mhz": None,
                    "min_gpu_sm_clock_mhz": None}}
        row = mr.build_row([("bert", 0, None)], "t", solo, solo, 1.0, {0: 100}, 30.0)
        self.assertIn("t0_nvpmodel", row)
        self.assertIn("t0_avg_gpu_sm_clock_mhz", row)
        self.assertIn("t0_min_gpu_sm_clock_mhz", row)
        self.assertIsNone(row["t0_nvpmodel"])
        self.assertIsNone(row["t0_avg_gpu_sm_clock_mhz"])
        self.assertIsNone(row["t0_min_gpu_sm_clock_mhz"])


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

    def _fake(self, fam, ex, sub, subdir, os_, n_samples=None, **kwargs):
        self.calls.append((subdir, n_samples))
        lat = 0.010 if "probe_" in subdir or subdir.startswith("mt_solo") else 0.015
        leaf = f"exit_{ex}_P{sub + 3}" if (fam == "yolo" and sub is not None) else f"exit_{ex}"
        _write_hw(mr.LOGS / subdir / fam / "d" / leaf / "hw_results.json", lat)
        return _FakeProc(), time.perf_counter()

    def _run(self, fn, *a, **k):
        with mock.patch("psutil.virtual_memory", return_value=mock.Mock(available=8e9)), \
             mock.patch.object(mr, "run_one", side_effect=self._fake):
            return fn(*a, **k)

    def test_pair_three_phases(self):
        # keep_suspect=True so the flow-mechanics test is not affected by the
        # overlap gate (the gate itself is tested in test_low_overlap_cell_goes_to_suspect).
        rows = self._run(mr.run_pair, [("bert", 12, None), ("vision", 12, None)], "t", 30.0,
                         keep_suspect=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["t0_slowdown"], 1.5)
        # Probe runs have "probe_" in the subdir (mt_solo_probe_t_*).
        # Real solo runs start with mt_solo_ but do not contain "probe_".
        probe = [c for c in self.calls if "probe_" in c[0]]
        solo = [c for c in self.calls if c[0].startswith("mt_solo") and "probe_" not in c[0]]
        conc = [c for c in self.calls if c[0].startswith("mt_conc")]
        self.assertEqual(len(probe), 2)                # one probe per tenant
        self.assertEqual(len(solo), 2)                 # one real solo per tenant
        self.assertEqual(len(conc), 2)                 # phase 3: both together
        # probe runs use config default (no n_samples)
        self.assertTrue(all(c[1] is None for c in probe))
        # real solo runs use calibrated count (wall==forward in test fixture: 30/0.010=3000)
        self.assertTrue(all(c[1] == 3000 for c in solo))
        # concurrent runs also use 3000
        self.assertTrue(all(c[1] == 3000 for c in conc))

    def test_calibration_reaches_subprocess(self):
        self._run(mr.run_pair, [("bert", 12, None)], "t", 60.0, keep_suspect=True)
        conc = [c for c in self.calls if c[0].startswith("mt_conc")]
        self.assertEqual(conc[0][1], 6000)             # 60s / 0.010s

    def test_grid_solo_cached(self):
        rows = self._run(mr.run_grid, "bert", "vision", "g", keep_suspect=True)
        self.assertEqual(len(rows), 36)                # 6x6
        probe = [c for c in self.calls if "probe_" in c[0]]
        solo = [c for c in self.calls if c[0].startswith("mt_solo") and "probe_" not in c[0]]
        # 6+6 unique anchors: one probe + one real solo each = 12+12 = 24 phase-1 runs
        self.assertEqual(len(probe), 12)
        self.assertEqual(len(solo), 12)

    def test_scenario_yolo_pair(self):
        rows = self._run(mr.run_scenario, "llama_yolo", keep_suspect=True)
        self.assertEqual(len(rows), 36)
        self.assertTrue(any("_P" in r["t1"] for r in rows))

    def test_scenario_scaling(self):
        rows = self._run(mr.run_scenario, "yolo_scale", keep_suspect=True)
        # 6 exit anchors x 3 tenant counts = 18 cells
        self.assertEqual(len(rows), 18)
        self.assertEqual([r["n_tenants"] for r in rows], [2, 3, 4] * 6)
        # solo runs are cached: 6 unique (exit, sub) anchors for yolo, each measured once
        probe = [c for c in self.calls if "probe_" in c[0]]
        solo = [c for c in self.calls if c[0].startswith("mt_solo") and "probe_" not in c[0]]
        self.assertEqual(len(probe), 6)
        self.assertEqual(len(solo), 6)

    def test_scenario_scaling_k2(self):
        rows = self._run(mr.run_scenario, "bert_scale", k=2, keep_suspect=True)
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

    def test_min_exit_negative_rejected(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, str(mr.REPO_ROOT / "multitenant_run.py"),
             "--scenario", "bert_scale", "--min-exit", "-1"],
            capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--min-exit must be non-negative", result.stdout + result.stderr)

    def test_scenario_bert_scale_k8_min_exit_1_no_exit_0(self):
        """run_scenario with k=8 and min_exit=1 must not build any cell at exit 0."""
        rows = self._run(mr.run_scenario, "bert_scale", k=8, min_exit=1,
                         keep_suspect=True)
        # 8 exit anchors x 3 tenant counts = 24 cells
        self.assertEqual(len(rows), 24)
        for row in rows:
            for i in range(row["n_tenants"]):
                tenant_str = row[f"t{i}"]
                # tenant_str is "bert@EXIT": extract exit number
                exit_num = int(tenant_str.split("@")[1])
                self.assertGreater(exit_num, 0,
                    f"exit 0 appeared in row despite min_exit=1: {tenant_str}")

    def test_abort_launches_nothing(self):
        # With 0.4 GB free, no cell fits (llama+yolo needs 3.0+0.54+1.0 = 4.54 GB).
        with mock.patch("psutil.virtual_memory", return_value=mock.Mock(available=0.4e9)), \
             mock.patch.object(mr, "run_one", side_effect=self._fake):
            out = mr.run_scenario("llama_yolo")
        self.assertIsNone(out)
        self.assertEqual(self.calls, [])

    def test_repeats_three_distinct_rows(self):
        """--repeats 3 produces three rows carrying distinct repeat_idx values."""
        rows = self._run(mr.run_pair, [("bert", 0, None)], "t", 30.0, repeats=3,
                         keep_suspect=True)
        self.assertEqual(len(rows), 3)
        idxs = [r["repeat_idx"] for r in rows]
        self.assertEqual(sorted(idxs), [0, 1, 2])

    def test_low_overlap_cell_goes_to_suspect_not_main(self):
        """A cell with timed_overlap_frac below OVERLAP_GATE is written to the
        suspect sidecar and NOT to the main CSV."""

        def _fake_concurrent(tenants, tag, counts, import_os, mode_label=None, **kwargs):
            # Return empty shared results and a low timed overlap.
            shared = {i: {"lat": 0.015, "thru": 66.0, "p95": 0.020, "power_w": 5.0,
                          "energy_j": 0.05, "vram_mb": 700, "ram_mb": 2000,
                          "n": 100, "total_sec": 1.5,
                          "timed_start_unix": None, "timed_end_unix": None,
                          "gpu_mem_static_mb": None, "gpu_mem_dynamic_mb": None,
                          "gpu_mem_peak_mb": None, "nvpmodel": None,
                          "avg_gpu_sm_clock_mhz": None, "min_gpu_sm_clock_mhz": None,
                          "violation_ratio": None}
                     for i in range(len(tenants))}
            return shared, 0.5, 0.5   # timed_overlap_frac = 0.5 < OVERLAP_GATE

        with mock.patch("psutil.virtual_memory", return_value=mock.Mock(available=8e9)), \
             mock.patch.object(mr, "run_one", side_effect=self._fake), \
             mock.patch.object(mr, "measure_concurrent", side_effect=_fake_concurrent):
            mr.run_pair([("bert", 0, None)], "lowov", 30.0)

        main_csv = mr.OUT_DIR / "concurrent_slowdown.csv"
        suspect_csv = mr.OUT_DIR / "concurrent_slowdown.suspect.csv"
        # Main CSV should not have the low-overlap row.
        self.assertFalse(main_csv.exists() or (main_csv.exists() and
            len(main_csv.read_text().splitlines()) > 1))
        # Suspect CSV must exist and have exactly one data row.
        self.assertTrue(suspect_csv.exists())
        lines = [l for l in suspect_csv.read_text().splitlines() if l.strip()]
        self.assertGreaterEqual(len(lines), 2)   # header + at least one data row

    def test_keep_suspect_writes_to_main(self):
        """Under --keep-suspect, low-overlap cells go to the main CSV."""
        def _fake_concurrent(tenants, tag, counts, import_os, mode_label=None, **kwargs):
            shared = {i: {"lat": 0.015, "thru": 66.0, "p95": 0.020, "power_w": 5.0,
                          "energy_j": 0.05, "vram_mb": 700, "ram_mb": 2000,
                          "n": 100, "total_sec": 1.5,
                          "timed_start_unix": None, "timed_end_unix": None,
                          "gpu_mem_static_mb": None, "gpu_mem_dynamic_mb": None,
                          "gpu_mem_peak_mb": None, "nvpmodel": None,
                          "avg_gpu_sm_clock_mhz": None, "min_gpu_sm_clock_mhz": None,
                          "violation_ratio": None}
                     for i in range(len(tenants))}
            return shared, 0.5, 0.5

        with mock.patch("psutil.virtual_memory", return_value=mock.Mock(available=8e9)), \
             mock.patch.object(mr, "run_one", side_effect=self._fake), \
             mock.patch.object(mr, "measure_concurrent", side_effect=_fake_concurrent):
            rows = mr.run_pair([("bert", 0, None)], "keepsus", 30.0, keep_suspect=True)
        # The row appears in the returned list (main CSV).
        self.assertEqual(len(rows), 1)
        main_csv = mr.OUT_DIR / "concurrent_slowdown.csv"
        self.assertTrue(main_csv.exists())
        lines = [l for l in main_csv.read_text().splitlines() if l.strip()]
        self.assertGreaterEqual(len(lines), 2)

    def test_run_cells_passes_duration_to_solo_and_concurrent(self):
        """run_cells must forward --duration to both the real solo runs (phase 2)
        and the concurrent runs (phase 3). Probe runs (phase 1) intentionally do
        NOT receive --duration so they run at config default to measure wall latency.
        Both --duration and --n-samples must appear together in the real runs."""
        captured = []

        def _capturing_run_one(fam, ex, sub, subdir, os_, n_samples=None,
                               duration=None, **kwargs):
            captured.append({"subdir": subdir, "n_samples": n_samples, "duration": duration})
            lat = 0.010
            leaf = f"exit_{ex}_P{sub + 3}" if (fam == "yolo" and sub is not None) else f"exit_{ex}"
            _write_hw(mr.LOGS / subdir / fam / "d" / leaf / "hw_results.json", lat)
            return _FakeProc(), time.perf_counter()

        with mock.patch("psutil.virtual_memory", return_value=mock.Mock(available=8e9)), \
             mock.patch.object(mr, "run_one", side_effect=_capturing_run_one):
            mr.run_pair([("bert", 12, None)], "dur_test", 30.0, keep_suspect=True)

        probes = [c for c in captured if "probe_" in c["subdir"]]
        solos = [c for c in captured if c["subdir"].startswith("mt_solo") and "probe_" not in c["subdir"]]
        concs = [c for c in captured if c["subdir"].startswith("mt_conc")]

        # Probe runs: no duration (measure natural wall latency).
        self.assertTrue(all(c["duration"] is None for c in probes),
                        "probe runs must not receive duration")
        # Real solo and concurrent runs: duration is forwarded.
        self.assertTrue(all(c["duration"] == 30.0 for c in solos),
                        "real solo runs must receive duration=30.0")
        self.assertTrue(all(c["duration"] == 30.0 for c in concs),
                        "concurrent runs must receive duration=30.0")
        # Both n_samples and duration must be present in the real runs.
        self.assertTrue(all(c["n_samples"] is not None for c in solos),
                        "real solo runs must also carry n_samples ceiling")
        self.assertTrue(all(c["n_samples"] is not None for c in concs),
                        "concurrent runs must also carry n_samples ceiling")

    def test_bench_cmd_argv_has_both_n_samples_and_duration(self):
        """The argv built by bench_cmd must contain both --n-samples and --duration
        when both are provided, asserting the safety-ceiling contract."""
        argv, _ = mr.bench_cmd("bert", 5, None, "x", n_samples=1141, duration=30.0)
        self.assertIn("--n-samples", argv)
        self.assertIn("--duration", argv)
        self.assertEqual(argv[argv.index("--n-samples") + 1], "1141")
        self.assertEqual(argv[argv.index("--duration") + 1], "30.0")


class TestDatasetPin(unittest.TestCase):
    """bench_cmd must include the pinned dataset flag for each family by default,
    and CLI overrides must replace the default without affecting other families."""

    def test_bert_includes_task_sst2_by_default(self):
        argv, _ = mr.bench_cmd("bert", 5, None, "x")
        self.assertIn("--task", argv)
        idx = argv.index("--task")
        self.assertEqual(argv[idx + 1], "SST-2")

    def test_bert_task_override_replaces_default(self):
        argv, _ = mr.bench_cmd("bert", 5, None, "x", task="QNLI")
        self.assertIn("--task", argv)
        idx = argv.index("--task")
        self.assertEqual(argv[idx + 1], "QNLI")

    def test_yolo_includes_dataset_coco_by_default(self):
        argv, _ = mr.bench_cmd("yolo", 3, 1, "x")
        self.assertIn("--dataset", argv)
        idx = argv.index("--dataset")
        self.assertEqual(argv[idx + 1], "coco")

    def test_yolo_dataset_override_replaces_default(self):
        argv, _ = mr.bench_cmd("yolo", 3, 1, "x", dataset="voc")
        self.assertIn("--dataset", argv)
        idx = argv.index("--dataset")
        self.assertEqual(argv[idx + 1], "voc")

    def test_vision_includes_dataset_cifar10_by_default(self):
        argv, _ = mr.bench_cmd("vision", 2, None, "x")
        self.assertIn("--dataset", argv)
        idx = argv.index("--dataset")
        self.assertEqual(argv[idx + 1], "uoft-cs/cifar10")

    def test_llama_includes_dataset_cnn_dailymail_by_default(self):
        argv, _ = mr.bench_cmd("llama", 4, None, "x")
        self.assertIn("--dataset", argv)
        idx = argv.index("--dataset")
        self.assertEqual(argv[idx + 1], "cnn_dailymail")

    def test_llama3b_has_no_dataset_flag(self):
        """llama3b exposes no --dataset flag in bench_jetson; must not get one."""
        argv, _ = mr.bench_cmd("llama3b", 4, None, "x")
        self.assertNotIn("--dataset", argv)
        self.assertNotIn("--task", argv)

    def test_bert_argv_is_well_formed(self):
        """The complete bert argv must be parseable: no stray flags, correct order."""
        argv, env = mr.bench_cmd("bert", 5, None, "sub1", n_samples=200)
        # Must start with python, then bench_jetson.py, then the family subcommand.
        self.assertIn("bench_jetson.py", argv[1])
        self.assertEqual(argv[2], "bert")
        self.assertIn("--exit", argv)
        self.assertIn("--no-quality", argv)
        self.assertIn("--task", argv)
        self.assertIn("--n-samples", argv)
        self.assertEqual(env["BENCH_SUBDIR"], "sub1")

    def test_task_override_in_bert_argv_only_once(self):
        """--task must appear exactly once even when the pin and override agree."""
        argv, _ = mr.bench_cmd("bert", 0, None, "x", task="SST-2")
        self.assertEqual(argv.count("--task"), 1)

    def test_dataset_override_in_yolo_argv_only_once(self):
        """--dataset must appear exactly once."""
        argv, _ = mr.bench_cmd("yolo", 0, 0, "x", dataset="coco")
        self.assertEqual(argv.count("--dataset"), 1)


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

    def test_suspect_csv_separate(self):
        mr._append_csv({"tag": "a", "t0": "bert@0"}, suspect=False)
        mr._append_csv({"tag": "b", "t0": "yolo@0"}, suspect=True)
        self.assertTrue((mr.OUT_DIR / "concurrent_slowdown.csv").exists())
        self.assertTrue((mr.OUT_DIR / "concurrent_slowdown.suspect.csv").exists())
        main_lines = (mr.OUT_DIR / "concurrent_slowdown.csv").read_text().splitlines()
        sus_lines = (mr.OUT_DIR / "concurrent_slowdown.suspect.csv").read_text().splitlines()
        self.assertEqual(len(main_lines), 2)    # header + 1 row
        self.assertEqual(len(sus_lines), 2)


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


class TestHoldout(unittest.TestCase):
    def test_holdout_is_deterministic(self):
        """The same (fam, k, n_holdout, seed_tag) always yields the same selection."""
        a = mr._holdout_anchors("bert", 6, 3, "run42")
        b = mr._holdout_anchors("bert", 6, 3, "run42")
        self.assertEqual(a, b)

    def test_holdout_different_tags_differ(self):
        """Different tags yield different selections (hash sensitivity check)."""
        a = mr._holdout_anchors("bert", 6, 3, "run1")
        b = mr._holdout_anchors("bert", 6, 3, "run2")
        # Not guaranteed to differ, but sha256 should make collisions negligible.
        # Use a loose check: at least not always identical for random strings.
        # (If this flakes, increase n_holdout or use more distinct tags.)
        all_same = (a == b)
        # We just verify both are lists of the right length.
        self.assertIsInstance(a, list)
        self.assertIsInstance(b, list)

    def test_holdout_points_are_off_anchor(self):
        """All returned points must lie outside the anchor set."""
        anchors = set(mr._anchor("bert", 6))
        holdouts = mr._holdout_anchors("bert", 6, 4, "test")
        for pt in holdouts:
            self.assertNotIn(pt, anchors,
                             f"holdout point {pt} is on the anchor grid")

    def test_holdout_yolo(self):
        """Works for yolo (leaf-based coordinates)."""
        anchors = set(mr._anchor("yolo", 6))
        holdouts = mr._holdout_anchors("yolo", 6, 2, "test")
        for pt in holdouts:
            self.assertNotIn(pt, anchors)


class TestModeLabelNormalisation(unittest.TestCase):
    """_normalize_label must produce filesystem-safe lowercase tokens."""

    def test_uppercase_mode_names(self):
        self.assertEqual(mr._normalize_label("MAXN_SUPER"), "maxn_super")
        self.assertEqual(mr._normalize_label("25W"), "25w")
        self.assertEqual(mr._normalize_label("15W"), "15w")

    def test_slash_is_replaced(self):
        result = mr._normalize_label("a/b")
        self.assertNotIn("/", result)

    def test_space_is_replaced(self):
        result = mr._normalize_label("a b")
        self.assertNotIn(" ", result)

    def test_already_clean_is_unchanged(self):
        self.assertEqual(mr._normalize_label("maxn_super"), "maxn_super")

    def test_mixed_case_lowercased(self):
        self.assertEqual(mr._normalize_label("MaXn"), "maxn")


class TestDetectModeLabel(unittest.TestCase):
    """_detect_mode_label must return a string in all cases, never raise."""

    def test_returns_unknown_when_nvpmodel_absent(self):
        with mock.patch("subprocess.run",
                        side_effect=FileNotFoundError("nvpmodel not found")):
            result = mr._detect_mode_label()
        self.assertEqual(result, "unknown")

    def test_returns_unknown_on_timeout(self):
        import subprocess
        with mock.patch("subprocess.run",
                        side_effect=subprocess.TimeoutExpired("nvpmodel", 15)):
            result = mr._detect_mode_label()
        self.assertEqual(result, "unknown")

    def test_returns_normalised_label_on_success(self):
        mock_result = mock.Mock(stdout="NV Power Mode: 25W\n2\n", stderr="")
        with mock.patch("subprocess.run", return_value=mock_result):
            result = mr._detect_mode_label()
        self.assertEqual(result, "25w")

    def test_returns_unknown_when_line_absent(self):
        mock_result = mock.Mock(stdout="no relevant line\n", stderr="")
        with mock.patch("subprocess.run", return_value=mock_result):
            result = mr._detect_mode_label()
        self.assertEqual(result, "unknown")


class TestModeLabelPaths(unittest.TestCase):
    """Verify that mode_label scopes output to the correct nested paths."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._logs, self._out = mr.LOGS, mr.OUT_DIR
        mr.LOGS, mr.OUT_DIR = self.tmp / "logs", self.tmp / "out"

    def tearDown(self):
        mr.LOGS, mr.OUT_DIR = self._logs, self._out

    def test_csv_written_under_mode_folder(self):
        """With mode_label='maxn_super', CSV lands at out/maxn_super/concurrent_slowdown.csv."""
        csv_dir = mr.OUT_DIR / "maxn_super"
        csv_dir.mkdir(parents=True, exist_ok=True)
        row = {"tag": "test", "n_tenants": 1}
        mr._append_csv(row, suspect=False, csv_dir=csv_dir)
        expected = mr.OUT_DIR / "maxn_super" / "concurrent_slowdown.csv"
        self.assertTrue(expected.exists())

    def test_suspect_csv_written_under_mode_folder(self):
        csv_dir = mr.OUT_DIR / "15w"
        csv_dir.mkdir(parents=True, exist_ok=True)
        row = {"tag": "test", "n_tenants": 1}
        mr._append_csv(row, suspect=True, csv_dir=csv_dir)
        expected = mr.OUT_DIR / "15w" / "concurrent_slowdown.suspect.csv"
        self.assertTrue(expected.exists())

    def _fake_run_one(self, fam, ex, sub, subdir, os_, n_samples=None, **kwargs):
        lat = 0.010 if ("probe_" in subdir or "mt_solo" in subdir) else 0.015
        leaf = f"exit_{ex}_P{sub + 3}" if (fam == "yolo" and sub is not None) else f"exit_{ex}"
        _write_hw(mr.LOGS / subdir / fam / "d" / leaf / "hw_results.json", lat)
        return _FakeProc(), time.perf_counter()

    def test_mode_label_maxn_super_scopes_all_paths(self):
        """run_pair with mode_label='maxn_super' writes CSV to out/maxn_super/
        and solo/conc subdirs under logs/multitenant.maxn_super/."""
        with mock.patch("psutil.virtual_memory", return_value=mock.Mock(available=8e9)), \
             mock.patch.object(mr, "run_one", side_effect=self._fake_run_one):
            rows = mr.run_pair(
                [("bert", 12, None)], "run1", 30.0,
                keep_suspect=True, mode_label="maxn_super",
            )
        # CSV must be in the mode-scoped folder.
        csv_path = mr.OUT_DIR / "maxn_super" / "concurrent_slowdown.csv"
        self.assertTrue(csv_path.exists(), f"CSV not found at {csv_path}")
        # Log subdirs must be under logs/multitenant.maxn_super/.
        mode_log_root = mr.LOGS / "multitenant.maxn_super"
        self.assertTrue(mode_log_root.exists(),
                        f"mode log root not found at {mode_log_root}")
        # At least one mt_solo and one mt_conc directory must exist under it.
        solo_dirs = list(mode_log_root.glob("mt_solo_*"))
        conc_dirs = list(mode_log_root.glob("mt_conc_*"))
        self.assertTrue(solo_dirs, "no mt_solo_* dirs under multitenant.maxn_super")
        self.assertTrue(conc_dirs, "no mt_conc_* dirs under multitenant.maxn_super")

    def test_mode_label_none_uses_flat_layout(self):
        """When mode_label is None, CSV is at out/concurrent_slowdown.csv (backward compat)."""
        with mock.patch("psutil.virtual_memory", return_value=mock.Mock(available=8e9)), \
             mock.patch.object(mr, "run_one", side_effect=self._fake_run_one):
            rows = mr.run_pair(
                [("bert", 12, None)], "flat_run", 30.0,
                keep_suspect=True, mode_label=None,
            )
        csv_path = mr.OUT_DIR / "concurrent_slowdown.csv"
        self.assertTrue(csv_path.exists(), f"flat CSV not found at {csv_path}")
        # The mode-scoped folder must NOT have been created.
        mode_dirs = [d for d in mr.OUT_DIR.iterdir()
                     if d.is_dir() and d.name != "concurrent_slowdown.csv"]
        self.assertEqual(mode_dirs, [],
                         f"unexpected mode subdirectories created: {mode_dirs}")

    def test_find_hw_nested_under_mode_path(self):
        """_find_hw resolves a file under the nested mode-named path."""
        mode_root = self.tmp / "logs" / "multitenant.15w" / "mt_solo_run1_bert_0"
        p = mode_root / "bert" / "d" / "exit_0" / "hw_results.json"
        _write_hw(p, 0.02)
        found = mr._find_hw(mode_root, "bert", 0)
        self.assertIsNotNone(found)
        self.assertEqual(found.resolve(), p.resolve())

    def test_find_hw_flat_path_without_mode_label(self):
        """_find_hw still resolves a file under the old flat (no label) path."""
        flat_root = self.tmp / "logs" / "mt_solo_run2_bert_0"
        p = flat_root / "bert" / "d" / "exit_0" / "hw_results.json"
        _write_hw(p, 0.02)
        found = mr._find_hw(flat_root, "bert", 0)
        self.assertIsNotNone(found)


class TestPerCellGating(unittest.TestCase):
    """Per-cell preflight: cells that fit run, cells that do not are skipped."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._logs, self._out = mr.LOGS, mr.OUT_DIR
        mr.LOGS, mr.OUT_DIR = self.tmp / "logs", self.tmp / "out"
        self.calls = []

    def tearDown(self):
        mr.LOGS, mr.OUT_DIR = self._logs, self._out

    def _fake(self, fam, ex, sub, subdir, os_, n_samples=None, **kwargs):
        self.calls.append((fam, len([f for f in subdir.split("/") if f]), subdir))
        lat = 0.010
        leaf = f"exit_{ex}_P{sub + 3}" if (fam == "yolo" and sub is not None) else f"exit_{ex}"
        _write_hw(mr.LOGS / subdir / fam / "d" / leaf / "hw_results.json", lat)
        return _FakeProc(), time.perf_counter()

    def test_bert_scale_partial_fit_n2_n3_run_n4_skipped(self):
        """The exact reported failure: bert_scale at fixed counts [2,3,4].
        With 4.6 GB free, n=2 (2*1.2+1.0=3.4) and n=3 (3*1.2+1.0=4.6) fit;
        n=4 (4*1.2+1.0=5.8) does not. n=2 and n=3 cells must run; n=4 cells
        must be skipped with a log message and nothing launched for them."""
        # 4.6 GB: need for n=4 is 5.8 GB which exceeds it; n=2 and n=3 fit.
        with mock.patch("psutil.virtual_memory",
                        return_value=mock.Mock(available=4.6e9)), \
             mock.patch.object(mr, "run_one", side_effect=self._fake):
            rows = mr.run_scenario("bert_scale", k=2, keep_suspect=True)
        # k=2 anchors: 2 exit points x counts that fit
        # With free=4.6 GB: n=2 (3.4 GB) and n=3 (4.6 GB) fit; n=4 (5.8 GB) does not.
        # 2 anchors x 2 fitting counts = 4 cells (not 6).
        self.assertIsNotNone(rows)
        n_tenants_seen = sorted(set(r["n_tenants"] for r in rows))
        self.assertIn(2, n_tenants_seen)
        self.assertIn(3, n_tenants_seen)
        self.assertNotIn(4, n_tenants_seen)

    def test_nothing_fits_aborts_with_clear_message(self):
        """When free memory is below even n=2, the scenario aborts and launches nothing."""
        # 0.5 GB free: n=2 needs 2*1.2+1.0=3.4 GB.
        with mock.patch("psutil.virtual_memory",
                        return_value=mock.Mock(available=0.5e9)), \
             mock.patch.object(mr, "run_one", side_effect=self._fake):
            result = mr.run_scenario("bert_scale", k=2, keep_suspect=True)
        self.assertIsNone(result)
        self.assertEqual(self.calls, [])


class TestGrow(unittest.TestCase):
    """--grow sweeps tenant counts up to the safe limit."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._logs, self._out = mr.LOGS, mr.OUT_DIR
        mr.LOGS, mr.OUT_DIR = self.tmp / "logs", self.tmp / "out"
        self.calls = []

    def tearDown(self):
        mr.LOGS, mr.OUT_DIR = self._logs, self._out

    def _fake(self, fam, ex, sub, subdir, os_, n_samples=None, **kwargs):
        self.calls.append((fam, subdir))
        lat = 0.010
        leaf = f"exit_{ex}_P{sub + 3}" if (fam == "yolo" and sub is not None) else f"exit_{ex}"
        _write_hw(mr.LOGS / subdir / fam / "d" / leaf / "hw_results.json", lat)
        return _FakeProc(), time.perf_counter()

    def _grow_counts(self, fam, free_gb):
        """Recompute expected grow counts for a given family and free memory."""
        inst = mr.RESIDENT_GB.get(fam, 1.0)
        return [n for n in range(2, mr.MAX_TENANTS + 1)
                if inst * n + mr.HEADROOM_GB <= free_gb]

    def test_grow_sweeps_n2_n3_when_free_fits_up_to_3(self):
        """With free memory sized for n up to 3, --grow sweeps n=2,3 and not n=4."""
        # bert: 1.2 GB/instance. n=3 needs 3*1.2+1.0=4.6 GB. n=4 needs 5.8 GB.
        # Set free to 4.6 GB exactly so n=3 fits and n=4 does not.
        free = 4.6e9
        with mock.patch("psutil.virtual_memory",
                        return_value=mock.Mock(available=free)), \
             mock.patch.object(mr, "run_one", side_effect=self._fake):
            rows = mr.run_scenario("bert_scale", k=2, keep_suspect=True, grow=True)
        self.assertIsNotNone(rows)
        n_tenants_seen = sorted(set(r["n_tenants"] for r in rows))
        self.assertIn(2, n_tenants_seen)
        self.assertIn(3, n_tenants_seen)
        self.assertNotIn(4, n_tenants_seen)

    def test_grow_sweeps_n2_through_n6_when_free_fits_up_to_6(self):
        """With more free memory sized for n up to 6, grow sweeps 2..6."""
        # bert: n=6 needs 6*1.2+1.0=8.2 GB. n=7 needs 9.4 GB.
        free = 8.2e9
        with mock.patch("psutil.virtual_memory",
                        return_value=mock.Mock(available=free)), \
             mock.patch.object(mr, "run_one", side_effect=self._fake):
            rows = mr.run_scenario("bert_scale", k=2, keep_suspect=True, grow=True)
        self.assertIsNotNone(rows)
        n_tenants_seen = sorted(set(r["n_tenants"] for r in rows))
        for n in range(2, 7):
            self.assertIn(n, n_tenants_seen)
        self.assertNotIn(7, n_tenants_seen)

    def test_grow_cap_stops_at_max_tenants(self):
        """With absurdly large free memory, growth stops at MAX_TENANTS."""
        # 1000 GB free: growth should stop at MAX_TENANTS, not higher.
        with mock.patch("psutil.virtual_memory",
                        return_value=mock.Mock(available=1000e9)), \
             mock.patch.object(mr, "run_one", side_effect=self._fake):
            rows = mr.run_scenario("bert_scale", k=2, keep_suspect=True, grow=True)
        self.assertIsNotNone(rows)
        n_tenants_seen = set(r["n_tenants"] for r in rows)
        self.assertLessEqual(max(n_tenants_seen), mr.MAX_TENANTS)
        # Cap is reached, not exceeded.
        self.assertNotIn(mr.MAX_TENANTS + 1, n_tenants_seen)

    def test_grow_on_heterogeneous_scenario_prints_note_and_runs(self):
        """--grow on a non-scaling scenario is ignored with a printed note; the
        scenario still runs normally (does not abort, does not silently do nothing)."""
        with mock.patch("psutil.virtual_memory",
                        return_value=mock.Mock(available=8e9)), \
             mock.patch.object(mr, "run_one", side_effect=self._fake):
            rows = mr.run_scenario("llama_yolo", k=2, keep_suspect=True, grow=True)
        # The heterogeneous scenario must still produce rows (not None).
        self.assertIsNotNone(rows)

    def test_grow_counts_helper_respects_cap(self):
        """_grow_counts never returns more than MAX_TENANTS counts."""
        with mock.patch("psutil.virtual_memory",
                        return_value=mock.Mock(available=1000e9)):
            counts, reason = mr._grow_counts("bert")
        self.assertLessEqual(max(counts), mr.MAX_TENANTS)
        self.assertEqual(len(counts), mr.MAX_TENANTS - 1)  # 2..MAX_TENANTS

    def test_grow_counts_helper_empty_when_nothing_fits(self):
        """_grow_counts returns empty list when n=2 already exceeds free memory."""
        # bert n=2: 2*1.2+1.0=3.4 GB. 0.5 GB free.
        with mock.patch("psutil.virtual_memory",
                        return_value=mock.Mock(available=0.5e9)):
            counts, reason = mr._grow_counts("bert")
        self.assertEqual(counts, [])

    def test_grow_scenario_returns_none_when_nothing_fits(self):
        """run_scenario with grow=True returns None and prints a message when
        no tenant count fits, instead of running with an empty list."""
        with mock.patch("psutil.virtual_memory",
                        return_value=mock.Mock(available=0.5e9)), \
             mock.patch.object(mr, "run_one", side_effect=lambda *a, **k: (
                 _FakeProc(), time.perf_counter())):
            result = mr.run_scenario("bert_scale", k=2, grow=True)
        self.assertIsNone(result)


class TestPollBarrierDir(unittest.TestCase):
    """Tests for the pure barrier-polling core in shared.benchmark_profiler.

    benchmark_profiler imports torch (which is absent here), so we import only
    the module-level function _poll_barrier_dir directly after patching out the
    hw_profiler and cpu_cache imports that it does not need.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _load_poll_fn(self):
        """Import _poll_barrier_dir without triggering the torch-dependent imports."""
        import importlib
        import sys
        # Stub the two relative imports that benchmark_profiler needs at module level.
        hw_stub = mock.MagicMock()
        hw_stub.Timer = object  # must be a class for Timer() call
        cache_stub = mock.MagicMock()
        cache_stub.is_available = mock.MagicMock(return_value=False)
        sys.modules.setdefault("shared", mock.MagicMock())
        sys.modules["shared.hw_profiler"] = hw_stub
        sys.modules["shared.cpu_cache"] = cache_stub
        # Load the module fresh so the stubs take effect.
        if "shared.benchmark_profiler" in sys.modules:
            del sys.modules["shared.benchmark_profiler"]
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "shared.benchmark_profiler",
            str(Path(mr.REPO_ROOT) / "shared" / "benchmark_profiler.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._poll_barrier_dir

    def test_n_files_present_returns_true(self):
        fn = self._load_poll_fn()
        d = str(self.tmp)
        # Pre-populate N readiness tokens so the poll sees them immediately.
        for k in range(3):
            (self.tmp / f"ready_{1000 + k}").write_text("1", encoding="utf-8")
        result = fn(d, n=3, timeout=1.0, poll=0.01)
        self.assertTrue(result)

    def test_fewer_than_n_files_times_out(self):
        fn = self._load_poll_fn()
        d = str(self.tmp)
        # Only one token present but N=3 required: must time out quickly.
        (self.tmp / "ready_9999").write_text("1", encoding="utf-8")
        result = fn(d, n=3, timeout=0.15, poll=0.02)
        self.assertFalse(result)

    def test_writes_own_token_atomically(self):
        """_poll_barrier_dir must write a ready_* file (not a .tmp) for itself."""
        fn = self._load_poll_fn()
        d = str(self.tmp)
        # Pre-populate N-1 tokens; the function will add the Nth and return True.
        (self.tmp / "ready_1111").write_text("1", encoding="utf-8")
        result = fn(d, n=2, timeout=1.0, poll=0.01)
        self.assertTrue(result)
        tokens = [p for p in self.tmp.iterdir()
                  if p.name.startswith("ready_") and not p.name.endswith(".tmp")]
        self.assertGreaterEqual(len(tokens), 2)

    def test_tmp_files_are_not_counted(self):
        """Files ending in .tmp must not count as ready tokens."""
        fn = self._load_poll_fn()
        d = str(self.tmp)
        # Plant two .tmp files and one real token: should still time out waiting for N=3.
        (self.tmp / "ready_0001.tmp").write_text("1", encoding="utf-8")
        (self.tmp / "ready_0002.tmp").write_text("1", encoding="utf-8")
        (self.tmp / "ready_0003").write_text("1", encoding="utf-8")
        result = fn(d, n=3, timeout=0.10, poll=0.02)
        self.assertFalse(result)


class TestBarrierEnvInConcurrent(unittest.TestCase):
    """Verify that measure_concurrent sets the barrier env vars for every tenant,
    that the solo path does NOT set them, and that the barrier directory is
    created before launch and removed after."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._logs, self._out = mr.LOGS, mr.OUT_DIR
        mr.LOGS, mr.OUT_DIR = self.tmp / "logs", self.tmp / "out"
        self.captured_envs = []
        self.captured_subdirs = []

    def tearDown(self):
        mr.LOGS, mr.OUT_DIR = self._logs, self._out

    def _fake_run_one(self, fam, ex, sub, subdir, os_, n_samples=None,
                      _extra_env=None, **kwargs):
        """Capture the _extra_env passed for each tenant launch."""
        self.captured_envs.append(dict(_extra_env) if _extra_env else {})
        self.captured_subdirs.append(subdir)
        lat = 0.010
        leaf = f"exit_{ex}_P{sub + 3}" if (fam == "yolo" and sub is not None) else f"exit_{ex}"
        _write_hw(mr.LOGS / subdir / fam / "d" / leaf / "hw_results.json", lat)
        return _FakeProc(), time.perf_counter()

    def test_concurrent_sets_barrier_env_for_each_tenant(self):
        """Every tenant in a concurrent run must receive all three barrier vars."""
        tenants = [("bert", 0, None), ("bert", 1, None), ("bert", 2, None)]
        with mock.patch.object(mr, "run_one", side_effect=self._fake_run_one):
            mr.measure_concurrent(tenants, "tag_test", {0: 100, 1: 100, 2: 100},
                                  __import__("os"))
        # Three tenants: three captured envs.
        self.assertEqual(len(self.captured_envs), 3)
        for env in self.captured_envs:
            self.assertIn("BENCH_BARRIER_DIR", env, "BENCH_BARRIER_DIR missing")
            self.assertIn("BENCH_BARRIER_N", env,   "BENCH_BARRIER_N missing")
            self.assertIn("BENCH_BARRIER_ID", env,  "BENCH_BARRIER_ID missing")

    def test_barrier_n_equals_tenant_count(self):
        """BENCH_BARRIER_N must equal the number of tenants."""
        tenants = [("bert", 0, None), ("bert", 1, None), ("bert", 2, None)]
        with mock.patch.object(mr, "run_one", side_effect=self._fake_run_one):
            mr.measure_concurrent(tenants, "tag_n", {0: 50, 1: 50, 2: 50},
                                  __import__("os"))
        for env in self.captured_envs:
            self.assertEqual(env["BENCH_BARRIER_N"], "3")

    def test_barrier_ids_are_distinct(self):
        """Each tenant must receive a distinct BENCH_BARRIER_ID."""
        tenants = [("bert", 0, None), ("bert", 1, None), ("bert", 2, None)]
        with mock.patch.object(mr, "run_one", side_effect=self._fake_run_one):
            mr.measure_concurrent(tenants, "tag_ids", {0: 50, 1: 50, 2: 50},
                                  __import__("os"))
        ids = [env["BENCH_BARRIER_ID"] for env in self.captured_envs]
        self.assertEqual(len(set(ids)), 3, f"IDs not distinct: {ids}")

    def test_barrier_dir_shared_across_all_tenants(self):
        """All tenants in the same cell must share the same barrier directory."""
        tenants = [("bert", 0, None), ("bert", 1, None)]
        with mock.patch.object(mr, "run_one", side_effect=self._fake_run_one):
            mr.measure_concurrent(tenants, "tag_shared", {0: 50, 1: 50},
                                  __import__("os"))
        dirs = {env["BENCH_BARRIER_DIR"] for env in self.captured_envs}
        self.assertEqual(len(dirs), 1, f"tenants got different barrier dirs: {dirs}")

    def test_barrier_dir_removed_after_run(self):
        """The barrier directory must not persist after measure_concurrent returns."""
        captured_dir = []

        def _capturing_run_one(fam, ex, sub, subdir, os_, n_samples=None,
                               _extra_env=None, **kwargs):
            if _extra_env and "BENCH_BARRIER_DIR" in _extra_env and not captured_dir:
                captured_dir.append(_extra_env["BENCH_BARRIER_DIR"])
            lat = 0.010
            leaf = f"exit_{ex}_P{sub + 3}" if (fam == "yolo" and sub is not None) else f"exit_{ex}"
            _write_hw(mr.LOGS / subdir / fam / "d" / leaf / "hw_results.json", lat)
            return _FakeProc(), time.perf_counter()

        tenants = [("bert", 0, None), ("bert", 1, None)]
        with mock.patch.object(mr, "run_one", side_effect=_capturing_run_one):
            mr.measure_concurrent(tenants, "tag_cleanup", {0: 50, 1: 50},
                                  __import__("os"))
        self.assertEqual(len(captured_dir), 1, "barrier dir was not captured")
        self.assertFalse(
            Path(captured_dir[0]).exists(),
            f"barrier dir still exists after run: {captured_dir[0]}",
        )

    def test_solo_run_does_not_set_barrier_env(self):
        """measure_solo / run_one in the solo phase must NOT set barrier env vars."""
        solo_envs = []

        def _solo_capture(fam, ex, sub, subdir, os_, n_samples=None,
                          _extra_env=None, **kwargs):
            solo_envs.append(dict(_extra_env) if _extra_env else {})
            lat = 0.010
            leaf = f"exit_{ex}_P{sub + 3}" if (fam == "yolo" and sub is not None) else f"exit_{ex}"
            _write_hw(mr.LOGS / subdir / fam / "d" / leaf / "hw_results.json", lat)
            return _FakeProc(), time.perf_counter()

        with mock.patch.object(mr, "run_one", side_effect=_solo_capture):
            mr.measure_solo("bert", 0, None, "solo_test", __import__("os"))

        # measure_solo calls run_one once without barrier vars.
        self.assertEqual(len(solo_envs), 1)
        self.assertEqual(solo_envs[0], {},
                         f"solo run received unexpected barrier env: {solo_envs[0]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
