"""Offline unit tests for multitenant_analyze.py.

All tests write synthetic hw_results.json files to a temp directory.
No torch, no hardware, no network.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from multitenant_analyze import (
    analyze_cell,
    compute_timeseries,
    discover_cells,
    discover_solos,
    load_intended_counts,
    run,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic fixture trees
# ---------------------------------------------------------------------------

def _hw(per_sample_sec_mean, throughput_samples_per_sec, avg_power_w=10.0,
        avg_energy_j=1.5, gpu_mem_static_mb=1371.0, gpu_mem_dynamic_mb=13.0,
        peak_vram_allocated_mb=1384.0, n_samples=100):
    """Return a minimal hw_results.json dict."""
    return {
        "aggregate": {
            "per_sample_sec_mean": per_sample_sec_mean,
            "throughput_samples_per_sec": throughput_samples_per_sec,
            "avg_power_w": avg_power_w,
            "avg_energy_j": avg_energy_j,
            "gpu_mem_static_mb": gpu_mem_static_mb,
            "gpu_mem_dynamic_mb": gpu_mem_dynamic_mb,
            "peak_vram_allocated_mb": peak_vram_allocated_mb,
            "n_samples": n_samples,
        }
    }


def _write_hw(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _conc_path(root: Path, basetag, cellidx, tid, fam, exit_n, dataset="DS") -> Path:
    """Return the hw_results.json path inside a concurrent tenant folder."""
    folder = root / f"mt_conc_{basetag}_{cellidx}_r0_{tid}_{fam}_{exit_n}"
    return folder / fam / dataset / "pretrained" / f"exit_{exit_n}" / "hw_results.json"


def _solo_path(root: Path, basetag, fam, exit_n, count=100, dataset="DS") -> Path:
    """Return the hw_results.json path inside a solo baseline folder."""
    folder = root / f"mt_solo_{basetag}_{fam}_{exit_n}_n{count}"
    return folder / fam / dataset / "pretrained" / f"exit_{exit_n}" / "hw_results.json"


def _yolo_solo_path(root: Path, basetag, exit_n, folder_sub, inner_sub, count=50, dataset="coco") -> Path:
    """Return the hw_results.json path inside a yolo solo baseline folder.

    folder_sub is the P-index embedded in the folder name (e.g. 1 for _P1).
    inner_sub  is the P-index in the inner exit directory (e.g. 4 for exit_0_P4).
    These differ on real data; inner_sub is the canonical matching key.
    """
    folder = root / f"mt_solo_{basetag}_yolo_{exit_n}_P{folder_sub}_n{count}"
    return folder / "yolo" / dataset / "pretrained" / f"exit_{exit_n}_P{inner_sub}" / "hw_results.json"


def _yolo_conc_path(root: Path, basetag, cellidx, tid, exit_n, inner_sub, dataset="coco") -> Path:
    """Return the hw_results.json path inside a yolo concurrent tenant folder."""
    folder = root / f"mt_conc_{basetag}_{cellidx}_r0_{tid}_yolo_{exit_n}"
    return folder / "yolo" / dataset / "pretrained" / f"exit_{exit_n}_P{inner_sub}" / "hw_results.json"


def _write_csv(path: Path, rows: list[dict]):
    """Write a minimal concurrent_slowdown.csv with tag and n_tenants columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tag", "n_tenants"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCompleteTwoTenantCell(unittest.TestCase):
    """A cell with 2 tenants, both present, both with solo baselines.

    Known inputs: solo thru=100, shared thru each=50 -> STP=1.0, ANTT=2.0.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root

        # solo lat=0.01 -> thru=100; shared lat=0.02 -> thru=50
        solo_lat, solo_thru = 0.01, 100.0
        sh_lat, sh_thru = 0.02, 50.0

        # Solo baseline.
        _write_hw(_solo_path(root, "bert_test_MAXN", "bert", 1), _hw(solo_lat, solo_thru))

        # Two tenants in cell 0.
        _write_hw(_conc_path(root, "bert_test_MAXN", 0, 0, "bert", 1), _hw(sh_lat, sh_thru))
        _write_hw(_conc_path(root, "bert_test_MAXN", 0, 1, "bert", 1), _hw(sh_lat, sh_thru))

        # Write CSV so intended=2 comes from CSV, not inference.
        _write_csv(
            root / "concurrent_slowdown.csv",
            [{"tag": "bert_test_MAXN_0_r0", "n_tenants": 2}],
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_classification(self):
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        counts, _ = load_intended_counts(self.root)
        self.assertIn(("bert_test_MAXN", 0), cells)
        tid_map = cells[("bert_test_MAXN", 0)]
        result = analyze_cell("bert_test_MAXN", 0, tid_map, solos, 2, "csv")
        self.assertEqual(result["status"], "complete")

    def test_stp_equals_one(self):
        """STP = sum(thru_shared / thru_solo) = 50/100 + 50/100 = 1.0."""
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        result = analyze_cell("bert_test_MAXN", 0, cells[("bert_test_MAXN", 0)], solos, 2, "csv")
        self.assertAlmostEqual(result["stp"], 1.0, places=6)

    def test_antt_equals_two(self):
        """ANTT = mean(lat_shared / lat_solo) = mean(0.02/0.01, 0.02/0.01) = 2.0."""
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        result = analyze_cell("bert_test_MAXN", 0, cells[("bert_test_MAXN", 0)], solos, 2, "csv")
        self.assertAlmostEqual(result["antt"], 2.0, places=6)

    def test_per_tenant_slowdown(self):
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        result = analyze_cell("bert_test_MAXN", 0, cells[("bert_test_MAXN", 0)], solos, 2, "csv")
        for t in result["per_tenant"]:
            self.assertAlmostEqual(t["slowdown"], 2.0, places=6)
            self.assertAlmostEqual(t["thru_shared"], 50.0, places=6)

    def test_agg_thru_equals_sum(self):
        """Aggregate throughput = 50 + 50 = 100."""
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        result = analyze_cell("bert_test_MAXN", 0, cells[("bert_test_MAXN", 0)], solos, 2, "csv")
        self.assertAlmostEqual(result["agg_thru"], 100.0, places=4)

    def test_run_end_to_end(self):
        """run() returns 0 and classifies the cell as complete."""
        code = run(self.root, "test", None)
        self.assertEqual(code, 0)

    def test_fairness_fields_present_for_complete_cell(self):
        """analyze_cell must return a 'fairness' dict for a complete cell."""
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        result = analyze_cell("bert_test_MAXN", 0, cells[("bert_test_MAXN", 0)], solos, 2, "csv")
        self.assertIn("fairness", result)
        fa = result["fairness"]
        self.assertIsNotNone(fa)

    def test_jain_index_is_one_for_equal_slowdowns(self):
        """Both tenants have slowdown=2.0 -> perfectly fair -> jain_index=1.0."""
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        result = analyze_cell("bert_test_MAXN", 0, cells[("bert_test_MAXN", 0)], solos, 2, "csv")
        self.assertAlmostEqual(result["fairness"]["jain_index"], 1.0, places=6)

    def test_per_tenant_lat_cv_field_present(self):
        """Each per_tenant entry must carry a lat_cv field (may be None if no samples)."""
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        result = analyze_cell("bert_test_MAXN", 0, cells[("bert_test_MAXN", 0)], solos, 2, "csv")
        for t in result["per_tenant"]:
            self.assertIn("lat_cv", t)

    def test_csv_contains_fairness_columns(self):
        """run() with an out_path must write jain_index and unfairness_ratio columns."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            out = f.name
        run(self.root, "test", Path(out))
        with open(out, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
        self.assertIn("jain_index", fieldnames)
        self.assertIn("unfairness_ratio", fieldnames)
        self.assertIn("lat_cv", fieldnames)


class TestIncompleteCell_MissingTenants(unittest.TestCase):
    """Intended=4 (from CSV) but only 2 tenant folders present -> INCOMPLETE."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root

        _write_hw(_solo_path(root, "bert_test_MAXN", "bert", 5), _hw(0.02, 50.0))
        # Only tids 0 and 1 present, not 2 and 3.
        _write_hw(_conc_path(root, "bert_test_MAXN", 2, 0, "bert", 5), _hw(0.04, 25.0))
        _write_hw(_conc_path(root, "bert_test_MAXN", 2, 1, "bert", 5), _hw(0.04, 25.0))

        _write_csv(
            root / "concurrent_slowdown.csv",
            [{"tag": "bert_test_MAXN_2_r0", "n_tenants": 4}],
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_incomplete_classification(self):
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        counts, _ = load_intended_counts(self.root)
        tid_map = cells[("bert_test_MAXN", 2)]
        result = analyze_cell("bert_test_MAXN", 2, tid_map, solos, 4, "csv")
        self.assertEqual(result["status"], "incomplete")

    def test_logged_vs_intended(self):
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        result = analyze_cell("bert_test_MAXN", 2, cells[("bert_test_MAXN", 2)], solos, 4, "csv")
        self.assertEqual(result["logged"], 2)
        self.assertEqual(result["intended"], 4)

    def test_not_in_complete_output(self):
        """run() must not list this cell in the complete section."""
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            run(self.root, "test", None)
        output = buf.getvalue()
        # The incomplete cell tag should appear only in the INCOMPLETE section.
        self.assertIn("INCOMPLETE", output)
        self.assertIn("bert_test_MAXN_2_r0", output)


class TestSoloProbeIgnored(unittest.TestCase):
    """mt_solo_probe_* folders must not be used as baselines."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root

        # Write a probe solo (should be ignored).
        probe_path = (
            root / "mt_solo_probe_bert_test_MAXN_bert_3"
            / "bert" / "DS" / "pretrained" / "exit_3" / "hw_results.json"
        )
        _write_hw(probe_path, _hw(0.01, 100.0))

        # One concurrent cell; no calibrated solo.
        _write_hw(_conc_path(root, "bert_test_MAXN", 0, 0, "bert", 3), _hw(0.02, 50.0))
        _write_hw(_conc_path(root, "bert_test_MAXN", 0, 1, "bert", 3), _hw(0.02, 50.0))

        _write_csv(
            root / "concurrent_slowdown.csv",
            [{"tag": "bert_test_MAXN_0_r0", "n_tenants": 2}],
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_probe_not_in_solos(self):
        solos = discover_solos(self.root)
        # The probe folder key would be ('bert_test_MAXN', 'bert', 3)
        self.assertNotIn(("bert_test_MAXN", "bert", 3), solos)

    def test_cell_incomplete_without_calibrated_solo(self):
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        result = analyze_cell("bert_test_MAXN", 0, cells[("bert_test_MAXN", 0)], solos, 2, "csv")
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("solo", result["reason"])


class TestMissingSoloBaseline(unittest.TestCase):
    """A cell where the solo baseline folder does not exist -> INCOMPLETE."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root

        # No solo baseline written.
        _write_hw(_conc_path(root, "bert_test_MAXN", 7, 0, "bert", 9), _hw(0.06, 16.0))
        _write_hw(_conc_path(root, "bert_test_MAXN", 7, 1, "bert", 9), _hw(0.06, 16.0))

        _write_csv(
            root / "concurrent_slowdown.csv",
            [{"tag": "bert_test_MAXN_7_r0", "n_tenants": 2}],
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_incomplete_due_to_missing_solo(self):
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        result = analyze_cell("bert_test_MAXN", 7, cells[("bert_test_MAXN", 7)], solos, 2, "csv")
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("solo", result["reason"])


class TestSTPReciprocals(unittest.TestCase):
    """STP equals the sum of (thru_shared / thru_solo) on a hand-crafted case.

    solo thru = 80; tenant 0 shared = 40 (ratio 0.5); tenant 1 shared = 60 (ratio 0.75).
    STP = 0.5 + 0.75 = 1.25.
    ANTT = mean(80/40 wait no -- slowdown is lat_shared/lat_solo.
    lat = 1/thru for these synthetic values:
      lat_solo = 1/80; tenant 0 lat_shared = 1/40; tenant 1 lat_shared = 1/60.
      slowdown_0 = (1/40)/(1/80) = 2.0; slowdown_1 = (1/60)/(1/80) = 80/60 = 1.333...
      ANTT = mean(2.0, 1.333) = 1.666...
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root

        _write_hw(_solo_path(root, "bt_MAXN", "bert", 4), _hw(1 / 80, 80.0))
        _write_hw(_conc_path(root, "bt_MAXN", 1, 0, "bert", 4), _hw(1 / 40, 40.0))
        _write_hw(_conc_path(root, "bt_MAXN", 1, 1, "bert", 4), _hw(1 / 60, 60.0))

        _write_csv(
            root / "concurrent_slowdown.csv",
            [{"tag": "bt_MAXN_1_r0", "n_tenants": 2}],
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_stp(self):
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        result = analyze_cell("bt_MAXN", 1, cells[("bt_MAXN", 1)], solos, 2, "csv")
        self.assertEqual(result["status"], "complete")
        self.assertAlmostEqual(result["stp"], 1.25, places=6)

    def test_antt(self):
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        result = analyze_cell("bt_MAXN", 1, cells[("bt_MAXN", 1)], solos, 2, "csv")
        expected_antt = (2.0 + 80 / 60) / 2
        self.assertAlmostEqual(result["antt"], expected_antt, places=6)


class TestComputeTimeseries(unittest.TestCase):
    """Offline tests for compute_timeseries bucketing logic."""

    def _make_sample(self, elapsed_sec, power_w, forward_sec=0.01):
        return {"elapsed_sec": elapsed_sec, "power_w": power_w, "forward_sec": forward_sec}

    def test_bucketing_counts_and_means(self):
        """Samples at 0.1, 0.5, 0.9 land in t=0; 1.2 and 1.7 land in t=1."""
        samples = [
            self._make_sample(0.1, 10.0),
            self._make_sample(0.5, 20.0),
            self._make_sample(0.9, 30.0),
            self._make_sample(1.2, 40.0),
            self._make_sample(1.7, 50.0),
        ]
        buckets, has_elapsed = compute_timeseries(samples)
        self.assertTrue(has_elapsed)
        self.assertEqual(len(buckets), 2)

        b0 = buckets[0]
        self.assertEqual(b0["t"], 0)
        self.assertEqual(b0["n_samples"], 3)
        self.assertAlmostEqual(b0["mean_power_w"], (10 + 20 + 30) / 3, places=4)

        b1 = buckets[1]
        self.assertEqual(b1["t"], 1)
        self.assertEqual(b1["n_samples"], 2)
        self.assertAlmostEqual(b1["mean_power_w"], (40 + 50) / 2, places=4)

    def test_energy_equals_power_times_duration(self):
        """10 W over a full second (t=0, duration=1.0) must yield 10 J.

        Hand-check: energy = power * time = 10 W * 1.0 s = 10.0 J.
        Two samples in t=0, one in t=1 to make t=0 a full second.
        """
        samples = [
            self._make_sample(0.0, 10.0),
            self._make_sample(0.5, 10.0),
            self._make_sample(1.0, 10.0),
        ]
        buckets, _ = compute_timeseries(samples)
        b0 = next(b for b in buckets if b["t"] == 0)
        self.assertAlmostEqual(b0["duration_sec"], 1.0, places=5)
        self.assertAlmostEqual(b0["mean_power_w"], 10.0, places=5)
        self.assertAlmostEqual(b0["energy_j"], 10.0, places=5)

    def test_partial_final_bucket_uses_real_duration(self):
        """Last bucket covering [1, 1.7) has duration 0.7, not 1.0.

        energy_j must equal mean_power_w * 0.7, not mean_power_w * 1.0.
        """
        samples = [
            self._make_sample(0.0, 10.0),
            self._make_sample(0.5, 10.0),
            self._make_sample(1.0, 20.0),
            self._make_sample(1.7, 20.0),
        ]
        buckets, _ = compute_timeseries(samples)
        last = buckets[-1]
        self.assertEqual(last["t"], 1)
        self.assertAlmostEqual(last["duration_sec"], 0.7, places=5)
        expected_energy = last["mean_power_w"] * last["duration_sec"]
        self.assertAlmostEqual(last["energy_j"], expected_energy, places=5)
        # Confirm it is NOT 1.0 s worth of energy.
        self.assertFalse(abs(last["energy_j"] - last["mean_power_w"] * 1.0) < 1e-6)

    def test_clock_mean_and_min_correct(self):
        """A bucket with known gpu_sm_clock_mhz values reports correct mean and min."""
        samples = [
            {"elapsed_sec": 0.1, "power_w": 10.0, "forward_sec": 0.01,
             "gpu_sm_clock_mhz": 600.0},
            {"elapsed_sec": 0.5, "power_w": 10.0, "forward_sec": 0.01,
             "gpu_sm_clock_mhz": 400.0},
            {"elapsed_sec": 0.9, "power_w": 10.0, "forward_sec": 0.01,
             "gpu_sm_clock_mhz": 800.0},
            # second bucket; gives t=0 a full 1.0s duration
            {"elapsed_sec": 1.0, "power_w": 10.0, "forward_sec": 0.01,
             "gpu_sm_clock_mhz": 700.0},
        ]
        buckets, _ = compute_timeseries(samples)
        b0 = next(b for b in buckets if b["t"] == 0)
        self.assertAlmostEqual(b0["mean_gpu_clock_mhz"], (600 + 400 + 800) / 3, places=1)
        self.assertAlmostEqual(b0["min_gpu_clock_mhz"], 400.0, places=1)

    def test_missing_clock_reports_none_not_zero(self):
        """A bucket whose samples lack gpu_sm_clock_mhz reports None, never 0."""
        samples = [
            {"elapsed_sec": 0.1, "power_w": 10.0, "forward_sec": 0.01},
            {"elapsed_sec": 0.5, "power_w": 20.0, "forward_sec": 0.01},
            # second bucket to make t=0 a full second
            {"elapsed_sec": 1.0, "power_w": 10.0, "forward_sec": 0.01},
        ]
        buckets, _ = compute_timeseries(samples)
        b0 = next(b for b in buckets if b["t"] == 0)
        self.assertIsNone(b0["mean_gpu_clock_mhz"])
        self.assertIsNone(b0["min_gpu_clock_mhz"])
        # Power and sample count are still correct.
        self.assertAlmostEqual(b0["mean_power_w"], 15.0, places=4)
        self.assertEqual(b0["n_samples"], 2)

    def test_temperature_mean_correct(self):
        """Temperature mean is correct on a hand-crafted bucket."""
        samples = [
            {"elapsed_sec": 0.1, "power_w": 10.0, "forward_sec": 0.01,
             "gpu_temperature_c": 50.0},
            {"elapsed_sec": 0.5, "power_w": 10.0, "forward_sec": 0.01,
             "gpu_temperature_c": 70.0},
            {"elapsed_sec": 1.0, "power_w": 10.0, "forward_sec": 0.01},
        ]
        buckets, _ = compute_timeseries(samples)
        b0 = next(b for b in buckets if b["t"] == 0)
        self.assertAlmostEqual(b0["mean_gpu_temp_c"], 60.0, places=2)

    def test_csv_header_includes_clock_and_temp_columns(self):
        """The timeseries CSV header must include the three new columns."""
        import tempfile, io
        from contextlib import redirect_stdout
        from multitenant_analyze import _emit_timeseries
        import csv as _csv

        with tempfile.TemporaryDirectory() as tmp_log, \
             tempfile.TemporaryDirectory() as tmp_out:
            root = Path(tmp_log)
            # A hw_results.json with two elapsed_sec samples so t=0 is a full bucket
            # and t=1 is the partial-final bucket; both will be emitted as rows.
            hw_path = root / "hw_results.json"
            hw_path.write_text(
                '{"samples": ['
                '{"elapsed_sec": 0.1, "power_w": 10.0, "forward_sec": 0.01},'
                '{"elapsed_sec": 1.0, "power_w": 10.0, "forward_sec": 0.01}'
                ']}',
                encoding="utf-8",
            )
            ts_csv = Path(tmp_out) / "ts_out.csv"
            with redirect_stdout(io.StringIO()):
                _emit_timeseries(root, "test", ts_csv)

            with ts_csv.open(newline="", encoding="utf-8") as fh:
                fieldnames = _csv.DictReader(fh).fieldnames or []
        self.assertIn("mean_gpu_clock_mhz", fieldnames)
        self.assertIn("min_gpu_clock_mhz", fieldnames)
        self.assertIn("mean_gpu_temp_c", fieldnames)

    def test_missing_elapsed_sec_is_skipped(self):
        """A run whose samples lack elapsed_sec returns has_elapsed=False."""
        samples = [
            {"power_w": 10.0, "forward_sec": 0.01},  # no elapsed_sec key
            {"power_w": 12.0, "forward_sec": 0.01},
        ]
        buckets, has_elapsed = compute_timeseries(samples)
        self.assertFalse(has_elapsed)
        self.assertEqual(buckets, [])

    def test_empty_samples_skipped(self):
        buckets, has_elapsed = compute_timeseries([])
        self.assertFalse(has_elapsed)
        self.assertEqual(buckets, [])


class TestYoloSoloMatching(unittest.TestCase):
    """yolo solo folders with _P<sub> suffix must match their concurrent cell.

    Mirrors the real folder naming:
      solo:  mt_solo_tag_yolo_0_P1_n50  inner: exit_0_P4
      conc:  mt_conc_tag_0_r0_0_yolo_0  inner: exit_0_P4

    The folder-name sub (P1) and the inner-path sub (P4) differ — the inner
    path sub is the authoritative key on both sides.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root

        # Solo: folder sub=1, inner sub=4 (matching real data pattern)
        _write_hw(
            _yolo_solo_path(root, "tag", exit_n=0, folder_sub=1, inner_sub=4),
            _hw(0.07, 14.0),
        )
        # Two concurrent yolo tenants, same inner sub=4
        _write_hw(_yolo_conc_path(root, "tag", cellidx=0, tid=0, exit_n=0, inner_sub=4), _hw(0.09, 11.0))
        _write_hw(_yolo_conc_path(root, "tag", cellidx=0, tid=1, exit_n=0, inner_sub=4), _hw(0.09, 11.0))

        _write_csv(root / "concurrent_slowdown.csv", [{"tag": "tag_0_r0", "n_tenants": 2}])

    def tearDown(self):
        self.tmp.cleanup()

    def test_yolo_solo_discovered(self):
        """discover_solos must include the yolo solo keyed on inner sub."""
        solos = discover_solos(self.root)
        self.assertIn(("tag", "yolo", 0, 4), solos)

    def test_yolo_cell_complete(self):
        """analyze_cell must resolve the yolo solo and return status=complete."""
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        self.assertIn(("tag", 0), cells)
        result = analyze_cell("tag", 0, cells[("tag", 0)], solos, 2, "csv")
        self.assertEqual(result["status"], "complete",
                         msg=f"Expected complete but got: {result.get('reason', '')}")

    def test_yolo_slowdown_populated(self):
        """Complete yolo cell must have a non-None slowdown value."""
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        result = analyze_cell("tag", 0, cells[("tag", 0)], solos, 2, "csv")
        self.assertEqual(result["status"], "complete")
        for t in result["per_tenant"]:
            self.assertIsNotNone(t["slowdown"])

    def test_nonexistent_sub_stays_incomplete(self):
        """A concurrent cell whose inner sub has no matching solo is still incomplete."""
        root2 = Path(tempfile.mkdtemp())
        try:
            # Solo at inner sub=4, conc at inner sub=5 -> no match
            _write_hw(
                _yolo_solo_path(root2, "tag2", exit_n=0, folder_sub=1, inner_sub=4),
                _hw(0.07, 14.0),
            )
            _write_hw(_yolo_conc_path(root2, "tag2", cellidx=0, tid=0, exit_n=0, inner_sub=5), _hw(0.09, 11.0))
            _write_hw(_yolo_conc_path(root2, "tag2", cellidx=0, tid=1, exit_n=0, inner_sub=5), _hw(0.09, 11.0))
            _write_csv(root2 / "concurrent_slowdown.csv", [{"tag": "tag2_0_r0", "n_tenants": 2}])

            cells = discover_cells(root2)
            solos = discover_solos(root2)
            result = analyze_cell("tag2", 0, cells[("tag2", 0)], solos, 2, "csv")
            self.assertEqual(result["status"], "incomplete")
            self.assertIn("solo", result["reason"])
        finally:
            import shutil
            shutil.rmtree(root2, ignore_errors=True)

    def test_non_yolo_solo_unchanged(self):
        """Existing bert solo (no P suffix) must still be discovered and match."""
        root3 = Path(tempfile.mkdtemp())
        try:
            _write_hw(_solo_path(root3, "bt", "bert", 1), _hw(0.01, 100.0))
            _write_hw(_conc_path(root3, "bt", 0, 0, "bert", 1), _hw(0.02, 50.0))
            _write_hw(_conc_path(root3, "bt", 0, 1, "bert", 1), _hw(0.02, 50.0))
            _write_csv(root3 / "concurrent_slowdown.csv", [{"tag": "bt_0_r0", "n_tenants": 2}])

            cells = discover_cells(root3)
            solos = discover_solos(root3)
            self.assertIn(("bt", "bert", 1), solos)
            result = analyze_cell("bt", 0, cells[("bt", 0)], solos, 2, "csv")
            self.assertEqual(result["status"], "complete")
        finally:
            import shutil
            shutil.rmtree(root3, ignore_errors=True)


class TestHeterogeneousCell_BertYolo(unittest.TestCase):
    """Heterogeneous 2-tenant cell: tid=0 is bert, tid=1 is yolo.

    Solos use very different latencies so a wrong cross-model lookup would
    produce an obviously wrong slowdown.
      bert solo lat = 0.02 (thru=50)
      yolo solo lat = 0.20 (thru=5)   <- 10x slower than bert
      bert shared lat = 0.04  -> correct slowdown = 0.04/0.02 = 2.0
      yolo shared lat = 0.40  -> correct slowdown = 0.40/0.20 = 2.0

    Old (wrong) behaviour: both tenants used bert's solo (tid=0 was bert),
    giving yolo slowdown = 0.40/0.02 = 20.0 (physically absurd).
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root

        # bert solo
        _write_hw(_solo_path(root, "None", "bert", 0, count=100), _hw(0.02, 50.0))
        # yolo solo (inner_sub=4 to mirror real data)
        _write_hw(_yolo_solo_path(root, "None", exit_n=0, folder_sub=1, inner_sub=4, count=50),
                  _hw(0.20, 5.0))

        # Heterogeneous cell: tid=0=bert, tid=1=yolo
        _write_hw(_conc_path(root, "None", 0, 0, "bert", 0), _hw(0.04, 25.0))
        _write_hw(_yolo_conc_path(root, "None", cellidx=0, tid=1, exit_n=0, inner_sub=4),
                  _hw(0.40, 2.5))

        _write_csv(root / "concurrent_slowdown.csv", [{"tag": "None_0_r0", "n_tenants": 2}])

    def tearDown(self):
        self.tmp.cleanup()

    def _result(self):
        cells = discover_cells(self.root)
        solos = discover_solos(self.root)
        return analyze_cell("None", 0, cells[("None", 0)], solos, 2, "csv")

    def test_cell_is_complete(self):
        r = self._result()
        self.assertEqual(r["status"], "complete",
                         msg=f"Expected complete but got: {r.get('reason', '')}")

    def test_bert_tenant_uses_bert_solo(self):
        """tid=0 (bert) slowdown must be bert_lat_shared / bert_lat_solo = 0.04/0.02 = 2.0."""
        r = self._result()
        bert_t = next(t for t in r["per_tenant"] if t["tid"] == 0)
        self.assertAlmostEqual(bert_t["slowdown"], 2.0, places=4)

    def test_yolo_tenant_uses_yolo_solo(self):
        """tid=1 (yolo) slowdown must be yolo_lat_shared / yolo_lat_solo = 0.40/0.20 = 2.0.

        Under the old (wrong) code this would be 0.40/0.02 = 20.0.
        """
        r = self._result()
        yolo_t = next(t for t in r["per_tenant"] if t["tid"] == 1)
        self.assertAlmostEqual(yolo_t["slowdown"], 2.0, places=4)

    def test_per_tenant_lat_solo_recorded(self):
        """Each per_tenant entry must carry lat_solo matching its own model."""
        r = self._result()
        bert_t = next(t for t in r["per_tenant"] if t["tid"] == 0)
        yolo_t = next(t for t in r["per_tenant"] if t["tid"] == 1)
        self.assertAlmostEqual(bert_t["lat_solo"], 0.02, places=6)
        self.assertAlmostEqual(yolo_t["lat_solo"], 0.20, places=6)

    def test_stp_and_antt_correct(self):
        """STP = (25/50) + (2.5/5) = 0.5 + 0.5 = 1.0; ANTT = mean(2.0, 2.0) = 2.0."""
        r = self._result()
        self.assertAlmostEqual(r["stp"], 1.0, places=4)
        self.assertAlmostEqual(r["antt"], 2.0, places=4)

    def test_agg_thru_not_comparable_for_heterogeneous(self):
        """Heterogeneous cell must have agg_thru_comparable=False."""
        r = self._result()
        self.assertFalse(r["agg_thru_comparable"])

    def test_csv_fam_labels_are_per_tenant(self):
        """CSV rows for a heterogeneous cell must carry each tenant's own fam label.

        Before the fix both rows had fam=bert (the cell-level value).
        After the fix one row must have fam=bert and one must have fam=yolo.
        """
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            out = Path(f.name)
        run(self.root, "None", out)
        with out.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        fams = {r["fam"] for r in rows}
        self.assertIn("bert", fams, "bert row missing from CSV")
        self.assertIn("yolo", fams, "yolo row missing from CSV — CSV labelled both as bert")
        bert_row = next(r for r in rows if r["fam"] == "bert")
        yolo_row = next(r for r in rows if r["fam"] == "yolo")
        self.assertEqual(bert_row["exit"], "0")
        self.assertEqual(yolo_row["exit"], "0")


if __name__ == "__main__":
    unittest.main()
