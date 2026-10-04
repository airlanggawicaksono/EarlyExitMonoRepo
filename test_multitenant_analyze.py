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


if __name__ == "__main__":
    unittest.main()
