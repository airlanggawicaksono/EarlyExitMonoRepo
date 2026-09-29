"""Unit tests for multitenant_report.py, all offline.

Covers:
  - CSV discovery (per-mode subfolders and legacy flat file).
  - Mode label read from t0_nvpmodel for the legacy flat file.
  - Truncated final line skipped without raising.
  - Cell key stability across two mode files.
  - Workbook written with exactly the expected sheet names.
  - Low-overlap cell flagged in the validity sheet.

No real Jetson, no real torch, no real result tree: all tests use temp dirs
and synthetic CSVs.

Run:  python -m unittest test_multitenant_report -v
"""

import tempfile
import unittest
from pathlib import Path

import multitenant_report as mr


# Minimal valid CSV header covering the columns the report cares about.
_HEADER = (
    "tag,n_tenants,overlap_frac,timed_overlap_frac,"
    "t0,t0_slowdown,t0_p95_ratio,"
    "t1,t1_slowdown,t1_p95_ratio,"
    "stp,antt,throughput_gain,agg_throughput,agg_throughput_comparable,"
    "t0_gpu_mem_static_mb,t0_gpu_mem_dynamic_mb,t0_gpu_mem_peak_mb,"
    "t1_gpu_mem_static_mb,t1_gpu_mem_dynamic_mb,t1_gpu_mem_peak_mb,"
    "pair_gpu_mem_static_mb,pair_gpu_mem_peak_mb,"
    "pair_power_w,pair_energy_j,"
    "t0_calib_fallback,t1_calib_fallback,"
    "t0_nvpmodel,t1_nvpmodel"
)

_ROW_MAXN = (
    "runMAXN,2,0.97,0.98,"
    "bert@0,1.10,1.15,"
    "yolo@3_P3,1.05,1.08,"
    "1.92,1.075,0.93,120.0,False,"
    "700.0,100.0,800.0,400.0,70.0,470.0,1100.0,1270.0,"
    "10.0,300.0,"
    "False,False,"
    "MAXN,MAXN"
)

_ROW_15W = (
    "run15w,2,0.93,0.95,"
    "bert@0,1.22,1.28,"
    "yolo@3_P3,1.15,1.20,"
    "1.75,1.185,0.88,100.0,False,"
    "700.0,115.0,815.0,400.0,78.0,478.0,1100.0,1293.0,"
    "6.0,180.0,"
    "False,False,"
    "15W,15W"
)

_ROW_LOW_OVERLAP = (
    "run15w_lo,2,0.35,0.38,"
    "bert@12,1.40,1.50,"
    "yolo@5_P5,1.30,1.35,"
    "1.55,1.35,0.78,85.0,False,"
    "700.0,210.0,910.0,400.0,95.0,495.0,1100.0,1405.0,"
    "5.8,174.0,"
    "False,False,"
    "15W,15W"
)


def _make_tree(base: Path, files: dict):
    """Create CSVs under base given {relative_path: content}."""
    for rel, content in files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


class TestDiscovery(unittest.TestCase):
    def test_discovers_per_mode_subfolders(self):
        """Two per-mode CSVs under subdirs are discovered with folder names as labels."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _make_tree(base, {
                "maxn/concurrent_slowdown.csv": _HEADER + "\n" + _ROW_MAXN,
                "15w/concurrent_slowdown.csv": _HEADER + "\n" + _ROW_15W,
            })
            found = mr.discover_csv_files(base)
            self.assertEqual(set(found.keys()), {"MAXN", "15W"})
            self.assertTrue(found["MAXN"].exists())
            self.assertTrue(found["15W"].exists())

    def test_folder_name_used_as_mode_label(self):
        """The folder name (upper-cased) becomes the mode label."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _make_tree(base, {
                "25w/concurrent_slowdown.csv": _HEADER + "\n" + _ROW_MAXN,
            })
            found = mr.discover_csv_files(base)
            self.assertIn("25W", found)

    def test_legacy_flat_file_discovered(self):
        """A flat concurrent_slowdown.csv at base level is picked up."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            # Flat file with t0_nvpmodel = 25W.
            content = _HEADER + "\n" + _ROW_MAXN.replace(",MAXN,MAXN", ",25W,25W")
            (base / "concurrent_slowdown.csv").write_text(content, encoding="utf-8")
            found = mr.discover_csv_files(base)
            self.assertIn("25W", found)

    def test_legacy_mode_read_from_nvpmodel_column(self):
        """The mode for a flat CSV is read from t0_nvpmodel, not guessed."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            content = _HEADER + "\n" + _ROW_15W  # t0_nvpmodel = 15W
            (base / "concurrent_slowdown.csv").write_text(content, encoding="utf-8")
            mode = mr._mode_from_flat_csv(base / "concurrent_slowdown.csv")
            self.assertEqual(mode, "15W")

    def test_per_mode_file_takes_priority_over_legacy(self):
        """When a per-mode subfolder and a legacy flat file share a mode, the
        subfolder entry wins."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _make_tree(base, {
                "15w/concurrent_slowdown.csv": _HEADER + "\n" + _ROW_15W,
            })
            legacy = base / "concurrent_slowdown.csv"
            legacy.write_text(_HEADER + "\n" + _ROW_15W, encoding="utf-8")
            found = mr.discover_csv_files(base)
            # Only one entry for 15W, pointing at the subfolder.
            self.assertEqual(len([k for k in found if k == "15W"]), 1)
            self.assertIn("15w", str(found["15W"]))


class TestLoadCsv(unittest.TestCase):
    def test_basic_load(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.csv"
            p.write_text(_HEADER + "\n" + _ROW_MAXN, encoding="utf-8")
            rows = mr.load_csv(p)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["t0"], "bert@0")

    def test_truncated_final_line_skipped(self):
        """A line with no comma (truncated write) is skipped, not raised."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trunc.csv"
            content = _HEADER + "\n" + _ROW_MAXN + "\npartial_no_comma"
            p.write_text(content, encoding="utf-8")
            rows = mr.load_csv(p)
            # Only the complete row survives.
            self.assertEqual(len(rows), 1)

    def test_empty_file_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "empty.csv"
            p.write_text("", encoding="utf-8")
            rows = mr.load_csv(p)
            self.assertEqual(rows, [])

    def test_header_only_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "hdr.csv"
            p.write_text(_HEADER, encoding="utf-8")
            rows = mr.load_csv(p)
            self.assertEqual(rows, [])


class TestCellKey(unittest.TestCase):
    def test_two_tenant_key(self):
        row = {"n_tenants": "2", "t0": "bert@0", "t1": "yolo@3_P3"}
        self.assertEqual(mr.cell_key(row), "bert@0 + yolo@3_P3 (n=2)")

    def test_single_tenant_key(self):
        row = {"n_tenants": "1", "t0": "bert@0"}
        self.assertEqual(mr.cell_key(row), "bert@0 (n=1)")

    def test_key_stable_across_modes(self):
        """The same logical cell produces the same key from two mode files."""
        row_a = {"n_tenants": "2", "t0": "bert@0", "t1": "yolo@3_P3"}
        row_b = {"n_tenants": "2", "t0": "bert@0", "t1": "yolo@3_P3"}
        self.assertEqual(mr.cell_key(row_a), mr.cell_key(row_b))

    def test_different_cells_give_different_keys(self):
        row_a = {"n_tenants": "2", "t0": "bert@0", "t1": "yolo@3_P3"}
        row_b = {"n_tenants": "2", "t0": "bert@12", "t1": "yolo@5_P5"}
        self.assertNotEqual(mr.cell_key(row_a), mr.cell_key(row_b))

    def test_key_includes_tenant_count(self):
        """n=2 and n=3 of the same model produce distinct keys."""
        row2 = {"n_tenants": "2", "t0": "bert@0", "t1": "bert@0"}
        row3 = {"n_tenants": "3", "t0": "bert@0", "t1": "bert@0", "t2": "bert@0"}
        self.assertNotEqual(mr.cell_key(row2), mr.cell_key(row3))


class TestWorkbook(unittest.TestCase):
    def _write_two_mode_tree(self, base: Path):
        _make_tree(base, {
            "maxn/concurrent_slowdown.csv": _HEADER + "\n" + _ROW_MAXN,
            "15w/concurrent_slowdown.csv": (
                _HEADER + "\n" + _ROW_15W + "\n" + _ROW_LOW_OVERLAP
            ),
        })

    def test_expected_sheet_names(self):
        """The workbook contains exactly the six expected sheets."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            self._write_two_mode_tree(base)
            out = base / "report.xlsx"
            ok = mr.write_report(base_dir=base, out_path=out)
            self.assertTrue(ok)
            import openpyxl
            wb = openpyxl.load_workbook(str(out))
            self.assertEqual(
                set(wb.sheetnames),
                {"legend", "interference", "throughput", "memory",
                 "power_energy", "validity"},
            )

    def test_low_overlap_flagged_in_validity(self):
        """A cell with timed_overlap_frac < OVERLAP_WARN has a numeric value
        below the threshold recorded in the validity sheet."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            self._write_two_mode_tree(base)
            out = base / "report.xlsx"
            mr.write_report(base_dir=base, out_path=out)
            import openpyxl
            wb = openpyxl.load_workbook(str(out))
            ws = wb["validity"]
            low_values = []
            for row in ws.iter_rows(min_row=3):
                for c in row[1:]:
                    if isinstance(c.value, float) and c.value < mr.OVERLAP_WARN:
                        low_values.append(c.value)
            self.assertTrue(
                low_values,
                "no numeric value below OVERLAP_WARN found in validity sheet",
            )
            self.assertLess(low_values[0], mr.OVERLAP_WARN)

    def test_no_data_returns_false(self):
        """write_report returns False when no CSVs are found."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            out = base / "report.xlsx"
            ok = mr.write_report(base_dir=base, out_path=out)
            self.assertFalse(ok)

    def test_cells_align_across_modes(self):
        """The same cell key appears in both mode columns (rows align)."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            self._write_two_mode_tree(base)
            mode_order, data = mr.load_all_modes(base)
            shared_key = "bert@0 + yolo@3_P3 (n=2)"
            self.assertIn(shared_key, data["MAXN"])
            self.assertIn(shared_key, data["15W"])


class TestLoadAllModes(unittest.TestCase):
    def test_mode_order_follows_preferred(self):
        """Known modes appear in preferred order: MAXN_SUPER, MAXN, 25W, 15W."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _make_tree(base, {
                "15w/concurrent_slowdown.csv": _HEADER + "\n" + _ROW_15W,
                "maxn/concurrent_slowdown.csv": _HEADER + "\n" + _ROW_MAXN,
                "25w/concurrent_slowdown.csv": _HEADER + "\n" + _ROW_MAXN,
            })
            mode_order, _ = mr.load_all_modes(base)
            self.assertEqual(mode_order, ["MAXN", "25W", "15W"])

    def test_empty_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            mode_order, data = mr.load_all_modes(Path(td))
            self.assertEqual(mode_order, [])
            self.assertEqual(data, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
