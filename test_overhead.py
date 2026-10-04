"""Offline unit tests for shared/overhead.py. No torch, no jtop required.

overhead.py needs only time, contextlib and typing at module level, and its
energy reader is a lazy import, so nothing heavy is required to test it. The
file is loaded directly via importlib rather than with `from shared.overhead
import ...`, because the package form would execute shared/__init__.py and
pull in pandas and torch. This is the loader pattern multitenant_analyze.py
already uses, and it keeps this test independent of whatever other test
modules leave behind in sys.modules.

Run:  python -m unittest test_overhead -v
"""

import importlib.util as _ilu
import json
import time
import unittest
from pathlib import Path

_spec = _ilu.spec_from_file_location(
    "_overhead_under_test",
    Path(__file__).resolve().parent / "shared" / "overhead.py",
)
_overhead = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_overhead)
OverheadTimer = _overhead.OverheadTimer


class TestOverheadTimer(unittest.TestCase):

    def _make(self):
        return OverheadTimer()

    def test_phase_records_plausible_duration(self):
        ov = self._make()
        with ov.phase("sleep"):
            time.sleep(0.05)
        d = ov.as_dict()
        self.assertGreaterEqual(d["overhead_sleep_sec"], 0.05)
        self.assertLess(d["overhead_sleep_sec"], 2.0)

    def test_as_dict_keys_flat_prefixed_json_serialisable(self):
        ov = self._make()
        with ov.phase("alpha"):
            pass
        with ov.phase("beta"):
            pass
        d = ov.as_dict()
        # All keys start with overhead_
        for k in d:
            self.assertTrue(k.startswith("overhead_"), f"unexpected key: {k}")
        # Round-trip JSON without error
        roundtrip = json.loads(json.dumps(d))
        self.assertEqual(set(d.keys()), set(roundtrip.keys()))

    def test_energy_is_none_not_zero_when_unavailable(self):
        # On the dev box pynvml is absent, so energy must be None, not 0.
        ov = self._make()
        with ov.phase("noenergy"):
            pass
        d = ov.as_dict()
        # Energy key exists and is None (not zero, not missing)
        self.assertIn("overhead_noenergy_j", d)
        val = d["overhead_noenergy_j"]
        self.assertIsNone(val, f"expected None, got {val!r}")

    def test_exception_phase_recorded_and_propagates(self):
        ov = self._make()
        with self.assertRaises(ValueError):
            with ov.phase("boom"):
                time.sleep(0.01)
                raise ValueError("test error")
        d = ov.as_dict()
        self.assertIn("overhead_boom_sec", d)
        self.assertGreaterEqual(d["overhead_boom_sec"], 0.01)
        # Exception path sets energy to None
        self.assertIsNone(d["overhead_boom_j"])

    def test_total_equals_sum_of_phases(self):
        ov = self._make()
        with ov.phase("a"):
            time.sleep(0.02)
        with ov.phase("b"):
            time.sleep(0.03)
        d = ov.as_dict()
        expected = round(d["overhead_a_sec"] + d["overhead_b_sec"], 3)
        self.assertAlmostEqual(d["overhead_total_sec"], expected, places=2)

    def test_duplicate_name_last_wins(self):
        ov = self._make()
        with ov.phase("dup"):
            time.sleep(0.01)
        with ov.phase("dup"):
            time.sleep(0.05)
        d = ov.as_dict()
        # Only one entry for "dup"; it must be >= 0.05 (the second call)
        keys_for_dup = [k for k in d if "dup" in k and k.endswith("_sec")]
        self.assertEqual(len(keys_for_dup), 1)
        self.assertGreaterEqual(d["overhead_dup_sec"], 0.05)

    def test_total_after_duplicate_uses_last_value(self):
        """Total must reflect last-wins, not accumulate both durations."""
        ov = self._make()
        with ov.phase("dup"):
            time.sleep(0.01)
        with ov.phase("dup"):
            time.sleep(0.05)
        d = ov.as_dict()
        # Total == the single surviving dup duration (last wins)
        self.assertAlmostEqual(d["overhead_total_sec"], d["overhead_dup_sec"], places=2)


if __name__ == "__main__":
    unittest.main()
