"""Offline unit tests for shared/overhead.py. No torch, no jtop required.

shared/__init__.py imports hw_profiler which imports torch. Stub all heavy
modules in sys.modules before any shared import so this file works offline.
The pattern matches how prior probes in this project handle torch absence.

Run:  python -m unittest test_overhead -v
"""

import json
import sys
import time
import types
import unittest
from unittest import mock

# Stub every heavy dep before any shared.* import.  shared/__init__.py pulls in
# hw_profiler (torch, psutil, pynvml), training/benchmark profilers (torch),
# csv_export, plotting (pandas, matplotlib, numpy), etc.
# Use MagicMock so attribute access (psutil.Process(), torch.cuda.is_available,
# etc.) returns cooperative fakes without AttributeError.
_HEAVY = [
    "torch", "torch.nn", "torch.cuda", "torch.utils", "torch.utils.data",
    "psutil", "pynvml", "tqdm", "pandas",
    "matplotlib", "matplotlib.pyplot", "matplotlib.colors", "matplotlib.ticker",
    "numpy", "sklearn", "sklearn.metrics",
    "transformers", "huggingface_hub",
    "shared.jetson_profiler",
    # extras pulled by csv_export / grouped_export / plotting
    "scipy", "scipy.stats",
]
for _m in _HEAVY:
    if _m not in sys.modules:
        sys.modules[_m] = mock.MagicMock()

# torch.cuda.is_available() must return False so hw_profiler Timer skips sync.
sys.modules["torch"].cuda.is_available.return_value = False  # type: ignore


class TestOverheadTimer(unittest.TestCase):

    def _make(self):
        from shared.overhead import OverheadTimer
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
