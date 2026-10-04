"""Offline unit tests for shared/fairness.py.

No torch, no hardware, no network.  All arithmetic is checkable by hand.
"""

import math
import sys
import unittest
from pathlib import Path

# Load without going through shared/__init__.py (which imports torch).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "shared.fairness",
    Path(__file__).parent / "shared" / "fairness.py",
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

jain_fairness = _mod.jain_fairness
latency_dispersion = _mod.latency_dispersion


class TestJainIndex(unittest.TestCase):

    def test_equal_slowdowns_give_one(self):
        """Identical slowdowns -> perfectly fair -> J = 1.0."""
        r = jain_fairness([2.0, 2.0, 2.0])
        self.assertAlmostEqual(r["jain_index"], 1.0, places=6)

    def test_known_uneven_case(self):
        """slowdowns [1.0, 3.0]: (4^2)/(2*10) = 16/20 = 0.8."""
        r = jain_fairness([1.0, 3.0])
        self.assertAlmostEqual(r["jain_index"], 0.8, places=6)

    def test_maximally_skewed_approaches_one_over_n(self):
        """[1, 1, 1, 1000] is very skewed; J must be > 1/n and well below 1.0.

        Hand-check: n=4, sum=1003, sum_sq=3+1000000=1000003.
        J = 1003^2 / (4 * 1000003) = 1006009 / 4000012 ~= 0.2515.
        The 1/n bound is 0.25; J is just above it and well below 1.0.
        """
        r = jain_fairness([1.0, 1.0, 1.0, 1000.0])
        n = 4
        self.assertGreater(r["jain_index"], 1.0 / n)
        # Should be close to 1/n, not close to 1.0.
        self.assertLess(r["jain_index"], 0.30)

    def test_single_tenant_is_trivial_one(self):
        """n=1: index is definitionally 1.0, trivial=True."""
        r = jain_fairness([5.0])
        self.assertAlmostEqual(r["jain_index"], 1.0, places=6)
        self.assertTrue(r["trivial"])

    def test_empty_returns_none(self):
        """Empty input must return None, not 1.0."""
        self.assertIsNone(jain_fairness([]))

    def test_all_nonphysical_returns_none(self):
        """All zero or negative -> no valid values -> None."""
        self.assertIsNone(jain_fairness([0.0, -1.0]))

    def test_nonphysical_excluded_and_counted(self):
        """Zero and negative slowdowns are excluded; exclusion is reported."""
        r = jain_fairness([2.0, 0.0, -0.5, 2.0])
        self.assertEqual(r["n_excluded"], 2)
        self.assertEqual(r["n_valid"], 2)
        # Remaining values are [2.0, 2.0] -> J = 1.0.
        self.assertAlmostEqual(r["jain_index"], 1.0, places=6)

    def test_unfairness_ratio_equals_max_over_min(self):
        """unfairness_ratio = max/min on a known case."""
        r = jain_fairness([1.0, 4.0])
        self.assertAlmostEqual(r["unfairness_ratio"], 4.0, places=6)

    def test_unfairness_ratio_none_for_single_tenant(self):
        """With only one tenant there is no ratio to compute."""
        r = jain_fairness([3.0])
        self.assertIsNone(r["unfairness_ratio"])

    def test_min_max_spread_fields(self):
        r = jain_fairness([1.0, 2.0, 3.0])
        self.assertAlmostEqual(r["min_slowdown"], 1.0, places=6)
        self.assertAlmostEqual(r["max_slowdown"], 3.0, places=6)


class TestLatencyDispersion(unittest.TestCase):

    def test_cv_equals_std_over_mean(self):
        """Hand-crafted list: [1.0, 2.0, 3.0].
        mean = 2.0; variance (ddof=1) = ((1+0+1)/2) = 1.0; std = 1.0; CV = 0.5.
        """
        r = latency_dispersion([1.0, 2.0, 3.0])
        self.assertAlmostEqual(r["mean"], 2.0, places=6)
        self.assertAlmostEqual(r["std"], 1.0, places=6)
        self.assertAlmostEqual(r["cv"], 0.5, places=6)

    def test_empty_returns_none(self):
        self.assertIsNone(latency_dispersion([]))

    def test_single_sample_std_is_none(self):
        """n=1: std and cv are undefined (ddof=1 requires n>=2)."""
        r = latency_dispersion([0.01])
        self.assertIsNone(r["std"])
        self.assertIsNone(r["cv"])
        self.assertAlmostEqual(r["mean"], 0.01, places=6)

    def test_nonpositive_excluded(self):
        """Zero and negative latencies are not physical; they must be excluded."""
        r = latency_dispersion([0.0, -0.01, 1.0, 2.0, 3.0])
        self.assertEqual(r["n"], 3)
        self.assertAlmostEqual(r["mean"], 2.0, places=6)

    def test_quartiles_uniform(self):
        """For [1, 2, 3, 4], Q2 (median) should be between 2 and 3."""
        r = latency_dispersion([1.0, 2.0, 3.0, 4.0])
        self.assertGreater(r["q2"], 1.5)
        self.assertLess(r["q2"], 3.5)
        self.assertLessEqual(r["q1"], r["q2"])
        self.assertLessEqual(r["q2"], r["q3"])

    def test_uniform_values_cv_zero(self):
        """Identical values: std=0, CV=0."""
        r = latency_dispersion([0.05, 0.05, 0.05])
        self.assertAlmostEqual(r["std"], 0.0, places=6)
        self.assertAlmostEqual(r["cv"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
