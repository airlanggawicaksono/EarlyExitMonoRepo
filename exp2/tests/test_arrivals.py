"""Tests for exp2.arrivals (slice S1a): seeded PCG64 open-loop schedules (BRIEF_v1 K2)."""
import json

import numpy as np
import pytest

from exp2.arrivals import schedule
from exp2.contracts import Workload

KEYS = {"req_id", "type_id", "t_arrival_ns", "t_deadline_ns", "k_override"}
N_BIG = 200_000


def _workload(**over):
    w = {"name": "W_hi", "types": ["bert:5", "llama:15"], "probs": [0.9, 0.1], "rho": 0.6, "rate_per_s": 5.636,
         "arrival": "poisson", "cv_a": 1.0}
    w.update(over)
    return w


def _gaps_s(items):
    t = np.array([it["t_arrival_ns"] for it in items], dtype=np.float64)
    return np.diff(np.concatenate(([0.0], t))) / 1e9


def _cv(x):
    return float(np.std(x) / np.mean(x))


# --------------------------------------------------------------------------- shape of the output
class TestScheduleItems:
    def test_item_keys_ids_and_placeholders(self):
        items = schedule(_workload(), 5, seed=0)
        assert len(items) == 5
        for i, it in enumerate(items, start=1):
            assert set(it) == KEYS
            assert it["req_id"] == "r%06d" % i
            assert it["type_id"] in ("bert:5", "llama:15")
            assert it["t_deadline_ns"] is None
            assert it["k_override"] is None

    def test_values_are_builtin_and_json_serialisable(self):
        items = schedule(_workload(), 50, seed=3)
        assert all(type(it["t_arrival_ns"]) is int and type(it["type_id"]) is str for it in items)
        assert json.loads(json.dumps(items)) == items

    def test_zero_requests_is_empty_list(self):
        assert schedule(_workload(), 0, seed=0) == []

    def test_one_request(self):
        (it,) = schedule(_workload(), 1, seed=0)
        assert it["req_id"] == "r000001" and it["t_arrival_ns"] > 0

    @pytest.mark.parametrize("over", [{}, {"arrival": "gamma", "cv_a": 2.0}])
    def test_contracts_workload_dataclass_equals_dict_workload(self, over):
        w = _workload(**over)
        assert schedule(Workload(**w), 100, seed=2) == schedule(w, 100, seed=2)

    def test_input_workload_is_not_mutated(self):
        w = _workload()
        before = json.dumps(w, sort_keys=True)
        schedule(w, 10, seed=0)
        assert json.dumps(w, sort_keys=True) == before


# --------------------------------------------------------------------------- determinism and ordering
class TestDeterminism:
    def test_same_seed_gives_identical_output(self):
        assert schedule(_workload(), 1000, seed=7) == schedule(_workload(), 1000, seed=7)

    def test_different_seed_gives_different_output(self):
        assert schedule(_workload(), 1000, seed=7) != schedule(_workload(), 1000, seed=8)

    def test_offsets_never_decrease_and_start_after_zero(self):
        t = [it["t_arrival_ns"] for it in schedule(_workload(rate_per_s=2000.0), N_BIG, seed=0)]
        assert t[0] >= 0
        assert all(b >= a for a, b in zip(t, t[1:]))

    def test_offsets_nondecreasing_even_when_gaps_round_to_zero_ns(self):
        t = [it["t_arrival_ns"] for it in schedule(_workload(rate_per_s=5e8), 5000, seed=0)]
        assert all(b >= a for a, b in zip(t, t[1:]))

    def test_poisson_gaps_are_the_first_pcg64_exponential_draws(self):
        rate, n = 5.636, 1000
        rng = np.random.Generator(np.random.PCG64(11))
        expected_ns = np.cumsum(rng.exponential(1.0 / rate, n)) * 1e9
        got = np.array([it["t_arrival_ns"] for it in schedule(_workload(rate_per_s=rate), n, seed=11)])
        assert np.max(np.abs(got - expected_ns)) <= 1.0

    def test_gamma_gaps_use_shape_1_over_cv2_and_scale_mean_cv2(self):
        rate, cv, n = 4.0, 2.0, 1000
        rng = np.random.Generator(np.random.PCG64(5))
        expected_ns = np.cumsum(rng.gamma(1.0 / cv ** 2, (1.0 / rate) * cv ** 2, n)) * 1e9
        w = _workload(rate_per_s=rate, arrival="gamma", cv_a=cv)
        got = np.array([it["t_arrival_ns"] for it in schedule(w, n, seed=5)])
        assert np.max(np.abs(got - expected_ns)) <= 1.0


# --------------------------------------------------------------------------- distribution (DoD)
class TestDistribution:
    def test_poisson_200k_mean_within_2pct_and_cv_within_3pct(self):
        gaps = _gaps_s(schedule(_workload(), N_BIG, seed=0))
        assert gaps.mean() == pytest.approx(1 / 5.636, rel=0.02)
        assert _cv(gaps) == pytest.approx(1.0, rel=0.03)

    def test_gamma_cv2_within_4pct_and_mean_preserved(self):
        w = _workload(arrival="gamma", cv_a=2.0)
        gaps = _gaps_s(schedule(w, N_BIG, seed=0))
        assert _cv(gaps) == pytest.approx(2.0, rel=0.04)
        assert gaps.mean() == pytest.approx(1 / 5.636, rel=0.03)

    def test_gamma_cv_half_is_smoother_than_poisson(self):
        w = _workload(arrival="gamma", cv_a=0.5)
        gaps = _gaps_s(schedule(w, N_BIG, seed=0))
        assert _cv(gaps) == pytest.approx(0.5, rel=0.03)
        assert gaps.mean() == pytest.approx(1 / 5.636, rel=0.02)

    def test_type_frequencies_within_0_01_absolute(self):
        probs = [0.5, 0.3, 0.2]
        w = _workload(types=["bert:5", "vision:24", "llama:15"], probs=probs)
        items = schedule(w, N_BIG, seed=0)
        for tid, p in zip(w["types"], probs):
            freq = sum(1 for it in items if it["type_id"] == tid) / N_BIG
            assert abs(freq - p) < 0.01

    def test_type_is_drawn_independently_of_gap_and_of_previous_type(self):
        items = schedule(_workload(probs=[0.5, 0.5]), N_BIG, seed=0)
        is_llama = np.array([it["type_id"] == "llama:15" for it in items], dtype=float)
        gaps = _gaps_s(items)
        assert abs(np.corrcoef(gaps, is_llama)[0, 1]) < 0.01
        assert abs(np.corrcoef(is_llama[:-1], is_llama[1:])[0, 1]) < 0.01

    def test_zero_probability_type_never_appears(self):
        w = _workload(types=["bert:5", "llama:15"], probs=[1.0, 0.0])
        assert {it["type_id"] for it in schedule(w, 2000, seed=0)} == {"bert:5"}

    def test_probs_summing_to_one_within_5e_7_are_accepted_and_renormalised(self):
        # numpy.choice itself rejects sums off by more than ~1.5e-8, so this fails without renormalising
        w = _workload(types=["a:1", "b:2", "c:3"], probs=[0.1, 0.2, 0.7000005])
        assert len(schedule(w, 10, seed=0)) == 10


# --------------------------------------------------------------------------- invalid input
class TestValidation:
    @pytest.mark.parametrize("over", [
        {"rate_per_s": 0.0},
        {"rate_per_s": -1.0},
        {"arrival": "uniform"},
        {"arrival": "gamma", "cv_a": 0.0},
        {"arrival": "gamma", "cv_a": -1.0},
        {"arrival": "poisson", "cv_a": 2.0},
        {"probs": [0.5]},
        {"probs": [0.5, 0.4]},
        {"probs": [1.2, -0.2]},
        {"types": []},
    ])
    def test_bad_workload_raises(self, over):
        w = _workload(**over)
        if over.get("types") == []:
            w["probs"] = []
        with pytest.raises(ValueError):
            schedule(w, 10, seed=0)

    def test_negative_n_raises(self):
        with pytest.raises(ValueError):
            schedule(_workload(), -1, seed=0)

    def test_missing_field_raises(self):
        w = _workload()
        del w["rate_per_s"]
        with pytest.raises((KeyError, ValueError)):
            schedule(w, 10, seed=0)
