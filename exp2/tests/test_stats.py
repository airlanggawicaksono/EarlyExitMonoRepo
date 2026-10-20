"""Tests for exp2.stats (slice S1a). numpy + stdlib only; every DoD item of BRIEF_v1 sec. 4 'S1a' has a test."""
import math
import pickle

import numpy as np
import pytest

from exp2.stats import (
    BootCI,
    block_bootstrap_ci,
    block_length,
    energy_j,
    energy_per_request,
    ess,
    lag1,
    mape,
    mean_ci,
    p_nearest_rank,
    request_window,
    trapezoid,
    welch_diff_ci,
)

S = 1_000_000_000  # ns per second

# Two-sided 95% Student-t quantiles, df = 1..30, typed independently of the implementation.
T975 = [12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2622, 2.2281,
        2.2010, 2.1788, 2.1604, 2.1448, 2.1315, 2.1199, 2.1098, 2.1009, 2.0930, 2.0860,
        2.0796, 2.0739, 2.0687, 2.0639, 2.0595, 2.0555, 2.0518, 2.0484, 2.0452, 2.0423]


# --------------------------------------------------------------------------- p_nearest_rank
class TestNearestRank:
    def test_dod_p95_of_1_to_20_is_19(self):
        assert p_nearest_rank(range(1, 21), 0.95) == 19

    def test_median_even_n_takes_lower_middle(self):
        assert p_nearest_rank([4, 1, 3, 2], 0.5) == 2

    def test_float_noise_does_not_push_rank_up(self):
        # 0.07 * 100 == 7.000000000000001, so a naive ceil gives rank 8
        for k in range(1, 101):
            assert p_nearest_rank(range(1, 101), k / 100) == k

    def test_q0_is_min_and_q1_is_max(self):
        assert p_nearest_rank([3, 1, 2], 0) == 1
        assert p_nearest_rank([3, 1, 2], 1) == 3

    def test_unsorted_input_and_single_element(self):
        assert p_nearest_rank([9.5, 0.5, 5.0], 0.5) == 5.0
        assert p_nearest_rank([7], 0.95) == 7

    def test_result_is_a_member_of_the_data(self):
        xs = [0.1, 0.4, 0.9, 1.7, 2.2, 3.0]
        for q in (0.05, 0.3, 0.5, 0.77, 0.99):
            assert p_nearest_rank(xs, q) in xs

    def test_large_input(self):
        xs = np.random.default_rng(1).random(200_000)
        assert p_nearest_rank(xs, 0.95) == np.sort(xs)[190_000 - 1]

    @pytest.mark.parametrize("xs, q", [([], 0.5), ([1, 2], -0.01), ([1, 2], 1.01), ([1, float("nan")], 0.5)])
    def test_invalid_input_raises(self, xs, q):
        with pytest.raises(ValueError):
            p_nearest_rank(xs, q)


# --------------------------------------------------------------------------- mape
class TestMape:
    def test_symmetric_error_ten_percent(self):
        assert mape([110, 90], [100, 100]) == pytest.approx(10.0, rel=1e-12)

    def test_divides_by_measured_not_predicted(self):
        assert mape([1.0], [2.0]) == pytest.approx(50.0, rel=1e-12)
        assert mape([3.0], [2.0]) == pytest.approx(50.0, rel=1e-12)

    def test_perfect_prediction_is_zero(self):
        assert mape([1.5, 2.5], [1.5, 2.5]) == 0.0

    def test_mean_over_items(self):
        # errors 10% and 30% -> mean 20%
        assert mape([11, 70], [10, 100]) == pytest.approx(20.0, rel=1e-12)

    @pytest.mark.parametrize("pred, meas", [([], []), ([1, 2], [1]), ([1.0], [0.0])])
    def test_invalid_input_raises(self, pred, meas):
        with pytest.raises(ValueError):
            mape(pred, meas)


# --------------------------------------------------------------------------- lag1 and ess
def _ar1(phi, n, seed):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


class TestLag1Ess:
    def test_lag1_hand_value(self):
        # centered [-2,-1,0,1,2]: sum of lag products 4, sum of squares 10
        assert lag1([1, 2, 3, 4, 5]) == pytest.approx(0.4, rel=1e-12)

    def test_lag1_alternating_is_negative(self):
        assert lag1([1, -1, 1, -1, 1, -1]) == pytest.approx(-5 / 6, rel=1e-12)

    def test_lag1_constant_series_is_zero(self):
        assert lag1([3.0] * 10) == 0.0

    def test_lag1_short_series_is_zero(self):
        assert lag1([1.0]) == 0.0
        assert lag1([]) == 0.0

    def test_lag1_iid_near_zero_and_ar1_near_phi(self):
        assert abs(lag1(np.random.default_rng(3).standard_normal(20_000))) < 0.03
        assert lag1(_ar1(0.8, 50_000, 4)) == pytest.approx(0.8, abs=0.03)

    def test_dod_ess_with_r1_half_is_n_over_3(self):
        xs = [1, 1, 1, -1, -1, -1]  # sum of lag products 3, sum of squares 6 -> r1 = 0.5 exactly
        assert lag1(xs) == pytest.approx(0.5, rel=1e-12)
        assert ess(xs) == pytest.approx(len(xs) / 3, rel=1e-12)

    def test_ess_formula_with_r1_0_4(self):
        assert ess([1, 2, 3, 4, 5]) == pytest.approx(5 * 0.6 / 1.4, rel=1e-12)

    def test_ess_of_constant_series_is_n(self):
        assert ess([2.0] * 8) == pytest.approx(8.0)

    def test_ess_of_ar1_is_about_n_over_9(self):
        n = 50_000
        assert ess(_ar1(0.8, n, 5)) == pytest.approx(n * 0.2 / 1.8, rel=0.15)

    def test_ess_single_value_is_one_and_empty_raises(self):
        assert ess([4.0]) == 1.0
        with pytest.raises(ValueError):
            ess([])


# --------------------------------------------------------------------------- trapezoid
def _ref_trapezoid(t, y):
    return sum((t[i + 1] - t[i]) * (y[i] + y[i + 1]) / 2 for i in range(len(t) - 1))


class TestTrapezoid:
    def test_ramp(self):
        assert trapezoid([0, 1, 2], [0, 1, 2]) == pytest.approx(2.0, rel=1e-12)

    def test_non_uniform_spacing(self):
        # 1*(0+2)/2 + 2*(2+2)/2 = 5
        assert trapezoid([0, 1, 3], [0, 2, 2]) == pytest.approx(5.0, rel=1e-12)

    def test_constant_over_offset_interval(self):
        assert trapezoid([1.0, 2.5, 5.0], [3.0, 3.0, 3.0]) == pytest.approx(12.0, rel=1e-12)

    def test_matches_loop_reference_on_random_data(self):
        rng = np.random.default_rng(7)
        t = np.cumsum(rng.random(500) + 0.01)
        y = rng.random(500) * 10
        assert trapezoid(t, y) == pytest.approx(_ref_trapezoid(t, y), rel=1e-12)

    def test_fewer_than_two_points_is_zero(self):
        assert trapezoid([], []) == 0.0
        assert trapezoid([1.0], [5.0]) == 0.0

    def test_returns_builtin_float(self):
        assert type(trapezoid([0, 1], [1, 1])) is float

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            trapezoid([0, 1, 2], [1, 2])

    def test_decreasing_time_raises(self):
        with pytest.raises(ValueError):
            trapezoid([0, 2, 1], [1, 1, 1])


# --------------------------------------------------------------------------- energy_j
def _power_rows(samples, base_ns=0):
    """samples: [(t_seconds, mW)] -> power.csv-like dict rows."""
    return [{"t_mono_ns": base_ns + int(round(t * S)), "t_wall_ns": 0, "vdd_in_mv": 5000, "vdd_in_ma": 0,
             "vdd_in_mw": float(mw)} for t, mw in samples]


TRACE = [(0, 1000), (1, 2000), (2, 2000), (3, 1000)]


class TestEnergyJ:
    def test_full_range_hand_value(self):
        # 1500 + 2000 + 1500 mW*s = 5.0 J
        assert energy_j(_power_rows(TRACE), 0, 3 * S) == pytest.approx(5.0, rel=1e-12)

    def test_clipped_both_ends_interpolates_linearly(self):
        # P(0.5)=1500, P(2.5)=1500: 875 + 2000 + 875 mW*s = 3.75 J
        assert energy_j(_power_rows(TRACE), S // 2, 5 * S // 2) == pytest.approx(3.75, rel=1e-12)

    def test_window_inside_a_single_segment(self):
        # P(0.25)=1250, P(0.75)=1750 -> mean 1500 for 0.5 s = 0.75 J
        assert energy_j(_power_rows(TRACE), S // 4, 3 * S // 4) == pytest.approx(0.75, rel=1e-12)

    def test_window_starting_on_a_sample(self):
        # 1 -> 2 s: 2000 mW*s; 2 -> 2.5 s: (2000+1500)/2*0.5 = 875 -> 2.875 J
        assert energy_j(_power_rows(TRACE), S, 5 * S // 2) == pytest.approx(2.875, rel=1e-12)

    def test_zero_length_window_is_zero(self):
        assert energy_j(_power_rows(TRACE), S, S) == 0.0

    def test_constant_power_is_power_times_duration(self):
        rows = _power_rows([(0.0, 5000), (7.0, 5000), (20.0, 5000)])
        t0, t1 = int(1.37 * S), int(13.71 * S)
        assert energy_j(rows, t0, t1) == pytest.approx(5.0 * (t1 - t0) / S, rel=1e-12)

    def test_additive_over_adjacent_windows(self):
        rng = np.random.default_rng(11)
        t = np.cumsum(rng.random(400) * 0.1 + 0.01)
        rows = _power_rows(list(zip(t, rng.random(400) * 9000 + 1000)))
        a, b, c = int(t[10] * S), int(t[200] * S) + 12345, int(t[390] * S)
        whole = energy_j(rows, a, c)
        assert energy_j(rows, a, b) + energy_j(rows, b, c) == pytest.approx(whole, rel=1e-10)

    def test_large_monotonic_offsets_keep_precision(self):
        base = 17_000_000_000_000  # ~4.7 h of uptime
        rows = _power_rows([(k * 0.05, 6000) for k in range(401)], base_ns=base)
        assert energy_j(rows, base + 3 * S, base + 17 * S) == pytest.approx(6.0 * 14, rel=1e-12)

    def test_200k_rows(self):
        rows = [{"t_mono_ns": k * 50_000_000, "vdd_in_mw": 7000.0} for k in range(200_000)]
        assert energy_j(rows, 100 * S, 9000 * S) == pytest.approx(7.0 * 8900, rel=1e-12)

    def test_csv_style_string_values(self):
        rows = [{"t_mono_ns": str(t * S), "vdd_in_mw": str(mw)} for t, mw in [(0, 1000), (1, 3000)]]
        assert energy_j(rows, 0, S) == pytest.approx(2.0, rel=1e-12)

    def test_duplicate_timestamps_do_not_break_integration(self):
        rows = _power_rows([(0, 1000), (1, 1000), (1, 3000), (2, 3000)])
        assert energy_j(rows, 0, 2 * S) == pytest.approx(4.0, rel=1e-12)

    def test_window_before_first_or_after_last_sample_raises(self):
        rows = _power_rows(TRACE)
        with pytest.raises(ValueError):
            energy_j(rows, -1, 2 * S)
        with pytest.raises(ValueError):
            energy_j(rows, S, 3 * S + 1)

    def test_reversed_window_raises(self):
        with pytest.raises(ValueError):
            energy_j(_power_rows(TRACE), 2 * S, S)

    def test_too_few_or_unsorted_rows_raise(self):
        with pytest.raises(ValueError):
            energy_j([], 0, 1)
        with pytest.raises(ValueError):
            energy_j(_power_rows([(0, 1000)]), 0, 0)
        with pytest.raises(ValueError):
            energy_j(_power_rows([(0, 1000), (2, 1000), (1, 1000)]), 0, S)


# --------------------------------------------------------------------------- energy_per_request
def _req(arrival_s, done_s=None, deadline_s=None, warm=False, outcome="complete"):
    def ns(v):
        return None if v is None else int(round(v * S))
    return {"req_id": "r", "t_arrival_ns": ns(arrival_s), "t_done_ns": ns(done_s), "t_deadline_ns": ns(deadline_s),
            "is_warmup": warm, "outcome": outcome}


FLAT_10W = _power_rows([(0.0, 10_000), (50.0, 10_000), (100.0, 10_000)])


class TestEnergyPerRequest:
    def test_window_runs_from_first_nonwarmup_arrival_to_last_nonwarmup_done(self):
        rows = [
            _req(1.0, 99.0, warm=True),                                   # warmup: must not move either end
            _req(10.0, 12.0),                                             # complete
            _req(20.0, 25.0, deadline_s=25.0, outcome="expired_in_queue"),
            _req(30.0, 31.0, outcome="miss"),
            _req(35.0, 40.0, outcome="truncated"),
        ]
        assert request_window(rows) == (10 * S, 40 * S, 4)
        # window 30 s at 10 W = 300 J, divided by all 4 non-warmup requests
        assert energy_per_request(rows, FLAT_10W) == pytest.approx(75.0, rel=1e-12)

    def test_n_counts_every_outcome_including_errors(self):
        rows = [_req(10.0, 20.0, outcome=o) for o in ("complete", "truncated", "miss", "expired_in_queue", "error")]
        assert request_window(rows)[2] == 5
        assert energy_per_request(rows, FLAT_10W) == pytest.approx(100.0 / 5, rel=1e-12)

    def test_row_order_does_not_matter(self):
        rows = [_req(10.0, 12.0), _req(20.0, 40.0), _req(5.0, 6.0, warm=True), _req(15.0, 16.0)]
        assert request_window(rows) == request_window(rows[::-1])
        assert request_window(rows) == (10 * S, 40 * S, 3)

    def test_missing_done_falls_back_to_deadline(self):
        rows = [_req(10.0, 20.0), _req(15.0, None, deadline_s=45.0, outcome="error")]
        assert request_window(rows) == (10 * S, 45 * S, 2)

    def test_deadline_is_not_used_when_done_exists(self):
        rows = [_req(10.0, 20.0, deadline_s=90.0)]
        assert request_window(rows)[1] == 20 * S

    def test_row_without_done_or_deadline_counts_but_does_not_extend_window(self):
        rows = [_req(10.0, 20.0), _req(12.0, None, outcome="error")]
        assert request_window(rows) == (10 * S, 20 * S, 2)

    def test_missing_is_warmup_key_means_not_warmup(self):
        row = {"t_arrival_ns": 10 * S, "t_done_ns": 20 * S}
        assert request_window([row]) == (10 * S, 20 * S, 1)

    def test_only_warmup_rows_raise(self):
        with pytest.raises(ValueError):
            request_window([_req(1.0, 2.0, warm=True)])
        with pytest.raises(ValueError):
            energy_per_request([], FLAT_10W)

    def test_no_end_time_at_all_raises(self):
        with pytest.raises(ValueError):
            request_window([_req(10.0, None, outcome="error")])

    def test_power_trace_not_covering_window_raises(self):
        with pytest.raises(ValueError):
            energy_per_request([_req(10.0, 200.0)], FLAT_10W)

    def test_varying_power_uses_interpolated_edges(self):
        power = _power_rows([(0, 1000), (10, 1000), (20, 3000), (30, 3000)])
        # window [5, 25] over 2 requests: 5..10 -> 1000*5; 10..20 -> mean 2000*10; 20..25 -> 3000*5 (mW*s)
        expected = (1000 * 5 + 2000 * 10 + 3000 * 5) / 1000 / 2
        assert energy_per_request([_req(5.0, 25.0), _req(6.0, 7.0)], power) == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------- mean_ci and welch_diff_ci
class TestMeanCi:
    def test_hand_value_df4(self):
        m, lo, hi = mean_ci([1, 2, 3, 4, 5])
        half = 2.7764 * math.sqrt(2.5) / math.sqrt(5)
        assert m == 3.0
        assert (m - lo) == pytest.approx(half, rel=1e-12)
        assert (hi - m) == pytest.approx(half, rel=1e-12)

    @pytest.mark.parametrize("df", range(1, 31))
    def test_uses_t_table_for_every_df_up_to_30(self, df):
        n = df + 1
        m, lo, hi = mean_ci(list(range(n)))
        se = math.sqrt(n * (n + 1) / 12) / math.sqrt(n)  # sample var of 0..n-1 is n(n+1)/12
        assert hi - m == pytest.approx(T975[df - 1] * se, rel=1e-12)

    def test_above_df_30_uses_1_96(self):
        n = 32  # df 31
        m, lo, hi = mean_ci(list(range(n)))
        se = math.sqrt(n * (n + 1) / 12) / math.sqrt(n)
        assert hi - m == pytest.approx(1.96 * se, rel=1e-12)

    def test_zero_variance_collapses_to_point(self):
        assert mean_ci([2.0, 2.0, 2.0]) == (2.0, 2.0, 2.0)

    def test_returns_builtin_floats(self):
        assert all(type(v) is float for v in mean_ci(np.array([1.0, 2.0, 4.0])))

    def test_fewer_than_two_values_raises(self):
        with pytest.raises(ValueError):
            mean_ci([1.0])
        with pytest.raises(ValueError):
            mean_ci([])


class TestWelch:
    def test_dod_df2_matches_table(self):
        # a=[1,3], b=[2,4]: var 2 each, s2/n = 1 each -> df = 4/2 = 2, t = 4.3027
        d, lo, hi = welch_diff_ci([1, 3], [2, 4])
        half = 4.3027 * math.sqrt(2.0)
        assert d == -1.0
        assert (d - lo) == pytest.approx(half, rel=1e-12)
        assert (hi - d) == pytest.approx(half, rel=1e-12)

    def test_df8_equal_variance_equal_n(self):
        d, lo, hi = welch_diff_ci([1, 2, 3, 4, 5], [11, 12, 13, 14, 15])
        assert d == -10.0
        assert hi - d == pytest.approx(2.3060 * 1.0, rel=1e-12)  # se = sqrt(0.5 + 0.5) = 1

    def test_fractional_df_is_floored(self):
        # s1^2/n1 = 4/3, s2^2/n2 = 56/21 -> df = 7.714, floored to 7 -> t = 2.3646 (rounding would give 2.3060)
        a, b = [1, 3, 5], [2, 4, 6, 8, 10, 12, 14]
        d, lo, hi = welch_diff_ci(a, b)
        se = math.sqrt(4 / 3 + 56 / 21)
        assert hi - d == pytest.approx(2.3646 * se, rel=1e-12)

    def test_large_df_uses_1_96(self):
        a, b = list(range(100)), list(range(100))
        d, lo, hi = welch_diff_ci(a, b)
        se = math.sqrt(2 * (100 * 101 / 12) / 100)
        assert d == 0.0
        assert hi - d == pytest.approx(1.96 * se, rel=1e-12)

    def test_sign_is_mean_a_minus_mean_b(self):
        d, lo, hi = welch_diff_ci([10, 12, 14], [1, 2, 3])
        assert d == 10.0
        assert lo < d < hi

    def test_zero_variance_in_both_groups_collapses(self):
        assert welch_diff_ci([5.0, 5.0], [3.0, 3.0]) == (2.0, 2.0, 2.0)

    def test_fewer_than_two_per_group_raises(self):
        with pytest.raises(ValueError):
            welch_diff_ci([1.0], [1.0, 2.0])
        with pytest.raises(ValueError):
            welch_diff_ci([1.0, 2.0], [])


# --------------------------------------------------------------------------- block bootstrap
class TestBlockBootstrap:
    @pytest.mark.parametrize("n, b", [(1, 1), (2, 2), (3, 2), (8, 2), (9, 3), (26, 3), (27, 3), (28, 4), (64, 4),
                                      (65, 5), (125, 5), (126, 6), (1000, 10), (1001, 11), (343, 7), (344, 8)])
    def test_block_length_is_exact_ceil_cube_root(self, n, b):
        assert block_length(n) == b

    def test_block_length_matches_float_formula_off_the_cubes(self):
        cubes = {i ** 3 for i in range(1, 30)}
        for n in range(1, 20_000):
            if n not in cubes:
                assert block_length(n) == math.ceil(n ** (1 / 3))

    def test_block_length_is_exposed_on_the_result(self):
        xs = np.random.default_rng(0).random(100)
        ci = block_bootstrap_ci(xs, np.mean, B=50, seed=0)
        assert ci.block == block_length(100) == 5

    def test_result_unpacks_as_lo_hi_and_keeps_attributes(self):
        ci = block_bootstrap_ci(np.arange(30.0), np.mean, B=100, seed=0)
        lo, hi = ci
        assert isinstance(ci, BootCI) and isinstance(ci, tuple) and len(ci) == 2
        assert (ci.lo, ci.hi) == (lo, hi) and lo <= hi
        assert type(lo) is float and type(hi) is float

    def test_result_survives_pickle(self):
        ci = block_bootstrap_ci(np.arange(30.0), np.mean, B=20, seed=0)
        back = pickle.loads(pickle.dumps(ci))
        assert back == ci and back.block == ci.block

    def test_resamples_contiguous_blocks_not_single_points(self):
        # xs = 0..63, b = 4: each replicate is 16 blocks, each block holds 3 consecutive pairs.
        # stat = number of consecutive pairs -> always >= 48 for blocks, ~1 for iid, < 48 if blocks wrap or are shorter
        xs = np.arange(64.0)

        def consecutive_pairs(r):
            return float(np.sum(np.diff(r) == 1.0))

        ci = block_bootstrap_ci(xs, consecutive_pairs, B=1000, seed=0)
        assert ci.block == 4
        assert ci.lo == 48.0
        assert ci.hi <= 63.0

    def test_replicates_have_length_n_when_n_is_not_a_multiple_of_block(self):
        lengths = set()

        def stat(r):
            lengths.add(len(r))
            return float(r.max())

        xs = np.arange(10.0)  # b = 3 -> 4 blocks = 12 values, truncated to 10
        ci = block_bootstrap_ci(xs, stat, B=200, seed=1)
        assert lengths == {10}
        assert 0.0 <= ci.lo <= ci.hi <= 9.0

    def test_same_seed_same_ci_different_seed_different_ci(self):
        xs = np.random.default_rng(2).standard_normal(200)
        a = block_bootstrap_ci(xs, np.mean, B=300, seed=5)
        b = block_bootstrap_ci(xs, np.mean, B=300, seed=5)
        c = block_bootstrap_ci(xs, np.mean, B=300, seed=6)
        assert a == b
        assert a != c

    def test_does_not_mutate_input(self):
        xs = np.random.default_rng(3).random(50)
        before = xs.copy()
        block_bootstrap_ci(xs, np.mean, B=50, seed=0)
        assert np.array_equal(xs, before)

    def test_accepts_plain_lists(self):
        ci = block_bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], np.median, B=100, seed=0)
        assert 1.0 <= ci.lo <= ci.hi <= 6.0

    def test_iid_width_agrees_with_normal_theory(self):
        xs = np.random.default_rng(4).standard_normal(400)
        ci = block_bootstrap_ci(xs, np.mean, B=1000, seed=0)
        _, lo, hi = mean_ci(xs)
        assert (ci.hi - ci.lo) == pytest.approx(hi - lo, rel=0.25)

    def test_autocorrelated_data_gets_a_wider_interval_than_iid_theory(self):
        xs = _ar1(0.9, 2000, 8)
        ci = block_bootstrap_ci(xs, np.mean, B=1000, seed=0)
        _, lo, hi = mean_ci(xs)
        assert (ci.hi - ci.lo) > 1.5 * (hi - lo)

    def test_higher_level_gives_wider_interval(self):
        xs = np.random.default_rng(9).standard_normal(300)
        narrow = block_bootstrap_ci(xs, np.mean, B=1000, seed=0, level=0.8)
        wide = block_bootstrap_ci(xs, np.mean, B=1000, seed=0, level=0.99)
        assert (wide.hi - wide.lo) > (narrow.hi - narrow.lo)

    def test_single_value_gives_degenerate_interval(self):
        ci = block_bootstrap_ci([4.0], np.mean, B=10, seed=0)
        assert (ci.lo, ci.hi, ci.block) == (4.0, 4.0, 1)

    @pytest.mark.parametrize("kwargs", [{"B": 0}, {"level": 0.0}, {"level": 1.0}, {"level": 1.5}])
    def test_invalid_parameters_raise(self, kwargs):
        with pytest.raises(ValueError):
            block_bootstrap_ci([1.0, 2.0, 3.0], np.mean, seed=0, **kwargs)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            block_bootstrap_ci([], np.mean, seed=0)


class TestWilsonAndPaired:
    def test_wilson_95_of_100(self):
        from exp2.stats import wilson_ci
        lo, hi = wilson_ci(95, 100)
        assert lo == pytest.approx(0.8882, abs=5e-5) and hi == pytest.approx(0.9785, abs=5e-5)

    def test_wilson_edges_stay_inside_unit_interval(self):
        from exp2.stats import wilson_ci
        lo, hi = wilson_ci(0, 10)
        assert lo == 0.0 and 0.0 < hi < 0.4
        lo, hi = wilson_ci(10, 10)
        assert hi == 1.0 and 0.6 < lo < 1.0

    @pytest.mark.parametrize("k,n", [(1, 0), (-1, 5), (6, 5), (0, -1)])
    def test_wilson_invalid_raises(self, k, n):
        from exp2.stats import wilson_ci
        with pytest.raises(ValueError):
            wilson_ci(k, n)

    def test_paired_diff_matches_mean_ci_of_differences(self):
        from exp2.stats import paired_diff_ci
        a, b = [2.0, 2.2, 2.4], [1.5, 1.7, 1.6]
        d, lo, hi = paired_diff_ci(a, b)
        m, mlo, mhi = mean_ci([x - y for x, y in zip(a, b)])
        assert (d, lo, hi) == pytest.approx((m, mlo, mhi))

    def test_paired_diff_is_tighter_than_welch_for_correlated_pairs(self):
        from exp2.stats import paired_diff_ci
        a, b = [1.0, 2.0, 3.0], [0.9, 1.9, 2.9]
        _, lo, hi = paired_diff_ci(a, b)
        _, wlo, whi = welch_diff_ci(a, b)
        assert hi - lo < whi - wlo

    @pytest.mark.parametrize("a,b", [([1.0, 2.0], [1.0]), ([1.0], [2.0]), ([], [])])
    def test_paired_diff_invalid_raises(self, a, b):
        from exp2.stats import paired_diff_ci
        with pytest.raises(ValueError):
            paired_diff_ci(a, b)
