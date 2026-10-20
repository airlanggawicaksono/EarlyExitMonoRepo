"""W2 screen_window: clock lock fractions, temperature, memory, swap, page faults over a window (mock rows only)."""
import pytest

from exp2 import contracts as C
from exp2.analyze import screen as SC

SEC = 1_000_000_000


def _row(t_s, gpu_mhz=1020, emc=3199, tj=50000, mem_kb=2048 * 1024, **extra):
    r = {"t_mono_ns": int(t_s * SEC), "gpu_cur_hz": gpu_mhz * 1_000_000, "emc_mhz": emc, "mem_available_kb": mem_kb,
         "gpu_load_permille": 0, "temp_tj_thermal_mc": tj, "temp_gpu_thermal_mc": tj - 100}
    r.update(extra)
    return r


def test_all_locked_is_ok_and_window_is_inclusive():
    rows = [_row(t) for t in range(10)]
    r = SC.screen_window(rows, 2 * SEC, 5 * SEC, gpu_mhz=1020, emc_mhz=3199)
    assert r["n_clock_rows"] == 4
    assert r["gpu_lock_frac"] == 1.0 and r["emc_lock_frac"] == 1.0 and r["screen_ok"] == 1
    assert r["max_tj_c"] == 50.0 and r["min_mem_available_mb"] == 2048.0


def test_gpu_unlock_below_threshold_flags():
    rows = [_row(t) for t in range(100)]
    rows[10]["gpu_cur_hz"] = 1_300_000_000     # 1 of 100 off -> 0.99 still ok
    assert SC.screen_window(rows, 0, 99 * SEC, gpu_mhz=1020, emc_mhz=None)["screen_ok"] == 1
    rows[11]["gpu_cur_hz"] = 1_300_000_000     # 2 of 100 off -> 0.98
    r = SC.screen_window(rows, 0, 99 * SEC, gpu_mhz=1020, emc_mhz=None)
    assert r["gpu_lock_frac"] == pytest.approx(0.98) and r["screen_ok"] == 0


def test_lock_tolerance_one_mhz():
    rows = [_row(t) for t in range(5)]
    rows[0]["gpu_cur_hz"] = 1_020_900_000      # 0.9 MHz off, inside tolerance
    assert SC.screen_window(rows, 0, 4 * SEC, gpu_mhz=1020, emc_mhz=None)["gpu_lock_frac"] == 1.0


def test_unpinned_arm_has_no_gpu_lock_and_is_ok():
    rows = [_row(t, gpu_mhz=300 + 50 * t) for t in range(5)]
    r = SC.screen_window(rows, 0, 4 * SEC, gpu_mhz=None, emc_mhz=None)
    assert r["gpu_lock_frac"] is None and r["emc_lock_frac"] is None and r["screen_ok"] == 1


def test_emc_lock_uses_only_rows_that_log_it():
    rows = [_row(0, emc=None), _row(1), _row(2), _row(3, emc=2000)]
    r = SC.screen_window(rows, 0, 3 * SEC, gpu_mhz=1020, emc_mhz=3199)
    assert r["emc_lock_frac"] == pytest.approx(2 / 3) and r["screen_ok"] == 0


def test_emc_never_logged_is_none_and_not_a_failure():
    rows = [_row(t, emc=None) for t in range(4)]
    r = SC.screen_window(rows, 0, 3 * SEC, gpu_mhz=1020, emc_mhz=3199)
    assert r["emc_lock_frac"] is None and r["screen_ok"] == 1


def test_tj_falls_back_to_max_of_temp_columns():
    rows = [_row(t) for t in range(3)]
    for r in rows:
        del r["temp_tj_thermal_mc"]
    rows[1]["temp_gpu_thermal_mc"] = 71500
    assert SC.screen_window(rows, 0, 2 * SEC, gpu_mhz=1020, emc_mhz=None)["max_tj_c"] == 71.5


def test_swap_and_pgmajfault_old_clocks_are_none():
    r = SC.screen_window([_row(t) for t in range(3)], 0, 2 * SEC, gpu_mhz=1020, emc_mhz=None)
    assert r["max_swap_used_mb"] is None and r["pgmajfault_delta"] is None


def test_swap_and_pgmajfault_new_clocks():
    rows = [_row(t, swap_total_kb=4096 * 1024, swap_free_kb=(4096 - 100 * t) * 1024, pgmajfault=1000 + 7 * t)
            for t in range(4)]
    r = SC.screen_window(rows, 0, 3 * SEC, gpu_mhz=1020, emc_mhz=None)
    assert r["max_swap_used_mb"] == 300.0 and r["pgmajfault_delta"] == 21


def test_empty_window_is_not_ok_and_does_not_raise():
    r = SC.screen_window([_row(t) for t in range(3)], 10 * SEC, 20 * SEC, gpu_mhz=1020, emc_mhz=None)
    assert r["n_clock_rows"] == 0 and r["screen_ok"] == 0
    assert r["gpu_lock_frac"] is None and r["max_tj_c"] is None and r["min_mem_available_mb"] is None


def test_result_keys_match_screen_columns():
    r = SC.screen_window([_row(0)], 0, SEC, gpu_mhz=1020, emc_mhz=None)
    assert set(r) <= set(C.SCREEN_CSV_COLS)
    assert {"n_clock_rows", "gpu_lock_frac", "emc_lock_frac", "max_tj_c", "min_mem_available_mb",
            "max_swap_used_mb", "pgmajfault_delta", "screen_ok"} == set(r)
