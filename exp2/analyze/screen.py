"""W2/B5 screening of one closed condition against clocks.csv (rules: exp2/contracts.py, "Run 1 variants and screening").

    screen_window(clocks, t0_ns, t1_ns, *, gpu_mhz, emc_mhz) -> dict

Report and flag only (D8): the caller writes the dict into screen.csv; nothing is dropped here.
"""
from exp2 import contracts as C


def _lock_frac(rows, key, scale, target):
    vals = [r[key] / scale for r in rows if r.get(key) is not None]
    if target is None or not vals:
        return None
    return sum(1 for v in vals if abs(v - target) <= C.SCREEN_LOCK_TOL_MHZ) / len(vals)


def _max_tj_c(rows):
    vals = [r[C.SCREEN_TJ_COL] for r in rows if r.get(C.SCREEN_TJ_COL) is not None]
    if not vals:
        vals = [v for r in rows for k, v in r.items() if k.startswith("temp_") and v is not None]
    return max(vals) / 1000 if vals else None


def screen_window(clocks, t0_ns, t1_ns, *, gpu_mhz, emc_mhz):
    """Screen the clock rows with t0_ns <= t_mono_ns <= t1_ns. gpu_mhz None = unpinned arm (no GPU lock check);
    emc_mhz None = no EMC target. Returns the screen.csv metrics (None = empty cell)."""
    rows = [r for r in clocks if t0_ns <= r["t_mono_ns"] <= t1_ns]
    gpu = _lock_frac(rows, "gpu_cur_hz", 1e6, gpu_mhz)
    emc = _lock_frac(rows, "emc_mhz", 1.0, emc_mhz)
    mem = [r["mem_available_kb"] for r in rows if r.get("mem_available_kb") is not None]
    swap = [(r["swap_total_kb"] - r["swap_free_kb"]) / 1024 for r in rows
            if r.get("swap_total_kb") is not None and r.get("swap_free_kb") is not None]
    maj = [r["pgmajfault"] for r in rows if r.get("pgmajfault") is not None]
    ok = bool(rows) and (gpu is None or gpu >= C.SCREEN_GPU_LOCK_MIN) and (emc is None or emc >= C.SCREEN_EMC_LOCK_MIN)
    return {"n_clock_rows": len(rows), "gpu_lock_frac": gpu, "emc_lock_frac": emc, "max_tj_c": _max_tj_c(rows),
            "min_mem_available_mb": min(mem) / 1024 if mem else None,
            "max_swap_used_mb": max(swap) if swap else None,
            "pgmajfault_delta": maj[-1] - maj[0] if len(maj) >= 2 else None,
            "screen_ok": 1 if ok else 0}
