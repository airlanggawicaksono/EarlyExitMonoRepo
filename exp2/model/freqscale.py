"""Frequency scaling tau(f) = a * f^-b + c by variable projection: b on a grid (then refined), a and c by least squares."""
import numpy as np


def _solve_ac(x, tau, b):
    """Least squares for a, c at fixed b. x is f / f_max, so the basis column is x^-b. Returns (a_scaled, c, sse)."""
    basis = np.column_stack([x ** (-b), np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(basis, tau, rcond=None)
    resid = tau - basis @ coef
    return float(coef[0]), float(coef[1]), float(resid @ resid)


def fit_tau(freqs_mhz, taus, b_range=(0.01, 4.0), n_grid=400):
    """Return (a, b, c) with tau(f) = a * f^-b + c, f in MHz. Needs at least 3 distinct levels."""
    f = np.asarray(freqs_mhz, dtype=float)
    tau = np.asarray(taus, dtype=float)
    if f.shape != tau.shape or len(np.unique(f)) < 3:
        raise ValueError("fit_tau needs matching arrays with at least 3 distinct frequencies")
    f_max = float(f.max())
    x = f / f_max
    grid = np.linspace(b_range[0], b_range[1], n_grid)
    sse = [_solve_ac(x, tau, b)[2] for b in grid]
    i = int(np.argmin(sse))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)]
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(200):  # golden section on the projected objective
        m1, m2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
        if _solve_ac(x, tau, m1)[2] < _solve_ac(x, tau, m2)[2]:
            hi = m2
        else:
            lo = m1
    b = 0.5 * (lo + hi)
    a_s, c, _ = _solve_ac(x, tau, b)
    return a_s * f_max ** b, b, c  # a_s multiplies (f/f_max)^-b = f^-b * f_max^b


def predict_tau(params, f_mhz):
    a, b, c = params
    return a * float(f_mhz) ** (-b) + c
