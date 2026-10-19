"""Small statistics helpers for Exp2 (BRIEF_v1 sec. 4 'S1a'). numpy + stdlib only.

Conventions
- Time columns are int nanoseconds on one clock (time.monotonic_ns); power is mW; energy is J.
- Anything that cannot be computed honestly raises ValueError (empty input, n < 2 for a CI, a window the power
  trace does not cover). Nothing is clamped or filled in silently.
- Confidence intervals are 95% only (Student-t table for df 1..30, then 1.96).
"""
import math

import numpy as np

# Two-sided 95% Student-t quantiles for df = 1..30. Above df 30 the normal value is used.
_T975 = (12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2622, 2.2281,
         2.2010, 2.1788, 2.1604, 2.1448, 2.1315, 2.1199, 2.1098, 2.1009, 2.0930, 2.0860,
         2.0796, 2.0739, 2.0687, 2.0639, 2.0595, 2.0555, 2.0518, 2.0484, 2.0452, 2.0423)
_Z975 = 1.96
_NS = 1e9


def _floats(xs):
    """1-D finite float array from any numeric iterable."""
    a = np.asarray(list(xs) if not hasattr(xs, "__len__") else xs, dtype=float)
    if a.ndim != 1:
        raise ValueError("expected a 1-D sequence of numbers")
    if not np.all(np.isfinite(a)):
        raise ValueError("input contains NaN or inf")
    return a


def _t975(df):
    """Quantile for a (possibly fractional) df: floor(df), which is the conservative (wider) choice."""
    d = max(1, int(math.floor(df + 1e-9)))
    return _T975[d - 1] if d <= len(_T975) else _Z975


# --------------------------------------------------------------------------- order statistics and error
def p_nearest_rank(xs, q):
    """sorted(xs)[ceil(q*n) - 1]; q=0 gives the minimum. Returns an element of xs."""
    vals = list(xs)
    n = len(vals)
    if n == 0:
        raise ValueError("p_nearest_rank: empty input")
    if not 0.0 <= q <= 1.0:
        raise ValueError("p_nearest_rank: q must be in [0, 1], got %r" % (q,))
    if any(v != v for v in vals):
        raise ValueError("p_nearest_rank: input contains NaN")
    rank = math.ceil(q * n - 1e-9)  # the 1e-9 absorbs float noise such as 0.07 * 100 == 7.000000000000001
    return sorted(vals)[min(max(rank, 1), n) - 1]


def mape(pred, meas):
    """mean(|pred - meas| / |meas|) * 100. meas must be non-zero."""
    p, m = _floats(pred), _floats(meas)
    if p.shape != m.shape or p.size == 0:
        raise ValueError("mape: pred and meas must be non-empty and the same length")
    if np.any(m == 0.0):
        raise ValueError("mape: meas contains 0")
    return float(np.mean(np.abs(p - m) / np.abs(m)) * 100.0)


# --------------------------------------------------------------------------- autocorrelation
def lag1(xs):
    """Lag-1 sample autocorrelation r1. A constant or shorter-than-2 series gives 0.0."""
    x = _floats(xs)
    if x.size < 2 or np.ptp(x) == 0.0:
        return 0.0
    c = x - x.mean()
    return float(np.dot(c[:-1], c[1:]) / np.dot(c, c))


def ess(xs):
    """Effective sample size n * (1 - r1) / (1 + r1). Not clamped to n (negative r1 gives ess > n)."""
    x = _floats(xs)
    if x.size == 0:
        raise ValueError("ess: empty input")
    r1 = lag1(x)
    return float(x.size * (1.0 - r1) / (1.0 + r1))


# --------------------------------------------------------------------------- integration and energy
def trapezoid(t, y):
    """Trapezoid rule, written by hand (numpy's name for it differs between 1.x and 2.x). t non-decreasing."""
    tt, yy = _floats(t), _floats(y)
    if tt.shape != yy.shape:
        raise ValueError("trapezoid: t and y must have the same length")
    if tt.size < 2:
        return 0.0
    dt = np.diff(tt)
    if np.any(dt < 0.0):
        raise ValueError("trapezoid: t must be non-decreasing")
    return float(np.sum(dt * (yy[1:] + yy[:-1])) / 2.0)


def _to_int(v):
    try:
        return int(v)
    except ValueError:  # CSV text such as "5000000.0"
        return int(float(v))


def _power_arrays(power_rows):
    rows = list(power_rows)
    if len(rows) < 2:
        raise ValueError("power trace needs at least 2 samples, got %d" % len(rows))
    t = np.array([_to_int(r["t_mono_ns"]) for r in rows], dtype=np.int64)
    p = np.array([float(r["vdd_in_mw"]) for r in rows], dtype=np.float64)
    if not np.all(np.isfinite(p)):
        raise ValueError("power trace contains NaN or inf")
    if np.any(np.diff(t) < 0):
        raise ValueError("power rows are not sorted by t_mono_ns")
    return t, p


def _interp(t, p, x):
    """Linear interpolation of the sample trace at int time x (exact integer time differences)."""
    j = min(max(int(np.searchsorted(t, x, side="right")) - 1, 0), len(t) - 2)
    dt = int(t[j + 1]) - int(t[j])
    if dt == 0:
        return float(p[j + 1])
    return float(p[j]) + (float(p[j + 1]) - float(p[j])) * ((x - int(t[j])) / dt)


def energy_j(power_rows, t0_ns, t1_ns):
    """Energy in J over [t0_ns, t1_ns] from power rows (dicts with t_mono_ns, vdd_in_mw).

    The trace is piecewise linear between samples; the clipped ends are linearly interpolated.
    mW * s / 1000 = J. A window not covered by the samples raises ValueError.
    """
    t, p = _power_arrays(power_rows)
    t0, t1 = _to_int(t0_ns), _to_int(t1_ns)
    if t1 < t0:
        raise ValueError("energy_j: t1_ns < t0_ns")
    if t0 < int(t[0]) or t1 > int(t[-1]):
        raise ValueError("energy_j: window [%d, %d] ns is not covered by power samples [%d, %d] ns"
                         % (t0, t1, int(t[0]), int(t[-1])))
    if t1 == t0:
        return 0.0
    inner = (t > t0) & (t < t1)
    ts = np.concatenate((np.array([t0], dtype=np.int64), t[inner], np.array([t1], dtype=np.int64)))
    ps = np.concatenate(([_interp(t, p, t0)], p[inner], [_interp(t, p, t1)]))
    return trapezoid((ts - t0) / _NS, ps) / 1000.0


def request_window(rows):
    """(t0_ns, t1_ns, n) for a condition's requests.rows (dicts of K8 keys), warmup rows excluded.

    t0 = first scheduled arrival, t1 = last t_done_ns (t_deadline_ns when a row has no t_done_ns),
    n = every non-warmup row whatever its outcome. A row with neither time counts in n but cannot extend t1.
    """
    starts, ends, n = [], [], 0
    for r in rows:
        if r.get("is_warmup"):
            continue
        n += 1
        starts.append(int(r["t_arrival_ns"]))
        end = r.get("t_done_ns")
        if end is None:
            end = r.get("t_deadline_ns")
        if end is not None:
            ends.append(int(end))
    if n == 0:
        raise ValueError("request_window: no non-warmup requests")
    if not ends:
        raise ValueError("request_window: no non-warmup request has t_done_ns or t_deadline_ns")
    return min(starts), max(ends), n


def energy_per_request(rows, power_rows):
    """Window energy (J) divided by the number of non-warmup requests (all outcomes count)."""
    t0, t1, n = request_window(rows)
    return energy_j(power_rows, t0, t1) / n


# --------------------------------------------------------------------------- confidence intervals
def mean_ci(xs):
    """(mean, lo, hi): Student-t 95% CI of the mean. Needs n >= 2."""
    x = _floats(xs)
    if x.size < 2:
        raise ValueError("mean_ci: needs at least 2 values")
    m = float(x.mean())
    half = _t975(x.size - 1) * float(x.std(ddof=1)) / math.sqrt(x.size)
    return m, m - half, m + half


def welch_diff_ci(a, b):
    """(mean(a) - mean(b), lo, hi): Welch 95% CI with the Satterthwaite df floored. Needs n >= 2 per group."""
    xa, xb = _floats(a), _floats(b)
    if xa.size < 2 or xb.size < 2:
        raise ValueError("welch_diff_ci: needs at least 2 values per group")
    va, vb = float(xa.var(ddof=1)) / xa.size, float(xb.var(ddof=1)) / xb.size
    d = float(xa.mean() - xb.mean())
    se2 = va + vb
    if se2 == 0.0:
        return d, d, d
    df = se2 * se2 / (va * va / (xa.size - 1) + vb * vb / (xb.size - 1))
    half = _t975(df) * math.sqrt(se2)
    return d, d - half, d + half


# --------------------------------------------------------------------------- block bootstrap
def block_length(n):
    """ceil(n ** (1/3)) computed exactly (no float error on perfect cubes)."""
    if n < 1:
        raise ValueError("block_length: n must be >= 1")
    b = max(1, round(n ** (1.0 / 3.0)))
    while b ** 3 < n:
        b += 1
    while b > 1 and (b - 1) ** 3 >= n:
        b -= 1
    return b


class BootCI(tuple):
    """(lo, hi) of a block-bootstrap CI. ``.block`` is the block length used; ``.lo`` and ``.hi`` alias the items."""

    def __new__(cls, lo, hi, block):
        self = super().__new__(cls, (float(lo), float(hi)))
        self.block = int(block)
        return self

    def __getnewargs__(self):
        return self[0], self[1], self.block

    @property
    def lo(self):
        return self[0]

    @property
    def hi(self):
        return self[1]


def block_bootstrap_ci(xs, stat, B=1000, seed=0, level=0.95):
    """Moving-block bootstrap percentile CI of stat(sample) for a dependent series.

    Blocks are overlapping, contiguous and do not wrap; length ceil(n ** (1/3)). Each replicate concatenates
    ceil(n / block) random blocks and truncates to n. ``stat`` receives a float ndarray of length n.
    Returns BootCI(lo, hi) with ``.block``. Deterministic for a given seed (numpy PCG64).
    """
    x = _floats(xs)
    n = x.size
    if n == 0:
        raise ValueError("block_bootstrap_ci: empty input")
    if B < 1:
        raise ValueError("block_bootstrap_ci: B must be >= 1")
    if not 0.0 < level < 1.0:
        raise ValueError("block_bootstrap_ci: level must be in (0, 1)")
    b = block_length(n)
    k = -(-n // b)
    offs = np.arange(b)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    reps = np.empty(int(B))
    for i in range(int(B)):
        starts = rng.integers(0, n - b + 1, size=k)
        reps[i] = stat(x[(starts[:, None] + offs).ravel()[:n]])
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(reps, [alpha, 1.0 - alpha])
    return BootCI(lo, hi, b)
