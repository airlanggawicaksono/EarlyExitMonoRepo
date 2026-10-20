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


def paired_diff_ci(a, b):
    """(mean(a - b), lo, hi): Student-t 95% CI of the paired differences. Equal length and n >= 2 are required."""
    xa, xb = _floats(a), _floats(b)
    if xa.size != xb.size:
        raise ValueError("paired_diff_ci: a and b must have the same length")
    if xa.size < 2:
        raise ValueError("paired_diff_ci: needs at least 2 pairs")
    return mean_ci(xa - xb)


def wilson_ci(k, n):
    """(lo, hi): Wilson score 95% interval (z 1.96) of k successes in n trials. n >= 1 and 0 <= k <= n."""
    if n < 1 or k < 0 or k > n:
        raise ValueError("wilson_ci: need n >= 1 and 0 <= k <= n, got k=%r n=%r" % (k, n))
    z2 = _Z975 * _Z975
    p = k / n
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = _Z975 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------- Student t, Holm, Spearman
def _betacf(a, b, x):
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (d if abs(d) > tiny else tiny)
        h *= d * (c if abs(c) > tiny else tiny)
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (d if abs(d) > tiny else tiny)
        delta = d * (c if abs(c) > tiny else tiny)
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b), 0 <= x <= 1."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(ln_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(ln_front) * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    """Two-sided p value of Student's t statistic with df > 0 degrees of freedom: I_{df/(df+t^2)}(df/2, 1/2)."""
    if not df > 0:
        raise ValueError("t_two_sided_p: df must be > 0, got %r" % (df,))
    t = float(t)
    if t != t:
        raise ValueError("t_two_sided_p: t is NaN")
    if math.isinf(t):
        return 0.0
    return min(1.0, max(0.0, _betainc(df / 2.0, 0.5, df / (df + t * t))))


def paired_t_test(a, b):
    """(mean(a - b), lo, hi, p): paired t test, 95% interval as paired_diff_ci, two-sided p from Student t with
    n - 1 df. Zero-variance differences give p 0.0 (non-zero mean) or 1.0 (all zero). Needs n >= 2."""
    m, lo, hi = paired_diff_ci(a, b)
    d = _floats(a) - _floats(b)
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return m, lo, hi, (1.0 if m == 0.0 else 0.0)
    return m, lo, hi, t_two_sided_p(m / (sd / math.sqrt(d.size)), d.size - 1)


def holm(pvals):
    """Holm step-down adjusted p values, same order as the input: sorted ascending, p_(i) * (m - i), running
    maximum, capped at 1. Reject where the adjusted value <= alpha."""
    ps = list(pvals)
    for p in ps:
        if not 0.0 <= p <= 1.0:
            raise ValueError("holm: p values must be in [0, 1], got %r" % (p,))
    m = len(ps)
    out, run = [0.0] * m, 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: ps[j])):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        out[i] = run
    return out


def _ranks(xs):
    """Average ranks (1-based) with ties."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(x, y):
    """Spearman rank correlation (Pearson on average ranks). None when n < 2 or either side is constant."""
    xa, ya = _floats(x), _floats(y)
    if xa.size != ya.size:
        raise ValueError("spearman: x and y must have the same length")
    if xa.size < 2:
        return None
    rx, ry = np.array(_ranks(list(xa))), np.array(_ranks(list(ya)))
    if np.ptp(rx) == 0.0 or np.ptp(ry) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


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
