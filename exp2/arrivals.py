"""Open-loop arrival schedules (BRIEF_v1 K2). Seeded numpy PCG64; no clock, no I/O.

schedule(workload, n, seed) -> list of ScheduleItem dicts, times relative to the condition start t0:
    {"req_id": "r000001", "type_id": "bert:5", "t_arrival_ns": 134999872, "t_deadline_ns": None, "k_override": None}

Draw order from one PCG64(seed) generator: n inter-arrival gaps first, then n independent type indices.
poisson: exponential gaps with mean 1/rate_per_s (cv_a must be 1).
gamma:   shape 1/cv_a**2, scale mean * cv_a**2, so the mean gap stays 1/rate_per_s and the CV is cv_a.
Deadlines are filled in later (contracts.with_deadlines), so t_deadline_ns is None here.
"""
import math
from collections.abc import Mapping

import numpy as np

ARRIVALS = ("poisson", "gamma")
_NS = 1e9
_PROB_SUM_TOL = 1e-6


def _field(workload, key, *default):
    """workload may be a dict or any object with attributes (contracts.Workload)."""
    if isinstance(workload, Mapping):
        if key in workload:
            return workload[key]
    elif hasattr(workload, key):
        return getattr(workload, key)
    if default:
        return default[0]
    raise ValueError("workload has no field %r" % key)


def _check(workload):
    types = list(_field(workload, "types"))
    probs = np.asarray(_field(workload, "probs"), dtype=float)
    rate = float(_field(workload, "rate_per_s"))
    arrival = _field(workload, "arrival")
    cv = float(_field(workload, "cv_a", 1.0))
    if not types or probs.shape != (len(types),):
        raise ValueError("workload needs one prob per type and at least one type")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0) or abs(probs.sum() - 1.0) > _PROB_SUM_TOL:
        raise ValueError("workload probs must be >= 0 and sum to 1, got %r" % (probs.tolist(),))
    if not (math.isfinite(rate) and rate > 0):
        raise ValueError("workload rate_per_s must be > 0, got %r" % (rate,))
    if arrival not in ARRIVALS:
        raise ValueError("workload arrival must be one of %r, got %r" % (ARRIVALS, arrival))
    if arrival == "poisson" and abs(cv - 1.0) > 1e-9:
        raise ValueError("poisson arrivals need cv_a == 1, got %r (use arrival='gamma')" % (cv,))
    if arrival == "gamma" and not (math.isfinite(cv) and cv > 0):
        raise ValueError("gamma arrivals need cv_a > 0, got %r" % (cv,))
    return types, probs / probs.sum(), rate, arrival, cv


def schedule(workload, n, seed):
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0, got %d" % n)
    types, probs, rate, arrival, cv = _check(workload)
    if n == 0:
        return []
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    mean = 1.0 / rate
    if arrival == "poisson":
        gaps = rng.exponential(mean, n)
    else:
        gaps = rng.gamma(1.0 / cv ** 2, mean * cv ** 2, n)
    t_ns = np.rint(np.cumsum(gaps) * _NS).astype(np.int64).tolist()
    picks = rng.choice(len(types), size=n, p=probs).tolist()
    return [{"req_id": "r%06d" % (i + 1), "type_id": types[j], "t_arrival_ns": t, "t_deadline_ns": None,
             "k_override": None} for i, (t, j) in enumerate(zip(t_ns, picks))]
