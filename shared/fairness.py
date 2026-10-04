"""Fairness and within-run dispersion metrics for multi-tenant analysis.

Two families of functions:

1. Jain's fairness index and unfairness ratio, applied to per-tenant slowdown
   values for one concurrent cell.

2. Within-run latency dispersion (mean, std, CV, quartiles) computed from the
   per-sample latency list inside one hw_results.json run.

References
----------
Jain's index: R. Jain, D. Chiu, W. Hawe, "A Quantitative Measure of Fairness
and Discrimination for Resource Allocation in Shared Computer Systems", DEC
Technical Report TR-301, 1984.

Unfairness ratio (max/min slowdown): standard in multiprogram scheduling
evaluation, see e.g. S. Eyerman and L. Eeckhout, "System-Level Performance
Metrics for Multiprogram Workloads", IEEE Micro, 2008.  That paper is already
cited elsewhere in this project.

Both Jain's index and the max/min ratio answer the same question (how equal are
the slowdowns?) but from opposite directions: J=1.0 and ratio=1.0 both mean
perfect fairness, while J=1/n and a large ratio both mean maximum unfairness.
"""

import math
from typing import Optional


def jain_fairness(slowdowns: list) -> Optional[dict]:
    """Compute Jain's fairness index over per-tenant slowdown values.

    Filters out non-physical (zero or negative) slowdowns before computing.
    Returns None for an empty input rather than a fabricated 1.0.
    Returns a trivially-labelled 1.0 for a single tenant (index is always 1.0
    with n=1, since there is no one to be unfair to).

    Return dict keys:
        jain_index      -- float in [1/n, 1.0], or None
        unfairness_ratio -- max/min slowdown, or None if fewer than 2 valid values
        min_slowdown    -- float or None
        max_slowdown    -- float or None
        n_valid         -- count of slowdowns that passed the physical filter
        n_excluded      -- count of zero-or-negative values excluded
        trivial         -- True when n_valid == 1 (index is definitionally 1.0)

    Args:
        slowdowns: list of per-tenant slowdown values (lat_shared / lat_solo).
                   Non-physical values (<= 0) are excluded with a count.
    """
    if not slowdowns:
        return None

    valid = [x for x in slowdowns if isinstance(x, (int, float)) and x > 0]
    excluded = len(slowdowns) - len(valid)

    if not valid:
        return None

    n = len(valid)
    s = sum(valid)
    sq = sum(x * x for x in valid)

    # ponytail: sq > 0 guaranteed because all valid > 0
    j = (s * s) / (n * sq)

    trivial = (n == 1)
    min_sd = min(valid)
    max_sd = max(valid)
    ratio = max_sd / min_sd if n >= 2 else None

    return {
        "jain_index": round(j, 6),
        "unfairness_ratio": round(ratio, 6) if ratio is not None else None,
        "min_slowdown": round(min_sd, 6),
        "max_slowdown": round(max_sd, 6),
        "n_valid": n,
        "n_excluded": excluded,
        "trivial": trivial,
    }


def latency_dispersion(latencies: list) -> Optional[dict]:
    """Compute within-run latency dispersion from a list of per-sample values.

    This is WITHIN-run dispersion only.  It cannot capture run-to-run drift
    such as thermal effects; repeated independent runs are required for that.
    The coefficient of variation (CV = std / mean) is the most useful output
    because it is unitless and comparable across exits and models.

    Filters out non-positive values.  Returns None for empty input.

    Return dict keys:
        mean        -- arithmetic mean (seconds, or whatever unit the input is)
        std         -- sample standard deviation (ddof=1); None if n < 2
        cv          -- std / mean; None if std is None or mean is 0
        q1, q2, q3 -- 25th, 50th, 75th percentiles
        n           -- count of valid samples used
    """
    valid = [x for x in latencies if isinstance(x, (int, float)) and x > 0]
    if not valid:
        return None

    n = len(valid)
    mean = sum(valid) / n

    if n >= 2:
        variance = sum((x - mean) ** 2 for x in valid) / (n - 1)
        std = math.sqrt(variance)
        cv = std / mean if mean > 0 else None
    else:
        std = None
        cv = None

    sorted_v = sorted(valid)

    def _percentile(p):
        # linear interpolation, same as numpy default
        idx = p / 100.0 * (n - 1)
        lo = int(idx)
        hi = lo + 1
        if hi >= n:
            return sorted_v[-1]
        return sorted_v[lo] + (idx - lo) * (sorted_v[hi] - sorted_v[lo])

    return {
        "mean": mean,
        "std": std,
        "cv": round(cv, 6) if cv is not None else None,
        "q1": _percentile(25),
        "q2": _percentile(50),
        "q3": _percentile(75),
        "n": n,
    }
