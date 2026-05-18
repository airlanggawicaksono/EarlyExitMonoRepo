"""Shared quality metrics for early-exit benchmarks."""

import numpy as np


def compute_ece(confidences, corrects, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE).

    When the model says X% confidence, does it actually get X% right?
    Critical for early-exit: shallow exits may be systematically over/under-confident.

    Args:
        confidences: 1-D float array, values in [0, 1].
        corrects:    1-D bool array, True if prediction was correct.
        n_bins:      Number of equal-width confidence bins.

    Returns:
        ECE scalar in [0, 1]. Lower = better calibrated.
    """
    confidences = np.asarray(confidences, dtype=float)
    corrects = np.asarray(corrects, dtype=bool)
    n = len(confidences)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        bin_acc = corrects[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += abs(bin_acc - bin_conf) * n_bin / n

    return float(ece)
