import numpy as np
import pytest

from exp2.model.freqscale import fit_tau, predict_tau

LEVELS = [408, 612, 816, 1020]
TRUTH = (900.0, 1.1, 0.012)


def test_noise_free_recovers_abc():
    taus = [predict_tau(TRUTH, f) for f in LEVELS]
    a, b, c = fit_tau(LEVELS, taus)
    for got, want in zip((a, b, c), TRUTH):
        assert got == pytest.approx(want, rel=1e-3)


def test_noise_heldout_level_prediction_within_2pct():
    rng = np.random.default_rng(3)
    taus = [predict_tau(TRUTH, f) * (1 + 0.01 * rng.standard_normal()) for f in LEVELS]
    params = fit_tau(LEVELS, taus)
    assert predict_tau(params, 714) == pytest.approx(predict_tau(TRUTH, 714), rel=0.02)
