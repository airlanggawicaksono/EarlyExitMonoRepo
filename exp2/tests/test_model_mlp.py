"""Tests for exp2.model.mlp (slice E1a)."""
import numpy as np

from exp2.model.mlp import MLP


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = 1.0 + 0.8 / (1.0 + np.exp(-X[:, 0]))
    return X, y


def test_fit_is_deterministic_and_bounded():
    X, y = _data()
    a = MLP(3, seed=5, hi=2.0).fit(X, y, epochs=300)
    b = MLP(3, seed=5, hi=2.0).fit(X, y, epochs=300)
    pa = a.predict(X)
    assert np.array_equal(pa, b.predict(X))
    assert pa.min() >= 1.0 and pa.max() <= 2.0
    assert np.mean(np.abs(pa - y)) < np.mean(np.abs(y - y.mean()))   # learned something


def test_dict_round_trip():
    X, y = _data()
    a = MLP(3, seed=1, hi=2.0).fit(X, y, epochs=50)
    assert np.array_equal(a.predict(X), MLP.from_dict(a.to_dict()).predict(X))
