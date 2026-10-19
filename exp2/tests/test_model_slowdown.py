"""Tests for exp2.model.slowdown (slice E1a, DoD D8). Synthetic M2 truth, eta = 0.93, 3% multiplicative noise."""
import itertools
import json

import numpy as np
import pytest

from exp2 import contracts as C
from exp2.model import slowdown as S
from exp2.tests.mocks import make_types

ETA = 0.93
U = {"bert:5": 0.9, "vision:24": 0.8, "yolo:1": 0.6, "llama:3": 0.95}
LEVELS = (612, 1020)
ALL = ("M0", "M1", "M2", "M3", "M4lin", "M4")


@pytest.fixture(scope="module")
def types():
    return make_types({t: {"tau": 0.05, "u": u} for t, u in U.items()}, f_levels=LEVELS)


def _rows():
    rng = np.random.default_rng(1)
    rows, k = [], 0
    for n in (2, 3, 4):
        for A in itertools.combinations(U, n):
            split = "heldout" if k % 3 == 0 else "train"
            k += 1
            for i in A:
                for f in LEVELS:
                    for rep in range(3):
                        UA = sum(U[t] for t in A)
                        s = (1 - U[i]) + U[i] * max(1.0, UA / ETA)
                        rows.append({"cell_id": f"c{k}", "split": split, "f_mhz": str(f), "repeat": str(rep),
                                     "type_id": i, "family": i.split(":")[0], "A": "|".join(A),
                                     "s_p50": str(s * (1 + rng.uniform(-0.03, 0.03)))})
    return rows


@pytest.fixture(scope="module")
def rows():
    r = _rows()
    return [x for x in r if x["split"] == "train"], [x for x in r if x["split"] == "heldout"]


def test_m2_recovers_eta_and_beats_m0(rows, types):
    train, held = rows
    m2 = S.M2().fit(train, [], types)
    assert all(abs(e - ETA) / ETA < 0.05 for e in m2.eta.values())
    mapes = S.evaluate_heldout(train, held, [], types, names=("M0", "M2"))
    assert mapes["M2"] <= 5.0 and mapes["M2"] < mapes["M0"]


def test_heldout_row_raises(rows, types):
    train, held = rows
    for name in ALL:
        with pytest.raises(C.HeldOutError):
            S.make(name).fit(train + held[:1], [], types)


def test_all_models_json_round_trip_and_identity(rows, types):
    train, held = rows
    for name in ALL:
        m = S.make(name, seed=3).fit(train, [], types)
        for i, f in (("bert:5", 612), ("llama:3", 1020)):
            assert m.predict(i, (i,), f) == 1.0
        obj = json.loads(json.dumps(m.to_json()))
        C.validate_slowdown(obj)
        m2 = S.from_json(obj, types)
        A = ("bert:5", "vision:24", "yolo:1")
        assert m2.predict("vision:24", A, 1020) == m.predict("vision:24", A, 1020)


def test_m4_same_seed_identical(rows, types):
    train, _ = rows
    a, b = S.M4(seed=7).fit(train, [], types), S.M4(seed=7).fit(train, [], types)
    assert a.to_json() == b.to_json()


def test_select_model_order_and_fallback():
    ok = {"M0": 14.0, "M1": 11.0, "M2": 4.1, "M3": 4.0, "M4lin": 5.2, "M4": 4.8}
    assert S.select_model(ok) == ("M2", True)                       # first under 10% and below M0, not the lowest
    assert S.select_model({"M0": 5.0, "M1": 6.0, "M2": 7.0, "M3": 8.0, "M4lin": 9.0, "M4": 9.5}) == ("M0", False)
    assert S.select_model({"M0": 30.0, "M1": 25.0, "M2": 20.0, "M3": 12.0, "M4lin": 15.0, "M4": 18.0}) == ("M3", False)


def test_sample_pool_fallback_and_determinism(rows, types):
    m = S.M2().fit(rows[0], [], types)
    m.residuals = {"bert|2": [0.5, 2.0], "vision|3": [3.0]}
    A2, A3 = ("bert:5", "vision:24"), ("bert:5", "vision:24", "yolo:1")
    p2 = m.predict("bert:5", A2, 612)
    assert {m.sample("bert:5", A2, 612, np.random.default_rng(k)) / p2 for k in range(30)} == {0.5, 2.0}
    # no bert|3 pool: falls back to the family pool (bert|2 only)
    assert m.sample("bert:5", A3, 612, np.random.default_rng(0)) / m.predict("bert:5", A3, 612) in (0.5, 2.0)
    assert m.sample("bert:5", A2, 612, np.random.default_rng(4)) == m.sample("bert:5", A2, 612, np.random.default_rng(4))
    assert m.sample("bert:5", ("bert:5",), 612, np.random.default_rng(0)) == 1.0
