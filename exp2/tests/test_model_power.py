import pytest

from exp2.model.power import PowerModel, fit_power
from exp2.tests import mocks

TYPES = mocks.make_types()
P_IDLE = {612: 5.1, 1020: 5.6}
DP = {612: 4.2, 1020: 7.9}
ETA = {612: 0.93, 1020: 0.91}
U_A = {612: [0.5, 0.9, 1.4, 2.0], 1020: [0.4, 0.8, 1.3, 1.9]}


def _rows():
    idle = [{"f_mhz": f, "kind": "resident", "p_idle_w": P_IDLE[f]} for f in P_IDLE]
    corun = []
    for f, us in U_A.items():
        for i, u in enumerate(us):
            p = P_IDLE[f] + DP[f] * min(1.0, u / ETA[f])
            corun.append({"cell_id": "c%d" % i, "f_mhz": f, "repeat": 0, "U_A": u, "p_board_w": p})
    return idle, corun


def test_synthetic_truth_recovered_within_2pct():
    idle, corun = _rows()
    m = fit_power(idle, corun, ETA, TYPES)
    for f in P_IDLE:
        assert m.dp_w[f] == pytest.approx(DP[f], rel=0.02)
        assert m.p_idle_w[f] == pytest.approx(P_IDLE[f], rel=0.02)


def test_empty_set_is_p_idle_exactly():
    idle, corun = _rows()
    m = fit_power(idle, corun, ETA, TYPES)
    assert m.p((), 612) == m.p_idle_w[612] == 5.1
    assert m.to_json()["schema"] == "exp2.power/1"


B, V = "bert:5", "vision:24"
SETS = [(B, V), (B, B), (V, V), (B, B, V), (B, V, V)]
TRUE_ETA = {612: 0.9, 1020: 0.85}


def _full_rows(split="train", dp_scale=1.0, s_scale=1.0):
    idle = [{"f_mhz": f, "kind": "resident", "p_idle_w": P_IDLE[f]} for f in P_IDLE]
    corun = []
    for f in P_IDLE:
        for j, A in enumerate(SETS):
            U = sum(TYPES.get(t).u[f] for t in A)
            p = P_IDLE[f] + dp_scale * DP[f] * min(1.0, U / TRUE_ETA[f])
            for t in sorted(set(A)):
                u = TYPES.get(t).u[f]
                s = ((1 - u) + u * max(1.0, U / TRUE_ETA[f])) * s_scale
                corun.append({"cell_id": "c%d" % j, "f_mhz": f, "repeat": 0, "U_A": U, "p_board_w": p,
                              "type_id": t, "family": t.split(":")[0], "A": "|".join(A), "s_p50": s,
                              "split": split})
    return idle, corun


def test_heldout_rows_do_not_change_the_fit():
    idle, train = _full_rows()
    _, held = _full_rows(split="heldout", dp_scale=3.0)
    for r in held:
        r["cell_id"] = "h" + r["cell_id"]
    a = fit_power(idle, train, ETA, TYPES)
    b = fit_power(idle, train + held, ETA, TYPES)
    assert a.dp_w == b.dp_w


def test_eta_none_falls_back_to_m2_on_train_rows():
    idle, train = _full_rows()
    _, held = _full_rows(split="heldout", s_scale=2.0)
    for r in held:
        r["cell_id"] = "h" + r["cell_id"]
    m = fit_power(idle, train + held, None, TYPES)
    for f in P_IDLE:
        assert m.eta[f] == pytest.approx(TRUE_ETA[f], abs=0.005)
        assert m.dp_w[f] == pytest.approx(DP[f], rel=0.02)


def test_from_json_round_trip():
    idle, corun = _rows()
    m = fit_power(idle, corun, ETA, TYPES)
    m2 = PowerModel.from_json(m.to_json(), TYPES)
    assert m2.to_json() == m.to_json()
    assert m2.p((), 612) == m.p((), 612)
    A = ("bert:5", "vision:24")
    assert m2.p(A, 1020) == pytest.approx(m.p(A, 1020))


def test_from_json_rejects_wrong_schema():
    with pytest.raises(ValueError):
        PowerModel.from_json({"schema": "exp2.slowdown/1"}, TYPES)
