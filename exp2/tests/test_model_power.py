import pytest

from exp2.model.power import fit_power
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
