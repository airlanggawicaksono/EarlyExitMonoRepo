"""Smoke test of exp2.analyze.explore on tiny synthetic input (no board data, no model)."""
import math

from exp2.analyze import explore as X

S = 1_000_000_000
TINFO = {"bert:5": {"phi": {3: 0.5, 5: 0.9}, "head": 0.1, "tau": {612: 2.0}},
         "yolo:1": {"phi": {0: 0.6, 1: 0.8}, "head": 0.2, "tau": {612: 1.0}}}


def _row(tid, t0, t1, outcome, ckpt=(), reached=None):
    return {"type_id": tid, "t_start_ns": t0 * S, "t_end_ns": t1 * S, "outcome": outcome, "ckpt": [list(c) for c in ckpt],
            "reached": reached}


def test_request_work_credit_and_waste():
    assert X.request_work(_row("bert:5", 0, 1, "complete"), TINFO, 612) == (2.0, 0.0)
    cr, wa = X.request_work(_row("bert:5", 0, 1, "truncated", [(3, 1), (5, 2)], 3), TINFO, 612)
    assert math.isclose(cr, 2.0 * (0.5 + 0.1)) and math.isclose(wa, 2.0 * (0.9 - 0.5))
    assert X.request_work(_row("yolo:1", 0, 1, "miss", [(0, 1)]), TINFO, 612) == (0.0, 1.0 * 0.6)
    assert X.request_work({"type_id": "yolo:1", "outcome": "expired_in_queue", "ckpt": []}, TINFO, 612) == (0.0, 0.0)


def test_set_occupancy_piecewise_constant():
    rows = [_row("bert:5", 0, 4, "complete"), _row("yolo:1", 2, 6, "complete")]
    occ, mx = X.set_occupancy(rows, 0, 10 * S)
    assert mx == 2
    assert math.isclose(occ[("bert:5",)], 2.0) and math.isclose(occ[("bert:5", "yolo:1")], 2.0)
    assert math.isclose(occ[("yolo:1",)], 2.0) and math.isclose(sum(occ.values()), 6.0)


def test_integrate_windows_streams_trapezoid(tmp_path):
    p = tmp_path / "power.csv"
    p.write_text("t_mono_ns,t_wall_ns,vdd_in_mv,vdd_in_ma,vdd_in_mw\n"
                 + "".join(f"{i * S},0,0,0,{(10 + i) * 1000}\n" for i in range(11)))
    (e1, c1), (e2, c2) = X.integrate_windows(p, [(1 * S, 3 * S), (4 * S + S // 2, 6 * S)])
    assert math.isclose(e1, 24.0, rel_tol=1e-9) and math.isclose(c1, 2.0)      # mean 12 W for 2 s = 24 J
    assert math.isclose(e2, 15.25 * 1.5, rel_tol=1e-9) and math.isclose(c2, 1.5)


def test_spearman_ties_and_sign():
    assert math.isclose(X.spearman([0, 0, 1, 2], [0, 1, 2, 3]), 0.9486832980505138, rel_tol=1e-9)
    assert X.spearman([1, 2, 3], [3, 2, 1]) == -1.0
    assert X.check_e8()["outcome"] == "not_computable"
