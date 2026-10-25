"""Tests for exp2.analyze.quality_sens: the floor-quality score Q(theta) of Run 3. Tiny synthetic trace rows only.

    Q(1)        = 1 - slo_violation_ratio of coexit_pilot.metrics on the same trace
    Q(0)        = share of measured requests served on time at the target exit
    floor       an on-time floor request scores theta
    break-even  linear interpolation of the mean difference; None without a sign change
    check_q1    a reps.csv violation ratio that differs by more than 1e-9 fails loudly
"""
import csv

import pytest

from exp2 import contracts as C
from exp2.analyze import quality_sens as QS
from exp2.model import coexit_pilot as P

DL = 100   # relative deadline, ns


def _row(i, fam, exit_, outcome="complete", late=False):
    """One trace row (run3.to_trace shape) of family fam that arrives at its target and runs at exit_."""
    t = i * 1000
    ran = exit_ is not None
    done = None if outcome == "expired_in_queue" else t + DL + (1 if late else 0)
    return {"req_id": "r%d" % i, "type_id": "%s:%d" % (fam, C.EXIT_MENU[fam][1]),
            "run_type_id": "%s:%d" % (fam, exit_) if ran else None, "family": fam, "outcome": outcome,
            "t_arrival_ns": t, "t_deadline_ns": t + DL, "t_done_ns": done}


def _trace():
    """1 warm-up row + 7 measured rows: 2 on time at the target, 2 on time at the floor, 3 failures."""
    return [_row(0, "bert", 23),                                # warm-up, excluded
            _row(1, "bert", 23), _row(2, "bert", 5),            # target, floor on time
            _row(3, "bert", 23, late=True),                     # complete but late
            _row(4, "yolo", 1), _row(5, "yolo", 5),             # floor, target on time
            _row(6, "bert", None, outcome="expired_in_queue"),  # never ran
            _row(7, "yolo", 5, outcome="truncated")]


def test_q1_is_one_minus_violation_ratio():
    c = QS.counts(_trace(), 1)
    viol = P.metrics(_trace(), 1, ["bert", "yolo"])["all"]["slo_violation_ratio"]
    assert c == (7, 2, 2)
    assert QS.q_score(c, 1.0) == pytest.approx(1.0 - viol, abs=1e-12)


def test_q0_is_on_time_target_share():
    c = QS.counts(_trace(), 1)
    assert QS.q_score(c, 0.0) == pytest.approx(2 / 7, abs=1e-12)
    assert QS.q_score(c, 0.5) == pytest.approx(3 / 7, abs=1e-12)


def test_floor_success_counts_theta():
    c = QS.counts([_row(0, "bert", 23), _row(1, "llama", 3)], 1)
    assert c == (1, 0, 1)
    for th in QS.THETAS:
        assert QS.q_score(c, th) == pytest.approx(th, abs=1e-12)


def test_break_even_interpolates_the_mean_difference():
    diffs = [0.3 - 0.8 * th for th in QS.THETAS]                 # zero at 0.375, between grid points 0.3 and 0.4
    assert QS.break_even(QS.THETAS, diffs) == pytest.approx(0.375, abs=1e-12)
    assert QS.break_even(QS.THETAS, [0.5 - 0.5 * th for th in QS.THETAS]) == pytest.approx(1.0)
    assert QS.break_even(QS.THETAS, [0.1 + th for th in QS.THETAS]) is None


def _reps_csv(path, viol):
    row = dict.fromkeys(C.COEXIT_PILOT_COLS, "")
    row.update(method="coexit", mix="A", arrival="poisson", cv_a="1.0", rho="0.9", deadline_scale="1.0", rep="0",
               seed="20261022", family="all", n="7", slo_violation_ratio=repr(viol), sim_runtime_s="1.0")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, C.COEXIT_PILOT_COLS, lineterminator="\n")
        wr.writeheader()
        wr.writerow(row)


def test_check_q1_fails_loudly(tmp_path):
    run = {"method": "coexit", "mix": "A", "arrival": "poisson", "cv_a": 1.0, "rho": 0.9, "rep": 0,
           "seed": 20261022, "condition_id": "A_p_rho090_coexit_r0", "n": 7, "n_target": 2, "n_floor": 2}
    _reps_csv(tmp_path / "reps.csv", 1.0 - 4 / 7)
    assert QS.check_q1([run], tmp_path / "reps.csv") <= 1e-9
    _reps_csv(tmp_path / "reps.csv", 1.0 - 4 / 7 + 1e-6)
    with pytest.raises(C.ContractError):
        QS.check_q1([run], tmp_path / "reps.csv")
