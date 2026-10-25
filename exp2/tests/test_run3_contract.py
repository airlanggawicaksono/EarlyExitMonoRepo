"""S0 tests for the Run 3 board contract (PLAN PL2 a, b; exp2.contracts 'Exit choice on the board').

Expected values are written by hand from the contract text. The dispatcher, plan builder and analysis are S1-S4.
"""
import copy
import json
from pathlib import Path

import pytest

from exp2 import arrivals
from exp2 import contracts as C
from exp2 import exits as X
from exp2.contracts import ContractError
from exp2.model import coexit_pilot as P
from exp2.tests import mocks

FIX = Path(__file__).resolve().parent / "fixtures"


def rows():
    return json.loads((FIX / "run3_rows.json").read_text(encoding="utf-8"))["rows"]


# ---------------------------------------------------------------- request rows
def test_run2_rows_without_req_type_id_still_validate():
    for p in (FIX / "session_open").glob("*/requests.jsonl"):
        for r in C.read_requests(p):
            assert "req_type_id" not in r


def test_run3_rows_validate_and_describe_the_run_type():
    rs = rows()
    assert len(rs) == 7
    for r in rs:
        C.validate_request_row(r)
    b2 = next(r for r in rs if r["req_id"] == "b2")
    assert (b2["type_id"], b2["exit"], b2["target"], b2["req_type_id"]) == ("bert:5", 5, 5, "bert:23")
    b3 = next(r for r in rs if r["req_id"] == "b3")
    assert b3["t_admit_ns"] is None and b3["type_id"] == b3["req_type_id"] == "bert:23"


@pytest.mark.parametrize("change, needle", [
    (lambda r: r.update(req_type_id="yolo:5"), "not of family"),
    (lambda r: r.pop("req_type_id"), "req_type_id is present exactly"),
    (lambda r: r.update(policy_spec=dict(r["policy_spec"], params={})), "req_type_id is present exactly"),
])
def test_bad_req_type_id_is_rejected(change, needle):
    r = copy.deepcopy(next(x for x in rows() if x["req_id"] == "b2"))
    change(r)
    with pytest.raises(ContractError, match=needle):
        C.validate_request_row(r)


def test_a_row_that_never_ran_keeps_its_arrival_type():
    r = copy.deepcopy(next(x for x in rows() if x["req_id"] == "b3"))
    r["req_type_id"] = "bert:5"
    with pytest.raises(ContractError, match="never ran"):
        C.validate_request_row(r)


# ---------------------------------------------------------------- deadlines
def test_with_family_deadlines_matches_the_pilot_schedule():
    T = mocks.make_types({"bert:5": {"tau": 0.02}, "bert:23": {"tau": 0.08}, "yolo:1": {"tau": 0.04},
                          "yolo:5": {"tau": 0.06}, "llama:3": {"tau": 0.6}, "llama:15": {"tau": 1.6}},
                         f_levels=(612, 816, 1020))
    wl = P._workload("B", "poisson", 1.0, 0.9, T)
    dl = {"bert": 0.1, "yolo": 0.1, "llama": 3.35}
    pilot = P._schedule(wl, 20, 7, dl, 1.0)
    plain = arrivals.schedule(wl, 20, 7)
    got = C.with_family_deadlines(plain, dl)
    assert [it["t_deadline_ns"] for it in got] == [it["t_deadline_ns"] for it in pilot]
    fam = C.parse_type_id(got[0]["type_id"])[0]
    assert got[0]["t_deadline_ns"] == got[0]["t_arrival_ns"] + int(round(dl[fam] * 1e9))


def _run3_plan(**cond):
    """One open Run 3 condition on the Run 2 plan fixture (freeze reference kept), re-hashed."""
    p = json.loads((FIX / "plan_run2.json").read_text(encoding="utf-8"))
    p["session"]["families"] = ["bert", "yolo"]
    p["session"]["exits"] = {"bert": [5, 23], "yolo": [1, 5]}
    c = next(dict(x) for x in p["conditions"] if x["mode"] == "open")
    c.update(condition_id="A_p_rho060_coexit_r0", policy_key="coexit", policy=X.pilot_specs(1020)["coexit"].to_json(),
             workload={"name": "mixA", "types": ["bert:23", "yolo:5"], "probs": [0.5, 0.5], "rho": 0.6,
                       "rate_per_s": 8.56, "arrival": "poisson", "cv_a": 1.0},
             alpha=None, deadlines_s={"bert": 0.1, "yolo": 0.1})
    c.update(cond)
    p["conditions"] = [c]
    p["sha256"] = C.plan_sha256(p)
    return p


def test_a_run3_plan_with_family_deadlines_validates():
    C.validate_plan(_run3_plan())


@pytest.mark.parametrize("cond, needle", [
    ({"deadlines_s": {"bert": 0.1}}, "workload families"),
    ({"deadlines_s": {"bert": 0.1, "yolo": 0.0}}, "deadlines_s.yolo"),
    ({"alpha": 5.0}, "alpha null"),
])
def test_bad_deadlines_are_rejected(cond, needle):
    with pytest.raises(ContractError, match=needle):
        C.validate_plan(_run3_plan(**cond))


def test_an_exit_policy_needs_both_menu_exits_loaded():
    p = _run3_plan()
    p["session"]["exits"]["bert"] = [23]
    p["sha256"] = C.plan_sha256(p)
    with pytest.raises(ContractError, match="must load the exits"):
        C.validate_plan(p)


# ---------------------------------------------------------------- columns
def test_run3_columns():
    for cols in (C.RUN3_RUNS_COLS, C.RUN3_COMPARE_COLS):
        assert len(cols) == len(set(cols))
    assert {"condition_id", "kept", "p99_dispatch_ms", "wall_s"} <= set(C.RUN3_RUNS_COLS)
    assert {"board_viol_mean", "sim_viol_mean", "same_sign"} <= set(C.RUN3_COMPARE_COLS)
    # pauses: limited preemption on the board (PLAN PL3, owner 2026-10-26), present only on rows of an LP spec
    assert C.REQUEST_OPTIONAL_KEYS == ("req_type_id", "pauses")
