"""S0 tests for limited preemption on the board (PLAN PL3 K1-K9; exp2.contracts "Limited preemption on the board").

Owner 2026-10-26: "lp saja mendingan dan coexit lp gak sih ?". Expected values are written by hand from the contract
text and the fixtures in fixtures/lp/ (the mocks the server, dispatcher and analysis slices build against).
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from exp2 import contracts as C
from exp2 import exits as X
from exp2.contracts import ContractError, DispatchView, InServiceReq, PolicySpec, QueuedReq
from exp2.policies import AdmissionCore, Decision, Policy
from exp2.tests import mocks

FIX = Path(__file__).resolve().parent / "fixtures"
LP = FIX / "lp"
NS = 1_000_000_000
TYPES = {"bert:5": {"tau": 0.02, "klass": "short"}, "bert:23": {"tau": 0.08, "klass": "short"},
         "yolo:1": {"tau": 0.04, "klass": "short"}, "yolo:5": {"tau": 0.06, "klass": "short"},
         "llama:3": {"tau": 0.6, "klass": "long"}, "llama:15": {"tau": 1.6, "klass": "long"}}


def load(name):
    return json.loads((LP / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def T():
    return mocks.make_types(TYPES, f_levels=(612, 816, 1020))


def Q(T, rid, tid, a_s, dl_s):
    t = T.get(tid)
    return QueuedReq(rid, tid, t.family, t.klass, int(a_s * NS), None if dl_s is None else int(dl_s * NS))


def S(T, rid, tid, admit_s, dl_s):
    t = T.get(tid)
    return InServiceReq(rid, tid, t.family, t.klass, int(admit_s * NS), None if dl_s is None else int(dl_s * NS))


def es_lp(T):
    return X.ExitPolicy(X.board_specs(1020)["edgeserving_lp"], T)


# ---------------------------------------------------------------- K1 policyspec, K8 registry
def test_lp_fixture_specs_are_the_board_specs():
    fx, board = load("policyspecs.json"), X.board_specs(1020)
    assert X.LP_METHODS == ("edgeserving_lp", "coexit_lp")
    for m in X.LP_METHODS:
        assert board[m].to_json() == fx[m]
        C.validate_policyspec(fx[m])
    assert fx["coexit_lp"]["params"]["cap_margin"] == C.COEXIT_CAP_MARGIN


def test_board_specs_add_the_lp_methods_and_leave_run3_and_sim_specs_alone():
    run3, sim, board = X.run3_specs(1020), X.sim_specs(1020), X.board_specs(1020)
    assert tuple(board) == X.METHODS + X.ABLATIONS + X.LP_METHODS
    assert all(board[m] == run3[m] for m in run3)
    assert tuple(run3) == X.METHODS + X.ABLATIONS
    assert tuple(sim) == X.METHODS + X.ABLATIONS + X.SIM_ONLY and "coexit_lp" not in sim
    assert board["edgeserving_lp"] == sim["edgeserving_lp"]
    want = dict(run3["coexit"].to_json(), name="coexit_lp")
    want["params"] = dict(want["params"], preempt_llama_at_token=True)
    assert board["coexit_lp"].to_json() == want


def spec_json(**params):
    return {"name": "x", "k_short": None, "k_long": None, "order": "fcfs", "lane_cap": None, "f_rule": "static",
            "f_mhz": 1020, "params": params}


@pytest.mark.parametrize("params", [
    {"exit_rule": "greedy", "preempt_llama_at_token": True},
    {"exit_rule": "target", "preempt_llama_at_token": True},
    {"exit_rule": "floor", "preempt_llama_at_token": True},
    {"preempt_llama_at_token": True},
    {"exit_rule": "coexit", "preempt_llama_at_token": False},
    {"exit_rule": "coexit", "preempt_llama_at_token": 1},
])
def test_preempt_needs_true_and_edgeserving_or_coexit(params):
    with pytest.raises(ContractError, match="preempt_llama_at_token"):
        C.validate_policyspec(spec_json(**params))


def test_coexit_with_preempt_is_a_valid_spec():
    C.validate_policyspec(spec_json(exit_rule="coexit", preempt_llama_at_token=True))
    C.validate_policyspec(spec_json(exit_rule="coexit", cap_margin=0.053, preempt_llama_at_token=True))


# ---------------------------------------------------------------- K2 wire, K3 done
def test_lp_wire_fixtures_validate():
    fx = load("wire.json")
    for key in ("pause", "resume", "paused", "done_paused"):
        assert C.validate_server_msg(fx[key]) is fx[key]
    assert fx["pause"] == {"op": "pause", "req_id": "r000001"} and fx["resume"] == {"op": "resume", "req_id": "r000001"}
    assert C.validate_paused(fx["paused"]) is fx["paused"]


def test_paused_tokens_done_range_is_1_to_target_minus_1():
    m = load("wire.json")["paused"]
    C.validate_paused(dict(m, tokens_done=1))
    C.validate_paused(dict(m, tokens_done=31))
    C.validate_paused(dict(m, type_id="llama:3", tokens_done=31))    # llama target is 32 tokens for every exit
    for bad in (0, 32, None, "3"):
        with pytest.raises(ContractError, match="tokens_done"):
            C.validate_paused(dict(m, tokens_done=bad))


def test_done_without_pauses_is_unchanged_and_pauses_need_order():
    fx = load("wire.json")["done_paused"]
    plain = {k: v for k, v in fx.items() if k != "pauses"}
    C.validate_done(plain)
    p, r = fx["pauses"][0][1:]
    tok = dict(fx["ckpt"])
    two = [[3, p, r], [10, tok[10] + 1, tok[10] + 2]]
    C.validate_done(dict(fx, pauses=two))
    for bad in ([[10, tok[10] + 1, tok[10] + 2], [3, p, r]],                  # tokens not increasing
                [[3, p, r], [3, r + 1, r + 2]],                                # same token twice
                [[3, p, tok[10] + 5], [10, tok[10] + 1, tok[10] + 6]]):        # next pause before the resume
        with pytest.raises(ContractError, match="strictly increasing"):
            C.validate_done(dict(fx, pauses=bad))
    with pytest.raises(ContractError, match="entry must be"):
        C.validate_done(dict(fx, pauses=[[3, p]]))


def test_done_of_a_miss_has_no_pauses():
    m = dict(load("wire.json")["done_paused"], outcome="miss", reached=None, completion=0.0, tokens_done=None)
    m["ckpt"] = m["ckpt"][:4]
    with pytest.raises(ContractError, match="miss has no pauses"):
        C.validate_done(m)


def test_server_msg_rejects_bad_pause_and_resume():
    for bad in ({"op": "pause"}, {"op": "resume", "req_id": ""}, {"op": "resume", "req_id": "r", "x": 1},
                {"op": "paused", "req_id": "r"}):
        with pytest.raises(ContractError):
            C.validate_server_msg(bad)


# ---------------------------------------------------------------- K4 request rows
def rows():
    return load("rows.json")["rows"]


def test_lp_rows_validate_and_carry_pauses():
    r1, r2, r3 = rows()
    for r in (r1, r2, r3):
        assert tuple(r) == C.REQUEST_KEYS + C.REQUEST_OPTIONAL_KEYS
        C.validate_request_row(r)
    assert r1["pauses"] == [[3, r1["pauses"][0][1], r1["pauses"][0][2], r1["pauses"][0][3]]]
    assert r2["pauses"] == [] and r2["n_in_service_at_admit"] == 1         # admitted while r000001 was paused
    assert r3["outcome"] == "error" and r3["pauses"][-1][3] is None
    done = load("wire.json")["done_paused"]
    assert [[k, tp, tr] for k, _, tp, tr in r1["pauses"]] == done["pauses"] and r1["ckpt"] == done["ckpt"]


def test_lp_row_time_rules_hold_in_the_fixture():
    r1, r2, _ = rows()
    tok, t_req, t_pause, t_resume = r1["pauses"][0]
    ck = dict(r1["ckpt"])
    assert t_req == r2["t_arrival_ns"] + 100_000 and ck[tok] <= t_pause < r2["t_admit_ns"]
    assert r2["t_done_ns"] < t_resume < ck[tok + 1]
    assert ck[tok + 1] - t_resume - (ck[tok + 2] - ck[tok + 1]) == 5_000_000      # first-token penalty 5 ms


@pytest.mark.parametrize("case", load("invalid.json")["cases"], ids=lambda c: c["name"])
def test_invalid_lp_fixtures_are_rejected(case):
    base = load("wire.json")[case["base"]] if case["kind"] == "wire" else rows()[case["base"]]
    obj = copy.deepcopy(base)
    obj.update(copy.deepcopy(case.get("set", {})))
    for k in case.get("drop", ()):
        del obj[k]
    check = C.validate_server_msg if case["kind"] == "wire" else C.validate_request_row
    with pytest.raises(ContractError, match=case["needle"]):
        check(obj)


def test_non_lp_rows_never_get_a_pauses_key():
    run3 = json.loads((FIX / "run3_rows.json").read_text(encoding="utf-8"))["rows"]
    for r in run3:
        assert "pauses" not in r
        C.validate_request_row(r)
        with pytest.raises(ContractError, match="pauses is present exactly"):
            C.validate_request_row(dict(r, pauses=[]))
    for p in (FIX / "session_open").glob("*/requests.jsonl"):
        for r in C.read_requests(p):
            assert "pauses" not in r


def test_a_row_that_never_ran_or_missed_has_no_pauses():
    r3 = rows()[2]
    with pytest.raises(ContractError, match="never ran or missed"):
        C.validate_request_row(dict(r3, t_admit_ns=None))
    r1 = rows()[0]
    miss = dict(r1, outcome="miss", reached=None, completion=0.0, tokens_done=None, ckpt=r1["ckpt"][:4])
    with pytest.raises(ContractError, match="never ran or missed"):
        C.validate_request_row(miss)


def test_only_the_last_entry_of_an_error_row_may_stay_open():
    r3 = rows()[2]
    _, t_req, t_pause, _ = r3["pauses"][0]
    two = [[1, t_req - 10, t_req - 5, None], [2, t_req, t_pause, None]]
    with pytest.raises(ContractError, match="t_resume_ns"):
        C.validate_request_row(dict(r3, pauses=two))
    C.validate_request_row(dict(r3, pauses=[[1, t_req - 10, t_req - 5, t_req - 1], [2, t_req, t_pause, None]]))


# ---------------------------------------------------------------- K6 lp.csv columns
def test_run3_lp_columns():
    cols = C.RUN3_LP_COLS
    assert len(cols) == len(set(cols)) == 25
    assert cols[:9] == C.RUN3_RUNS_COLS[:9] == ("session_id", "condition_id", "method", "mix", "arrival", "cv_a",
                                                "rho", "rep", "seed")
    assert cols[9:] == ("n_llama", "n_paused_requests", "n_pauses", "n_empty_pauses", "paused_ms_p50",
                        "paused_ms_max", "boundary_wait_ms_p50", "boundary_wait_ms_p95", "boundary_wait_ms_max",
                        "handoff_ms_p50", "handoff_ms_p95", "resume_gap_ms_p50", "resume_gap_ms_p95",
                        "first_token_penalty_ms_p50", "first_token_penalty_ms_p95", "n_overlap")


# ---------------------------------------------------------------- K7 Decision.pause_ids, AdmissionCore.tick
def test_decision_pause_ids_default_is_empty():
    assert Decision().pause_ids == ()
    assert Decision(("a",), (), None, (0,), ()).pause_ids == ()


class _Stub:
    """Exit policy stub: admits every queued request at its own type; pause_ids records the view it saw."""
    family_exclusive = True

    def __init__(self, preempt, out=("L",)):
        self.preempt, self.out, self.seen = preempt, out, []

    def admit_exits(self, view):
        return [(q.req_id, q.type_id) for q in view.queue if q.family == "llama"]

    def admit(self, view):
        return [rid for rid, _ in self.admit_exits(view)]

    def desired_f(self, view):
        return None

    def pause_ids(self, view):
        if not self.preempt:
            raise AssertionError("pause_ids called on a policy that does not preempt")
        self.seen.append(view)
        return list(self.out)


def test_pause_ids_read_the_view_after_the_admits(T):
    pol = _Stub(True)
    core = AdmissionCore(pol, T, 1020)
    core.arrive(Q(T, "s", "bert:23", 0.0, 5.0), 0)
    d = core.arrive(Q(T, "L", "llama:15", 0.1, 10.0), int(0.1 * NS))
    assert d.admitted == ("L",) and d.pause_ids == ("L",) and isinstance(d.pause_ids, tuple)
    view = pol.seen[-1]
    assert [r.req_id for r in view.in_service] == ["L"] and [q.req_id for q in view.queue] == ["s"]


def test_pause_ids_are_not_asked_without_preempt(T):
    core = AdmissionCore(_Stub(False), T, 1020)
    d = core.arrive(Q(T, "L", "llama:15", 0.0, 10.0), 0)
    assert d.admitted == ("L",) and d.pause_ids == ()
    plain = AdmissionCore(Policy(PolicySpec.from_json(spec_json()), T), T, 1020)
    assert plain.arrive(Q(T, "a", "bert:23", 0.0, 1.0), 0).pause_ids == ()


def test_tick_drops_expired_requests_and_decides_again(T):
    admit_now = {"on": False}
    pol = _Stub(False)
    pol.admit_exits = lambda v: [(q.req_id, q.type_id) for q in v.queue] if admit_now["on"] else []
    core = AdmissionCore(pol, T, 1020)
    core.arrive(Q(T, "a", "bert:23", 0.0, 0.1), 0)
    core.arrive(Q(T, "b", "yolo:5", 0.0, 0.5), 0)
    admit_now["on"] = True
    d = core.tick(int(0.2 * NS))
    assert d.expired == ("a",) and d.admitted == ("b",) and d.n_in_service_at_admit == (0,)
    assert core.tick(int(0.3 * NS)) == Decision()
    with pytest.raises(ValueError, match="duplicate"):
        core.arrive(Q(T, "b", "yolo:5", 0.3, 0.9), int(0.3 * NS))
    core.arrive(Q(T, "a", "bert:23", 0.3, 0.9), int(0.3 * NS))     # an expired id is free again


# ---------------------------------------------------------------- K8 ExitPolicy LP state and edgeserving_lp
def test_exit_policy_lp_state_and_coexit_lp_builds(T):
    p = es_lp(T)
    assert p.preempt is True and p.paused == {} and p.pause_pending == set() and p.paused_ns == {}
    plain = X.ExitPolicy(X.board_specs(1020)["edgeserving"], T)
    assert plain.preempt is False and plain.pause_ids(DispatchView(0, (Q(T, "s", "bert:23", 0, 1),),
                                                                  (S(T, "L", "llama:15", 0, 9),), 1020)) == ()
    co = X.ExitPolicy(X.board_specs(1020)["coexit_lp"], T, slowdown=object())
    assert co.preempt is True and co.rule == "coexit" and co.pause_ids(DispatchView(0, (), (), 1020)) == ()


NOW = 0.30


def _views(T):
    L = S(T, "L", "llama:15", 0.0, 10.0)
    s = Q(T, "s", "bert:23", 0.25, 5.25)
    return [   # (name, in_service, queue, paused, pending, expected pause_ids)
        ("short_waits", (L,), (s,), (), (), ("L",)),
        ("short_expired", (L,), (Q(T, "s", "bert:23", 0.2, NOW),), (), (), ()),
        ("short_without_deadline", (L,), (Q(T, "s", "bert:23", 0.25, None),), (), (), ("L",)),
        ("expired_and_live_short", (L,), (Q(T, "s", "bert:23", 0.2, 0.29), Q(T, "y", "yolo:5", 0.26, 0.4)), (),
         (), ("L",)),
        ("only_llama_queued", (L,), (Q(T, "q", "llama:3", 0.26, 9.0),), (), (), ()),
        ("empty_queue", (L,), (), (), (), ()),
        ("llama_without_deadline", (S(T, "L", "llama:15", 0.0, None),), (s,), (), (), ()),
        ("llama_paused", (L,), (s,), ("L",), (), ()),
        ("llama_pending", (L,), (s,), (), ("L",), ()),
        ("no_llama_in_service", (S(T, "b", "bert:5", 0.29, 0.4),), (s,), (), (), ()),
        ("nothing_in_service", (), (s,), (), (), ()),
        ("paused_llama_and_a_short_running", (L, S(T, "b", "bert:5", 0.29, 0.4)), (s,), ("L",), (), ()),
    ]


@pytest.mark.parametrize("i", range(12))
def test_edgeserving_lp_pause_ids_on_hand_made_views(T, i):
    from exp2.model import simulator
    name, ins, queue, paused, pending, want = _views(T)[i]
    pol = es_lp(T)
    pol.paused.update({rid: 3 for rid in paused})
    pol.pause_pending.update(pending)
    view = DispatchView(int(NOW * NS), queue, ins, 1020)
    assert pol.pause_ids(view) == want, name
    # the queue part of the trigger is the simulator's own (_short_waiting) on the same queue
    fake = SimpleNamespace(now=view.now_ns, core=SimpleNamespace(view=lambda now: view))
    if want:
        assert simulator._Rep._short_waiting(fake) is True
    elif ins and ins[0].req_id == "L" and ins[0].t_deadline_ns is not None and not paused and not pending:
        assert simulator._Rep._short_waiting(fake) is False, name


def test_edgeserving_lp_flow_through_the_core(T):
    """Plan 1.1: run L, a short arrives -> pause L; paused -> tick admits the short alone; done -> resume."""
    pol = es_lp(T)
    core = AdmissionCore(pol, T, 1020)
    d = core.arrive(Q(T, "L", "llama:15", 0.0, 10.0), 0)
    assert (d.admitted, d.run_type_ids, d.pause_ids) == (("L",), ("llama:15",), ())
    d = core.arrive(Q(T, "s", "bert:23", 0.25, 5.25), int(0.25 * NS))
    assert (d.admitted, d.pause_ids) == ((), ("L",))
    pol.pause_pending.add("L")                                          # dispatcher sent pause
    assert core.tick(int(0.26 * NS)).pause_ids == () and core.tick(int(0.26 * NS)).admitted == ()
    pol.pause_pending.discard("L")
    pol.paused["L"] = 3                                                 # server: paused after token 3
    d = core.tick(int(0.30 * NS))
    assert (d.admitted, d.run_type_ids, d.n_in_service_at_admit, d.pause_ids) == (("s",), ("bert:23",), (1,), ())
    d = core.done("s", int(0.38 * NS))
    assert (d.admitted, d.pause_ids) == ((), ())
    del pol.paused["L"]                                                 # dispatcher sent resume
    pol.paused_ns["L"] = int(0.08 * NS)
    d = core.arrive(Q(T, "s2", "bert:23", 1.0, 1.1), int(1.0 * NS))
    assert (d.admitted, d.pause_ids) == ((), ("L",))                   # a second pause of the same request


# ---------------------------------------------------------------- K9 simulator guard
def test_the_simulator_refuses_coexit_lp(T):
    from exp2.model import simulator
    spec = X.board_specs(1020)["coexit_lp"]
    with pytest.raises(ValueError, match="coexit_lp board only"):
        simulator.simulate({"rate_per_s": 1.0}, spec, T, None, None, policy=object())
