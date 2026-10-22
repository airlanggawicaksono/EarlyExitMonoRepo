"""S0 tests for the CoExit pilot contract (PLAN PL1 b1-b4, b7; exp2.contracts 'Exit choice').

Expected values are written by hand from the contract text, not by calling the code under test. The exit rules
themselves (exits.ExitPolicy.admit_exits) are S1 and are tested in test_exits.py.
"""
import json
from pathlib import Path

import pytest

from exp2 import contracts as C
from exp2 import exits as X
from exp2.contracts import ContractError, DispatchView, InServiceReq, PolicySpec, QueuedReq
from exp2.policies import AdmissionCore, Policy
from exp2.tests import mocks

FIX = Path(__file__).resolve().parent / "fixtures"
NS = 1_000_000_000
TYPES = {"bert:5": {"tau": 0.02, "klass": "short"}, "bert:23": {"tau": 0.08, "klass": "short"},
         "yolo:1": {"tau": 0.04, "klass": "short"}, "yolo:5": {"tau": 0.06, "klass": "short"},
         "llama:3": {"tau": 0.6, "klass": "long"}, "llama:15": {"tau": 1.6, "klass": "long"}}


@pytest.fixture(scope="module")
def T():
    return mocks.make_types(TYPES, f_levels=(612, 816, 1020))


def spec_json(**kw):
    d = {"name": "x", "k_short": None, "k_long": None, "order": "fcfs", "lane_cap": None, "f_rule": "static",
         "f_mhz": 1020, "params": {"exit_rule": "coexit"}}
    d.update(kw)
    return d


def Q(T, rid, tid, a_s, dl_s):
    t = T.get(tid)
    return QueuedReq(rid, tid, t.family, t.klass, int(a_s * NS), int(dl_s * NS))


class _Pairs:
    """An exit policy stub: admit_exits returns whatever fn(view) returns."""
    family_exclusive = True

    def __init__(self, fn):
        self.fn = fn

    def admit_exits(self, view):
        return self.fn(view)

    def admit(self, view):
        return [rid for rid, _ in self.fn(view)]

    def desired_f(self, view):
        return None


# ---------------------------------------------------------------- b1 policyspec
def test_pilot_specs_are_the_five_methods_and_valid():
    specs = X.pilot_specs(1020)
    assert tuple(specs) == X.METHODS == ("all_final", "all_early", "edgeserving", "greedy", "coexit")
    rules = {k: s.params["exit_rule"] for k, s in specs.items()}
    assert rules == {"all_final": "target", "all_early": "floor", "edgeserving": "edgeserving",
                     "greedy": "greedy", "coexit": "coexit"}
    es = specs["edgeserving"]
    assert (es.k_short, es.k_long, es.order, es.params["C"]) == (1, 1, "edgeserving", C.EDGESERVING_C)
    for name in ("all_final", "all_early", "greedy", "coexit"):
        assert (specs[name].k_short, specs[name].k_long, specs[name].order) == (C.K_INF, C.K_INF, "fcfs")
        assert specs[name].f_mhz == 1020 and specs[name].f_rule == "static"


@pytest.mark.parametrize("bad, needle", [
    (spec_json(params={"exit_rule": "x"}), "exit_rule"),
    (spec_json(params={"exit_rule": "edgeserving"}), "disagree"),
    (spec_json(order="edgeserving", k_short=1, k_long=1, params={"exit_rule": "coexit"}), "disagree"),
    (spec_json(params={"exit_rule": "coexit", "tau_stat": "p99"}), "tau_stat"),
    (spec_json(k_short=2), "concurrency limit"),
    (spec_json(lane_cap={"short": 1, "long": 1}), "concurrency limit"),
    (spec_json(f_rule="governor", f_mhz=None), "static"),
    (spec_json(params={"exit_rule": "coexit", "cap_margin": -0.1}), "cap_margin"),
    (spec_json(params={"exit_rule": "greedy", "cap_margin": 0.05}), "cap_margin"),
])
def test_validate_rejects_bad_exit_specs(bad, needle):
    with pytest.raises(ContractError, match=needle):
        C.validate_policyspec(bad)


def test_validate_accepts_exit_specs_and_leaves_old_specs_alone():
    C.validate_policyspec(spec_json())
    C.validate_policyspec(spec_json(params={"exit_rule": "greedy", "tau_stat": "p50"}))
    C.validate_policyspec(spec_json(order="edgeserving", k_short=1, k_long=1,
                                    params={"exit_rule": "edgeserving", "C": 10.0}))
    # without exit_rule the old rules apply unchanged: a plain corun spec with k = 2 is fine
    C.validate_policyspec(spec_json(k_short=2, params={}))


def test_exit_menu_and_rules():
    assert C.EXIT_MENU == {"bert": (5, 23), "yolo": (1, 5), "llama": (3, 15)}
    for fam, (lo, hi) in C.EXIT_MENU.items():
        rlo, rhi = C.EXIT_RANGE[fam]
        assert rlo <= lo < hi <= rhi
    assert C.EXIT_RULES == ("target", "floor", "edgeserving", "greedy", "coexit")


# ---------------------------------------------------------------- b3 InServiceReq
def test_in_service_round_trip_with_and_without_deadline():
    a = InServiceReq("r", "bert:5", "bert", "short", 7)
    assert a.t_deadline_ns is None
    assert a.to_json() == {"req_id": "r", "type_id": "bert:5", "family": "bert", "klass": "short", "t_admit_ns": 7}
    assert InServiceReq.from_json(a.to_json()) == a
    b = InServiceReq("r", "bert:5", "bert", "short", 7, 99)
    assert b.to_json()["t_deadline_ns"] == 99
    assert InServiceReq.from_json(b.to_json()) == b


# ---------------------------------------------------------------- b2 Decision / AdmissionCore
def test_core_records_run_types_and_puts_them_in_service(T):
    pol = _Pairs(lambda v: [(q.req_id, "bert:5" if q.family == "bert" else q.type_id) for q in v.queue])
    core = AdmissionCore(pol, T, 1020)
    d1 = core.arrive(Q(T, "a", "bert:23", 1.0, 1.25), int(1.0 * NS))
    assert d1.admitted == ("a",) and d1.run_type_ids == ("bert:5",)
    d2 = core.arrive(Q(T, "b", "yolo:5", 1.01, 1.2), int(1.01 * NS))
    assert d2.admitted == ("b",) and d2.run_type_ids == ("yolo:5",)
    ins = {r.req_id: r for r in core.view(int(1.02 * NS)).in_service}
    assert ins["a"].type_id == "bert:5" and ins["a"].t_deadline_ns == int(1.25 * NS)
    assert ins["b"].type_id == "yolo:5" and ins["b"].t_deadline_ns == int(1.2 * NS)


def test_core_rejects_a_run_type_of_another_family_and_stays_unchanged(T):
    core = AdmissionCore(_Pairs(lambda v: [(q.req_id, "yolo:1") for q in v.queue]), T, 1020)
    with pytest.raises(RuntimeError, match="another family"):
        core.arrive(Q(T, "a", "bert:23", 1.0, 1.25), int(1.0 * NS))
    view = core.view(int(1.0 * NS))
    assert view.queue == () and view.in_service == ()


def test_old_policy_decisions_have_no_run_types(T):
    spec = PolicySpec.from_json(spec_json(params={}))
    core = AdmissionCore(Policy(spec, T), T, 1020)
    d = core.arrive(Q(T, "a", "bert:23", 1.0, 1.25), int(1.0 * NS))
    assert d.admitted == ("a",) and d.run_type_ids == ()
    assert core.view(int(1.0 * NS)).in_service[0].type_id == "bert:23"


def test_wait_pairs_admit_nothing(T):
    core = AdmissionCore(_Pairs(lambda v: []), T, 1020)
    d = core.arrive(Q(T, "a", "bert:23", 1.0, 1.25), int(1.0 * NS))
    assert d.admitted == () and d.run_type_ids == ()
    assert [q.req_id for q in core.view(int(1.0 * NS)).queue] == ["a"]


# ---------------------------------------------------------------- ExitPolicy skeleton
def test_exit_policy_constructor_rules(T):
    specs = X.pilot_specs(1020)
    p = X.ExitPolicy(specs["all_final"], T)
    assert p.family_exclusive is True and p.desired_f(None) is None and p.rule == "target"
    assert X.ExitPolicy(specs["edgeserving"].to_json(), T).C == C.EDGESERVING_C
    with pytest.raises(ValueError, match="slowdown"):
        X.ExitPolicy(specs["coexit"], T)
    with pytest.raises(ContractError, match="exit_rule"):
        X.ExitPolicy(PolicySpec.from_json(spec_json(params={})), T)


# ---------------------------------------------------------------- b7 columns and fixtures
def test_pilot_columns():
    for cols in (C.COEXIT_PILOT_COLS, C.COEXIT_SUMMARY_COLS):
        assert len(cols) == len(set(cols))
    assert {"method", "mix", "rho", "rep", "family", "slo_violation_ratio", "mean_norm_depth",
            "share_floor"} <= set(C.COEXIT_PILOT_COLS)
    assert {"viol_mean", "depth_mean", "d_viol_vs_edgeserving_mean"} <= set(C.COEXIT_SUMMARY_COLS)


def test_exit_cases_fixture_is_well_formed(T):
    fx = json.loads((FIX / "exit_cases.json").read_text(encoding="utf-8"))
    assert set(fx["types"]) == set(TYPES)
    assert len(fx["cases"]) == 11
    for case in fx["cases"]:
        assert case["spec"] in X.METHODS
        view = DispatchView.from_json(case["view"])
        for q in view.queue:
            assert T.get(q.type_id).family == q.family
        for rid, run in case["expect"]:
            q = next(q for q in view.queue if q.req_id == rid)
            assert run in ("%s:%d" % (q.family, e) for e in C.EXIT_MENU[q.family])


def test_exit_trace_example_fixture_is_well_formed():
    fx = json.loads((FIX / "exit_trace_example.json").read_text(encoding="utf-8"))
    sim_keys = {"req_id", "type_id", "family", "klass", "t_arrival_ns", "t_deadline_ns", "t_admit_ns", "t_end_ns",
                "t_done_ns", "outcome", "reached", "completion", "f_mhz", "n_in_service_at_admit",
                "queue_len_at_arrival", "k_override", "ckpt"}
    for row in fx["rows"]:
        assert set(row) == sim_keys | {"run_type_id"}
        assert row["outcome"] in C.OUTCOMES
        assert (row["run_type_id"] is None) == (row["outcome"] == "expired_in_queue")
    assert set(fx["expected"]) == {"all", "bert", "yolo"}
