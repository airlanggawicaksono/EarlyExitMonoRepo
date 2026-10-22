"""S1 tests for exp2.exits.ExitPolicy.admit_exits (exp2.contracts 'Exit choice', PLAN PL1 b5, d, e).

Every expected value is computed by hand in the comment next to it (s = n / c(n) from mocks.StubSlowdown), never by
calling the code under test. Times in the comments are seconds; views are built in integer ns.
"""
import random

import pytest

from exp2 import contracts as C
from exp2 import exits as X
from exp2.contracts import ContractError, DispatchView, InServiceReq, PolicySpec, QueuedReq
from exp2.policies import Policy
from exp2.tests import mocks

NS = 1_000_000_000
FX = mocks.load_fixture("exit_cases.json")
LEVELS = (612, 816, 1020)
TARGETS = tuple("%s:%d" % (fam, ex[1]) for fam, ex in C.EXIT_MENU.items())     # bert:23, yolo:5, llama:15
FLOORS = tuple("%s:%d" % (fam, ex[0]) for fam, ex in C.EXIT_MENU.items())      # bert:5, yolo:1, llama:3


@pytest.fixture(scope="module")
def T():
    return mocks.make_types(FX["types"], f_levels=LEVELS)


def stub(c=None):
    c = FX["slowdown"]["stub_c"] if c is None else c
    return mocks.StubSlowdown({int(k): v for k, v in c.items()})


def policy(T, method, slowdown=None, **params):
    spec = X.pilot_specs(FX["f_mhz"])[method]
    if params:
        spec = PolicySpec.from_json(dict(spec.to_json(), params=dict(spec.params, **params)))
    return X.ExitPolicy(spec, T, stub() if slowdown is None else slowdown)


def ns(x):
    return None if x is None else int(round(x * NS))


def Q(T, rid, tid, a_s, dl_s):
    t = T.get(tid)
    return QueuedReq(rid, tid, t.family, t.klass, ns(a_s), ns(dl_s))


def S(T, rid, tid, admit_s, dl_s):
    t = T.get(tid)
    return InServiceReq(rid, tid, t.family, t.klass, ns(admit_s), ns(dl_s))


def view(now_s, queue=(), in_service=(), f=1020):
    return DispatchView(ns(now_s), tuple(queue), tuple(in_service), f)


def pairs(pol, v):
    return [list(p) for p in pol.admit_exits(v)]


# ---------------------------------------------------------------- (1) the frozen fixture cases
@pytest.mark.parametrize("case", FX["cases"], ids=[c["name"] for c in FX["cases"]])
def test_fixture_case_matches_exactly(T, case):
    pol = X.ExitPolicy(X.pilot_specs(FX["f_mhz"])[case["spec"]], T, stub(case.get("stub_c")))
    v = DispatchView.from_json(case["view"])
    assert pairs(pol, v) == case["expect"], case["why"]
    assert pol.admit(v) == [rid for rid, _ in case["expect"]]


def test_fixture_has_the_eleven_cases():
    assert len(FX["cases"]) == 11


# ---------------------------------------------------------------- (2) seeded property: all_final == corun
def _random_view(rng, T):
    now = rng.randint(2 * NS, 20 * NS)
    in_service = []
    for i, fam in enumerate(rng.sample(sorted(C.EXIT_MENU), rng.randint(0, 2))):
        tid = rng.choice([t for t in FX["types"] if t.startswith(fam + ":")])      # any exit may be running
        ts = T.get(tid)
        dl = rng.choice([None, now + rng.randint(1, 2 * NS)])
        in_service.append(InServiceReq("s%d" % i, tid, fam, ts.klass, now - rng.randint(0, 2 * NS), dl))
    queue, a = [], now - rng.randint(0, 2 * NS)
    for i in range(rng.randint(0, 6)):
        tid = rng.choice(TARGETS)                       # the pilot queues every request at its target type
        ts = T.get(tid)
        a = min(now, a + rng.randint(0, NS // 4))
        dl = rng.choice([None, a + rng.randint(1, 3 * NS)])
        queue.append(QueuedReq("q%d" % i, tid, ts.family, ts.klass, a, dl))
    return DispatchView(now, tuple(queue), tuple(in_service), rng.choice([None, 612, 816, 1020]))


def test_all_final_equals_corun_and_all_early_runs_only_floors_on_200_random_views(T):
    corun = Policy(PolicySpec.from_json({"name": "corun", "k_short": None, "k_long": None, "order": "fcfs",
                                         "lane_cap": None, "f_rule": "static", "f_mhz": 1020, "params": {}}), T)
    final, early = policy(T, "all_final"), policy(T, "all_early")
    rng = random.Random(20261022)
    n_admit = n_skip = 0
    for _ in range(200):
        v = _random_view(rng, T)
        want = corun.admit(v)
        got_final, got_early = final.admit_exits(v), early.admit_exits(v)
        assert final.admit(v) == want and [rid for rid, _ in got_final] == want, v
        assert early.admit(v) == want and [rid for rid, _ in got_early] == want, v
        queued = {q.req_id: q for q in v.queue}
        for rid, run in got_final:
            assert run == queued[rid].type_id
        for rid, run in got_early:
            assert run in FLOORS and run == "%s:%d" % (queued[rid].family, C.EXIT_MENU[queued[rid].family][0])
        n_admit += len(want)
        n_skip += len(v.queue) - len(want)
    assert n_admit >= 100 and n_skip >= 50          # the draw exercises both admits and busy-family skips


# ---------------------------------------------------------------- (3) edge cases
@pytest.mark.parametrize("method", X.METHODS)
def test_a_family_outside_the_exit_menu_is_a_contract_error(method):
    T2 = mocks.make_types(dict(FX["types"], **{"vision:24": {"tau": 0.05, "klass": "short"}}), f_levels=LEVELS)
    v = view(1.0, [Q(T2, "v1", "vision:24", 0.9, 2.0)])
    with pytest.raises(ContractError, match="EXIT_MENU"):
        policy(T2, method).admit_exits(v)


def test_greedy_without_deadline_runs_the_target(T):
    # G3 without the deadline: 1 + 1.6 x 2 = 4.2 is never late -> llama:15 (with dl 4.0 it is llama:3).
    v = view(1.0, [Q(T, "r", "llama:15", 1.0, None)], [S(T, "Y", "yolo:5", 0.97, 1.04)])
    assert pairs(policy(T, "greedy"), v) == [["r", "llama:15"]]


def test_coexit_in_service_job_without_deadline_is_never_late(T):
    # C1 with L's deadline removed. target: r1 done 1.16 <= 1.25, L never late -> L=0. floor: L=0.
    # wait: L alone done 2.2, r1 at floor 2.22 > 1.25 -> L=1. min 0, first option -> target.
    # cap_margin 0: the rollout itself is under test (with s = n the capacity is 1.0, which the gate would stop).
    v = view(1.0, [Q(T, "r1", "bert:23", 1.0, 1.25)], [S(T, "L", "llama:15", 0.6, None)])
    assert pairs(policy(T, "coexit", cap_margin=0.0), v) == [["r1", "bert:23"]]


def test_coexit_offers_no_wait_when_nothing_is_in_service(T):
    # r1 bert (a 0.9, dl 1.01), q2 yolo (a 1.0, dl 1.05); nothing in service, s = n.
    # r1 target: q2 starts at yolo:1 too; q2 done 1.08, r1 done 1.12 > 1.01 -> target not offered.
    # r1 floor: r1 done 1.04, q2 w 0.02 left -> 1.06 > 1.05: L=2. WAIT would give q2 1.04, r1 1.06: L=1 (it would win),
    # but it is not offered -> r1 at bert:5.
    # q2 (r1 admitted, w 0.02): target q2 1.08 late -> not offered; floor L=2; WAIT r1 1.02, q2 1.06: L=2 -> floor.
    # cap_margin 0 keeps the rollout under test; with the pilot margin q2 would wait at capacity 1.0.
    v = view(1.0, [Q(T, "r1", "bert:23", 0.9, 1.01), Q(T, "q2", "yolo:5", 1.0, 1.05)])
    assert pairs(policy(T, "coexit", cap_margin=0.0), v) == [["r1", "bert:5"], ["q2", "yolo:1"]]
    assert pairs(policy(T, "coexit"), v) == [["r1", "bert:5"]]


def test_coexit_wait_counts_jobs_admitted_earlier_in_the_call(T):
    # Contract: "WAIT if >= 1 job in service"; a job admitted earlier in this call is in service (as for greedy's A).
    # r1 llama (a 0.99, dl 1.61), q2 yolo (a 1.0, dl 3.0), nothing in service, s = n.
    # r1 target: q2 yolo:1 co-runs, done 1.08; r1 1.56 left -> 2.64 > 1.61, not offered.
    # r1 floor: q2 1.08; r1 0.56 left -> 1.64 (late), L=1. No WAIT -> r1 at llama:3.
    # q2 (r1 in service, w 0.6): target q2 1.12, r1 0.54 left -> 1.66 late: L=1. floor q2 1.08, r1 1.64: L=1.
    # WAIT: r1 alone 1.6 <= 1.61, q2 starts at floor 1.6 -> 1.64 <= 3.0: L=0 -> q2 waits.
    v = view(1.0, [Q(T, "r1", "llama:15", 0.99, 1.61), Q(T, "q2", "yolo:5", 1.0, 3.0)])
    assert pairs(policy(T, "coexit", cap_margin=0.0), v) == [["r1", "llama:3"]]


@pytest.mark.parametrize("c2, expect", [
    # capacity c(2) = 1.04 < 1 + 0.053 -> r1 waits (serial), whatever the rollout says.
    (1.04, []),
    # capacity 1.06 >= 1.053 -> no gate; s = 2/1.06 = 1.8868 for both. w_L = 1.6 - 0.4 = 1.2.
    # target: r1 done 1.0 + 0.08 x 1.8868 = 1.151 <= 1.25; L did 0.08 -> 1.12 alone -> 2.271 > 2.25: L=1.
    # floor: r1 done 1.0 + 0.02 x 1.8868 = 1.038; L did 0.02 -> 1.18 alone -> 2.218 <= 2.25: L=0.
    # wait: L alone 2.2, r1 at floor 2.22 > 1.25: L=1. min L = 0 -> floor.
    (1.06, [["r1", "bert:5"]]),
])
def test_coexit_capacity_gate(T, c2, expect):
    v = view(1.0, [Q(T, "r1", "bert:23", 1.0, 1.25)], [S(T, "L", "llama:15", 0.6, 2.25)])
    assert pairs(policy(T, "coexit", stub({"1": 1.0, "2": c2})), v) == expect


def test_pilot_coexit_uses_the_frozen_margin():
    assert X.pilot_specs(1020)["coexit"].params["cap_margin"] == C.COEXIT_CAP_MARGIN == 0.053


def test_coexit_wait_keeps_the_turn_of_its_family(T):
    # c = {1: 1.0, 2: 1.6}: two jobs run at s = 2 / 1.6 = 1.25. Y yolo:5 in service (admit 0.97, dl 1.035), w_Y = 0.03.
    # Queue r llama (a 0.99, dl 4.0), r2 llama (a 1.0, dl 1.62).
    # r target: Y done 0.03 x 1.25 -> 1.0375 > 1.035 late; r 1.57 left -> 2.6075 on time; r2 -> 3.2075 late: L=2.
    # r floor: Y late; r 1.6075; r2 -> 2.2075 late: L=2.
    # r WAIT: Y alone 1.03 on time; r at floor 1.03 -> 1.63; r2 -> 2.23 late: L=1 -> r waits.
    # r2 is behind r in the llama queue, so it is not a candidate in this call. (Were it one: r2 floor gives Y late,
    # r2 1.6075 <= 1.62, r 2.2075: L=1; WAIT gives r2 1.63 late: L=1; tie -> r2 would be admitted at llama:3.)
    v = view(1.0, [Q(T, "r", "llama:15", 0.99, 4.0), Q(T, "r2", "llama:15", 1.0, 1.62)],
             [S(T, "Y", "yolo:5", 0.97, 1.035)])
    assert pairs(policy(T, "coexit", stub({"1": 1.0, "2": 1.6})), v) == []


def test_all_final_runs_the_menu_target_even_for_a_queued_floor_type(T):
    # "target: run the target type", target of family m = m:EXIT_MENU[m][1], whatever type the request queued as.
    v = view(1.0, [Q(T, "r1", "bert:5", 0.99, 2.0), Q(T, "r2", "yolo:1", 0.995, 2.0)])
    assert pairs(policy(T, "all_final"), v) == [["r1", "bert:23"], ["r2", "yolo:5"]]
    assert pairs(policy(T, "all_early"), v) == [["r1", "bert:5"], ["r2", "yolo:1"]]


def test_edgeserving_admits_nothing_while_a_job_is_in_service(T):
    v = view(1.0, [Q(T, "r1", "bert:23", 0.95, 1.10)], [S(T, "Y", "yolo:1", 0.99, 1.2)])
    assert pairs(policy(T, "edgeserving"), v) == []


def test_edgeserving_needs_a_deadline_on_every_queued_request(T):
    v = view(1.0, [Q(T, "r1", "bert:23", 0.95, 1.10), Q(T, "r2", "yolo:5", 0.96, None)])
    with pytest.raises(ContractError, match="deadline"):
        policy(T, "edgeserving").admit_exits(v)


def test_edgeserving_admits_at_most_one_and_breaks_ties_by_arrival(T):
    # Two bert-free families with mirrored numbers. r1 bert (a 0.90, dl 1.10), r2 yolo (a 0.92, dl 1.12), now 1.0.
    # e*(r1) = bert:23 (1.08 <= 1.10). e*(r2) = yolo:5 (1.06 <= 1.12).
    # S(r1) = exp((1.0 - 0.92 + 0.08) / 0.20 - 1) = exp(-0.2); S(r2) = exp((1.0 - 0.90 + 0.06) / 0.20 - 1) = exp(-0.2).
    # Tie -> earliest arrival -> r1 at bert:23, and only one admit.
    v = view(1.0, [Q(T, "r1", "bert:23", 0.90, 1.10), Q(T, "r2", "yolo:5", 0.92, 1.12)])
    assert pairs(policy(T, "edgeserving"), v) == [["r1", "bert:23"]]


def test_edgeserving_clips_each_term_at_C(T):
    # r1 llama (a 0.0, dl 10.0) waits 1.0; r2 bert (a 0.9995, dl 1.0005) has D = 0.001.
    # e*(r1) = llama:15 (2.6 <= 10). S(r1) = min(exp((0.0005 + 1.6) / 0.001 - 1), 10) = min(exp(1599.5), 10) = 10:
    # the clip must not overflow. e*(r2): 1.08 and 1.02 both > 1.0005 -> floor bert:5.
    # S(r2) = exp((1.0 + 0.02) / 10 - 1) = exp(-0.898) = 0.407 -> r2 at bert:5.
    v = view(1.0, [Q(T, "r1", "llama:15", 0.0, 10.0), Q(T, "r2", "bert:23", 0.9995, 1.0005)])
    assert pairs(policy(T, "edgeserving"), v) == [["r2", "bert:5"]]
    # With C = 0.1 both scores clip to 0.1 -> tie -> earliest arrival r1 at llama:15.
    assert pairs(policy(T, "edgeserving", C=0.1), v) == [["r1", "llama:15"]]


def test_greedy_reads_tau_at_the_commanded_level_or_f_max():
    # bert:23 tau 0.2 s at 612 MHz, 0.08 s at 1020 MHz; L in service, s = 2.
    # f 612: 1 + 0.2 x 2 = 1.4 > 1.25 -> bert:5. f 1020 or None (-> f_max 1020): 1.16 <= 1.25 -> bert:23.
    spec = dict(FX["types"], **{"bert:23": {"tau": {612: 0.2, 1020: 0.08}, "klass": "short"}})
    T2 = mocks.make_types(spec, f_levels=LEVELS)
    pol = policy(T2, "greedy")
    q, srv = [Q(T2, "r1", "bert:23", 1.0, 1.25)], [S(T2, "L", "llama:15", 0.6, 2.25)]
    assert pairs(pol, view(1.0, q, srv, f=612)) == [["r1", "bert:5"]]
    assert pairs(pol, view(1.0, q, srv, f=1020)) == [["r1", "bert:23"]]
    assert pairs(pol, view(1.0, q, srv, f=None)) == [["r1", "bert:23"]]


@pytest.mark.parametrize("stat, expect", [("p95", "bert:5"), ("p50", "bert:23")])
def test_greedy_reads_the_tau_stat_of_the_spec(stat, expect):
    # bert:23 p50 0.08, p95 0.15; L in service, s = 2. p95: 1 + 0.30 = 1.30 > 1.25 -> floor. p50: 1.16 -> target.
    tau = {"p50": 0.08, "mean": 0.1, "p95": 0.15}
    T2 = mocks.make_types(dict(FX["types"], **{"bert:23": {"tau": {1020: tau}, "klass": "short"}}), f_levels=LEVELS)
    v = view(1.0, [Q(T2, "r1", "bert:23", 1.0, 1.25)], [S(T2, "L", "llama:15", 0.6, 2.25)])
    assert pairs(policy(T2, "greedy", tau_stat=stat), v) == [["r1", expect]]


class _CountingSlowdown(mocks.StubSlowdown):
    def __init__(self):
        super().__init__({1: 1.0})
        self.calls = []

    def predict(self, i, A, f_mhz):
        self.calls.append((i, tuple(A), f_mhz))
        return super().predict(i, A, f_mhz)


def test_slowdown_is_memoised_on_type_sorted_run_set_and_level(T):
    sd = _CountingSlowdown()
    pol = policy(T, "coexit", sd, cap_margin=0.0)      # the rollout (not the gate) is what exercises the memo
    case = next(c for c in FX["cases"] if c["name"] == "C1_coexit_gate_waits_at_capacity_one")
    v = DispatchView.from_json(case["view"])
    assert pairs(pol, v) == [["r1", "bert:5"]]
    first = list(sd.calls)
    assert first and len(first) == len(set(first))                      # never the same key twice
    for i, A, f in first:
        assert i in A and list(A) == sorted(A) and f == 1020
    assert pairs(pol, v) == [["r1", "bert:5"]]
    assert sd.calls == first                                            # the second call is served from the memo


def test_empty_queue_admits_nothing(T):
    for method in X.METHODS:
        assert policy(T, method).admit_exits(view(1.0)) == []
