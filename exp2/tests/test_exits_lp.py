"""S3 tests for coexit_lp in exp2.exits (PLAN PL3 section 1.5; exp2.contracts "Limited preemption on the board").

Owner 2026-10-26: "lp saja mendingan dan coexit lp gak sih ?"; PREEMPT = r waits for the LLaMA token boundary, then
runs alone; a paused LLaMA resumes when no request in service runs. Every expected value is computed by hand in the
comment next to it (never by calling the code under test). Times in comments are seconds relative to now unless
stated; views are built in integer ns. Mock only: no torch, no model, no simulator (coexit_lp is board only).

Hand setup (plan 1.5): llama:15 = 2.0 s, 32 tokens x 62.5 ms (ckpt_frac[c] = c / 32, head_frac 0); bert:23 80 ms,
bert:5 20 ms. Slowdown PerType: s = 1 alone; in any co-run s(bert:23) = 1.5, every other type 1.2.
"""
import random

import pytest

from exp2 import contracts as C
from exp2 import exits as X
from exp2.contracts import DispatchView, InServiceReq, QueuedReq
from exp2.policies import AdmissionCore
from exp2.tests import mocks

NS = 1_000_000_000
TYPES = {"bert:5": {"tau": 0.02, "klass": "short"}, "bert:23": {"tau": 0.08, "klass": "short"},
         "yolo:1": {"tau": 0.04, "klass": "short"}, "yolo:5": {"tau": 0.06, "klass": "short"},
         "llama:3": {"tau": 0.5, "klass": "long"}, "llama:15": {"tau": 2.0, "klass": "long"}}
CO = {"bert:23": 1.5, "bert:5": 1.2, "yolo:5": 1.2, "yolo:1": 1.2, "llama:15": 1.2, "llama:3": 1.2}


class PerType:
    """s = 1 alone; s = co[type] in any co-run, whatever the partners."""

    def __init__(self, co=CO):
        self.co = co

    def predict(self, i, A, f_mhz):
        return 1.0 if len(A) == 1 else self.co[i]


@pytest.fixture(scope="module")
def T():
    return mocks.make_types(TYPES, f_levels=(612, 816, 1020))


def ns(x):
    return None if x is None else int(round(x * NS))


def Q(T, rid, tid, a_s, dl_s):
    t = T.get(tid)
    return QueuedReq(rid, tid, t.family, t.klass, ns(a_s), ns(dl_s))


def S(T, rid, tid, admit_s, dl_s):
    t = T.get(tid)
    return InServiceReq(rid, tid, t.family, t.klass, ns(admit_s), ns(dl_s))


def view(now_s, queue=(), in_service=()):
    return DispatchView(ns(now_s), tuple(queue), tuple(in_service), 1020)


def lp(T, slowdown=None):
    return X.ExitPolicy(X.board_specs(1020)["coexit_lp"], T, PerType() if slowdown is None else slowdown)


def coexit(T, slowdown=None):
    return X.ExitPolicy(X.board_specs(1020)["coexit"], T, PerType() if slowdown is None else slowdown)


def decide(pol, v):
    """(admit pairs, pause_ids) of one core step: admit_exits, then pause_ids on the view after the admits."""
    pairs = [list(p) for p in pol.admit_exits(v)]
    if pairs:
        taken = {rid for rid, _ in pairs}
        extra = tuple(InServiceReq(q.req_id, run, q.family, q.klass, v.now_ns, q.t_deadline_ns)
                      for q in v.queue for rid, run in pairs if rid == q.req_id)
        v = DispatchView(v.now_ns, tuple(q for q in v.queue if q.req_id not in taken), v.in_service + extra,
                         v.f_current_mhz)
    return pairs, pol.pause_ids(v)


# ---------------------------------------------------------------- spec and state
def test_coexit_lp_builds_and_holds_the_lp_state(T):
    pol = lp(T)
    assert pol.rule == "coexit" and pol.preempt is True and pol.cap_margin == C.COEXIT_CAP_MARGIN
    assert pol.paused == {} and pol.pause_pending == set() and pol.paused_ns == {}
    assert pol.pause_ids(view(1.0, [Q(T, "r", "bert:23", 1.0, 1.1)], [S(T, "L", "llama:15", 0.76, 10.0)])) == ()


# ---------------------------------------------------------------- the four hand cases of plan 1.5
def test_preempt_wins(T):
    # now 1.0. L llama:15 admitted 0.76 (dl 10.0): served 0.24, p 0.12, c* = 4 (4/32 = 0.125 > 0.12), d = 0.01,
    # L after the pause w = 2.0 x (1 - 0.125) = 1.75. r bert:23 (a 1.0, dl 1.1 -> 0.1).
    # PREEMPT(target): r starts at 0.01 alone, done 0.09 <= 0.1; L resumes 0.09, done 1.84 <= 9.0: late 0.
    # co-run(target): gate 1/1.2 + 1/1.5 = 1.5 >= 1.053 passes; r done 0.08 x 1.5 = 0.12 > 0.1 -> not offered.
    # PREEMPT(floor) 0, co-run(floor) 0 (r 0.024), WAIT 1 (r after L at 1.78). First with min 0 -> PREEMPT(target).
    v = view(1.0, [Q(T, "r", "bert:23", 1.0, 1.1)], [S(T, "L", "llama:15", 0.76, 10.0)])
    assert decide(lp(T), v) == ([], ("L",))


def test_preempt_floor(T):
    # Same with d = 0.04: L admitted 0.79, served 0.21, p 0.105, c* = 4, d = (0.125 - 0.105) x 2.0 = 0.04.
    # PREEMPT(target): r done 0.04 + 0.08 = 0.12 > 0.1 -> not offered. co-run(target): 0.12 -> not offered.
    # PREEMPT(floor): r done 0.06; L resumes 0.06, done 1.81: late 0. co-run(floor): r 0.024, L 1.794: late 0.
    # WAIT: r after L (1.79) -> late 1. PREEMPT(floor) is before co-run(floor) -> PREEMPT.
    v = view(1.0, [Q(T, "r", "bert:23", 1.0, 1.1)], [S(T, "L", "llama:15", 0.79, 10.0)])
    assert decide(lp(T), v) == ([], ("L",))


def test_corun_wins(T):
    # now 2.0. L admitted 1.0 (dl 3.015 -> 1.015): served 1.0, left 1.0, p 0.5, c* = 17 (0.53125), d = 0.0625,
    # w after the pause 0.9375. r bert:23 (a 2.0, dl 2.1 -> 0.1).
    # PREEMPT(target): r 0.1425 > 0.1 -> not offered. co-run(target): r 0.12 -> not offered.
    # PREEMPT(floor): r 0.0825; L resumes 0.0825, done 1.02 > 1.015: late 1.
    # co-run(floor): r 0.02 x 1.2 = 0.024; L did 0.02, 0.98 alone -> 1.004 <= 1.015: late 0.
    # WAIT: L 1.0, r floor 1.02 > 0.1: late 1. min 0 -> co-run(floor), no pause.
    v = view(2.0, [Q(T, "r", "bert:23", 2.0, 2.1)], [S(T, "L", "llama:15", 1.0, 3.015)])
    assert decide(lp(T), v) == ([["r", "bert:5"]], ())


def _mix_a_view(rng, T, fams=("bert", "yolo")):
    now = rng.randint(2 * NS, 20 * NS)
    in_service = []
    for i, fam in enumerate(rng.sample(fams, rng.randint(0, min(2, len(fams))))):
        tid = rng.choice([t for t in TYPES if t.startswith(fam + ":")])
        dl = rng.choice([None, now + rng.randint(1, 3 * NS)])
        in_service.append(InServiceReq("s%d" % i, tid, fam, T.get(tid).klass, now - rng.randint(0, NS), dl))
    queue, a = [], now - rng.randint(0, NS)
    for i in range(rng.randint(0, 6)):
        fam = rng.choice(fams)
        tid = "%s:%d" % (fam, C.EXIT_MENU[fam][1])
        a = min(now, a + rng.randint(0, NS // 8))
        dl = rng.choice([None, a + rng.randint(1, 3 * NS // 2)])
        queue.append(QueuedReq("q%d" % i, tid, fam, T.get(tid).klass, a, dl))
    return DispatchView(now, tuple(queue), tuple(in_service), rng.choice([None, 612, 1020]))


@pytest.mark.parametrize("slowdown, min_admit", [
    (PerType(), 100),
    (mocks.StubSlowdown({1: 1.0}), 40),                 # s = n: the gate blocks every co-run, admits onto an empty GPU
    (mocks.StubSlowdown({1: 1.0, 2: 1.5, 3: 2.0}), 100),
], ids=["per_type", "gate_closed", "gate_open"])
def test_mix_a_without_llama_equals_coexit_on_200_random_views(T, slowdown, min_admit):
    # No LLaMA: no request can be L or paused, so coexit_lp must make coexit's decisions exactly and never pause.
    a, b = coexit(T, slowdown), lp(T, slowdown)
    rng = random.Random(20261026)
    n_admit = 0
    for _ in range(200):
        v = _mix_a_view(rng, T)
        want = a.admit_exits(v)
        assert b.admit_exits(v) == want, v
        assert b.pause_ids(v) == ()
        n_admit += len(want)
    assert n_admit >= min_admit                         # the draw exercises admits, not only empty queues


# ---------------------------------------------------------------- pending, paused, WAIT
def test_a_pending_pause_admits_nothing(T):
    # test_corun_wins would admit r at bert:5; with L's pause pending nothing is admitted and nothing new is paused.
    v = view(2.0, [Q(T, "r", "bert:23", 2.0, 2.1)], [S(T, "L", "llama:15", 1.0, 3.015)])
    pol = lp(T)
    pol.pause_pending.add("L")
    assert decide(pol, v) == ([], ())


def test_a_pending_id_that_is_no_longer_in_service_does_not_block(T):
    # Decision S3: only a pending pause of a request in service blocks (the dispatcher drops it on done anyway).
    v = view(2.0, [Q(T, "r", "bert:23", 2.0, 2.1)], [S(T, "L", "llama:15", 1.0, 3.015)])
    pol = lp(T)
    pol.pause_pending.add("gone")
    assert decide(pol, v) == ([["r", "bert:5"]], ())


def test_wait_is_not_offered_when_only_paused_jobs_are_in_service(T):
    # now 2.0. L paused after token 16: w = 2.0 x (1 - 0.5) = 1.0, dl 3.01 -> 1.01. r bert:23 (dl 7.0 -> 5.0),
    # q llama:3 queued (dl 9.0 -> 7.0; llama is busy with the paused L, so q is no candidate). s = n (c(n) = 1).
    # No job runs: A is empty, the gate is open, WAIT is not offered.
    # target: r alone 0.08; L resumes 0.08, done 1.08 > 1.01 (late); q 1.58: L = 1.
    # floor: r 0.02; L 1.02 > 1.01: L = 1. -> target, the first with min 1.
    # (Offered, WAIT would resume L at once: L 1.0 on time, then r and q start: 0 late, and r would not run.
    # Counting the paused L in A would close the gate: capacity 2 x 1/2 = 1.0 < 1.053 -> WAIT as well.)
    v = view(2.0, [Q(T, "r", "bert:23", 2.0, 7.0), Q(T, "q", "llama:3", 1.9, 9.0)],
             [S(T, "L", "llama:15", 0.5, 3.01)])
    pol = lp(T, mocks.StubSlowdown({1: 1.0}))
    pol.paused["L"] = 16
    assert decide(pol, v) == ([["r", "bert:23"]], ())


def test_paused_ns_is_taken_off_the_served_time_of_l(T):
    # test_corun_wins with L admitted 0.7 and paused 0.3 s before: served = 2.0 - 0.7 - 0.3 = 1.0, the same view
    # -> co-run(floor). Ignoring paused_ns would give served 1.3, p 0.65, c* = 21, d 0.0125, w 0.6875:
    # PREEMPT(target) r done 0.0925 <= 0.1, L done 0.0925 + 0.6875 = 0.78 <= 1.015 (late 0) and PREEMPT would win.
    v = view(2.0, [Q(T, "r", "bert:23", 2.0, 2.1)], [S(T, "L", "llama:15", 0.7, 3.015)])
    pol = lp(T)
    pol.paused_ns["L"] = ns(0.3)
    assert decide(pol, v) == ([["r", "bert:5"]], ())


# ---------------------------------------------------------------- when L may not be paused (decision S3)
@pytest.mark.parametrize("name, now, L, r", [
    # test_preempt_wins without L's deadline: coexit -> target late (0.12); floor 0 (r 0.024); WAIT 1 -> bert:5.
    ("no_deadline", 1.0, ("llama:15", 0.76, None), ("bert:23", 1.0, 1.1)),
    # L's deadline is now (now < deadline fails): L is late in every option; floor 1, WAIT 2 -> bert:5.
    # (Were L preemptable, PREEMPT(target) with late 1 would come first.)
    ("deadline_passed", 1.0, ("llama:15", 0.76, 1.0), ("bert:23", 1.0, 1.1)),
    # L at its last token: served 1.96, p 0.98, c* = 32 = the last token. r (dl 0.09):
    # target L done 0.048, r 0.048 + 0.048 = 0.096 > 0.09 -> not offered; floor 0 (r 0.024, L 0.044) -> bert:5.
    # (Were L preemptable: PREEMPT(floor) r 0.04 + 0.02 = 0.06, L 0.06: late 0, it would win.)
    ("last_token", 3.0, ("llama:15", 1.04, 10.0), ("bert:23", 3.0, 3.09)),
])
def test_no_preempt_when_l_is_not_preemptable(T, name, now, L, r):
    v = view(now, [Q(T, "r", *r)], [S(T, "L", *L)])
    assert decide(lp(T), v) == ([["r", "bert:5"]], ()), name
    assert decide(coexit(T), v) == ([["r", "bert:5"]], ()), name


def test_no_preempt_when_another_job_runs_or_was_admitted_in_the_call(T):
    # Only-L-runs fails -> the coexit rule. Here coexit_lp must equal coexit on 200 random views with LLaMA
    # (views where exactly one LLaMA with a deadline is in service are skipped: there PREEMPT may apply).
    a, b = coexit(T), lp(T)
    rng = random.Random(20261027)
    n_cmp = n_admit = 0
    while n_cmp < 200:
        v = _mix_a_view(rng, T, ("bert", "yolo", "llama"))
        if len(v.in_service) == 1 and v.in_service[0].family == "llama" and v.in_service[0].t_deadline_ns:
            continue
        want = a.admit_exits(v)
        assert b.admit_exits(v) == want, v
        assert b.pause_ids(v) == ()
        n_cmp += 1
        n_admit += len(want)
    assert n_admit >= 100


def test_only_l_runs_views_either_pause_or_admit_never_both(T):
    # 200 random views where one LLaMA with a deadline runs alone: no rollout stalls; a pause comes with no admit and
    # names L; both outcomes occur.
    pol = lp(T)
    rng = random.Random(20261028)
    n_pause = n_admit = 0
    for i in range(200):
        now = rng.randint(2 * NS, 20 * NS)
        L = InServiceReq("L", "llama:15", "llama", "long", now - rng.randint(0, 2 * NS),
                         now + rng.randint(-NS // 10, 3 * NS))
        queue = tuple(QueuedReq("q%d" % k, tid, tid.split(":")[0], "short", now - rng.randint(0, NS // 20),
                                now + rng.randint(1, NS // 4))
                      for k, tid in enumerate(rng.sample(("bert:23", "yolo:5"), rng.randint(1, 2))))
        pairs, pause = decide(pol, DispatchView(now, queue, (L,), 1020))
        assert not (pairs and pause), (pairs, pause)
        assert pause in ((), ("L",))
        if pause:
            assert L.t_deadline_ns > now                # decision S3: never pause an L that is already late
        n_pause += bool(pause)
        n_admit += bool(pairs)
    assert n_pause >= 20 and n_admit >= 20


def test_with_only_a_paused_llama_the_first_short_head_is_always_admitted(T):
    # The tick after `paused`: nothing runs, A is empty (gate open) and WAIT is not offered -> the first non-LLaMA
    # queued request is admitted, whatever the rollout says; queued LLaMA requests wait (family busy).
    pol = lp(T, mocks.StubSlowdown({1: 1.0}))
    rng = random.Random(20261029)
    for i in range(200):
        now = rng.randint(2 * NS, 20 * NS)
        pol.paused.clear()
        pol.paused["L"] = rng.randint(1, 31)
        L = InServiceReq("L", "llama:15", "llama", "long", now - rng.randint(0, 3 * NS),
                         now + rng.randint(1, 3 * NS))
        tids = [rng.choice(("bert:23", "yolo:5", "llama:15")) for _ in range(rng.randint(1, 4))]
        queue = tuple(QueuedReq("q%d" % k, tid, tid.split(":")[0], T.get(tid).klass, now - rng.randint(0, NS // 20),
                                now + rng.randint(1, NS))
                      for k, tid in enumerate(tids))
        pairs, pause = decide(pol, DispatchView(now, queue, (L,), 1020))
        heads = [q.req_id for q in queue if q.family != "llama"]
        assert pause == ()
        assert [rid for rid, _ in pairs][:1] == heads[:1], (pairs, queue)
        assert all(rid in heads for rid, _ in pairs)


# ---------------------------------------------------------------- the tick and the flow through the core
def test_tick_runs_r_alone_after_the_pause(T):
    # test_preempt_floor, at the tick: now 1.04, L paused after token 4 (w 1.75), r (dl 1.1 -> 0.06).
    # target: r 0.08 > 0.06 -> not offered; floor: r 0.02, L resumes 0.02, done 1.77: late 0 -> bert:5.
    v = view(1.04, [Q(T, "r", "bert:23", 1.0, 1.1)], [S(T, "L", "llama:15", 0.79, 10.0)])
    pol = lp(T)
    pol.paused["L"] = 4
    assert decide(pol, v) == ([["r", "bert:5"]], ())


def test_coexit_lp_flow_through_the_core(T):
    """Plan 1.1 with coexit_lp: run L; r arrives -> PREEMPT (pause L); pending -> nothing; paused -> tick admits r
    alone at its target (test_preempt_wins: done 0.09 after its arrival); r done -> nothing to admit or pause."""
    pol = lp(T)
    core = AdmissionCore(pol, T, 1020)
    # L alone: target 2.0 <= 9.24 on time, late 0 -> llama:15.
    d = core.arrive(Q(T, "L", "llama:15", 0.76, 10.0), ns(0.76))
    assert (d.admitted, d.run_type_ids, d.pause_ids) == (("L",), ("llama:15",), ())
    d = core.arrive(Q(T, "r", "bert:23", 1.0, 1.1), ns(1.0))          # test_preempt_wins
    assert (d.admitted, d.pause_ids) == ((), ("L",))
    pol.pause_pending.add("L")                                          # dispatcher sent pause
    # y llama:3 arrives while the pause is pending: no admit and no new pause (without the pending rule r would
    # PREEMPT again and pause_ids would be ("L",)).
    d = core.arrive(Q(T, "y", "llama:3", 1.005, 9.0), ns(1.005))
    assert (d.admitted, d.pause_ids) == ((), ())
    pol.pause_pending.discard("L")
    pol.paused["L"] = 4                                                 # server: paused after token 4, at 1.01
    # Tick at 1.01: nothing runs, gate open, no WAIT. r target alone 0.08 -> 1.09 <= 1.1; L (w 1.75) resumes at
    # 0.08, done 1.83 <= 8.99; y waits for the llama family, 1.83 + 0.5 = 2.33 <= 7.99: late 0 -> bert:23.
    # y: llama is busy (L paused) -> no candidate. n_in_service_at_admit counts the paused L.
    d = core.tick(ns(1.01))
    assert (d.admitted, d.run_type_ids, d.n_in_service_at_admit, d.pause_ids) == (("r",), ("bert:23",), (1,), ())
    d = core.done("r", ns(1.09))                                        # the dispatcher resumes L after this
    assert (d.admitted, d.pause_ids) == ((), ())
    del pol.paused["L"]
    pol.paused_ns["L"] = ns(0.08)
    d = core.done("L", ns(1.84))
    # y alone: target llama:15 1.84 + 2.0 = 3.84 <= 9.0 -> llama:15.
    assert (d.admitted, d.run_type_ids, d.pause_ids) == (("y",), ("llama:15",), ())
