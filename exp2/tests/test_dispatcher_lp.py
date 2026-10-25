"""S2 tests (board limited preemption, PLAN PL3 sections 1.2 and 3): the dispatcher pauses and resumes a LLaMA request.

Rig: VirtualLPDispatcher, the VirtualDispatcher of test_serve_dispatcher (the real Dispatcher, AdmissionCore, ExitPolicy
and wire; only _poll is virtual) with the llama server emulated token by token as server.py does it (slice S1):
32 tokens, one every service/32 ns; a `pause` read at t_req stops the run at the first on-time, non-last token
boundary t_k >= t_req and the server answers `paused`; a `resume` runs token k+1 from the time it is read.
Every expected value is derived by hand from the plan, not by calling the code under test.  Fakes only.
"""
import heapq

import pytest

from exp2 import contracts as C
from exp2.exits import ExitPolicy, board_specs
from exp2.serve import dispatcher as dispatcher_mod
from exp2.tests import mocks
from exp2.tests.test_serve_dispatcher import MS, NS, ORIGIN, VirtualDispatcher, by_id, item, run_open

TOKENS = C.LLAMA_TOKENS
TAU_S = {"bert:5": 0.02, "bert:23": 0.08, "yolo:1": 0.04, "yolo:5": 0.06, "llama:3": 0.6, "llama:15": 3.2}
KLASS = {"bert": "short", "yolo": "short", "llama": "long"}
SPECS = board_specs(1020)


@pytest.fixture(scope="module")
def types():
    return mocks.make_types({tid: {"tau": tau, "klass": KLASS[C.parse_type_id(tid)[0]]} for tid, tau in TAU_S.items()})


class VirtualLPDispatcher(VirtualDispatcher):
    """VirtualDispatcher whose llama server pauses at token boundaries (the server semantics of PLAN PL3 1.2).

    A llama run of service s runs token j at t_seg + (j - k) * s/32 (k tokens done when the segment started at t_seg).
    The first late token or token 32 ends the run (K7).  A `pause` read at t_req (the scan time) stops the run at the
    first token boundary t_j >= t_req that is on time and not the last: `paused` (tokens_done j, t_pause_ns t_j) is sent
    and the server waits.  A `resume` read at t restarts it at t + resume_lag_ns.  A pause or resume that does not
    apply (other req_id, already paused, no boundary left) is stale: counted in .stale, otherwise ignored.  The done
    carries pauses [[j, t_pause, t_resume], ...] when non-empty.  die_at_ns: the llama connection breaks then.
    done_hook(done) -> done edits the llama done before it is sent.  .ctl lists (op, req_id, t_ns) of every
    pause/resume the dispatcher sent."""

    def __init__(self, types, *, resume_lag_ns=0, die_at_ns=None, done_hook=None, **kw):
        super().__init__(types, **kw)
        self._resume_lag = resume_lag_ns
        self._die_at = die_at_ns
        self._done_hook = done_hook
        self._job = None
        self.ctl = []
        self.stale = 0

    # ------------------------------------------------------------------------------------------- llama server
    def _scan(self):
        a = self.pairs["llama"][0]
        msgs = a.sent_msgs()
        now = self.vclock()
        for m in msgs[self._scanned["llama"]:]:
            self._llama_msg(m, now)
        self._scanned["llama"] = len(msgs)
        super()._scan()                                   # other families: answered at t_admit + service

    def _llama_msg(self, m, now):
        job = self._job
        if m["op"] == "run":
            C.validate_run(m)
            assert job is None, "a second llama run while one is in service"
            self.runs.append(m)
            self._job = {"run": m, "step": self.service_ns(m["type_id"]) // TOKENS, "k": 0, "t_seg": m["t_admit_ns"],
                         "times": [], "pend": None, "paused": False, "pauses": []}
            return
        C.validate_server_msg(m)
        self.ctl.append((m["op"], m["req_id"], now))
        mine = job is not None and job["run"]["req_id"] == m["req_id"]
        if m["op"] == "pause" and mine and not job["paused"] and job["pend"] is None:
            job["pend"] = now
        elif m["op"] == "resume" and mine and job["paused"]:
            t = now + self._resume_lag
            job.update(paused=False, t_seg=t)
            job["pauses"][-1][2] = t
        else:
            self.stale += 1

    def _llama_event(self):
        """-> (t_ns, "paused" | "done", token) of the next llama message, or None (no run, or paused)."""
        job = self._job
        if job is None or job["paused"]:
            return None
        dl, k = job["run"]["deadline_ns"], job["k"]
        for j in range(k + 1, TOKENS + 1):
            t = job["t_seg"] + (j - k) * job["step"]
            if j == TOKENS or (dl is not None and t > dl):
                return t, "done", j
            if job["pend"] is not None and t >= job["pend"]:
                return t, "paused", j
        raise AssertionError("unreachable")

    def _llama_fire(self, t, kind, j):
        job = self._job
        run, k = job["run"], job["k"]
        job["times"] += [job["t_seg"] + (i - k) * job["step"] for i in range(k + 1, j + 1)]
        job["k"] = j
        if kind == "paused":
            job.update(paused=True, pend=None)
            job["pauses"].append([j, t, None])
            return {"op": "paused", "req_id": run["req_id"], "type_id": run["type_id"], "tokens_done": j,
                    "t_pause_ns": t}
        fam, ex = C.parse_type_id(run["type_id"])
        ckpts = C.default_checkpoints(fam, ex)
        times = job["times"]
        reached, outcome = C.truncation_outcome(ckpts[:len(times)], times, run["deadline_ns"])
        done = {"op": "done", "req_id": run["req_id"], "type_id": run["type_id"], "t_recv_ns": run["t_admit_ns"],
                "t_start_ns": run["t_admit_ns"], "t_end_ns": t, "reached": reached, "target": ckpts[-1],
                "completion": C.completion(fam, ex, reached), "outcome": outcome,
                "ckpt": [[c, x] for c, x in zip(ckpts, times)], "tokens_done": reached if reached else None,
                "err": None}
        if job["pauses"]:
            done["pauses"] = [list(p) for p in job["pauses"]]
        if job["pend"] is not None:                       # a pause that found no boundary: read later, ignored
            self.stale += 1
        self._job = None
        return done if self._done_hook is None else self._done_hook(done)

    # ------------------------------------------------------------------------------------------- virtual wait
    def _poll(self, timeout_ns):
        self._scan()
        now = self.vclock()
        limit = now + timeout_ns
        cands = []
        if self._heap:
            cands.append((self._heap[0][0], 0, None))
        ev = self._llama_event()
        if ev is not None:
            cands.append((ev[0], 1, ev))
        if self._die_at is not None:
            cands.append((self._die_at, 2, None))
        if not cands or min(cands)[0] > limit:
            self.vclock.set_ns(limit + self._lag)
            return []
        t, which, ev = min(cands)
        self.vclock.set_ns(max(t, now))
        if which == 0:
            _, _, fam, msg = heapq.heappop(self._heap)
            return [(fam, msg, self.vclock())]
        if which == 2:
            self._die_at, self._job = None, None
            return [("llama", EOFError("the llama server died"), self.vclock())]
        return [("llama", self._llama_fire(*ev), self.vclock())]


def rel(t0, pauses):
    """Row pauses with times in ms after t0 (None kept)."""
    return [[p[0]] + [None if x is None else (x - t0) / MS for x in p[1:]] for p in pauses]


def ctl_ms(d, t0):
    return [(op, rid, (t - t0) / MS) for op, rid, t in d.ctl]


def lp_policy(types):
    return ExitPolicy(SPECS["edgeserving_lp"], types)


def assert_idle(pol):
    assert pol.paused == {} and pol.pause_pending == set()


# ================================================================================================ ES vs ES-LP
def test_edgeserving_runs_the_short_after_the_whole_llama_and_never_pauses(types, tmp_path):
    # ES: L llama:15 runs 0..3200 ms; b (bert:23, arrives 250) waits for it and starts at 3.2 s.
    d = VirtualLPDispatcher(types)
    sched = [item("L", "llama:15", 0, 5000), item("b", "bert:23", 250, 4000)]
    summary, rows, _ = run_open(d, sched, ExitPolicy(SPECS["edgeserving"], types), tmp_path)
    g, t0 = by_id(rows), summary["t0_ns"]
    assert g["L"]["t_done_ns"] - t0 == 3200 * MS and g["L"]["outcome"] == "complete"
    assert g["b"]["t_admit_ns"] - t0 == 3200 * MS and g["b"]["t_done_ns"] - t0 == 3280 * MS
    assert all("pauses" not in r for r in rows)                                 # non-LP rows: no pauses key
    assert d.ctl == [] and d.stale == 0 and summary["unexpected_msgs"] == 0


def test_edgeserving_lp_pauses_the_llama_at_the_next_token_and_resumes_when_the_short_is_done(types, tmp_path):
    # pause sent at 250 (b queued); token 3 ends at 300 -> paused; tick admits b (bert:23, 80 ms) at 300, done 380;
    # nothing runs -> resume at 380; tokens 4..32 at 380 + (j - 3) * 100 -> L ends at 3200 + 80 ms.
    d = VirtualLPDispatcher(types)
    pol = lp_policy(types)
    sched = [item("L", "llama:15", 0, 5000), item("b", "bert:23", 250, 4000)]
    summary, rows, _ = run_open(d, sched, pol, tmp_path)
    g, t0 = by_id(rows), summary["t0_ns"]
    L, b = g["L"], g["b"]
    assert rel(t0, L["pauses"]) == [[3, 250, 300, 380]]
    assert L["outcome"] == "complete" and L["reached"] == TOKENS and L["tokens_done"] == TOKENS
    assert L["t_end_ns"] - t0 == L["t_done_ns"] - t0 == 3280 * MS
    assert L["ckpt"][2] == [3, t0 + 300 * MS] and L["ckpt"][3] == [4, t0 + 480 * MS]
    assert (b["t_admit_ns"] - t0, b["t_done_ns"] - t0) == (300 * MS, 380 * MS)
    assert b["type_id"] == "bert:23" and b["n_in_service_at_admit"] == 1 and b["pauses"] == []
    assert ctl_ms(d, t0) == [("pause", "L", 250), ("resume", "L", 380)]
    assert d.stale == 0 and summary["unexpected_msgs"] == 0 and summary["dead_servers"] == {}
    assert_idle(pol)


def test_a_llama_admitted_while_a_short_waits_is_paused_in_the_same_decision(types, tmp_path):
    # b0 runs 0..80.  At 80 ES picks L over b1: S(L) = exp((0.06 + 3.2)/1000 - 1) = 0.369 < S(b1) =
    # exp((0.07 + 0.08)/3.4 - 1) = 0.384.  b1 still waits, so the run and the pause of L go out at 80 ms.
    # Token 1 ends at 180 -> paused; b1 runs 180..260; resume at 260; token 32 at 260 + 31 * 100 = 3360 <= 3410.
    d = VirtualLPDispatcher(types)
    sched = [item("b0", "bert:23", 0, 1_000_000), item("L", "llama:15", 10, 3410), item("b1", "bert:23", 20, 1_000_020)]
    summary, rows, _ = run_open(d, sched, lp_policy(types), tmp_path)
    g, t0 = by_id(rows), summary["t0_ns"]
    assert g["L"]["t_admit_ns"] - t0 == 80 * MS
    assert [m["op"] for m in d.pairs["llama"][0].sent_msgs()] == ["run", "pause", "resume"]
    assert rel(t0, g["L"]["pauses"]) == [[1, 80, 180, 260]]
    assert g["b1"]["t_admit_ns"] - t0 == 180 * MS and g["b1"]["n_in_service_at_admit"] == 1
    assert g["L"]["outcome"] == "complete" and g["L"]["t_done_ns"] - t0 == 3360 * MS


def test_a_second_short_while_the_pause_is_pending_sends_no_second_pause(types, tmp_path):
    # b1 (250) and b2 (270) wait; one pause.  Paused at 300: b1 runs 300..380, then b2 (only the paused L is in
    # service) 380..460, then resume at 460; L ends at 3200 + 160 ms.
    d = VirtualLPDispatcher(types)
    pol = lp_policy(types)
    sched = [item("L", "llama:15", 0, 5000), item("b1", "bert:23", 250, 4000), item("b2", "bert:5", 270, 4000)]
    summary, rows, _ = run_open(d, sched, pol, tmp_path)
    g, t0 = by_id(rows), summary["t0_ns"]
    assert ctl_ms(d, t0) == [("pause", "L", 250), ("resume", "L", 460)]
    assert rel(t0, g["L"]["pauses"]) == [[3, 250, 300, 460]]
    assert (g["b1"]["t_admit_ns"] - t0, g["b2"]["t_admit_ns"] - t0) == (300 * MS, 380 * MS)
    assert g["L"]["t_done_ns"] - t0 == 3360 * MS
    assert_idle(pol)


# ================================================================================================ races
def test_a_done_while_the_pause_is_pending_cancels_the_pause(types, tmp_path):
    # b arrives at 3150: pause sent; the next boundary is token 32 (the last) -> no pause, done at 3200.
    # The done cancels the pending pause; b is admitted at 3200; the server reads the stale pause and ignores it.
    d = VirtualLPDispatcher(types)
    pol = lp_policy(types)
    sched = [item("L", "llama:15", 0, 5000), item("b", "bert:23", 3150, 4000)]
    summary, rows, _ = run_open(d, sched, pol, tmp_path)
    g, t0 = by_id(rows), summary["t0_ns"]
    assert ctl_ms(d, t0) == [("pause", "L", 3150)] and d.stale == 1
    assert g["L"]["outcome"] == "complete" and g["L"]["pauses"] == [] and g["L"]["t_done_ns"] - t0 == 3200 * MS
    assert g["b"]["t_admit_ns"] - t0 == 3200 * MS and g["b"]["outcome"] == "complete"
    assert summary["unexpected_msgs"] == 0
    assert_idle(pol)


def test_a_short_that_expires_while_the_pause_is_pending_gives_an_immediate_resume(types, tmp_path):
    # b (deadline 290) is queued when the pause goes out at 250, but expired at the boundary (300): the tick admits
    # nothing, so the llama resumes at once (an empty pause) and ends at 3200.
    d = VirtualLPDispatcher(types)
    pol = lp_policy(types)
    sched = [item("L", "llama:15", 0, 5000), item("b", "bert:23", 250, 290)]
    summary, rows, _ = run_open(d, sched, pol, tmp_path)
    g, t0 = by_id(rows), summary["t0_ns"]
    assert g["b"]["outcome"] == "expired_in_queue" and g["b"]["t_done_ns"] - t0 == 290 * MS
    assert g["b"]["pauses"] == []
    assert rel(t0, g["L"]["pauses"]) == [[3, 250, 300, 300]]
    assert ctl_ms(d, t0) == [("pause", "L", 250), ("resume", "L", 300)]
    assert g["L"]["outcome"] == "complete" and g["L"]["t_done_ns"] - t0 == 3200 * MS
    assert_idle(pol)


# ================================================================================================ timing and the row
def test_the_row_keeps_the_server_pause_times_and_the_dispatcher_t_req(types, tmp_path):
    # The server reads the resume 2 ms after it is sent: the row holds the server's t_resume (382), L ends at 3282.
    d = VirtualLPDispatcher(types, resume_lag_ns=2 * MS)
    sched = [item("L", "llama:15", 0, 5000), item("b", "bert:23", 250, 4000)]
    summary, rows, _ = run_open(d, sched, lp_policy(types), tmp_path)
    g, t0 = by_id(rows), summary["t0_ns"]
    assert rel(t0, g["L"]["pauses"]) == [[3, 250, 300, 382]]
    assert ctl_ms(d, t0) == [("pause", "L", 250), ("resume", "L", 380)]
    assert g["L"]["t_done_ns"] - t0 == 3282 * MS


def test_a_done_whose_pauses_disagree_with_the_paused_messages_is_an_error_row(types, tmp_path):
    def drop_pauses(done):
        done = dict(done)
        done.pop("pauses", None)
        return done

    d = VirtualLPDispatcher(types, done_hook=drop_pauses)
    sched = [item("L", "llama:15", 0, 5000), item("b", "bert:23", 250, 4000)]
    summary, rows, _ = run_open(d, sched, lp_policy(types), tmp_path)
    g, t0 = by_id(rows), summary["t0_ns"]
    assert g["L"]["outcome"] == "error" and "invalid done" in g["L"]["err"]
    assert rel(t0, g["L"]["pauses"]) == [[3, 250, 300, 380]]                  # the dispatcher's own record
    assert g["b"]["outcome"] == "complete"


def test_a_llama_paused_longer_than_the_hang_timeout_is_not_killed(types, tmp_path):
    # b1 and b2 take 100 s each.  L pauses at 300 ms, b1 runs 0.3..100.3 s, b2 (arrives 50 s, while L is paused: no
    # second pause) runs 100.3..200.3 s, resume at 200.3 s, L ends at 200.3 + 2.9 = 203.2 s.  L is paused for 200 s
    # (skipped by the hang timer) and its timer restarts at the resume (203.2 s - t_admit > 120 s).
    assert dispatcher_mod.HANG_TIMEOUT_NS == 120 * NS
    d = VirtualLPDispatcher(types, service={"bert:23": 100 * NS})
    pol = lp_policy(types)
    sched = [item("L", "llama:15", 0, 300_000), item("b1", "bert:23", 250, 400_000),
             item("b2", "bert:23", 50_000, 400_000)]
    summary, rows, _ = run_open(d, sched, pol, tmp_path)
    g, t0 = by_id(rows), summary["t0_ns"]
    assert summary["dead_servers"] == {} and summary["outcomes"] == {"complete": 3}
    assert ctl_ms(d, t0) == [("pause", "L", 250), ("resume", "L", 200_300)]
    assert rel(t0, g["L"]["pauses"]) == [[3, 250, 300, 200_300]]
    assert g["L"]["t_done_ns"] - t0 == 203_200 * MS
    assert g["b2"]["t_admit_ns"] - t0 == 100_300 * MS
    assert_idle(pol)


def test_a_llama_server_that_dies_while_paused_gives_an_error_row(types, tmp_path):
    # L paused at 300 (b runs 300..380); the llama connection breaks at 350: L is an error row whose last pause has
    # no resume; b completes, nothing is resumed; L2 (arrives 1 s) meets a dead server.
    d = VirtualLPDispatcher(types, die_at_ns=ORIGIN + 350 * MS)
    pol = lp_policy(types)
    sched = [item("L", "llama:15", 0, 5000), item("b", "bert:23", 250, 4000), item("L2", "llama:15", 1000, 6000)]
    summary, rows, _ = run_open(d, sched, pol, tmp_path)
    g, t0 = by_id(rows), summary["t0_ns"]
    assert t0 == ORIGIN
    L = g["L"]
    assert L["outcome"] == "error" and "llama" in L["err"] and L["t_admit_ns"] == t0
    assert L["t_done_ns"] - t0 == 350 * MS and L["reached"] is None
    assert rel(t0, L["pauses"]) == [[3, 250, 300, None]]
    assert g["b"]["outcome"] == "complete" and g["b"]["t_done_ns"] - t0 == 380 * MS
    assert g["L2"]["outcome"] == "error" and g["L2"]["pauses"] == []
    assert ctl_ms(d, t0) == [("pause", "L", 250)]
    assert "llama" in summary["dead_servers"]
    assert_idle(pol)
