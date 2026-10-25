"""exp2.exits: exit-choosing admission policies for the CoExit pilot (PLAN PL1; S0 skeleton, rules filled by S1).

Semantics are frozen in the docstring of exp2.contracts ("Exit choice"). An ExitPolicy offers
admit_exits(view) -> [(req_id, run_type_id)]; AdmissionCore (exp2.policies) records the pairs in
Decision.run_type_ids and puts the run type in service. Policies in exp2.policies are not touched.

    ExitPolicy(spec, types, slowdown=None)   .admit_exits(view) -> [(req_id, run_type_id)]   .admit(view) -> [req_id]
    pilot_specs(f_mhz) -> {method: PolicySpec}
    run3_specs(f_mhz)  -> pilot_specs(f_mhz) + the ABLATIONS specs (Run 3 board)
    sim_specs(f_mhz)   -> run3_specs(f_mhz) + the SIM_ONLY specs (coexit_pilot --methods)
    board_specs(f_mhz) -> run3_specs(f_mhz) + the LP_METHODS specs (board plan with limited preemption, PLAN PL3)
    METHODS = the five pilot methods in report order
    ABLATIONS = ("coexit_nogate", "greedy_gate"): coexit with params.cap_margin -1 (the capacity gate never blocks);
                greedy with params.cap_margin COEXIT_CAP_MARGIN (coexit's capacity gate in front of the greedy rule)
    SIM_ONLY = ("edgeserving_lp",): edgeserving with params.preempt_llama_at_token true (EdgeServing-LP, review B1).
                In the simulator the pause at a LLaMA token boundary lives in exp2.model.simulator, which marks the
                paused request in ExitPolicy.paused. Not in run3_specs.
    LP_METHODS = ("edgeserving_lp", "coexit_lp"): the two limited-preemption methods of the board (owner 2026-10-26:
                "lp saja mendingan dan coexit lp gak sih ?"). coexit_lp = coexit with params.preempt_llama_at_token
                true, board only (the simulator refuses it); its rule (slice S3) is in _coexit, _lp_cut and _rollout.
    LP state (contracts "Limited preemption on the board"; the dispatcher writes it, the policy reads it):
        preempt (bool), paused {req_id: tokens_done}, pause_pending {req_id}, paused_ns {req_id: ns paused}.
        pause_ids(view) -> tuple of req_id to pause; () for a policy that does not preempt.

Points the contract leaves open, decided here (S1):
    - A WAIT keeps its family's turn: the requests of that family behind it are not candidates in the same call
      (FIFO per family, as in the other four methods; so "r heads its family queue" always holds).
    - "WAIT if >= 1 job in service" counts the jobs admitted earlier in the same call (as greedy's A does).
    - Every queued family must be in EXIT_MENU, checked at the start of each call (ContractError).
    - edgeserving: every queued request needs a deadline with D > 0 (ContractError, as exp2.policies); S values
      closer than 1e-9 are a tie (as exp2.policies).

Points the contract leaves open for coexit_lp, decided here (S3, owner 2026-10-26: PREEMPT = r waits for the token
boundary, then runs alone):
    - L may be paused only when now < its deadline and its next token boundary c* is not its last token (the server
      pauses only after an on-time, non-last token; a pause it would ignore must not block the admits).
    - "A pending pause -> no admit" counts a pending req_id only while that request is in service.
    - served time of every running request in service = now - t_admit - paused_ns (0 for coexit), in all options.
    - pause_ids(view) = (L,) when the admit_exits call at the same now chose PREEMPT; else ().
"""
from __future__ import annotations

import math
from collections import deque

from exp2 import contracts as C
from exp2.contracts import ContractError, PolicySpec

__all__ = ["ABLATIONS", "LP_METHODS", "METHODS", "SIM_ONLY", "ExitPolicy", "board_specs", "pilot_specs", "run3_specs",
           "sim_specs"]

METHODS = ("all_final", "all_early", "edgeserving", "greedy", "coexit")
ABLATIONS = ("coexit_nogate", "greedy_gate")
SIM_ONLY = ("edgeserving_lp",)
LP_METHODS = ("edgeserving_lp", "coexit_lp")
_NS = 1e9
_TIE_EPS = 1e-9      # edgeserving: S values closer than this are a tie (broken by earliest arrival)
_MENU = {fam: (f"{fam}:{lo}", f"{fam}:{hi}") for fam, (lo, hi) in C.EXIT_MENU.items()}   # family -> (floor, target)
_PREEMPT = object()  # coexit_lp option: pause L at its next token boundary, r runs alone after it (not admitted now)

_ROWS = {
    "all_final": {"params": {"exit_rule": "target"}},
    "all_early": {"params": {"exit_rule": "floor"}},
    "edgeserving": {"k_short": 1, "k_long": 1, "order": "edgeserving",
                    "params": {"C": C.EDGESERVING_C, "exit_rule": "edgeserving", "tau_stat": "p95"}},
    "greedy": {"params": {"exit_rule": "greedy", "tau_stat": "p95"}},
    "coexit": {"params": {"exit_rule": "coexit", "tau_stat": "p95", "cap_margin": C.COEXIT_CAP_MARGIN}},
    "coexit_nogate": {"params": {"exit_rule": "coexit", "tau_stat": "p95", "cap_margin": -1.0}},   # gate threshold 0
    "greedy_gate": {"params": {"exit_rule": "greedy", "tau_stat": "p95", "cap_margin": C.COEXIT_CAP_MARGIN}},
    "edgeserving_lp": {"k_short": 1, "k_long": 1, "order": "edgeserving",
                       "params": {"C": C.EDGESERVING_C, "exit_rule": "edgeserving", "tau_stat": "p95",
                                  "preempt_llama_at_token": True}},
    "coexit_lp": {"params": {"exit_rule": "coexit", "tau_stat": "p95", "cap_margin": C.COEXIT_CAP_MARGIN,
                             "preempt_llama_at_token": True}},
}


def _specs(names, f_mhz):
    base = {"k_short": None, "k_long": None, "order": "fcfs", "lane_cap": None, "f_rule": "static", "f_mhz": f_mhz}
    out = {}
    for name in names:
        spec = PolicySpec.from_json(dict(base, name=name, **_ROWS[name]))
        C.validate_policyspec(spec)
        out[name] = spec
    return out


def pilot_specs(f_mhz: int) -> dict:
    """The five pilot PolicySpecs (PLAN PL1 b1), validated, keyed by method name in METHODS order."""
    return _specs(METHODS, f_mhz)


def run3_specs(f_mhz: int) -> dict:
    """pilot_specs(f_mhz) plus the ABLATIONS specs, validated, keyed in METHODS + ABLATIONS order."""
    return _specs(METHODS + ABLATIONS, f_mhz)


def sim_specs(f_mhz: int) -> dict:
    """run3_specs(f_mhz) plus the SIM_ONLY specs, validated, keyed in METHODS + ABLATIONS + SIM_ONLY order."""
    return _specs(METHODS + ABLATIONS + SIM_ONLY, f_mhz)


def board_specs(f_mhz: int) -> dict:
    """run3_specs(f_mhz) plus the LP_METHODS specs, validated, keyed in METHODS + ABLATIONS + LP_METHODS order."""
    return _specs(METHODS + ABLATIONS + LP_METHODS, f_mhz)


class ExitPolicy:
    """Exit-choosing admission rule for one PolicySpec with params.exit_rule. Holds memo caches and the LP state
    .paused, .pause_pending, .paused_ns (filled by the dispatcher, or by the simulator for edgeserving_lp; empty
    otherwise)."""

    family_exclusive = True

    def __init__(self, spec, types, slowdown=None):
        if isinstance(spec, dict):
            spec = PolicySpec.from_json(spec)
        C.validate_policyspec(spec)
        rule = spec.params.get("exit_rule")
        if rule is None:
            raise ContractError("policyspec: ExitPolicy needs params.exit_rule")
        if rule in ("greedy", "coexit") and slowdown is None:
            raise ValueError(f"exit_rule {rule!r} needs a slowdown model")
        self.spec = spec
        self.types = types
        self.slowdown = slowdown
        self.rule = rule
        self.tau_stat = spec.params.get("tau_stat", "p95")
        self.C = float(spec.params.get("C", C.EDGESERVING_C))
        self.cap_margin = float(spec.params.get("cap_margin", 0.0))
        self.greedy_gate = rule == "greedy" and "cap_margin" in spec.params    # the greedy_gate ablation
        self.preempt = bool(spec.params.get("preempt_llama_at_token", False))   # LP_METHODS
        self.paused = {}         # req_id -> tokens_done of a paused in-service request (simulator: the paused job)
        self.pause_pending = set()   # req_id with a pause sent and neither paused nor done received yet
        self.paused_ns = {}      # req_id -> ns paused in its closed pauses (sum of t_resume - t_pause)
        self._s_memo = {}        # (type_id, sorted run types A, f) -> predicted slowdown
        self._pause_pick = None  # coexit_lp: (now_ns, req_id of L) when the last admit_exits chose PREEMPT

    def admit(self, view) -> list:
        return [rid for rid, _ in self.admit_exits(view)]

    def desired_f(self, view):
        return None

    def pause_ids(self, view) -> tuple:
        """req_ids to pause at the next token boundary, read on the view after the admits. () without preempt.
        edgeserving_lp: every LLaMA request in service with a deadline, not paused and with no pause pending, when a
        non-LLaMA request with no deadline or now < its deadline is queued (simulator._short_waiting).
        coexit_lp: (L,) when the admit_exits call at this now chose PREEMPT (it then admitted nothing), else ()."""
        if not self.preempt:
            return ()
        now = view.now_ns
        if self.rule == "coexit":
            pick = self._pause_pick
            return (pick[1],) if pick is not None and pick[0] == now else ()
        if not any(q.family != "llama" and (q.t_deadline_ns is None or now < q.t_deadline_ns) for q in view.queue):
            return ()
        return tuple(x.req_id for x in view.in_service if x.family == "llama" and x.t_deadline_ns is not None
                     and x.req_id not in self.paused and x.req_id not in self.pause_pending)

    def admit_exits(self, view) -> list:
        self._pause_pick = None
        queue = view.queue
        if not queue:
            return []
        for q in queue:
            _menu(q.family)
        f = view.f_current_mhz or self.types.f_max_mhz
        if self.rule == "edgeserving":
            return self._edgeserving(view, f)
        if self.preempt and any(x.req_id in self.pause_pending for x in view.in_service):
            return []                                   # coexit_lp: a pending pause -> no admit until the tick
        busy = {x.family for x in view.in_service}      # a paused request keeps its family busy
        run = [x.type_id for x in view.in_service]       # run types in service, incl. this call's admits
        admitted, out = [], []
        for r in queue:
            if r.family in busy:
                continue
            busy.add(r.family)                          # an admit makes it busy; a WAIT keeps the family's turn
            floor, target = _menu(r.family)
            if self.rule == "target":
                pick = target
            elif self.rule == "floor":
                pick = floor
            elif self.rule == "greedy":
                if self.greedy_gate and self._gate_waits(run, target, f):
                    continue                            # WAIT; the family keeps its turn
                done = view.now_ns + self._tau(target, f) * self._s(target, run + [target], f) * _NS
                pick = target if r.t_deadline_ns is None or done <= r.t_deadline_ns else floor
            else:
                pick = self._coexit(view, f, r, admitted)
                if pick is _PREEMPT:
                    break                               # pause L; r keeps its turn and later candidates WAIT
                if pick is None:
                    continue
            out.append((r.req_id, pick))
            admitted.append((r, pick))
            run.append(pick)
        return out

    # ---------------------------------------------------------------------------------------- helpers
    def _tau(self, type_id, f):
        return self.types.tau(type_id, f, self.tau_stat)

    def _s(self, type_id, A, f):
        key = (type_id, tuple(sorted(A)), f)
        v = self._s_memo.get(key)
        if v is None:
            v = self._s_memo[key] = float(self.slowdown.predict(type_id, key[1], f))
        return v

    def _gate_waits(self, A, target, f):
        """Capacity gate (coexit, greedy_gate): True (WAIT) when >= 1 job runs (A = run types in service incl. this
        call's admits) and the predicted capacity of A + target, sum of 1/s_t, is below 1 + cap_margin."""
        if not A:
            return False
        A = A + [target]
        return sum(1.0 / self._s(t, A, f) for t in A) < 1.0 + self.cap_margin

    # ---------------------------------------------------------------------------------------- coexit
    def _coexit(self, view, f, r, admitted):
        """Run type for r, None for WAIT, or _PREEMPT (coexit_lp: pause L, recorded in self._pause_pick).
        coexit: options = [target if r on time] + [floor] + [WAIT if a job runs].
        coexit_lp: a paused request (self.paused) is not in A, the gate or the running jobs, and is no job for WAIT;
        it waits in the rollout as held. While only L runs (_lp_cut): PREEMPT(target) if r on time, co-run(target)
        if the gate passes and r on time, PREEMPT(floor), co-run(floor) if the gate passes, WAIT."""
        floor, target = _menu(r.family)
        now = view.now_ns
        live = [x for x in view.in_service if x.req_id not in self.paused]
        a0 = [x.type_id for x in live]
        gate = not self._gate_waits(a0 + [t for _, t in admitted], target, f)
        cut = self._lp_cut(f, now, live, a0) if self.preempt and not admitted else None
        if not gate and cut is None:
            return None                             # co-running gains less than the margin: wait (serial)
        running = []                                    # [type, family, solo seconds left, deadline - now (s)]
        for x in live:
            served = (now - x.t_admit_ns - self.paused_ns.get(x.req_id, 0)) / _NS
            left = self._tau(x.type_id, f) - served / self._s(x.type_id, a0, f)
            running.append((x.type_id, x.family, max(0.0, left), _rel(x.t_deadline_ns, now)))
        for q, tid in admitted:
            running.append((tid, q.family, self._tau(tid, f), _rel(q.t_deadline_ns, now)))
        held = [(x.type_id, x.family, self._tau(x.type_id, f) * (1.0 - self._frac(x.type_id, self.paused[x.req_id])),
                 _rel(x.t_deadline_ns, now)) for x in view.in_service if x.req_id in self.paused]
        taken = {q.req_id for q, _ in admitted}
        others = [q for q in view.queue if q is not r and q.req_id not in taken]
        options = []
        if cut is not None:                             # PREEMPT: on [0, d] only L runs, then L is held, r starts
            d, w_after, L = cut
            held_cut = held + [(L.type_id, L.family, w_after, _rel(L.t_deadline_ns, now))]
            late_p, r_on_time = self._rollout(f, now, [], r, target, others, held_cut, d)
            if r_on_time:
                options.append((late_p, _PREEMPT))
        if gate:
            late_t, r_on_time = self._rollout(f, now, running, r, target, others, held)
            if r_on_time:
                options.append((late_t, target))
        if cut is not None:
            options.append((self._rollout(f, now, [], r, floor, others, held_cut, d)[0], _PREEMPT))
        if gate:
            options.append((self._rollout(f, now, running, r, floor, others, held)[0], floor))
        if running:
            options.append((self._rollout(f, now, running, r, None, others, held)[0], None))
        best = min(n for n, _ in options)
        pick = next(o for n, o in options if n == best)
        if pick is _PREEMPT:
            self._pause_pick = (now, cut[2].req_id)
        return pick

    def _frac(self, type_id, ckpt):
        return self.types.get(type_id).ckpt_frac[ckpt]

    def _lp_cut(self, f, now, live, a0):
        """coexit_lp, only L runs: (d, w_after, L) or None. L = the one running request, LLaMA, with a deadline and
        now < its deadline. served_L = (now - t_admit - paused_ns[L]) / s_L(A0), p = served_L / tau_L, c* = first ckpt
        with ckpt_frac > p, which must not be the last token. d = (frac[c*] - p) * tau_L * s_L(A0) seconds to the
        boundary; w_after = tau_L * (1 - frac[c*]) solo seconds left after the pause."""
        if len(live) != 1:
            return None
        L = live[0]
        if L.family != "llama" or L.t_deadline_ns is None or now >= L.t_deadline_ns:
            return None
        t = self.types.get(L.type_id)
        tau, s = self._tau(L.type_id, f), self._s(L.type_id, a0, f)
        p = (now - L.t_admit_ns - self.paused_ns.get(L.req_id, 0)) / _NS / s / tau
        c = next((c for c in t.ckpts if t.ckpt_frac[c] > p), None)
        if c is None or c == t.ckpts[-1]:
            return None                                 # L is at its last token: the server will not pause it
        frac = t.ckpt_frac[c]
        return (frac - p) * tau * s, tau * (1.0 - frac), L

    def _rollout(self, f, now, running, r, r_type, others, held=(), t0=0.0):
        """Deterministic fluid look-ahead, no future arrivals. r_type None = WAIT: r heads its family queue and the
        family is blocked until the first completion. Every other queued request starts at its floor exit at the
        first instant its family has no active job. held (coexit_lp) = paused jobs [type, family, solo seconds left,
        deadline - now]: they do not advance, keep their family busy and resume when no job is active (after the
        free families start). t0 (PREEMPT) = the time L pauses: r and the queued requests start at t0, L is in held.
        -> (number of late jobs, r on time)."""
        active = [[tid, fam, w, dl, False] for tid, fam, w, dl in running]
        held = [[tid, fam, w, dl, False] for tid, fam, w, dl in held]
        blocked = None
        if r_type is None:
            blocked = r.family
            waiting = [r] + list(others)
        else:
            active.append([r_type, r.family, self._tau(r_type, f), _rel(r.t_deadline_ns, now), True])
            waiting = list(others)
        pending = {}                                    # family -> FIFO of queued requests (r first when it waits)
        for q in waiting:
            pending.setdefault(q.family, deque()).append(q)

        def start_free():
            busy = {j[1] for j in active} | {j[1] for j in held}
            free = [fam for fam, fifo in pending.items() if fifo and fam not in busy and fam != blocked]
            for fam in sorted(free, key=lambda fam: pending[fam][0].t_arrival_ns):     # order of head arrival
                q = pending[fam].popleft()
                tid = _menu(fam)[0]
                active.append([tid, fam, self._tau(tid, f), _rel(q.t_deadline_ns, now), q is r])
            if held and not active:                     # a paused job resumes when no job is active
                active.extend(held)
                del held[:]

        n_late, r_on_time, t = 0, None, t0
        start_free()
        while active:
            A = [j[0] for j in active]
            s = [self._s(j[0], A, f) for j in active]
            finish = [j[2] * sj for j, sj in zip(active, s)]
            dt = min(finish)
            t += dt
            still = []
            for j, sj, fin in zip(active, s, finish):
                if fin <= dt:
                    late = j[3] is not None and t > j[3]
                    n_late += late
                    if j[4]:
                        r_on_time = not late
                else:
                    j[2] -= dt / sj
                    still.append(j)
            active = still
            blocked = None
            start_free()
        if any(pending.values()):
            raise RuntimeError("coexit rollout stalled with queued requests left")
        return n_late, r_on_time

    # ---------------------------------------------------------------------------------------- edgeserving
    def _edgeserving(self, view, f):
        """k = 1: one admit, only when nothing is in service. Family head h runs at e*(h) = target if
        now + tau(target) <= deadline(h), else floor; admit argmin S(h), tie -> earliest arrival.
        edgeserving_lp: a request in self.paused does not count as in service, and its family is not a candidate."""
        if any(x.req_id not in self.paused for x in view.in_service):
            return []
        now = view.now_ns
        info = []
        for q in view.queue:
            if q.t_deadline_ns is None:
                raise ContractError(f"edgeserving needs a deadline on every queued request; {q.req_id} has none")
            d = (q.t_deadline_ns - q.t_arrival_ns) / _NS
            if not d > 0:
                raise ContractError(f"edgeserving: {q.req_id} has a non-positive relative deadline {d}")
            info.append(((now - q.t_arrival_ns) / _NS, d))
        clip = self.C
        log_clip = math.log(clip)
        best, seen = None, {x.family for x in view.in_service}    # only paused jobs are left in service
        for h, r in enumerate(view.queue):
            if r.family in seen:
                continue
            seen.add(r.family)
            floor, target = _menu(r.family)
            e = target if now + self._tau(target, f) * _NS <= r.t_deadline_ns else floor
            tau = self._tau(e, f)
            score = 0.0
            for j, (wait, d) in enumerate(info):
                if j != h:
                    x = (wait + tau) / d - 1.0
                    score += clip if x >= log_clip else math.exp(x)
            key = (round(score / _TIE_EPS), r.t_arrival_ns, h)
            if best is None or key < best[0]:
                best = (key, r.req_id, e)
        return [] if best is None else [(best[1], best[2])]


def _menu(family):
    """(floor type, target type) of a family; ContractError outside EXIT_MENU."""
    try:
        return _MENU[family]
    except KeyError:
        raise ContractError(f"exit choice: family {family!r} is not in EXIT_MENU {sorted(C.EXIT_MENU)}") from None


def _rel(t_ns, now_ns):
    """Seconds from now to t_ns; None stays None (no deadline: never late)."""
    return None if t_ns is None else (t_ns - now_ns) / _NS
