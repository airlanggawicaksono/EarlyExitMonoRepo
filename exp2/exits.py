"""exp2.exits: exit-choosing admission policies for the CoExit pilot (PLAN PL1; S0 skeleton, rules filled by S1).

Semantics are frozen in the docstring of exp2.contracts ("Exit choice"). An ExitPolicy offers
admit_exits(view) -> [(req_id, run_type_id)]; AdmissionCore (exp2.policies) records the pairs in
Decision.run_type_ids and puts the run type in service. Policies in exp2.policies are not touched.

    ExitPolicy(spec, types, slowdown=None)   .admit_exits(view) -> [(req_id, run_type_id)]   .admit(view) -> [req_id]
    pilot_specs(f_mhz) -> {method: PolicySpec}
    run3_specs(f_mhz)  -> pilot_specs(f_mhz) + the ABLATIONS specs (Run 3 board)
    METHODS = the five pilot methods in report order
    ABLATIONS = ("coexit_nogate",): coexit with params.cap_margin -1 (the capacity gate never blocks)

Points the contract leaves open, decided here (S1):
    - A WAIT keeps its family's turn: the requests of that family behind it are not candidates in the same call
      (FIFO per family, as in the other four methods; so "r heads its family queue" always holds).
    - "WAIT if >= 1 job in service" counts the jobs admitted earlier in the same call (as greedy's A does).
    - Every queued family must be in EXIT_MENU, checked at the start of each call (ContractError).
    - edgeserving: every queued request needs a deadline with D > 0 (ContractError, as exp2.policies); S values
      closer than 1e-9 are a tie (as exp2.policies).
"""
from __future__ import annotations

import math
from collections import deque

from exp2 import contracts as C
from exp2.contracts import ContractError, PolicySpec

__all__ = ["ABLATIONS", "METHODS", "ExitPolicy", "pilot_specs", "run3_specs"]

METHODS = ("all_final", "all_early", "edgeserving", "greedy", "coexit")
ABLATIONS = ("coexit_nogate",)
_NS = 1e9
_TIE_EPS = 1e-9      # edgeserving: S values closer than this are a tie (broken by earliest arrival)
_MENU = {fam: (f"{fam}:{lo}", f"{fam}:{hi}") for fam, (lo, hi) in C.EXIT_MENU.items()}   # family -> (floor, target)

_ROWS = {
    "all_final": {"params": {"exit_rule": "target"}},
    "all_early": {"params": {"exit_rule": "floor"}},
    "edgeserving": {"k_short": 1, "k_long": 1, "order": "edgeserving",
                    "params": {"C": C.EDGESERVING_C, "exit_rule": "edgeserving", "tau_stat": "p95"}},
    "greedy": {"params": {"exit_rule": "greedy", "tau_stat": "p95"}},
    "coexit": {"params": {"exit_rule": "coexit", "tau_stat": "p95", "cap_margin": C.COEXIT_CAP_MARGIN}},
    "coexit_nogate": {"params": {"exit_rule": "coexit", "tau_stat": "p95", "cap_margin": -1.0}},   # gate threshold 0
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


class ExitPolicy:
    """Exit-choosing admission rule for one PolicySpec with params.exit_rule. Holds only memo caches."""

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
        self._s_memo = {}        # (type_id, sorted run types A, f) -> predicted slowdown

    def admit(self, view) -> list:
        return [rid for rid, _ in self.admit_exits(view)]

    def desired_f(self, view):
        return None

    def admit_exits(self, view) -> list:
        queue = view.queue
        if not queue:
            return []
        for q in queue:
            _menu(q.family)
        f = view.f_current_mhz or self.types.f_max_mhz
        if self.rule == "edgeserving":
            return self._edgeserving(view, f)
        busy = {x.family for x in view.in_service}
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
                done = view.now_ns + self._tau(target, f) * self._s(target, run + [target], f) * _NS
                pick = target if r.t_deadline_ns is None or done <= r.t_deadline_ns else floor
            else:
                pick = self._coexit(view, f, r, admitted)
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

    # ---------------------------------------------------------------------------------------- coexit
    def _coexit(self, view, f, r, admitted):
        """Run type for r, or None for WAIT. options = [target if r on time] + [floor] + [WAIT if a job runs]."""
        floor, target = _menu(r.family)
        A = [x.type_id for x in view.in_service] + [t for _, t in admitted]
        if A:
            A.append(target)
            if sum(1.0 / self._s(t, A, f) for t in A) < 1.0 + self.cap_margin:
                return None                         # co-running gains less than the margin: wait (serial)
        now = view.now_ns
        a0 = [x.type_id for x in view.in_service]
        running = []                                    # [type, family, solo seconds left, deadline - now (s), is_r]
        for x in view.in_service:
            left = self._tau(x.type_id, f) - (now - x.t_admit_ns) / _NS / self._s(x.type_id, a0, f)
            running.append((x.type_id, x.family, max(0.0, left), _rel(x.t_deadline_ns, now)))
        for q, tid in admitted:
            running.append((tid, q.family, self._tau(tid, f), _rel(q.t_deadline_ns, now)))
        taken = {q.req_id for q, _ in admitted}
        others = [q for q in view.queue if q is not r and q.req_id not in taken]
        late_t, r_on_time = self._rollout(f, now, running, r, target, others)
        options = [(late_t, target)] if r_on_time else []
        options.append((self._rollout(f, now, running, r, floor, others)[0], floor))
        if running:
            options.append((self._rollout(f, now, running, r, None, others)[0], None))
        best = min(n for n, _ in options)
        return next(o for n, o in options if n == best)

    def _rollout(self, f, now, running, r, r_type, others):
        """Deterministic fluid look-ahead, no future arrivals. r_type None = WAIT: r heads its family queue and the
        family is blocked until the first completion. Every other queued request starts at its floor exit at the
        first instant its family has no active job. -> (number of late jobs, r on time)."""
        active = [[tid, fam, w, dl, False] for tid, fam, w, dl in running]
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
            busy = {j[1] for j in active}
            free = [fam for fam, fifo in pending.items() if fifo and fam not in busy and fam != blocked]
            for fam in sorted(free, key=lambda fam: pending[fam][0].t_arrival_ns):     # order of head arrival
                q = pending[fam].popleft()
                tid = _menu(fam)[0]
                active.append([tid, fam, self._tau(tid, f), _rel(q.t_deadline_ns, now), q is r])

        n_late, r_on_time, t = 0, None, 0.0
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
        now + tau(target) <= deadline(h), else floor; admit argmin S(h), tie -> earliest arrival."""
        if view.in_service:
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
        best, seen = None, set()
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
        return [(best[1], best[2])]


def _menu(family):
    """(floor type, target type) of a family; ContractError outside EXIT_MENU."""
    try:
        return _MENU[family]
    except KeyError:
        raise ContractError(f"exit choice: family {family!r} is not in EXIT_MENU {sorted(C.EXIT_MENU)}") from None


def _rel(t_ns, now_ns):
    """Seconds from now to t_ns; None stays None (no deadline: never late)."""
    return None if t_ns is None else (t_ns - now_ns) / _NS
