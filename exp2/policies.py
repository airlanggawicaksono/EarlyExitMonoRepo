"""exp2.policies: admission policies and the single state-to-view object (BRIEF_v1 K4, S1b).

Semantics are frozen in the docstring of exp2.contracts ("Admission semantics").  The dispatcher
(exp2.serve.dispatcher) and the simulator (exp2.model.simulator) both call AdmissionCore, so a trace admits
identically in the fake server and in the model.

    Policy(spec, types, *, family_exclusive=True)   .admit(view) -> [req_id]   .desired_f(view) -> int | None
    AdmissionCore(policy, types, f_initial_mhz)     .arrive / .done -> Decision      .view(now_ns) -> DispatchView
    registry(types) -> {key: PolicySpec}

Decision fields (decided here, contracts left the element types open):
    admitted               tuple of req_id, in admit order
    expired                tuple of req_id dropped from the queue at this event (queue order); the caller records
                           outcome expired_in_queue with t_done_ns = that request's t_deadline_ns
    commanded_f            the NEW commanded GPU level in MHz, or None when no new command was issued
    n_in_service_at_admit  tuple of int aligned with `admitted`: requests in service just before that admit
"""
from __future__ import annotations

import math
from operator import attrgetter
from dataclasses import dataclass

from exp2 import contracts as C
from exp2.contracts import ContractError, DispatchView, InServiceReq, PolicySpec

__all__ = ["Policy", "AdmissionCore", "Decision", "registry"]

_FAMILY_SET = frozenset(C.FAMILIES)
_TIE_EPS = 1e-9      # edgeserving: S values closer than this are a tie (broken by earliest arrival)
_NS = 1e9
_K_OVERRIDE = attrgetter("k_override")
_NO_OVERRIDE = {None}


class Policy:
    """Pure, deterministic admission rule for one PolicySpec.  Holds no per-call state."""

    def __init__(self, spec, types, *, family_exclusive: bool = True):
        if isinstance(spec, dict):
            spec = PolicySpec.from_json(spec)
        C.validate_policyspec(spec)
        self.spec = spec
        self.types = types
        self.family_exclusive = bool(family_exclusive)
        self._p50_fmax = {}      # type_id -> tau p50 at f_max (stf key cache)
        self._tau_ns = {}        # (type_id, f) -> tau p50 in integer ns (greenllm cache)

    # ---------------------------------------------------------------------------------------- admit
    def admit(self, view: DispatchView) -> list:
        queue = view.queue
        if not queue:
            return []
        spec = self.spec
        n = len(view.in_service)
        k_cls = {"short": C.k_of(spec, "short"), "long": C.k_of(spec, "long")}
        k_max = max(k_cls.values())
        # no request can pass `n < k` once n >= k_max, unless a queued request carries a k_override
        if n >= k_max and set(map(_K_OVERRIDE, queue)) == _NO_OVERRIDE:
            return []
        exclusive = self.family_exclusive
        busy = {r.family for r in view.in_service} if exclusive else None
        if exclusive and busy >= _FAMILY_SET:
            return []
        lane_cap = spec.lane_cap
        used = None
        if lane_cap is not None:
            used = {"short": 0, "long": 0}
            for r in view.in_service:
                used[r.klass] = used.get(r.klass, 0) + 1
        out = []
        for r in self._candidates(view):
            if exclusive:
                if r.family in busy:
                    continue
            ko = r.k_override
            if n >= (k_cls[r.klass] if ko is None else ko):
                continue
            if used is not None and used.get(r.klass, 0) >= lane_cap[r.klass]:
                continue
            out.append(r.req_id)
            n += 1
            if used is not None:
                used[r.klass] = used.get(r.klass, 0) + 1
            if exclusive:
                busy.add(r.family)
                if busy >= _FAMILY_SET:
                    break
        return out

    def _candidates(self, view):
        order = self.spec.order
        queue = view.queue
        if order == "fcfs":
            return queue
        if order == "class":
            return sorted(queue, key=lambda r: 0 if r.klass == "short" else 1)      # stable: then arrival
        if order == "stf":
            return sorted(queue, key=lambda r: self._stf_key(r.type_id))             # stable: then arrival
        return self._edgeserving_candidates(view)

    def _stf_key(self, type_id):
        v = self._p50_fmax.get(type_id)
        if v is None:
            v = self._p50_fmax[type_id] = self.types.tau(type_id, self.types.f_max_mhz, "p50")
        return v

    def _edgeserving_candidates(self, view):
        """Per-type queue heads ordered by (S, arrival); S(h) = sum over the other queued r of
        min(exp((now - a_r + tau_h^p95(f)) / D_r - 1), C)."""
        now = view.now_ns
        f = view.f_current_mhz if view.f_current_mhz is not None else self.types.f_max_mhz
        clip = float(self.spec.params.get("C", C.EDGESERVING_C))
        log_clip = math.log(clip)
        info = []
        for r in view.queue:
            if r.t_deadline_ns is None:
                raise ContractError(f"edgeserving needs a deadline on every queued request; {r.req_id} has none")
            d = (r.t_deadline_ns - r.t_arrival_ns) / _NS
            if not d > 0:
                raise ContractError(f"edgeserving: {r.req_id} has a non-positive relative deadline {d}")
            info.append(((now - r.t_arrival_ns) / _NS, d))
        heads, seen = [], set()
        for i, r in enumerate(view.queue):
            if r.type_id not in seen:
                seen.add(r.type_id)
                heads.append(i)
        scored = []
        for h in heads:
            r = view.queue[h]
            tau = self.types.tau(r.type_id, f, "p95")
            s = 0.0
            for j, (wait, d) in enumerate(info):
                if j == h:
                    continue
                x = (wait + tau) / d - 1.0
                s += clip if x >= log_clip else math.exp(x)
            scored.append((round(s / _TIE_EPS), r.t_arrival_ns, h, r))
        scored.sort(key=lambda t: t[:3])
        return [t[3] for t in scored]

    # ---------------------------------------------------------------------------------------- desired_f
    def desired_f(self, view: DispatchView):
        rule = self.spec.f_rule
        if rule == "occupancy":
            return self._occupancy(view)
        if rule == "greenllm":
            return self._greenllm(view)
        return None                                  # static, governor

    def _occupancy(self, view):
        p = self.spec.params
        n_sys = len(view.queue) + len(view.in_service)
        threshold = p["f_base_mhz"] * n_sys ** (1.0 / p["alpha_p"])
        levels = sorted(self.types.corun_levels_mhz)
        for f in levels:
            if f >= threshold:
                return f
        return levels[-1]

    def _tau_p50_ns(self, type_id, f):
        key = (type_id, f)
        v = self._tau_ns.get(key)
        if v is None:
            v = self._tau_ns[key] = int(round(self.types.tau(type_id, f, "p50") * _NS))
        return v

    def _greenllm(self, view):
        types = self.types
        levels = sorted(types.corun_levels_mhz)
        queue = view.queue
        if not queue:
            return levels[0]
        now = view.now_ns
        f_cur = view.f_current_mhz if view.f_current_mhz is not None else types.f_max_mhz
        # (type_id, unfinished share) for every request in service: 1 - elapsed / tau_in(f_current)
        busy = []
        for r in view.in_service:
            tau_cur = types.tau(r.type_id, f_cur, "p50")
            busy.append((r.type_id, 1.0 - ((now - r.t_admit_ns) / _NS) / tau_cur))
        for f in levels:
            # the GPU is shared: the unfinished work of EVERY request in service delays a queued request,
            # whatever its lane (golden k20c, T6)
            residual = 0
            for tid, left in busy:
                if left > 0.0:
                    residual += int(round(types.tau(tid, f, "p50") * left * _NS))
            ahead = {}
            feasible = True
            for r in queue:
                ahead[r.klass] = ahead.get(r.klass, 0) + self._tau_p50_ns(r.type_id, f)
                if r.t_deadline_ns is None:
                    continue
                lhs = (now - r.t_arrival_ns) + ahead[r.klass] + residual
                if lhs > r.t_deadline_ns - r.t_arrival_ns:
                    feasible = False
                    break
            if feasible:
                return f
        return types.f_max_mhz


# ------------------------------------------------------------------------------------------------ core
@dataclass(frozen=True)
class Decision:
    admitted: tuple = ()
    expired: tuple = ()
    commanded_f: int | None = None
    n_in_service_at_admit: tuple = ()


class AdmissionCore:
    """The one place that turns events into policy views.  Not thread safe: the caller serialises events.

    Per event: apply the event, drop queued requests with now_ns >= t_deadline_ns, policy.admit(view),
    policy.desired_f(view after the admit).  A raising policy leaves the core unchanged (the event is not applied).
    """

    def __init__(self, policy, types, f_initial_mhz):
        self.policy = policy
        self.types = types
        self._queue = []                 # QueuedReq, arrival order
        self._in_service = {}            # req_id -> InServiceReq
        self._ids = set()                # req_id of every queued or in-service request
        self._min_dl = None              # earliest t_deadline_ns in the queue (None: no queued deadline)
        self._f_cmd = f_initial_mhz      # last COMMANDED level, never a read-back

    def view(self, now_ns: int) -> DispatchView:
        return DispatchView(now_ns, tuple(self._queue), tuple(self._in_service.values()), self._f_cmd)

    def arrive(self, req, now_ns: int) -> Decision:
        if req.req_id in self._ids:
            raise ValueError(f"duplicate req_id {req.req_id!r}")
        min_dl = self._min_dl
        if req.t_deadline_ns is not None and (min_dl is None or req.t_deadline_ns < min_dl):
            min_dl = req.t_deadline_ns
        dec = self._step(self._queue + [req], dict(self._in_service), min_dl, now_ns)
        self._ids.add(req.req_id)
        self._ids.difference_update(dec.expired)
        return dec

    def done(self, req_id: str, now_ns: int) -> Decision:
        if req_id not in self._in_service:
            raise KeyError(f"done() for {req_id!r}, which is not in service")
        in_service = dict(self._in_service)
        del in_service[req_id]
        dec = self._step(list(self._queue), in_service, self._min_dl, now_ns)
        self._ids.discard(req_id)
        self._ids.difference_update(dec.expired)
        return dec

    def _step(self, queue, in_service, min_dl, now_ns):
        """Decide on copies, then commit.  Returns the Decision."""
        f_cmd = self._f_cmd
        expired = ()
        if min_dl is not None and now_ns >= min_dl:
            expired = tuple(r.req_id for r in queue if r.t_deadline_ns is not None and now_ns >= r.t_deadline_ns)
            gone = set(expired)
            queue = [r for r in queue if r.req_id not in gone]
        view = DispatchView(now_ns, tuple(queue), tuple(in_service.values()), f_cmd)
        chosen = list(self.policy.admit(view))
        admitted, n_at = [], []
        if chosen:
            by_id = {r.req_id: r for r in queue}
            for rid in chosen:
                r = by_id.get(rid)
                if r is None or rid in in_service:
                    raise RuntimeError(f"policy admitted {rid!r}, which is not a queued request (or admitted twice)")
                n_at.append(len(in_service))
                in_service[rid] = InServiceReq(rid, r.type_id, r.family, r.klass, now_ns)
                admitted.append(rid)
            taken = set(admitted)
            queue = [r for r in queue if r.req_id not in taken]
            view = DispatchView(now_ns, tuple(queue), tuple(in_service.values()), f_cmd)
        want = self.policy.desired_f(view)
        commanded = want if want is not None and want != f_cmd else None
        if expired or admitted:
            min_dl = min((r.t_deadline_ns for r in queue if r.t_deadline_ns is not None), default=None)
        self._queue, self._in_service, self._min_dl = queue, in_service, min_dl
        if commanded is not None:
            self._f_cmd = commanded
        return Decision(tuple(admitted), expired, commanded, tuple(n_at))


# ------------------------------------------------------------------------------------------------ registry
def registry(types) -> dict:
    """The named policies of the study (policyspecs.json), built for this type table (f_max, lowest corun level)."""
    fmax = types.f_max_mhz
    lo = min(types.corun_levels_mhz)

    def mk(name, **kw):
        d = {"name": name, "k_short": None, "k_long": None, "order": "fcfs", "lane_cap": None,
             "f_rule": "static", "f_mhz": fmax, "params": {}}
        d.update(kw)
        spec = PolicySpec.from_json(d)
        C.validate_policyspec(spec)
        return spec

    specs = [
        mk("default", f_rule="governor", f_mhz=None),
        mk("default_stock", f_rule="governor", f_mhz=None),
        mk("serial", k_short=1, k_long=1),
        mk("serial_stf", k_short=1, k_long=1, order="stf"),
        mk("serial_class", k_short=1, k_long=1, order="class"),
        mk("serial_edgeserving", k_short=1, k_long=1, order="edgeserving", params={"C": 10}),
        mk("corun"),
        mk("greenllm_lanes", k_short=2, k_long=2, lane_cap={"short": 1, "long": 1}, f_rule="greenllm", f_mhz=None),
        mk("occupancy_f", f_rule="occupancy", f_mhz=None, params={"f_base_mhz": lo, "alpha_p": 2}),
    ]
    return {s.name: s for s in specs}
