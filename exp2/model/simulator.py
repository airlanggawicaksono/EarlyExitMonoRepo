"""exp2.model.simulator: agent-based simulation of the serving system (BRIEF_v1 K14, slice E2).

    simulate(workload, spec, types, slowdown, power, cfg, policy=None) -> contracts.SimResult

What is simulated (every admit decision goes through exp2.policies.AdmissionCore, every outcome through
contracts.truncation_outcome, so the simulator and the real dispatcher cannot disagree on K4 or K7):

    arrivals    arrivals.schedule(workload, n, seed) (PCG64) + contracts.with_deadlines(alpha); or cfg.schedule (replay)
    job size    one uniform quantile per request, then tau_s[f].samples at that quantile (comonotone over f)
    run type    a request runs as the type it queued with, or as Decision.run_type_ids[k] when the policy offers
                admit_exits() (exp2.exits). tau, checkpoints, head, A and the completion are those of the run type;
                per_type stays keyed by the queued type; trace rows add "run_type_id" (None for expired_in_queue).
    progress    a job advances at 1 / s solo-seconds per second, s = slowdown(type, A, f), A = sorted tuple of the
                run type ids in service (with multiplicity). s is re-sampled for every type present each time A or f
                changes.
    truncation  checkpoints are crossed at ckpt_frac x solo time; the first checkpoint later than the deadline stops
                the segments (that segment keeps the GPU), then the head (head_frac) runs for `reached`; a miss has no
                head; a complete job ends at fraction 1.0. A request with no deadline only tracks its last checkpoint.
    preempt     spec.params.preempt_llama_at_token (edgeserving_lp, contracts "Exit choice"; needs exits.ExitPolicy):
                a LLaMA job that crosses a token checkpoint on time (not its last) while a non-LLaMA request with
                now < deadline is queued pauses there: it leaves A, the power set and its heap (progress and run
                type kept, still counted in N), its req_id goes to policy.paused, and the core decides at once
                (_LPCore.tick). After a completion with no job running, it resumes from the same token. Cost 0.
    expiry      a queued request with now >= deadline is dropped lazily at the next event (AdmissionCore), recorded
                with t_done = t_deadline, completion 0. Nothing is served for it.
    frequency   static spec -> spec.f_mhz; governor/greenllm/occupancy start at f_max (a governor is not modelled: it
                runs at f_max). Commands come from Decision.commanded_f and take effect cfg.switch_lag_s later; while a
                switch is in flight only the newest command waits and is applied right after it (actuator thread).
    power       energy += power.p(A, f) dt over the whole run, idle included.
    window      from the first non-warmup arrival to the last non-warmup t_done (same as stats.request_window).
                energy_per_req_j = window energy / non-warmup requests (all outcomes); mean_power_w = energy / window;
                mean_in_system, utilization (fraction of the window with >= 1 job in service) and throughput
                (outcome == complete per second) use the same window.
    CRN         rep r uses seed cfg.seed + r for three independent PCG64 streams: arrivals, job-size quantiles and
                slowdown sampling. Arrivals and sizes depend only on (workload, seed), never on the policy.
    n_slope     OLS slope (requests/s) of N(t) = queued + in service over the second half of [0, last arrival]
                (exact for the piecewise-constant path). unstable iff cfg.alpha is None and at least
                ceil(0.8 * n_reps) reps have n_slope > UNSTABLE_SLOPE_FRAC * workload.rate_per_s.
    aggregation per_type / tagged statistics are the mean over reps of the per-rep statistics (n is the total over
                reps); response time R = t_done - t_arrival for every non-warmup request, whatever its outcome.

The event clock is an int ns. Jobs of one type share a virtual clock W (solo-seconds served) so the cost of an event
is O(types present + log jobs) however many jobs are in service. numpy + stdlib only.
"""
import heapq
import math
import time
from bisect import bisect_right

import numpy as np

from exp2 import arrivals
from exp2 import contracts as C
from exp2 import stats
from exp2.contracts import QueuedReq, SimResult, TypeStats
from exp2.policies import AdmissionCore, Policy

__all__ = ["simulate"]

_NS = 1e9
_EPS_FRAC = 1e-12           # a head shorter than this share of the job is not simulated
_STAT_KEYS = ("mean_R", "p50_R", "p95_R", "p99_R", "complete_rate", "truncation_rate", "miss_rate", "expired_rate",
              "mean_completion", "mean_shortfall")
_CODES = {"complete": 0, "truncated": 1, "miss": 2, "expired_in_queue": 3}


class _TypeInfo:
    """Everything the event loop needs about one request type, read once from contracts.TypeSpec."""
    __slots__ = ("tid", "family", "exit", "klass", "ckpts", "fr", "head", "samples", "_compl")

    def __init__(self, ts):
        self.tid, self.family, self.exit, self.klass = ts.type_id, ts.family, ts.exit, ts.klass
        self.ckpts = tuple(ts.ckpts)
        self.fr = tuple(float(ts.ckpt_frac[c]) for c in self.ckpts)
        self.head = float(ts.head_frac)
        self.samples = {int(f): sorted(float(x) for x in v["samples"]) for f, v in ts.tau_s.items()}
        self._compl = {}

    def tau(self, f_mhz, u):
        """Solo duration (s) at f_mhz of the job with size quantile u in [0, 1)."""
        lst = self.samples.get(f_mhz)
        if lst is None:
            raise C.ContractError("%s has no tau samples at %s MHz (have %s)" % (self.tid, f_mhz, sorted(self.samples)))
        return lst[int(u * len(lst))]

    def completion(self, reached):
        v = self._compl.get(reached)
        if v is None:
            v = self._compl[reached] = C.completion(self.family, self.exit, reached)
        return v


class _Job:
    """A request in service. Progress is p0 + (W[type] - w0) / tau, a fraction of the solo time (tau is solo time at
    the current f). kind 0: next event is checkpoint ck (target tgt); kind 1: next event is the end of the job."""
    __slots__ = ("i", "ti", "u", "tau", "p0", "w0", "tgt", "ck", "kind", "dl", "t_ck", "reached", "outcome")


class _LPCore(AdmissionCore):
    """AdmissionCore plus a decision at a LLaMA token boundary (edgeserving_lp only): the _step of arrive/done."""

    def tick(self, now_ns):
        dec = self._step(list(self._queue), dict(self._in_service), self._min_dl, now_ns)
        self._ids.difference_update(dec.expired)
        return dec


class _Sim:
    """Per-simulate() context shared by all reps."""

    def __init__(self, spec, types, slowdown, power, cfg, policy):
        self.spec, self.types, self.slowdown, self.power, self.cfg = spec, types, slowdown, power, cfg
        self.policy = policy
        self.lp = bool(spec.params.get("preempt_llama_at_token"))           # edgeserving_lp
        if self.lp and not getattr(policy, "preempt", False):
            raise ValueError("spec %r pauses LLaMA at token boundaries: it needs the exits.ExitPolicy of that spec"
                             % spec.name)
        if spec.f_mhz is not None:
            self.f_cmd0 = spec.f_mhz
        else:
            self.f_cmd0 = None if spec.f_rule == "governor" else types.f_max_mhz
        self.f0 = self.f_cmd0 if self.f_cmd0 is not None else types.f_max_mhz
        self._info = {}

    def info(self, tid):
        v = self._info.get(tid)
        if v is None:
            v = self._info[tid] = _TypeInfo(self.types.get(tid))
        return v


class _Rep:
    """One replication of the event loop."""

    def __init__(self, sim, items, u, rng, tagged, lam):
        self.sim, self.items, self.u, self.rng, self.tagged, self.lam = sim, items, u, rng, tagged, lam
        n = self.n = len(items)
        self.arr = [it["t_arrival_ns"] for it in items]
        if any(b < a for a, b in zip(self.arr, self.arr[1:])):
            raise ValueError("schedule t_arrival_ns must not decrease")
        self.idx = {it["req_id"]: i for i, it in enumerate(items)}
        if len(self.idx) != n:
            raise ValueError("schedule req_id values must be unique")
        self.tids = sorted({it["type_id"] for it in items} | set(sim.types.type_ids()))   # a slot for any run type
        self.tix = {t: k for k, t in enumerate(self.tids)}
        self.infos = [sim.info(t) for t in self.tids]
        nt = len(self.tids)
        self.W, self.sd, self.cnt = [0.0] * nt, [1.0] * nt, [0] * nt
        self.heaps = [[] for _ in range(nt)]
        self.seq = 0
        self.core = (_LPCore if sim.lp else AdmissionCore)(sim.policy, sim.types, sim.f_cmd0)
        self.paused = {}                  # edgeserving_lp: req_id -> paused LLaMA job (out of A, still in nserv)
        if sim.lp:
            sim.policy.paused = self.paused
        self.tag_k = sim.cfg.tag.k_dev if sim.cfg.tag is not None else None
        self.lag = int(round(sim.cfg.switch_lag_s * _NS))
        self.f_eff = sim.f0
        self.act_time = self.act_target = self.act_pending = None
        self.now = 0
        self.nq = self.nserv = 0
        self.dirty = False
        self.E = self.NI = self.BI = 0.0
        self.P = sim.power.p((), self.f_eff)
        self.ev_t, self.ev_e, self.ev_n, self.ev_b = [], [], [], []
        # instability detector: OLS of N(t) on [slope_a, slope_b] (second half of [0, last arrival])
        self.slope_b = self.arr[-1]
        self.slope_a = self.slope_b // 2
        self.slope_num = 0.0
        # per-request results
        self.out = [None] * n
        self.done = [None] * n
        self.t_admit = [None] * n
        self.t_end = [None] * n
        self.reached = [None] * n
        self.nsa = [None] * n
        self.f_adm = [None] * n
        self.qlen = [None] * n
        self.t_cks = [None] * n
        self.run_ti = [None] * n          # slot of the type the request runs as (None: never admitted)

    # ------------------------------------------------------------------------------------------- event loop
    def run(self):
        n, nt, arr = self.n, len(self.tids), self.arr
        heaps, W, sd = self.heaps, self.W, self.sd
        self._refresh()
        i_next = 0
        while True:
            t_arr = arr[i_next] if i_next < n else None
            t_job, ti_best = None, -1
            for ti in range(nt):
                h = heaps[ti]
                if h:
                    eta = (h[0][0] - W[ti]) * sd[ti]
                    tj = self.now + (int(eta * _NS + 0.5) if eta > 0.0 else 0)
                    if t_job is None or tj < t_job:
                        t_job, ti_best = tj, ti
            t, kind = None, -1
            for k, tk in ((0, self.act_time), (1, t_job), (2, t_arr)):     # tie order: f switch, job, arrival
                if tk is not None and (t is None or tk < t):
                    t, kind = tk, k
            if t is None:
                break
            self._advance(t)
            if kind == 0:
                self._act_event()
            elif kind == 1:
                self._job_event(ti_best)
            else:
                self._arrival(i_next)
                i_next += 1
            if self.dirty:
                self._refresh()
        if self.nq or self.nserv:
            raise RuntimeError("simulation ended with %d queued and %d in service: the policy admits nothing while the "
                               "system is idle" % (self.nq, self.nserv))
        return self

    def _advance(self, t):
        dt_ns = t - self.now
        if dt_ns > 0:
            dt = dt_ns / _NS
            cnt, W, sd = self.cnt, self.W, self.sd
            for ti in range(len(cnt)):
                if cnt[ti]:
                    W[ti] += dt / sd[ti]
            N = self.nq + self.nserv
            self.E += self.P * dt
            self.NI += N * dt
            if self.nserv:
                self.BI += dt
            lo, hi = max(self.now, self.slope_a), min(t, self.slope_b)
            if hi > lo:
                # integral of N(t) (t - t_mid) dt over [lo, hi]; t_mid is the middle of [slope_a, slope_b]
                self.slope_num += N * ((hi - lo) / _NS) * ((lo + hi - self.slope_a - self.slope_b) / 2.0 / _NS)
            self.now = t
        self.ev_t.append(self.now)
        self.ev_e.append(self.E)
        self.ev_n.append(self.NI)
        self.ev_b.append(self.BI)

    def _refresh(self):
        """A or f changed: sample a new slowdown for every type present, and the new power level."""
        sim = self.sim
        A = []
        for ti, c in enumerate(self.cnt):
            if c:
                A.extend([self.tids[ti]] * c)
        A = tuple(A)
        f = self.f_eff
        for ti, c in enumerate(self.cnt):
            if c:
                tid = self.tids[ti]
                if sim.cfg.sample_slowdown:
                    s = sim.slowdown.sample(tid, A, f, self.rng)
                else:
                    s = sim.slowdown.predict(tid, A, f)
                if not (0.0 < s < math.inf):
                    raise ValueError("slowdown for %s with A=%s at %s MHz is %r; it must be finite and > 0"
                                     % (tid, A, f, s))
                self.sd[ti] = s
        self.P = sim.power.p(A, f)
        self.dirty = False

    # ------------------------------------------------------------------------------------------- events
    def _arrival(self, i):
        it = self.items[i]
        info = self.infos[self.tix[it["type_id"]]]
        ko = self.tag_k if i in self.tagged else it["k_override"]
        req = QueuedReq(it["req_id"], info.tid, info.family, info.klass, it["t_arrival_ns"], it["t_deadline_ns"], ko)
        self.qlen[i] = self.nq
        self.nq += 1
        self._decide(self.core.arrive(req, self.now))

    def _decide(self, d):
        items, idx = self.items, self.idx
        for rid in d.expired:
            i = idx[rid]
            self.nq -= 1
            self.out[i] = "expired_in_queue"
            self.done[i] = items[i]["t_deadline_ns"]
        for k, (rid, nsa) in enumerate(zip(d.admitted, d.n_in_service_at_admit)):
            i = idx[rid]
            rt = d.run_type_ids[k] if d.run_type_ids else items[i]["type_id"]
            ti = self.run_ti[i] = self.tix[rt]
            self.nq -= 1
            self.nserv += 1
            self.cnt[ti] += 1
            self._start(i, ti, nsa)
            self.dirty = True
        if d.commanded_f is not None:
            self._command(d.commanded_f)

    def _start(self, i, ti, nsa):
        info = self.infos[ti]
        job = _Job()
        job.i, job.ti, job.u = i, ti, self.u[i]
        job.tau = info.tau(self.f_eff, job.u)
        job.dl = self.items[i]["t_deadline_ns"]
        job.t_ck, job.reached, job.outcome = [], None, None
        job.ck = 0 if job.dl is not None else len(info.ckpts) - 1      # no deadline: only the last checkpoint matters
        job.kind = 0
        job.tgt = info.fr[job.ck]
        self.t_admit[i], self.nsa[i], self.f_adm[i] = self.now, nsa, self.f_eff
        self._push(job, 0.0)

    def _push(self, job, p_now):
        ti = job.ti
        job.p0, job.w0 = p_now, self.W[ti]
        heapq.heappush(self.heaps[ti], (job.w0 + (job.tgt - p_now) * job.tau, self.seq, job))
        self.seq += 1

    def _job_event(self, ti):
        _, _, job = heapq.heappop(self.heaps[ti])
        if job.kind == 1:
            self._finish(job)
            return
        info, now, k = self.infos[ti], self.now, job.ck
        if job.dl is None:
            job.reached, job.outcome = info.ckpts[-1], "complete"
            remaining = 1.0 - info.fr[k]
        else:
            job.t_ck.append(now)
            if now <= job.dl and k < len(info.ckpts) - 1:           # on time and not the last: next segment
                job.ck = k + 1
                job.tgt = info.fr[k + 1]
                if self.sim.lp and info.family == "llama" and self._short_waiting():
                    self._pause(job, info.fr[k])
                    return
                self._push(job, info.fr[k])
                return
            job.reached, job.outcome = C.truncation_outcome(info.ckpts, job.t_ck, job.dl)
            if job.outcome == "complete":
                remaining = 1.0 - info.fr[k]
            elif job.reached is None:
                remaining = 0.0
            else:
                remaining = info.head
        if remaining > _EPS_FRAC:                                   # the head runs after the decision
            job.kind = 1
            job.tgt = info.fr[k] + remaining
            self._push(job, info.fr[k])
        else:
            self._finish(job)

    def _finish(self, job):
        i, ti, now = job.i, job.ti, self.now
        self.cnt[ti] -= 1
        self.nserv -= 1
        self.out[i], self.reached[i] = job.outcome, job.reached
        self.done[i] = self.t_end[i] = now
        self.t_cks[i] = job.t_ck
        self.dirty = True
        self._decide(self.core.done(self.items[i]["req_id"], now))
        self._resume_if_idle()

    # ------------------------------------------------------------------------------------------- edgeserving_lp
    def _short_waiting(self):
        now = self.now
        return any(q.family != "llama" and (q.t_deadline_ns is None or now < q.t_deadline_ns)
                   for q in self.core.view(now).queue)

    def _pause(self, job, p):
        """The LLaMA job stops at token boundary p; the core admits a queued short request at once."""
        job.p0 = p
        self.cnt[job.ti] -= 1
        self.paused[self.items[job.i]["req_id"]] = job
        self.dirty = True
        self._decide(self.core.tick(self.now))
        self._resume_if_idle()

    def _resume_if_idle(self):
        """No job runs: the paused job resumes from its token boundary (same run type; tau at the current f)."""
        if self.paused and not any(self.cnt):
            _, job = self.paused.popitem()
            job.tau = self.infos[job.ti].tau(self.f_eff, job.u)
            self.cnt[job.ti] += 1
            self._push(job, job.p0)
            self.dirty = True

    # ------------------------------------------------------------------------------------------- frequency actuator
    def _command(self, f):
        if self.lag == 0:
            self._set_f(f)
        elif self.act_time is not None:
            self.act_pending = f                                    # newest command only
        elif f != self.f_eff:
            self.act_time, self.act_target = self.now + self.lag, f

    def _act_event(self):
        self._set_f(self.act_target)
        pending, self.act_pending = self.act_pending, None
        if pending is not None and pending != self.f_eff:
            self.act_time, self.act_target = self.now + self.lag, pending
        else:
            self.act_time = self.act_target = None

    def _set_f(self, f):
        """The GPU now runs at f: keep every job's progress fraction, change its solo time."""
        if f == self.f_eff:
            return
        for ti, h in enumerate(self.heaps):
            if not h:
                continue
            info, Wt, new = self.infos[ti], self.W[ti], []
            for _, seq, job in h:
                p = min(job.p0 + (Wt - job.w0) / job.tau, job.tgt)
                job.tau = info.tau(f, job.u)
                job.p0, job.w0 = p, Wt
                new.append((Wt + (job.tgt - p) * job.tau, seq, job))
            heapq.heapify(new)
            self.heaps[ti] = new
        self.f_eff = f
        self.dirty = True

    # ------------------------------------------------------------------------------------------- results
    def _cum(self, t, values):
        """Cumulative quantity at ns time t (piecewise linear between events)."""
        ev_t = self.ev_t
        k = bisect_right(ev_t, t) - 1
        if k < 0:
            return 0.0
        if k + 1 >= len(ev_t) or ev_t[k + 1] == ev_t[k]:
            return values[k]
        return values[k] + (values[k + 1] - values[k]) * ((t - ev_t[k]) / (ev_t[k + 1] - ev_t[k]))

    def n_slope(self):
        span = (self.slope_b - self.slope_a) / _NS
        return 12.0 * self.slope_num / span ** 3 if span > 0 else 0.0

    def summary(self, n_warmup, tag):
        n, items, infos, tix = self.n, self.items, self.infos, self.tix
        w0 = self.arr[n_warmup]
        w1 = max(self.done[i] for i in range(n_warmup, n))
        win = (w1 - w0) / _NS
        e = self._cum(w1, self.ev_e) - self._cum(w0, self.ev_e)
        n_meas = n - n_warmup
        by_type, tagged = {}, ([], [], []) if tag is not None else None
        n_complete = 0
        for i in range(n_warmup, n):
            ti = tix[items[i]["type_id"]]
            code = _CODES[self.out[i]]
            r = self.reached[i]
            comp = infos[self.run_ti[i]].completion(r) if r is not None else 0.0
            row = ((self.done[i] - self.arr[i]) / _NS, code, comp)
            for c, v in zip(by_type.setdefault(ti, ([], [], [])), row):
                c.append(v)
            if tagged is not None and i in self.tagged:
                for c, v in zip(tagged, row):
                    c.append(v)
            n_complete += code == 0
        out = {
            "energy_per_req_j": e / n_meas,
            "mean_power_w": e / win if win > 0 else 0.0,
            "throughput": n_complete / win if win > 0 else 0.0,
            "mean_in_system": (self._cum(w1, self.ev_n) - self._cum(w0, self.ev_n)) / win if win > 0 else 0.0,
            "utilization": (self._cum(w1, self.ev_b) - self._cum(w0, self.ev_b)) / win if win > 0 else 0.0,
            "n_slope": self.n_slope(),
            "per_type": {self.tids[ti]: _type_stats(*cols) for ti, cols in by_type.items()},
            "tagged": None,
        }
        if tagged is not None and tagged[0]:
            out["tagged"] = _type_stats(*tagged)
        return out

    def trace(self):
        rows = []
        for i, it in enumerate(self.items):
            info = self.infos[self.tix[it["type_id"]]]
            run = self.infos[self.run_ti[i]] if self.run_ti[i] is not None else None
            r = self.reached[i]
            ck = self.t_cks[i] or []
            rows.append({
                "req_id": it["req_id"], "type_id": it["type_id"], "run_type_id": run.tid if run is not None else None,
                "family": info.family, "klass": info.klass,
                "t_arrival_ns": it["t_arrival_ns"], "t_deadline_ns": it["t_deadline_ns"], "t_admit_ns": self.t_admit[i],
                "t_end_ns": self.t_end[i], "t_done_ns": self.done[i], "outcome": self.out[i], "reached": r,
                "completion": run.completion(r) if r is not None else 0.0, "f_mhz": self.f_adm[i],
                "n_in_service_at_admit": self.nsa[i], "queue_len_at_arrival": self.qlen[i],
                "k_override": self.tag_k if i in self.tagged else it["k_override"],
                "ckpt": [[c, t] for c, t in zip(run.ckpts, ck)] if run is not None else []})
        return tuple(rows)


def _type_stats(R, codes, comp):
    """Per-request response times, outcome codes and completions -> TypeStats-shaped dict (one rep)."""
    n = len(R)
    srt = sorted(R)
    mean_c = min(1.0, math.fsum(comp) / n)
    out = {"n": n, "mean_R": math.fsum(R) / n,
           "p50_R": stats.p_nearest_rank(srt, 0.50), "p95_R": stats.p_nearest_rank(srt, 0.95),
           "p99_R": stats.p_nearest_rank(srt, 0.99)}
    for name, code in (("complete_rate", 0), ("truncation_rate", 1), ("miss_rate", 2), ("expired_rate", 3)):
        out[name] = codes.count(code) / n
    out["mean_completion"] = mean_c
    out["mean_shortfall"] = max(0.0, 1.0 - mean_c)
    return out


def _mean_stats(rows):
    """Mean over reps of TypeStats-shaped dicts; n is the total."""
    out = {"n": int(sum(r["n"] for r in rows))}
    for k in _STAT_KEYS:
        out[k] = float(np.mean([r[k] for r in rows]))
    out["mean_completion"] = min(1.0, out["mean_completion"])
    return out


def _tag_indices(items, types, tag):
    """Index of every `tag.every`-th request of class tag.klass (1-based count within the class)."""
    if tag is None:
        return frozenset()
    out, m = set(), 0
    for i, it in enumerate(items):
        if types.get(it["type_id"]).klass == tag.klass:
            m += 1
            if m % tag.every == 0:
                out.add(i)
    return frozenset(out)


def _field(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def simulate(workload, spec, types, slowdown, power, cfg=None, policy=None):
    """Simulate `spec` on `workload`. See the module docstring for the model. policy None builds
    policies.Policy(spec, types, family_exclusive=cfg.family_exclusive)."""
    t_start = time.perf_counter()
    cfg = cfg if cfg is not None else C.SimConfig()
    if cfg.n_reps < 1 or cfg.seed < 0 or cfg.n_warmup < 0:
        raise ValueError("SimConfig needs n_reps >= 1, seed >= 0, n_warmup >= 0")
    if cfg.tag is not None and (cfg.tag.klass not in C.CLASSES or cfg.tag.every < 1):
        raise ValueError("TagSpec needs klass in %s and every >= 1, got %r" % (C.CLASSES, cfg.tag))
    if policy is None:
        policy = Policy(spec, types, family_exclusive=cfg.family_exclusive)
    sim = _Sim(spec, types, slowdown, power, cfg, policy)
    lam = float(_field(workload, "rate_per_s"))
    reps = []
    for r in range(cfg.n_reps):
        seed = cfg.seed + r
        if cfg.schedule is not None:
            items = [dict(it) for it in cfg.schedule]
        else:
            items = C.with_deadlines(arrivals.schedule(workload, cfg.n_requests, seed), types, cfg.alpha)
        if len(items) <= cfg.n_warmup:
            raise ValueError("%d requests leave nothing to measure after n_warmup=%d" % (len(items), cfg.n_warmup))
        u = np.random.Generator(np.random.PCG64([seed, 1])).random(len(items)).tolist()
        rng = np.random.Generator(np.random.PCG64([seed, 2]))
        rep = _Rep(sim, items, u, rng, _tag_indices(items, types, cfg.tag), lam).run()
        reps.append((seed, rep, rep.summary(cfg.n_warmup, cfg.tag)))
    return _assemble(spec, cfg, lam, reps, time.perf_counter() - t_start)


def _assemble(spec, cfg, lam, reps, runtime_s):
    sums = [s for _, _, s in reps]
    energies = [s["energy_per_req_j"] for s in sums]
    if len(energies) >= 2:
        e_mean, lo, hi = stats.mean_ci(energies)
    else:
        e_mean = lo = hi = energies[0]
    per_type = {}
    for tid in sorted({t for s in sums for t in s["per_type"]}):
        per_type[tid] = TypeStats(**_mean_stats([s["per_type"][tid] for s in sums if tid in s["per_type"]]))
    tagged = None
    if cfg.tag is not None and any(s["tagged"] is not None for s in sums):
        tagged = dict(klass=cfg.tag.klass, k_dev=cfg.tag.k_dev,
                      **_mean_stats([s["tagged"] for s in sums if s["tagged"] is not None]))
    need = math.ceil(0.8 * cfg.n_reps - 1e-9)
    flagged = sum(1 for s in sums if s["n_slope"] > C.UNSTABLE_SLOPE_FRAC * lam)
    def tagged_json(st):
        return None if st is None else dict(klass=cfg.tag.klass, k_dev=cfg.tag.k_dev, **st)

    rep_rows = tuple({"seed": seed, "energy_per_req_j": s["energy_per_req_j"], "n_slope": s["n_slope"],
                      "per_type": {t: dict(v) for t, v in s["per_type"].items()}, "tagged": tagged_json(s["tagged"])}
                     for seed, _, s in reps)
    trace = reps[0][1].trace() if cfg.schedule is not None else None
    return SimResult(
        spec_name=spec.name, per_type=per_type, energy_per_req_j=e_mean, energy_ci=(lo, hi),
        mean_power_w=float(np.mean([s["mean_power_w"] for s in sums])),
        throughput=float(np.mean([s["throughput"] for s in sums])),
        mean_in_system=float(np.mean([s["mean_in_system"] for s in sums])),
        utilization=float(np.mean([s["utilization"] for s in sums])),
        unstable=bool(cfg.alpha is None and flagged >= need), tagged=tagged, reps=rep_rows, trace=trace,
        runtime_s=runtime_s)
