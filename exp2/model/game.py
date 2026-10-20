"""exp2.model.game: the two-class concurrency game (BRIEF_v1 slice E3a).

Players are the classes short and long. A strategy is the class limit k in K = C.K_SET. A profile is
(k_short, k_long); everything else in the PolicySpec stays as given.

    nash_equilibria(...) all pure Nash profiles of K x K (9 for the default K), tolerance C.GAME_TIE_TOL (D32-8)
    best_response(...)   secondary: one best-response dynamic from `start`; round = short moves, then long moves
    equilibria(...)      secondary: the dynamic from all |K|^2 starts
    poa(...)             freeze game block; poa_eps = worst Nash energy / constrained optimum energy

Nash (D32-8): a profile is an equilibrium if no class lowers its cost by more than tau = GAME_TIE_TOL through a
unilateral change of its k. Cost = mean shortfall; when the shortfalls differ by at most tau the class compares mean_R
(strictly lower is an improvement), unless player_cost is "shortfall" (ties never improve) or "rt" (mean_R only).
Constrained optimum: the lowest-energy profile in which every type meets C.COMPLETE_TARGET with the one-sided lower
bound of D32-1 (mean - t_{0.95,R-1} sd / sqrt(R) of the per-replication complete_rate over result.reps). A result
without per-replication data (fewer than 2 reps, or a rep without the type) uses the point estimate complete_rate.

Player cost: "rt" = mean_R, "shortfall" = mean_shortfall, "shortfall_then_rt" = shortfall unless it is a tie
(difference < SHORTFALL_TIE or overlapping 95% CIs over the reps), then mean_R. Full tie: keep the current k, else
the smaller k. PoA_R (worst equilibrium mean R / optimum mean R) exists only in simulation: cfg.alpha None and
player_cost "rt"; otherwise it is None.

sim_fn(workload, spec, types, slowdown, power, cfg) -> SimResult (default: exp2.model.simulator.simulate).
"""
import dataclasses

from exp2 import contracts as C
from exp2 import stats

SHORTFALL_TIE = C.GAME_TIE_TOL
PLAYER_COSTS = ("rt", "shortfall", "shortfall_then_rt")
DEFAULT_K = C.K_SET
# one-sided t quantile, level 0.95, df 1..20, then 25, 30 (use the nearest LOWER df: conservative), then normal
_T95 = (6.3138, 2.9200, 2.3534, 2.1318, 2.0150, 1.9432, 1.8946, 1.8595, 1.8331, 1.8125, 1.7959, 1.7823, 1.7709,
        1.7613, 1.7531, 1.7459, 1.7396, 1.7341, 1.7291, 1.7247)
_EPS = 1e-12


class _Eval:
    """Memoised simulation of profiles (shared by every start of one game)."""

    def __init__(self, workload, base_spec, types, slowdown, power, cfg, sim_fn):
        if sim_fn is None:
            from exp2.model.simulator import simulate as sim_fn
        self._args = (workload, types, slowdown, power, cfg)
        self._base, self._sim, self.types, self.cfg = base_spec, sim_fn, types, cfg
        self._cache = {}

    def __call__(self, ks, kl):
        if (ks, kl) not in self._cache:
            name = "%s_k%s_%s" % (self._base.name, "inf" if ks >= C.K_INF else ks, "inf" if kl >= C.K_INF else kl)
            spec = dataclasses.replace(self._base, name=name, k_short=ks, k_long=kl)
            w, t, s, p, c = self._args
            self._cache[(ks, kl)] = self._sim(w, spec, t, s, p, c)
        return self._cache[(ks, kl)]


def _class_tids(res, types, klass):
    return [t for t in res.per_type if types.get(t).klass == klass]


def _class_value(res, types, klass, field):
    """n-weighted mean of a TypeStats field over the types of the class (None if the class has no type)."""
    tids = _class_tids(res, types, klass)
    w = [res.per_type[t].n for t in tids]
    if not tids or sum(w) <= 0:
        return None
    return sum(getattr(res.per_type[t], field) * n for t, n in zip(tids, w)) / sum(w)


def _class_ci(res, types, klass, field):
    """95% CI of the class value over the reps; None when the reps do not carry the class (or fewer than 2)."""
    tids = _class_tids(res, types, klass)
    if len(res.reps) < 2 or not tids:
        return None
    vals = []
    for rep in res.reps:
        pt = rep.get("per_type") or {}
        if any(t not in pt for t in tids):
            return None
        w = [res.per_type[t].n for t in tids]
        vals.append(sum(pt[t][field] * n for t, n in zip(tids, w)) / sum(w))
    _, lo, hi = stats.mean_ci(vals)
    return lo, hi


def _t95(df):
    if df <= len(_T95):
        return _T95[df - 1]
    return 1.7081 if df < 30 else (1.6973 if df < 60 else 1.6711)


def _complete_lower(res, tid):
    """One-sided 95% lower bound of the complete rate of a type over the replications (D32-1)."""
    point = res.per_type[tid].complete_rate
    vals = [(rep.get("per_type") or {}).get(tid, {}).get("complete_rate") for rep in (res.reps or [])]
    if len(vals) < 2 or any(v is None for v in vals):
        return point
    n = len(vals)
    m = sum(vals) / n
    sd = (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5
    return m - _t95(n - 1) * sd / n ** 0.5


def _meets_constraint(res):
    return all(_complete_lower(res, t) >= C.COMPLETE_TARGET - _EPS for t in res.per_type)


def _improves(ra, rb, types, klass, player_cost):
    """True if result ra lowers the cost of `klass` below rb by more than tau (see module docstring)."""
    if player_cost == "rt":
        return _class_value(ra, types, klass, "mean_R") < _class_value(rb, types, klass, "mean_R") - _EPS
    d = _class_value(rb, types, klass, "mean_shortfall") - _class_value(ra, types, klass, "mean_shortfall")
    if d > C.GAME_TIE_TOL + _EPS:
        return True
    if player_cost == "shortfall" or d < -C.GAME_TIE_TOL - _EPS:
        return False
    return _class_value(ra, types, klass, "mean_R") < _class_value(rb, types, klass, "mean_R") - _EPS


def _nash(ev, K, player_cost):
    out = []
    for a in K:
        for b in K:
            cur = ev(a, b)
            if any(_improves(ev(x, b), cur, ev.types, "short", player_cost) for x in K if x != a):
                continue
            if any(_improves(ev(a, y), cur, ev.types, "long", player_cost) for y in K if y != b):
                continue
            out.append((a, b))
    return out


def _cmp(a, b):
    return 0 if abs(a - b) <= _EPS else (-1 if a < b else 1)


def _compare(ra, rb, types, klass, player_cost):
    """-1 if result ra is strictly better for the class than rb, +1 if worse, 0 if tied."""
    if player_cost == "rt":
        return _cmp(_class_value(ra, types, klass, "mean_R"), _class_value(rb, types, klass, "mean_R"))
    sa, sb = (_class_value(r, types, klass, "mean_shortfall") for r in (ra, rb))
    ca, cb = (_class_ci(r, types, klass, "mean_shortfall") for r in (ra, rb))
    overlap = ca is not None and cb is not None and ca[0] <= cb[1] and cb[0] <= ca[1]
    if abs(sa - sb) >= SHORTFALL_TIE and not overlap:
        return -1 if sa < sb else 1
    if player_cost == "shortfall":
        return 0
    return _cmp(_class_value(ra, types, klass, "mean_R"), _class_value(rb, types, klass, "mean_R"))


def _reply(ev, klass, state, K, player_cost):
    """Best k of `klass` against the other class's k in `state`; ties keep the current k, then the smaller k."""
    idx = 0 if klass == "short" else 1

    def res(k):
        s = list(state)
        s[idx] = k
        return ev(*s)

    best = state[idx]
    for k in sorted(K):
        if k != best and _compare(res(k), res(best), ev.types, klass, player_cost) < 0:
            best = k
    return best


def _best_response(ev, player_cost, K, start, max_iter):
    state = (start[0], start[1])
    history = [state]
    for i in range(1, max_iter + 1):
        ks = _reply(ev, "short", state, K, player_cost)
        kl = _reply(ev, "long", (ks, state[1]), K, player_cost)
        new = (ks, kl)
        if new == state:
            return _entry(start, state, True, i, None)
        if new in history:
            return _entry(start, new, False, i, [list(s) for s in history[history.index(new):]])
        history.append(new)
        state = new
    return _entry(start, state, False, max_iter, None)


def _entry(start, state, converged, iters, cycle):
    return {"start": [start[0], start[1]], "k_short": state[0], "k_long": state[1], "converged": converged,
            "iters": iters, "cycle": cycle}


def _check(player_cost, K):
    if player_cost not in PLAYER_COSTS:
        raise ValueError("player_cost must be one of %s, got %r" % (PLAYER_COSTS, player_cost))
    if len(set(K)) != len(K) or not K:
        raise ValueError("K must be non-empty and without duplicates, got %r" % (K,))


def best_response(workload, base_spec, types, slowdown, power, cfg=None, *, player_cost="shortfall_then_rt",
                  K=DEFAULT_K, start, max_iter=20, sim_fn=None):
    """Best-response dynamic from `start` = (k_short, k_long). Returns one K-contract equilibria entry:
    {start, k_short, k_long, converged, iters, cycle}. cycle is the list of [k_short, k_long] states of the
    repeating loop, or None. A converged entry is a pure equilibrium; a cycle or max_iter leaves converged False."""
    _check(player_cost, K)
    ev = _Eval(workload, base_spec, types, slowdown, power, cfg, sim_fn)
    return _best_response(ev, player_cost, tuple(K), tuple(start), max_iter)


def equilibria(workload, base_spec, types, slowdown, power, cfg=None, *, player_cost="shortfall_then_rt",
               K=DEFAULT_K, max_iter=20, sim_fn=None, _ev=None):
    """best_response from every (k_short, k_long) in K x K (16 starts for the default K)."""
    _check(player_cost, K)
    K = tuple(K)
    ev = _ev or _Eval(workload, base_spec, types, slowdown, power, cfg, sim_fn)
    return [_best_response(ev, player_cost, K, (a, b), max_iter) for a in K for b in K]


def _mean_r(res):
    n = sum(s.n for s in res.per_type.values())
    return sum(s.mean_R * s.n for s in res.per_type.values()) / n


def nash_equilibria(workload, base_spec, types, slowdown, power, cfg=None, *, player_cost="shortfall_then_rt",
                    K=DEFAULT_K, sim_fn=None, _ev=None):
    """All pure Nash profiles of K x K as [{k_short, k_long}] (empty if none exists)."""
    _check(player_cost, K)
    ev = _ev or _Eval(workload, base_spec, types, slowdown, power, cfg, sim_fn)
    return [{"k_short": a, "k_long": b} for a, b in _nash(ev, tuple(K), player_cost)]


def poa(workload, base_spec, types, slowdown, power, cfg=None, *, player_cost="shortfall_then_rt",
        K=DEFAULT_K, max_iter=20, sim_fn=None):
    """The freeze `game` block: {player_cost, equilibria, worst, optimum, poa_E, poa_R, optimum_constrained,
    poa_eps, tie_tol}. equilibria = all pure Nash profiles [{k_short, k_long}]; worst = Nash profile with the highest
    energy_per_req_j; optimum = lowest energy over ALL profiles; poa_E = worst / optimum; optimum_constrained = lowest
    energy profile meeting the 95% rule (None if none); poa_eps = worst / optimum_constrained (None if either is
    missing); tie_tol = C.GAME_TIE_TOL. poa_R = worst-Nash mean R / lowest mean R, only when cfg.alpha is None and
    player_cost == "rt". No Nash profile: worst is {} and poa_E, poa_R, poa_eps are None.
    The best-response dynamics are not part of this block (the freeze contract has no key for them): call
    equilibria() / best_response() for that secondary description."""
    _check(player_cost, K)
    K = tuple(K)
    ev = _Eval(workload, base_spec, types, slowdown, power, cfg, sim_fn)
    nash = _nash(ev, K, player_cost)
    profiles = [(a, b) for a in K for b in K]
    energy = lambda p: ev(*p).energy_per_req_j  # noqa: E731
    opt = min(profiles, key=energy)
    feas = [p for p in profiles if _meets_constraint(ev(*p))]
    oc = min(feas, key=energy) if feas else None
    out = {"player_cost": player_cost, "equilibria": [{"k_short": a, "k_long": b} for a, b in nash], "worst": {},
           "optimum": {"k_short": opt[0], "k_long": opt[1]}, "poa_E": None, "poa_R": None,
           "optimum_constrained": None if oc is None else {"k_short": oc[0], "k_long": oc[1]},
           "poa_eps": None, "tie_tol": C.GAME_TIE_TOL}
    if not nash:
        return out
    worst = max(nash, key=energy)
    out["worst"] = {"k_short": worst[0], "k_long": worst[1]}
    out["poa_E"] = energy(worst) / energy(opt)
    if oc is not None:
        out["poa_eps"] = energy(worst) / energy(oc)
    if player_cost == "rt" and cfg is not None and cfg.alpha is None:
        worst_r = max(nash, key=lambda p: _mean_r(ev(*p)))
        out["poa_R"] = _mean_r(ev(*worst_r)) / min(_mean_r(ev(*p)) for p in profiles)
    return out
