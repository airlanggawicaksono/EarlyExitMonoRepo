"""exp2.model.game: the two-class concurrency game (BRIEF_v1 slice E3a).

Players are the classes short and long. A strategy is the class limit k in K = (1, 2, 3, K_INF). A profile is
(k_short, k_long); everything else in the PolicySpec stays as given.

    best_response(...)  one best-response dynamic from `start`; round = short moves, then long moves
    equilibria(...)     the dynamic from all |K|^2 starts (16 for the default K)
    poa(...)            equilibria + worst/optimum energy; poa_E = WORST equilibrium energy / optimum energy

Player cost: "rt" = mean_R, "shortfall" = mean_shortfall, "shortfall_then_rt" = shortfall unless it is a tie
(difference < SHORTFALL_TIE or overlapping 95% CIs over the reps), then mean_R. Full tie: keep the current k, else
the smaller k. PoA_R (worst equilibrium mean R / optimum mean R) exists only in simulation: cfg.alpha None and
player_cost "rt"; otherwise it is None.

sim_fn(workload, spec, types, slowdown, power, cfg) -> SimResult (default: exp2.model.simulator.simulate).
"""
import dataclasses

from exp2 import contracts as C
from exp2 import stats

SHORTFALL_TIE = 0.005
PLAYER_COSTS = ("rt", "shortfall", "shortfall_then_rt")
DEFAULT_K = (1, 2, 3, C.K_INF)
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


def poa(workload, base_spec, types, slowdown, power, cfg=None, *, player_cost="shortfall_then_rt",
        K=DEFAULT_K, max_iter=20, sim_fn=None):
    """The K `game` block: {player_cost, equilibria, worst, optimum, poa_E, poa_R}.
    worst = converged profile with the highest energy_per_req_j, optimum = lowest energy over ALL profiles in K x K,
    poa_E = worst / optimum. poa_R = worst-equilibrium mean R / lowest mean R, only when cfg.alpha is None and
    player_cost == "rt". No converged start: worst is {} and both ratios are None."""
    _check(player_cost, K)
    K = tuple(K)
    ev = _Eval(workload, base_spec, types, slowdown, power, cfg, sim_fn)
    eqs = equilibria(workload, base_spec, types, slowdown, power, cfg, player_cost=player_cost, K=K,
                     max_iter=max_iter, _ev=ev)
    profiles = [(a, b) for a in K for b in K]
    opt = min(profiles, key=lambda p: ev(*p).energy_per_req_j)
    conv = sorted({(e["k_short"], e["k_long"]) for e in eqs if e["converged"]})
    out = {"player_cost": player_cost, "equilibria": eqs, "worst": {},
           "optimum": {"k_short": opt[0], "k_long": opt[1]}, "poa_E": None, "poa_R": None}
    if not conv:
        return out
    worst = max(conv, key=lambda p: ev(*p).energy_per_req_j)
    out["worst"] = {"k_short": worst[0], "k_long": worst[1]}
    out["poa_E"] = ev(*worst).energy_per_req_j / ev(*opt).energy_per_req_j
    if player_cost == "rt" and cfg is not None and cfg.alpha is None:
        worst_r = max(conv, key=lambda p: _mean_r(ev(*p)))
        out["poa_R"] = _mean_r(ev(*worst_r)) / min(_mean_r(ev(*p)) for p in profiles)
    return out
