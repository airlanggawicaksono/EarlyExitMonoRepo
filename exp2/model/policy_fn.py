"""exp2.model.policy_fn: the policy function F (paper Algorithm 1, BRIEF_v1 slice E3b).

    search_F(types, type_ids, sim_fn, *, constraint, mem_budget_mb, f_grid, ks, orders, target) -> FResult

Exhaustive search over (k_short, k_long) x order x f.  `sim_fn(spec) -> SimResult` is a one-argument callable (bind the
workload, slowdown model, power model and config in a closure or functools.partial; StubSim works as is).
A point is feasible when the minimum over types of the constraint metric is >= target:
    constraint "complete_rate"    metric = TypeStats.complete_rate   (the paper constraint)
    constraint "mean_completion"  metric = TypeStats.mean_completion (sensitivity)
The feasible point with the lowest energy_per_req_j wins (ties: first in grid order).  With no feasible point the
point with the highest minimum metric wins (ties: lower energy) and FResult.feasible is False.
Only levels of `f_grid` (default types.corun_levels_mhz) are simulated; a level outside corun_levels is a ContractError.
"""
import time
from dataclasses import dataclass

from exp2 import contracts as C
from exp2.contracts import ContractError, K_INF, PolicySpec

__all__ = ["search_F", "FResult", "mem_footprint_mb", "K_GRID", "ORDERS"]

K_GRID = (1, 2, 3, K_INF)
ORDERS = ("fcfs", "stf")
_METRICS = ("complete_rate", "mean_completion")


@dataclass(frozen=True)
class FResult:
    spec: PolicySpec
    result: object            # SimResult of the chosen point
    feasible: bool
    min_metric: float         # min over types of the metric at the chosen point
    n_sim: int                # simulations run
    n_skipped_mem: int        # grid points removed by the memory filter
    compute_s: float          # wall time of the whole search (item 26)


def mem_footprint_mb(types, type_ids, k_short, k_long):
    """Worst-case memory (MB) of the policy: one resident copy of every family of the workload, plus the activation
    memory of the largest co-running set the limits allow.  With family exclusivity at most one request per family
    runs, at most k_short short-class and k_long long-class requests."""
    fams = {types.get(t).family for t in type_ids}
    resident = sum(types.families[f]["resident_mb"] for f in fams)
    best = {}                                  # family -> (act_mb, klass) of its largest type
    for t in type_ids:
        ts = types.get(t)
        if ts.family not in best or ts.act_mb > best[ts.family][0]:
            best[ts.family] = (ts.act_mb, ts.klass)
    left = {"short": k_short, "long": k_long}
    act = 0.0
    for a, kl in sorted(best.values(), reverse=True):
        if left[kl] > 0:
            act += a
            left[kl] -= 1
    return resident + act


def _min_metric(res, metric):
    if not res.per_type:
        raise ContractError("SimResult has no per-type statistics")
    return min(getattr(s, metric) for s in res.per_type.values())


def _grid(types, f_grid, ks, orders):
    levels = tuple(types.corun_levels_mhz if f_grid is None else f_grid)
    bad = [f for f in levels if f not in types.corun_levels_mhz]
    if bad:
        raise ContractError("levels %s are not corun levels %s; F never simulates them" % (bad, types.corun_levels_mhz))
    return [(f, o, a, b) for f in levels for o in orders for a in ks for b in ks]


def make_spec(name, k_short, k_long, order, f_mhz):
    spec = PolicySpec(name, k_short, k_long, order, None, "static", int(f_mhz), {})
    C.validate_policyspec(spec)
    return spec


def search(types, type_ids, sim_fn, score_fn, target, *, mem_budget_mb=None, f_grid=None, ks=K_GRID, orders=ORDERS,
           name="F"):
    """Core search; score_fn(SimResult) -> float, feasible iff score >= target."""
    t0 = time.perf_counter()
    feas, infeas, n_sim, n_mem = [], [], 0, 0
    for f, o, a, b in _grid(types, f_grid, ks, orders):
        if mem_budget_mb is not None and mem_footprint_mb(types, type_ids, a, b) > mem_budget_mb:
            n_mem += 1
            continue
        spec = make_spec(name, a, b, o, f)
        res = sim_fn(spec)
        n_sim += 1
        score = score_fn(res)
        (feas if score >= target else infeas).append((score, res, spec))
    if feas:
        score, res, spec = min(feas, key=lambda t: t[1].energy_per_req_j)       # min is stable: first wins ties
        ok = True
    elif infeas:
        score, res, spec = max(infeas, key=lambda t: (t[0], -t[1].energy_per_req_j))
        ok = False
    else:
        raise ContractError("no grid point satisfies the memory budget %s MB" % mem_budget_mb)
    return FResult(spec, res, ok, score, n_sim, n_mem, time.perf_counter() - t0)


def search_F(types, type_ids, sim_fn, *, constraint="complete_rate", mem_budget_mb=None, f_grid=None, ks=K_GRID,
             orders=ORDERS, target=C.COMPLETE_TARGET, name="F"):
    if constraint not in _METRICS:
        raise ContractError("constraint %r not in %s" % (constraint, _METRICS))
    return search(types, type_ids, sim_fn, lambda r: _min_metric(r, constraint), target, mem_budget_mb=mem_budget_mb,
                  f_grid=f_grid, ks=ks, orders=orders, name=name)
