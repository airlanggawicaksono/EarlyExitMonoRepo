"""exp2.model.baselines: the selection rule of every compared method (paper baselines table, BRIEF_v1 slice E3b).

    select_baseline(name, types, type_ids, *, sim_fn=None, probe_margins=None, mem_budget_mb=None,
                    constraint="complete_rate") -> Selection

Static named policies come from exp2.policies.registry(types), unchanged (priority_serial = registry serial_class).
The rest pick a grid point:

    capacity_one        F restricted to f_max, feasible iff mean_R_j <= D_j for every type; sim_fn must use s = |A|
    solo_profile_freq   k=1, FCFS, lowest corun level with solo tau p95 <= D_j for every type (f_max if none)
    probe_margin        k=inf, FCFS, lowest corun level with 1.035 tau_j(f) + M_j f_max / f <= D_j (f_max if none)
    F_M1, F_M3          F; sim_fn must be built with slowdown model M1 / M3
    freq_only           F with (k_short, k_long) in {1, inf}^2
    conc_only           F at f_max
"""
from dataclasses import dataclass

from exp2 import contracts as C
from exp2.contracts import ContractError, K_INF
from exp2.model.policy_fn import make_spec, search, search_F
from exp2.policies import registry

__all__ = ["Selection", "select_baseline", "BASELINES", "PROBE_FACTOR"]

PROBE_FACTOR = 1.035
_ALIASES = {"priority_serial": "serial_class"}             # paper name -> registry name
_STATIC = ("default", "default_stock", "serial", "serial_stf", "serial_edgeserving", "corun", "priority_serial",
           "greenllm_lanes", "occupancy_f")
_FREQ = ("solo_profile_freq", "probe_margin")
_SEARCH = ("capacity_one", "F_M1", "F_M3", "freq_only", "conc_only")
BASELINES = _STATIC + _FREQ + _SEARCH


@dataclass(frozen=True)
class Selection:
    name: str
    spec: object              # PolicySpec
    feasible: bool            # False when no grid point meets the rule (spec is then the best infeasible / f_max)
    fresult: object = None    # FResult for search rules, else None


def _deadlines(types, type_ids):
    out = {}
    for t in type_ids:
        d = C.deadline_s(types, t)
        if d is None:
            raise ContractError("rule needs deadlines but types.alpha is None")
        out[t] = d
    return out


def _lowest_level(types, ok):
    for f in sorted(types.corun_levels_mhz):
        if ok(f):
            return f, True
    return types.f_max_mhz, False


def select_baseline(name, types, type_ids, *, sim_fn=None, probe_margins=None, mem_budget_mb=None,
                    constraint="complete_rate"):
    if name not in BASELINES:
        raise ContractError("unknown baseline %r; known: %s" % (name, BASELINES))
    if name in _STATIC:
        return Selection(name, registry(types)[_ALIASES.get(name, name)], True)
    fmax = types.f_max_mhz
    if name == "solo_profile_freq":
        d = _deadlines(types, type_ids)
        f, ok = _lowest_level(types, lambda f: all(types.tau(t, f, "p95") <= d[t] for t in type_ids))
        return Selection(name, make_spec(name, 1, 1, "fcfs", f), ok)
    if name == "probe_margin":
        if probe_margins is None or any(t not in probe_margins for t in type_ids):
            raise ContractError("probe_margin needs probe_margins {type_id: M_j seconds} for every type")
        d = _deadlines(types, type_ids)
        f, ok = _lowest_level(types, lambda f: all(
            PROBE_FACTOR * types.tau(t, f, "p50") + probe_margins[t] * fmax / f <= d[t] for t in type_ids))
        return Selection(name, make_spec(name, K_INF, K_INF, "fcfs", f), ok)
    if sim_fn is None:
        raise ContractError("%s needs sim_fn" % name)
    common = dict(mem_budget_mb=mem_budget_mb, name=name)
    if name == "capacity_one":
        d = _deadlines(types, type_ids)
        fr = search(types, type_ids, sim_fn, lambda r: min(d[t] - r.per_type[t].mean_R for t in type_ids), 0.0,
                    f_grid=(fmax,), **common)
    elif name == "freq_only":
        fr = search_F(types, type_ids, sim_fn, constraint=constraint, ks=(1, K_INF), **common)
    elif name == "conc_only":
        fr = search_F(types, type_ids, sim_fn, constraint=constraint, f_grid=(fmax,), **common)
    else:                                                  # F_M1, F_M3: the model lives in sim_fn
        fr = search_F(types, type_ids, sim_fn, constraint=constraint, **common)
    return Selection(name, fr.spec, fr.feasible, fr)
