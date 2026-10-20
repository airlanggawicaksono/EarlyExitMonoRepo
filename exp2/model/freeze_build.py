"""Freeze builder for Run 2 (slice W6): fit nothing, search and simulate everything, write the freeze.

    python -m exp2.model.freeze_build --run1-csv DIR --types T --slowdown M --power P --conditions C
        --run1-plan P1 [--run1b-plan P2] --seed S [--n-requests N] [--n-warmup W] [--n-reps R] [--workers K] --out DIR
    N, W, R default to the contract: C.RUN2_N_MEASURED (500), C.RUN2_N_WARMUP (50), C.R_SIM (10); a flag overrides.
    exit 0 (DIR/freeze.json + freeze.json.sha256 written), 2 (usage or any input / contract error)

Per condition of C (list of {condition_id, alpha, workload}):
    SimConfig(n_requests=N, n_warmup=W, n_reps=R >= 3 (D32-1: the feasibility bound needs R >= 2; the floor is 3), seed=S + 11, alpha=condition, family_exclusive=True,
    sample_slowdown=True). Arrival schedule of rep r = Run 2 repeat r (arrival seed sim.seed + 11 + r).
    choices   F, F_sens, then the baselines in CHOICE_ORDER (default_pinned = the registry corun spec renamed, D2;
              duplicates stay listed, freeze.measured_key_map aliases them, D3). No default_stock.
    searches  F and every search use ks = K_SET; F_M1 / F_M3 use M1 / M3 fitted on train rows; capacity_one
              simulates with s = |A|; probe_margin M_j from train pair cells at f_max (D6); memory budget None (D5).
    grid      freeze.make_grid() = K_SET^2 x {fcfs, stf} x C.GRID_LEVELS_MHZ (612, 1020; D32-2: 36 points; F and the
              frequency baselines still use the co-run levels); predictions for every choice and every grid point.
    game      game.poa(base = corun, K = K_SET, shortfall_then_rt, max_iter 20): the block (equilibria, optimum_constrained,
              poa_eps, tie_tol, ...) is passed through unchanged; only worst / optimum are enriched with order,
              f_mhz and policy_key = measured_key_map[grid name of that profile].
Simulations are memoised per condition, keyed (model tag, PolicySpec without name). --workers K runs K conditions at a
time. stdout: one line per condition with the F choice, then runtime_estimate_s. Run it on the board, not the laptop.
"""
import argparse
import dataclasses
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from exp2 import contracts as C
from exp2.model import freeze as FZ
from exp2.model import game
from exp2.model import slowdown as SD
from exp2.model.baselines import select_baseline
from exp2.model.policy_fn import search_F
from exp2.model.power import PowerModel
from exp2.model.simulator import simulate
from exp2.policies import registry

CHOICE_ORDER = ("F", "F_sens", "default", "default_pinned", "serial", "serial_stf", "serial_edgeserving", "corun",
                "priority_serial", "greenllm_lanes", "occupancy_f", "solo_profile_freq", "probe_margin",
                "capacity_one", "F_M1", "F_M3", "freq_only", "conc_only")
CONSTRAINT = {"headline": "complete_rate>=0.95 per type", "sensitivity": "mean_completion>=0.95 per type"}
SEED_OFFSET = 11
MAX_ITER = 20
_NO_GIT = {"commit": "unknown", "dirty": True, "diff_sha256": None}


class CapacityOne:
    """Slowdown stand-in for the capacity_one baseline: s = |A| (n jobs share one unit of capacity)."""

    def predict(self, i, A, f_mhz):
        return float(len(A))

    def sample(self, i, A, f_mhz, rng):
        return float(len(A))


def _sig(spec):
    j = spec.to_json() if hasattr(spec, "to_json") else dict(spec)
    j.pop("name", None)
    return json.dumps(j, sort_keys=True)


class Sims:
    """Memoised simulations of one condition: key (model tag, spec without name)."""

    def __init__(self, workload, cfg, types, power):
        self.workload, self.cfg, self.types, self.power = workload, cfg, types, power
        self.cache = {}

    def fn(self, tag, slowdown):
        def run(spec):
            key = (tag, _sig(spec))
            if key not in self.cache:
                self.cache[key] = simulate(self.workload, spec, self.types, slowdown, self.power, self.cfg)
            return self.cache[key]
        return run


def probe_margins(corun_rows, types, type_ids):
    """D6: M_j = max over train pair cells (|A| = 2) at f_max of (s_p50 - 1) * tau_solo_p50, seconds. A type with no
    such row gets 0.0 and a warning on stderr."""
    margins, seen = {t: 0.0 for t in type_ids}, set()
    for r in corun_rows:
        t = r["type_id"]
        if r.get("split") == "heldout" or int(float(r["f_mhz"])) != types.f_max_mhz or t not in margins:
            continue
        A = r["A"].split("|") if isinstance(r["A"], str) else list(r["A"])
        if len(A) != 2:
            continue
        seen.add(t)
        margins[t] = max(margins[t], (float(r["s_p50"]) - 1.0) * float(r["tau_solo_p50"]))
    for t in sorted(set(type_ids) - seen):
        print("warning: probe_margin has no train pair row at f_max for %s; M = 0" % t, file=sys.stderr)
    return margins


def _pred(res):
    j = res.to_json()
    return {"per_type": j["per_type"], "energy_per_req_j": j["energy_per_req_j"]}


def _grid_name(grid, spec):
    for g in grid:
        if _sig(g) == _sig(spec):
            return g["name"]
    raise C.ContractError("no grid point equals %s; is f_max %s a corun level?" % (_sig(spec), spec.f_mhz))


def build_condition(cfg_cond, ctx, n_requests, n_warmup, n_reps, seed):
    """One freeze `conditions[]` entry plus a one-line summary. ctx = Inputs."""
    t_all = time.perf_counter()
    wl = cfg_cond["workload"]
    C.validate_workload(wl, ctx.types)
    alpha = cfg_cond["alpha"]
    types = ctx.types if ctx.types.alpha == alpha else dataclasses.replace(ctx.types, alpha=alpha)
    type_ids = list(wl["types"])
    cfg = C.SimConfig(n_requests=n_requests, n_warmup=n_warmup, n_reps=n_reps, seed=seed + SEED_OFFSET, alpha=alpha,
                      family_exclusive=True, sample_slowdown=True)
    workload = C.Workload.from_json(wl)
    sims = Sims(workload, cfg, types, ctx.power)
    sim = sims.fn("main", ctx.slowdown)
    ks = C.K_SET

    t0 = time.perf_counter()
    f_res = search_F(types, type_ids, sim, constraint="complete_rate", ks=ks, name="F")
    f_sens = search_F(types, type_ids, sim, constraint="mean_completion", ks=ks, name="F_sens")
    t_f = time.perf_counter() - t0

    t0 = time.perf_counter()
    reg = registry(types)
    sel_sim = {"capacity_one": sims.fn("capacity_one", CapacityOne()),
               "F_M1": sims.fn("M1", ctx.models["M1"]), "F_M3": sims.fn("M3", ctx.models["M3"])}
    margins = probe_margins(ctx.corun_rows, types, type_ids)
    choices = {"F": f_res.spec, "F_sens": f_sens.spec}
    for name in CHOICE_ORDER[2:]:
        if name == "default_pinned":
            choices[name] = dataclasses.replace(reg["corun"], name=name)
            continue
        choices[name] = select_baseline(name, types, type_ids, sim_fn=sel_sim.get(name, sim),
                                        probe_margins=margins, mem_budget_mb=None).spec
    t_base = time.perf_counter() - t0

    grid = FZ.make_grid(C.GRID_LEVELS_MHZ)
    cond = {"condition_id": cfg_cond["condition_id"], "workload": wl, "rho": wl["rho"], "arrival": wl["arrival"],
            "cv_a": wl["cv_a"], "alpha": alpha, "choices": {k: v.to_json() for k, v in choices.items()},
            "grid": grid, "predictions": {}, "game": None, "compute_s": {}}

    t0 = time.perf_counter()
    base = dataclasses.replace(reg["corun"], name="corun")
    gm = game.poa(workload, base, types, ctx.slowdown, ctx.power, cfg, player_cost="shortfall_then_rt", K=ks,
                  max_iter=MAX_ITER, sim_fn=lambda w, spec, t, s, p, c: sim(spec))
    key_map = FZ.measured_key_map(cond)
    for which in ("worst", "optimum"):
        if gm[which]:
            prof = dataclasses.replace(base, k_short=gm[which]["k_short"], k_long=gm[which]["k_long"])
            gm[which] = {"k_short": prof.k_short, "k_long": prof.k_long, "order": prof.order, "f_mhz": prof.f_mhz,
                         "policy_key": key_map[_grid_name(grid, prof)]}
    cond["game"] = gm
    t_game = time.perf_counter() - t0

    t0 = time.perf_counter()
    for key, spec in list(cond["choices"].items()) + [(g["name"], g) for g in grid]:
        cond["predictions"][key] = _pred(sim(C.PolicySpec.from_json(spec)))
    t_pred = time.perf_counter() - t0
    cond["compute_s"] = {"F": round(t_f, 3), "baselines": round(t_base, 3), "game": round(t_game, 3),
                         "predictions": round(t_pred, 3), "total": round(time.perf_counter() - t_all, 3)}
    fs = f_res.spec
    line = "%s F k=%s/%s order=%s f=%s feasible=%s min_complete=%.3f compute_s=%.1f" % (
        cond["condition_id"], fs.k_short, fs.k_long, fs.order, fs.f_mhz, f_res.feasible, f_res.min_metric,
        cond["compute_s"]["total"])
    return cond, line


class Inputs:
    """Everything read from disk, loaded once per process."""

    def __init__(self, p):
        self.types = C.load_types(p["types"])
        with open(p["slowdown"], encoding="utf-8") as fh:
            self.slowdown_obj = json.load(fh)
        self.slowdown = SD.from_json(self.slowdown_obj, self.types)
        with open(p["power"], encoding="utf-8") as fh:
            self.power = PowerModel.from_json(json.load(fh), self.types)
        d = Path(p["run1_csv"])
        self.corun_rows = C.read_csv(d / "corun.csv", "corun")
        solo_rows = C.read_csv(d / "solo.csv", "solo")
        train = [r for r in self.corun_rows if r.get("split") != "heldout"]
        self.models = {n: SD.make(n).fit(train, solo_rows, self.types) for n in ("M1", "M3")}


_CTX = {}


def _inputs(p):
    key = json.dumps(p, sort_keys=True)
    if key not in _CTX:
        _CTX[key] = Inputs(p)
    return _CTX[key]


def _job(args):
    paths, cond, size = args
    return build_condition(cond, _inputs(paths), *size)


def _slowdown_block(obj):
    mapes = dict(obj["heldout_mape"])
    sel = obj["name"]
    met = sel in mapes and mapes[sel] <= SD.TARGET_MAPE and ("M0" not in mapes or mapes[sel] < mapes["M0"])
    return {"selected": sel, "heldout_mape": mapes, "target_met": bool(met)}


def _git():
    try:
        from exp2.serve.run_session import git_info
        return git_info()
    except Exception as e:                                    # import or git failure must not stop the build
        print("warning: git info unavailable (%s)" % e, file=sys.stderr)
        return dict(_NO_GIT)


def build(a, git=None):
    """Build and return the validated freeze dict from the parsed CLI namespace `a`."""
    if a.n_reps < 3:
        raise C.ContractError("--n-reps must be >= 3, got %d" % a.n_reps)
    paths = {k: str(getattr(a, k)) for k in ("types", "slowdown", "power", "run1_csv")}
    ctx = _inputs(paths)
    with open(a.conditions, encoding="utf-8") as fh:
        conds = json.load(fh)
    if not isinstance(conds, list) or not conds:
        raise C.ContractError("--conditions must hold a non-empty list")
    size = (a.n_requests, a.n_warmup, a.n_reps, a.seed)
    jobs = [(paths, c, size) for c in conds]
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(_job, jobs))
    else:
        results = [_job(j) for j in jobs]
    for _, line in results:
        print(line)
    d = Path(a.run1_csv)
    inputs = {"types_sha256": C.sha256_file(a.types), "corun_sha256": C.sha256_file(d / "corun.csv"),
              "solo_sha256": C.sha256_file(d / "solo.csv"), "run1_plan_sha256": C.sha256_file(a.run1_plan),
              "slowdown_sha256": C.sha256_file(a.slowdown), "power_sha256": C.sha256_file(a.power)}
    if a.run1b_plan:
        inputs["run1b_plan_sha256"] = C.sha256_file(a.run1b_plan)
    return FZ.build_freeze(
        git=(git or _git)(), inputs=inputs, slowdown_model=_slowdown_block(ctx.slowdown_obj),
        sim={"n_requests": a.n_requests, "n_warmup": a.n_warmup, "n_reps": a.n_reps, "seed": a.seed},
        constraint=dict(CONSTRAINT), f_grid_mhz=sorted(ctx.types.corun_levels_mhz),
        conditions=[c for c, _ in results])


def main(argv=None, *, git=None):
    ap = argparse.ArgumentParser(prog="exp2.model.freeze_build")
    for name in ("run1-csv", "types", "slowdown", "power", "conditions", "run1-plan", "out"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--run1b-plan", default=None)
    ap.add_argument("--seed", type=int, required=True)
    for name, default in (("n-requests", C.RUN2_N_MEASURED), ("n-warmup", C.RUN2_N_WARMUP), ("n-reps", C.R_SIM)):
        ap.add_argument("--" + name, type=int, default=default)
    ap.add_argument("--workers", type=int, default=1)
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    try:
        obj = build(a, git)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        FZ.write_freeze(obj, str(out / "freeze.json"))
    except (C.ContractError, ValueError, KeyError, OSError, RuntimeError) as e:
        print("freeze_build: %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 2
    print("runtime_estimate_s=%s (%.1f h)" % (obj["runtime_estimate_s"], obj["runtime_estimate_s"] / 3600.0))
    print("freeze %s content_sha256=%s" % (out / "freeze.json", obj["content_sha256"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
