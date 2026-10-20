"""Freeze file for Run 2 (BRIEF_v1 K16, slice E3c).

    build_freeze(...)      -> exp2.freeze/1 dict with content_sha256
    write_freeze(obj, p)   -> p (LF JSON) + p.sha256 (`<hex>  <name>\\n`)
    make_grid(f_levels)    -> the {1,2,inf}^2 x {fcfs,stf} x GRID_LEVELS_MHZ grid as PolicySpec json
    evaluate_condition     -> one `conditions[]` entry from injected F / rules / game / sim functions
    CLI                    python -m exp2.model.freeze verify --freeze F      (exit 0 ok, 3 FreezeError, 2 usage)

`build` on the command line needs the real search_F, baselines, game and simulate (E3a, E3b, E2); those are wired by
the INT slice, so this module only takes them as arguments. Standard library + numpy-free + exp2.contracts.
"""
import argparse
import datetime
import json
import sys
import time

from exp2 import contracts as C

GRID_K = (1, 2, C.K_INF)
GRID_ORDERS = ("fcfs", "stf")


def make_grid(f_levels=C.GRID_LEVELS_MHZ):
    """Grid points g0, g1, ...: order x k_short x k_long x f (static clock). k values K_SET; f defaults to
    C.GRID_LEVELS_MHZ (D32-2: 3 x 3 x 2 x 2 = 36 points). F and the frequency baselines use the co-run levels instead."""
    out = []
    for order in GRID_ORDERS:
        for ks in GRID_K:
            for kl in GRID_K:
                for f in f_levels:
                    spec = C.PolicySpec(f"g{len(out)}", ks, kl, order, None, "static", int(f), {})
                    out.append(spec.to_json())
    return out


def _same_policy(a, b):
    return {k: v for k, v in a.items() if k != "name"} == {k: v for k, v in b.items() if k != "name"}


def measured_runs(cond):
    """[(key, spec json, n_rep)] measured for one freeze condition (D11). Keys are the choices in order. A choice equal
    to an earlier measured spec (ignoring the name) is its alias; a grid point equal to a measured spec is that key's
    alias; other grid points are new keys. n_rep = max of C.REPEATS over the key and its aliases."""
    runs = []                                  # [key, spec, n_rep]
    for name, spec in list(cond["choices"].items()) + [(g["name"], g) for g in cond["grid"]]:
        for run in runs:
            if _same_policy(run[1], spec):
                run[2] = max(run[2], C.REPEATS.get(name, 1))
                break
        else:
            runs.append([name, spec, C.REPEATS.get(name, 1)])
    return [(k, s, n) for k, s, n in runs]


def measured_key_map(cond):
    """{every choice key and grid name: the measured key that runs its policy}."""
    out = {}
    runs = measured_runs(cond)
    for name, spec in list(cond["choices"].items()) + [(g["name"], g) for g in cond["grid"]]:
        out[name] = next(k for k, s, _ in runs if _same_policy(s, spec))
    return out


def policy_list(cond):
    """[(key, spec json)] of the measured keys of one freeze condition (see measured_runs)."""
    return [(k, s) for k, s, _ in measured_runs(cond)]


def n_runs(cond):
    return sum(n for _, _, n in measured_runs(cond))


def estimate_runtime_s(conditions, sim, settle_s=5.0, idle_window_s=30.0):
    """Sum over measured runs of settle + n_requests / rate, plus the two idle windows (README D12)."""
    total = 2 * idle_window_s
    for c in conditions:
        total += n_runs(c) * (settle_s + sim["n_requests"] / c["workload"]["rate_per_s"])
    return round(total, 1)


def _utc(ns):
    dt = datetime.datetime.fromtimestamp(ns / 1e9, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_freeze(*, git, inputs, slowdown_model, sim, constraint, f_grid_mhz, conditions, created_wall_ns=None,
                 runtime_estimate_s=None):
    """Assemble and hash the freeze. created_wall_ns defaults to the wall clock now; runtime_estimate_s defaults to
    estimate_runtime_s. The result passes contracts.validate_freeze or ContractError is raised."""
    ns = int(time.time_ns() if created_wall_ns is None else created_wall_ns)
    if runtime_estimate_s is None:
        runtime_estimate_s = estimate_runtime_s(conditions, sim)
    obj = {
        "schema": C.SCHEMAS["freeze"], "created_utc": _utc(ns), "created_wall_ns": ns, "git": git, "inputs": inputs,
        "slowdown_model": slowdown_model, "sim": sim, "constraint": constraint, "f_grid_mhz": list(f_grid_mhz),
        "conditions": list(conditions), "runtime_estimate_s": runtime_estimate_s}
    obj = C.with_content_sha256(obj)
    return C.validate_freeze(obj)


def write_freeze(obj, path):
    """Validate, write LF JSON to `path` and the sidecar `path.sha256`. Returns the sidecar hash."""
    C.validate_freeze(obj)
    text = json.dumps(obj, indent=2, allow_nan=False) + "\n"
    with open(path, "wb") as fh:
        fh.write(text.encode("utf-8"))
    return C.write_sidecar(path)


def _pred(res):
    j = res.to_json() if hasattr(res, "to_json") else res
    return {"per_type": j["per_type"], "energy_per_req_j": j["energy_per_req_j"]}


def evaluate_condition(cfg, *, f_fn, rules_fn, game_fn, sim_fn, grid):
    """One `conditions[]` entry. cfg = {condition_id, workload (K2 json), alpha}.
    f_fn(cfg, constraint) -> PolicySpec, constraint in ("complete_rate", "mean_completion");
    rules_fn(cfg) -> {key: PolicySpec} (default, serial, ...); game_fn(cfg) -> K16 `game` dict;
    sim_fn(spec, cfg) -> SimResult; grid = make_grid(...) output."""
    wl = cfg["workload"]
    t0 = time.perf_counter()
    f_spec = f_fn(cfg, "complete_rate")
    f_sens = f_fn(cfg, "mean_completion")
    t_f = time.perf_counter() - t0
    choices = {"F": f_spec, "F_sens": f_sens, **rules_fn(cfg)}
    choices = {k: (v.to_json() if hasattr(v, "to_json") else v) for k, v in choices.items()}
    t0 = time.perf_counter()
    game = game_fn(cfg)
    t_g = time.perf_counter() - t0
    cond = {"condition_id": cfg["condition_id"], "workload": wl, "rho": wl["rho"], "arrival": wl["arrival"],
            "cv_a": wl["cv_a"], "alpha": cfg["alpha"], "choices": choices, "grid": list(grid), "predictions": {},
            "game": game, "compute_s": {"F": round(t_f, 3), "game": round(t_g, 3)}}
    for key, spec in policy_list(cond):
        cond["predictions"][key] = _pred(sim_fn(C.PolicySpec.from_json(spec), cfg))
    return cond


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exp2.model.freeze")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--freeze", required=True)
    b = sub.add_parser("build")
    for a in ("types", "slowdown", "power", "conditions", "out"):
        b.add_argument("--" + a, required=True)
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    if args.cmd == "build":
        print("freeze build needs search_F, game and simulate (E3a, E3b, E2); wire them via build_freeze()",
              file=sys.stderr)
        return 2
    try:
        obj = C.verify_freeze_file(args.freeze)
    except C.FreezeError as e:
        print(f"FreezeError: {e}", file=sys.stderr)
        return 3
    print(f"ok {obj['content_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
