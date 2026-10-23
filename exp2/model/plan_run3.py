"""Run-3 plan: the CoExit pilot grid on the board (PLAN PL2 c, slice S4).

    plan_run3(freeze, types, freeze_path="freeze.json", reps=3, methods=exits.METHODS, smoke=False)
                                -> exp2.plan/1 kind open, validated
    skip_done(plan, sessions)   -> the plan without the conditions that have a valid manifest in a session dir,
                                   validated; None when nothing is left
    CLI  python -m exp2.model.plan_run3 --freeze F --types T --out P [--reps 3]
         [--methods all_final,all_early,edgeserving,greedy,coexit] [--smoke] [--skip-done S [S ..]]
         exit 0 ok (also when --skip-done leaves nothing: then no file is written and a line says so),
         2 usage or unreadable input, 3 freeze refused or sha256(--types) != freeze.inputs.types_sha256

Cells       CELLS (mix, arrival, cv_a, rho): mix A Poisson rho 0.6, 0.9, 1.2; mix B Poisson 0.6, 0.9, 1.2; mix B
            Gamma cv 2.0 rho 0.9. Workload = coexit_pilot._workload (rate from contracts.rate_for_load on --types).
Condition   one per (repeat, cell, method). policy = exits.pilot_specs(F_MHZ)[method], arm pinned at F_MHZ.
            alpha null, deadlines_s = DEADLINES_S of the families in the mix. n N_REQUESTS, warm-up N_WARMUP.
            seed = SEED + repeat: every method of a repeat sees the same arrivals (common random numbers), equal to
            the arrivals of pilot rep `repeat`. condition_id "<cell>_<method>_r<repeat>", freeze_condition_id
            "run3_<cell>", cell "<mix>_<p | g<10 x cv>>_rho<100 x rho, 3 digits>" (e.g. B_g20_rho090_coexit_r2).
Session     plan_run2.SESSION; families = the families of the mixes in C.FAMILIES order (bert, llama, yolo);
            exits = contracts.EXIT_MENU, so the floor and target exit of every family are loaded.
Order       block r = every (cell, method) of repeat r, in CELLS x methods order. One numpy PCG64(order_seed)
            permutes block 0, then block 1, then block 2. Blocks run in order. order_seed = SEED + ORDER_SEED_OFFSET.
Wall        estimated_wall_s = 2 x idle_window_s + sum over the conditions of (settle_s + n_requests / rate_per_s).
--methods   kept in exits.METHODS order. --reps R plans repeats 0 .. R-1 (R blocks).
--smoke     n SMOKE_N, warm-up SMOKE_WARMUP, repeat 0 only (--reps is ignored), cells SMOKE_CELLS,
            condition_id prefix "smoke_" (freeze_condition_id keeps "run3_<cell>").
--skip-done drops every condition whose <S>/<condition_id>/manifest.json passes contracts.validate_manifest with
            valid true and the same condition_id, for any S. The order of the rest is kept; the session is kept;
            estimated_wall_s is recomputed. A session dir that does not exist exits 2.
"""
import argparse
import json
import os
import sys

import numpy as np

from exp2 import contracts as C
from exp2 import exits
from exp2.model.coexit_pilot import _workload
from exp2.model.plan_run2 import SESSION

CELLS = (("A", "poisson", 1.0, 0.6), ("A", "poisson", 1.0, 0.9), ("A", "poisson", 1.0, 1.2),
         ("B", "poisson", 1.0, 0.6), ("B", "poisson", 1.0, 0.9), ("B", "poisson", 1.0, 1.2),
         ("B", "gamma", 2.0, 0.9))
SMOKE_CELLS = (("A", "poisson", 1.0, 1.2), ("B", "poisson", 1.0, 1.2), ("B", "gamma", 2.0, 0.9))
DEADLINES_S = {"bert": 0.1, "yolo": 0.1, "llama": 3.35}     # PLAN PL2 b: per family, scale 1.0
F_MHZ = 1020
N_REQUESTS, N_WARMUP = 550, 50
SMOKE_N, SMOKE_WARMUP = 60, 10
SEED = 20261022                                               # = coexit_pilot --seed of the pilot v2 run
REPS = 3


def cell_id(mix, arrival, cv, rho):
    """'A_p_rho060' for (A, poisson, 1.0, 0.6); 'B_g20_rho090' for (B, gamma, 2.0, 0.9)."""
    arr = "p" if arrival == "poisson" else "g%d" % round(cv * 10)
    return "%s_%s_rho%03d" % (mix, arr, round(rho * 100))


def _wall(conds):
    runs = sum(SESSION["settle_s"] + c["n_requests"] / c["workload"]["rate_per_s"] for c in conds)
    return round(2 * SESSION["idle_window_s"] + runs, 1)


def plan_run3(freeze, types, freeze_path="freeze.json", reps=REPS, methods=exits.METHODS, smoke=False):
    C.validate_freeze(freeze)
    cells, n, w, n_reps, prefix = ((SMOKE_CELLS, SMOKE_N, SMOKE_WARMUP, 1, "smoke_") if smoke
                                   else (CELLS, N_REQUESTS, N_WARMUP, reps, ""))
    specs = exits.pilot_specs(F_MHZ)
    order_seed = SEED + C.ORDER_SEED_OFFSET
    rng = np.random.Generator(np.random.PCG64(order_seed))
    conds = []
    for rep in range(n_reps):
        block = []
        for cell in cells:
            cid = cell_id(*cell)
            for m in methods:
                wl = _workload(*cell, types)
                fams = sorted({C.parse_type_id(t)[0] for t in wl["types"]})
                block.append({
                    "condition_id": f"{prefix}{cid}_{m}_r{rep}", "mode": "open", "arm": "pinned", "gpu_mhz": F_MHZ,
                    "policy_key": m, "freeze_condition_id": f"run3_{cid}", "policy": specs[m].to_json(),
                    "workload": wl, "alpha": None, "deadlines_s": {f: DEADLINES_S[f] for f in fams},
                    "n_requests": n, "n_warmup": w, "seed": SEED + rep, "repeat": rep})
        conds += [block[i] for i in rng.permutation(len(block))]
    used = {C.parse_type_id(t)[0] for c in conds for t in c["workload"]["types"]}
    fams = [f for f in C.FAMILIES if f in used]
    plan = {
        "schema": C.SCHEMAS["plan"], "kind": "open",
        "session": {"families": fams, "exits": {f: list(C.EXIT_MENU[f]) for f in fams}, **SESSION},
        "freeze": {"path": freeze_path, "content_sha256": freeze["content_sha256"]},
        "conditions": conds, "estimated_wall_s": _wall(conds), "order_seed": order_seed}
    return C.validate_plan(C.with_plan_sha256(plan))


def _done(session_dir, cid):
    try:
        with open(os.path.join(session_dir, cid, "manifest.json"), encoding="utf-8") as fh:
            m = C.validate_manifest(json.load(fh))
    except (OSError, ValueError):                               # missing, partial or invalid (ContractError)
        return False
    return m["valid"] is True and m["condition_id"] == cid


def skip_done(plan, sessions):
    conds = [c for c in plan["conditions"] if not any(_done(s, c["condition_id"]) for s in sessions)]
    if not conds:
        return None
    out = {k: v for k, v in plan.items() if k != "sha256"}
    out.update(conditions=conds, estimated_wall_s=_wall(conds))
    return C.validate_plan(C.with_plan_sha256(out))


def _methods(s):
    ms = [x.strip() for x in s.split(",") if x.strip()]
    bad = [m for m in ms if m not in exits.METHODS]
    if not ms or bad or len(set(ms)) != len(ms):
        raise argparse.ArgumentTypeError("expected distinct methods from %s, got %r" % (",".join(exits.METHODS), s))
    return tuple(m for m in exits.METHODS if m in ms)


def _reps(s):
    try:
        v = int(s)
    except ValueError:
        v = 0
    if v < 1:
        raise argparse.ArgumentTypeError("expected an integer >= 1, got %r" % s)
    return v


def _rel(path, out):
    try:
        return os.path.relpath(path, os.path.dirname(os.path.abspath(out))).replace(os.sep, "/")
    except ValueError:                                          # another drive on Windows
        return os.path.abspath(path).replace(os.sep, "/")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exp2.model.plan_run3")
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--types", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=_reps, default=REPS)
    ap.add_argument("--methods", type=_methods, default=exits.METHODS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-done", nargs="+", default=[], metavar="SESSION_DIR")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    missing = [s for s in args.skip_done if not os.path.isdir(s)]
    if missing:
        print("plan_run3: --skip-done: no such session dir %s" % ", ".join(missing), file=sys.stderr)
        return 2
    try:
        freeze = C.verify_freeze_file(args.freeze)
    except C.FreezeError as e:
        print(f"FreezeError: {e}", file=sys.stderr)
        return 3
    try:
        got = C.sha256_file(args.types)
    except OSError as e:
        print(f"plan_run3: {e}", file=sys.stderr)
        return 2
    if got != freeze["inputs"]["types_sha256"]:
        print("FreezeError: sha256(--types) %s != freeze.inputs.types_sha256 %s"
              % (got, freeze["inputs"]["types_sha256"]), file=sys.stderr)
        return 3
    try:
        types = C.load_types(args.types)
        plan = plan_run3(freeze, types, _rel(args.freeze, args.out), args.reps, args.methods, args.smoke)
        n_all = len(plan["conditions"])
        if args.skip_done:
            plan = skip_done(plan, args.skip_done)
            if plan is None:
                print("nothing left to run: all %d conditions have a valid manifest in %s; no plan written"
                      % (n_all, " ".join(args.skip_done)))
                return 0
        with open(args.out, "wb") as fh:
            fh.write((json.dumps(plan, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    except (OSError, ValueError, KeyError) as e:
        print(f"plan_run3: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print("wrote %s: %d conditions (%d skipped as done), estimated_wall_s %.1f"
          % (args.out, len(plan["conditions"]), n_all - len(plan["conditions"]), plan["estimated_wall_s"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
