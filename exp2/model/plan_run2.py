"""Run-2 plan from a freeze (BRIEF_v1 K15, fixtures README D11, slice E3c).

    plan_run2(freeze, freeze_path)  -> exp2.plan/1 kind open, one condition per (freeze condition, policy, repeat)
    plan_confirm(freeze, open_summary, out_path=None, freeze_path="freeze.json")
                                    -> Run 2b plan (D32-5) or None when nothing needs a confirm run
    CLI  python -m exp2.model.plan_run2 --freeze F --out P      (exit 0, 2 usage, 3 freeze rejected)
         python -m exp2.model.plan_run2 --freeze F --confirm-from OPEN_SUMMARY --out P     (Run 2b; exit 0 also when
         nothing needs a confirm run: then no file is written and a line says so)

Confirm plan (D32-5): for every freeze condition whose cheapest measured feasible key (open_summary.oracle[fcid])
differs from the measured key of F (freeze.measured_key_map), and that has fewer than 3 repeats in open_summary
.per_condition, the key is re-measured: one open condition per repeat in C.CONFIRM_REPEATS (1, 2) with
"confirm_of" = the key, arrival seed = sim.seed + ARRIVAL_SEED_OFFSET + repeat (paired with Run 2), condition_id
"<fcid>_<key>_r<repeat>". Order: repeat 1 first, then repeat 2, each permuted with PCG64(order_seed + 1).

Policies per freeze condition: measured keys of freeze.measured_runs (aliases are not planned); n_rep from
C.REPEATS. Run order (D11): a key with several repeats puts repeat r in block r; the single runs are shuffled and
dealt round-robin to the blocks; each block is permuted; blocks run in order. All draws come from one
numpy PCG64(order_seed) in this order: shuffle of the singles, then block 0, 1, 2. order_seed = freeze.sim.seed + 23.
Arrival seed = freeze.sim.seed + 11 + repeat (paired across keys).
"""
import argparse
import json
import os
import sys

from exp2 import contracts as C
import numpy as np

from exp2.model.freeze import measured_key_map, measured_runs

SESSION = {"emc_mhz": 3199.0, "settle_s": 5.0, "idle_window_s": 30.0, "n_inputs": 32, "preprocess": "per_request"}


def _order(runs, order_seed):
    """runs: [(fc index, key, spec, n_rep, repeat)] in canonical order. Returns them in execution order."""
    rng = np.random.Generator(np.random.PCG64(order_seed))
    blocks = [[] for _ in range(C.N_BLOCKS)]
    singles = []
    for r in runs:
        if r[3] > C.N_BLOCKS:
            raise ValueError(f"{r[1]}: {r[3]} repeats do not fit {C.N_BLOCKS} blocks")
        (blocks[r[4]] if r[3] > 1 else singles).append(r)
    for i, j in enumerate(rng.permutation(len(singles))):
        blocks[i % C.N_BLOCKS].append(singles[j])
    out = []
    for blk in blocks:
        out += [blk[j] for j in rng.permutation(len(blk))]
    return out


def plan_run2(freeze, freeze_path="freeze.json"):
    C.validate_freeze(freeze)
    sim = freeze["sim"]
    exits = {}
    fcs = freeze["conditions"]
    runs = []
    for i, fc in enumerate(fcs):
        for t in fc["workload"]["types"]:
            fam, ex = C.parse_type_id(t)
            exits.setdefault(fam, set()).add(ex)
        for key, spec, n_rep in measured_runs(fc):
            runs += [(i, key, spec, n_rep, rep) for rep in range(n_rep)]
    order_seed = sim["seed"] + C.ORDER_SEED_OFFSET
    conds = []
    for i, key, spec, _, rep in _order(runs, order_seed):
        fc = fcs[i]
        pinned = spec["f_rule"] == "static"
        conds.append({
            "condition_id": f"{fc['condition_id']}_{key}_r{rep}", "mode": "open",
            "arm": "pinned" if pinned else "governor", "gpu_mhz": spec["f_mhz"] if pinned else None,
            "policy_key": key, "freeze_condition_id": fc["condition_id"], "policy": spec,
            "workload": fc["workload"], "alpha": fc["alpha"], "n_requests": sim["n_requests"],
            "n_warmup": sim["n_warmup"], "seed": sim["seed"] + C.ARRIVAL_SEED_OFFSET + rep, "repeat": rep})
    fams = [f for f in C.FAMILIES if f in exits]
    plan = {
        "schema": C.SCHEMAS["plan"], "kind": "open",
        "session": {"families": fams, "exits": {f: sorted(exits[f]) for f in fams}, **SESSION},
        "freeze": {"path": freeze_path, "content_sha256": freeze["content_sha256"]},
        "conditions": conds, "estimated_wall_s": freeze["runtime_estimate_s"], "order_seed": order_seed}
    return C.validate_plan(C.with_plan_sha256(plan))


def plan_confirm(freeze, open_summary, out_path=None, freeze_path="freeze.json"):
    """The Run 2b plan (see module docstring), validated; written to out_path when given. None if no condition needs
    a confirm run. A condition whose oracle is missing (no feasible measured key) is skipped."""
    C.validate_freeze(freeze)
    sim = freeze["sim"]
    oracle = open_summary.get("oracle") or {}
    per = open_summary.get("per_condition") or {}
    exits, jobs, wall = {}, [], 2 * SESSION["idle_window_s"]
    for fc in freeze["conditions"]:
        fcid = fc["condition_id"]
        best = (oracle.get(fcid) or {}).get("policy_key")
        if best is None:
            continue
        kmap = measured_key_map(fc)
        best = kmap.get(best, best)
        if best == kmap["F"]:
            continue
        if (per.get(f"{fcid}_{best}") or {}).get("n_repeats", 1) >= max(C.CONFIRM_REPEATS) + 1:
            continue                                  # already measured as often as F: nothing to equalise
        spec = next(sp for k, sp, _ in measured_runs(fc) if k == best)
        for t in fc["workload"]["types"]:
            fam, ex = C.parse_type_id(t)
            exits.setdefault(fam, set()).add(ex)
        jobs += [(fc, best, spec, rep) for rep in C.CONFIRM_REPEATS]
        wall += len(C.CONFIRM_REPEATS) * (SESSION["settle_s"] + sim["n_requests"] / fc["workload"]["rate_per_s"])
    if not jobs:
        return None
    order_seed = sim["seed"] + C.ORDER_SEED_OFFSET + 1
    rng = np.random.Generator(np.random.PCG64(order_seed))
    conds = []
    for rep in C.CONFIRM_REPEATS:
        group = [j for j in jobs if j[3] == rep]
        for i in rng.permutation(len(group)):
            fc, key, spec, _ = group[i]
            pinned = spec["f_rule"] == "static"
            conds.append({
                "condition_id": f"{fc['condition_id']}_{key}_r{rep}", "mode": "open",
                "arm": "pinned" if pinned else "governor", "gpu_mhz": spec["f_mhz"] if pinned else None,
                "policy_key": key, "freeze_condition_id": fc["condition_id"], "policy": spec,
                "workload": fc["workload"], "alpha": fc["alpha"], "n_requests": sim["n_requests"],
                "n_warmup": sim["n_warmup"], "seed": sim["seed"] + C.ARRIVAL_SEED_OFFSET + rep, "repeat": rep,
                "confirm_of": key})
    fams = [f for f in C.FAMILIES if f in exits]
    plan = {
        "schema": C.SCHEMAS["plan"], "kind": "open",
        "session": {"families": fams, "exits": {f: sorted(exits[f]) for f in fams}, **SESSION},
        "freeze": {"path": freeze_path, "content_sha256": freeze["content_sha256"]},
        "conditions": conds, "estimated_wall_s": round(wall, 1), "order_seed": order_seed}
    plan = C.validate_plan(C.with_plan_sha256(plan))
    if out_path is not None:
        with open(out_path, "wb") as fh:
            fh.write((json.dumps(plan, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return plan


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exp2.model.plan_run2")
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--confirm-from", default=None, help="open_summary.json: write the Run 2b confirm plan instead")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    try:
        freeze = C.verify_freeze_file(args.freeze)
    except C.FreezeError as e:
        print(f"FreezeError: {e}", file=sys.stderr)
        return 3
    rel = os.path.relpath(args.freeze, os.path.dirname(os.path.abspath(args.out))).replace(os.sep, "/")
    if args.confirm_from:
        try:
            with open(args.confirm_from, encoding="utf-8") as fh:
                summary = json.load(fh)
        except (OSError, ValueError) as e:
            print(f"plan_run2: {e}", file=sys.stderr)
            return 2
        if plan_confirm(freeze, summary, args.out, rel) is None:
            print("no confirm run needed: the best measured key is F's key or already has 3 repeats")
        return 0
    plan = plan_run2(freeze, rel)
    with open(args.out, "wb") as fh:
        fh.write((json.dumps(plan, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
