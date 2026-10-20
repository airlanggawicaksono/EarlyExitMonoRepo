"""Run-2 plan from a freeze (BRIEF_v1 K15, fixtures README D11, slice E3c).

    plan_run2(freeze, freeze_path)  -> exp2.plan/1 kind open, one condition per (freeze condition, policy, repeat)
    CLI  python -m exp2.model.plan_run2 --freeze F --out P      (exit 0, 2 usage, 3 freeze rejected)

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

from exp2.model.freeze import measured_runs

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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exp2.model.plan_run2")
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--out", required=True)
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
    plan = plan_run2(freeze, rel)
    with open(args.out, "wb") as fh:
        fh.write((json.dumps(plan, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
