"""Run-2 plan from a freeze (BRIEF_v1 K15, fixtures README D11, slice E3c).

    plan_run2(freeze, freeze_path)  -> exp2.plan/1 kind open, one condition per (freeze condition, policy, repeat)
    CLI  python -m exp2.model.plan_run2 --freeze F --out P      (exit 0, 2 usage, 3 freeze rejected)

Policies per freeze condition: `choices` in order, then the grid points that differ from a choice. F and default are
measured 3 times, all others once. seed = freeze.sim.seed + 11 + repeat.
"""
import argparse
import json
import os
import sys

from exp2 import contracts as C
from exp2.model.freeze import REPEATS, policy_list

SESSION = {"emc_mhz": 3199.0, "settle_s": 5.0, "idle_window_s": 30.0, "n_inputs": 32, "preprocess": "per_request"}


def plan_run2(freeze, freeze_path="freeze.json"):
    C.validate_freeze(freeze)
    sim = freeze["sim"]
    exits = {}
    conds = []
    for fc in freeze["conditions"]:
        for t in fc["workload"]["types"]:
            fam, ex = C.parse_type_id(t)
            exits.setdefault(fam, set()).add(ex)
        for key, spec in policy_list(fc):
            pinned = spec["f_rule"] == "static"
            for rep in range(REPEATS.get(key, 1)):
                conds.append({
                    "condition_id": f"{fc['condition_id']}_{key}_r{rep}", "mode": "open",
                    "arm": "pinned" if pinned else "governor", "gpu_mhz": spec["f_mhz"] if pinned else None,
                    "policy_key": key, "freeze_condition_id": fc["condition_id"], "policy": spec,
                    "workload": fc["workload"], "alpha": fc["alpha"], "n_requests": sim["n_requests"],
                    "n_warmup": sim["n_warmup"], "seed": sim["seed"] + 11 + rep, "repeat": rep})
    fams = [f for f in C.FAMILIES if f in exits]
    plan = {
        "schema": C.SCHEMAS["plan"], "kind": "open",
        "session": {"families": fams, "exits": {f: sorted(exits[f]) for f in fams}, **SESSION},
        "freeze": {"path": freeze_path, "content_sha256": freeze["content_sha256"]},
        "conditions": conds, "estimated_wall_s": freeze["runtime_estimate_s"]}
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
