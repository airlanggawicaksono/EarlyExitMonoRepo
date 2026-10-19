"""Run 1 plan (exp2.plan/1, kind closed): solo, idle and co-run conditions (rules D1, D2, D10 of the fixtures README)."""
import argparse
import itertools
import json
import sys

import numpy as np

from exp2 import contracts as C


def make_cells(exits):
    """All distinct-family co-run cells of size 2..len(families): one exit per family. Order: size, then families."""
    fams = [f for f in C.FAMILIES if exits.get(f)]
    cells = []
    for size in range(2, len(fams) + 1):
        for combo in itertools.combinations(fams, size):
            for pick in itertools.product(*[exits[f] for f in combo]):
                cells.append(tuple(C.type_id(f, e) for f, e in zip(combo, pick)))
    return cells


def split_cells(cells, heldout_frac, seed):
    """Stratified by |A|: round(frac * n) cells per size are 'heldout'. Returns {cell: split}."""
    rng = np.random.default_rng(seed)
    out = {}
    for size in sorted({len(c) for c in cells}):
        group = [c for c in cells if len(c) == size]
        k = int(round(heldout_frac * len(group)))
        held = set(int(i) for i in rng.permutation(len(group))[:k])
        for i, c in enumerate(group):
            out[c] = "heldout" if i in held else "train"
    return out


def plan_run1(exits, solo_levels, corun_levels, heldout_level=None, repeats=3, heldout_frac=0.3, duration_s=30.0,
              seed=0, settle_s=5.0, idle_window_s=30.0, n_warmup=5, emc_mhz=3199.0, duration_fn=None):
    """duration_fn(kind, cell) with kind in idle|solo|corun overrides duration_s (fixture scale)."""
    dur = duration_fn or (lambda kind, cell: duration_s)
    levels = list(solo_levels) + ([heldout_level] if heldout_level is not None else [])
    cells = make_cells(exits)
    splits = split_cells(cells, heldout_frac, seed)
    conds = [{"condition_id": "idle_f%d" % f, "mode": "idle", "arm": "pinned", "gpu_mhz": f,
              "duration_s": dur("idle", None)} for f in levels]
    tids = [C.type_id(f, e) for f in C.FAMILIES for e in exits.get(f, [])]

    def closed(cid, f, cell, split, rep, kind):
        return {"condition_id": cid, "mode": "closed", "arm": "pinned", "gpu_mhz": f, "cell": list(cell),
                "duration_s": dur(kind, cell), "n_warmup": n_warmup, "split": split, "repeat": rep}

    for tid in tids:
        for f in levels:
            split = "solo_heldout_level" if f == heldout_level else "solo"
            for r in range(repeats):
                cid = "solo_%s_f%d_r%d" % (tid.replace(":", "-"), f, r)
                conds.append(closed(cid, f, (tid,), split, r, "solo"))
    for i, cell in enumerate(cells, start=1):
        for f in corun_levels:
            for r in range(repeats):
                conds.append(closed("c%03d_f%d_r%d" % (i, f, r), f, cell, splits[cell], r, "corun"))
    wall = 2 * idle_window_s + sum(settle_s + c["duration_s"] for c in conds)
    plan = {"schema": C.SCHEMAS["plan"], "kind": "closed",
            "session": {"families": [f for f in C.FAMILIES if exits.get(f)],
                        "exits": {f: list(exits[f]) for f in C.FAMILIES if exits.get(f)},
                        "emc_mhz": emc_mhz, "settle_s": settle_s, "idle_window_s": idle_window_s, "n_inputs": 32,
                        "preprocess": "per_request"},
            "freeze": None, "conditions": conds, "estimated_wall_s": wall}
    return C.with_plan_sha256(plan)


def _levels(s):
    return [int(x) for x in s.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exp2.model.plan_run1")
    ap.add_argument("--exits", nargs="+", required=True, help="family=e1,e2 ...")
    ap.add_argument("--solo-levels", type=_levels, default=[408, 612, 816, 1020])
    ap.add_argument("--heldout-level", type=int, default=714)
    ap.add_argument("--corun-levels", type=_levels, default=[612, 1020])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--heldout-frac", type=float, default=0.3)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    try:
        args = ap.parse_args(argv)
        exits = {}
        for item in args.exits:
            fam, _, es = item.partition("=")
            exits[fam] = _levels(es)
    except (SystemExit, ValueError):
        return 2
    plan = plan_run1(exits, args.solo_levels, args.corun_levels, args.heldout_level, args.repeats,
                     args.heldout_frac, args.duration, args.seed)
    C.validate_plan(plan)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(plan, fh, indent=2)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
