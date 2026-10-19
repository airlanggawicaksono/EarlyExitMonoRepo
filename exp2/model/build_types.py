"""Build the TypeTable (exp2.types/1) from the Run 1 CSVs (rules D9 of tests/fixtures/README.md)."""
import argparse
import json
import statistics
import sys

from exp2 import contracts as C
from exp2 import stats

KLASS_RULE = "short iff tau_mean(f_max) <= median over types"
SOURCES = ("ckpt.csv", "corun.csv", "mem.csv", "solo.csv", "solo_samples.csv")


def build_types(run1_dir, alpha=5.0, heldout_level_mhz=None):
    rd = str(run1_dir)
    solo = C.read_csv(rd + "/solo.csv", "solo")
    samples = C.read_csv(rd + "/solo_samples.csv", "solo_samples")
    corun = C.read_csv(rd + "/corun.csv", "corun")
    ckpt = C.read_csv(rd + "/ckpt.csv", "ckpt")
    mem = C.read_csv(rd + "/mem.csv", "mem")
    f_levels = sorted({r["f_mhz"] for r in solo})
    corun_levels = sorted({r["f_mhz"] for r in corun})
    f_max = f_levels[-1]
    type_ids = list(dict.fromkeys(r["type_id"] for r in solo))
    mem_by = {(m["family"], m["exit"]): m for m in mem}

    types, tau_mean_max = [], {}
    for tid in type_ids:
        fam, ex = C.parse_type_id(tid)
        tau_s = {}
        for f in f_levels:
            xs = [r["tau_s"] for r in samples if r["type_id"] == tid and r["f_mhz"] == f]
            if xs:
                tau_s[str(f)] = {"p50": stats.p_nearest_rank(xs, 0.5), "mean": sum(xs) / len(xs),
                                 "p95": stats.p_nearest_rank(xs, 0.95), "samples": xs}
        tau_mean_max[tid] = tau_s[str(f_max)]["mean"]
        rows = [r for r in ckpt if r["type_id"] == tid and r["f_mhz"] == f_max]
        u = {}
        for f in f_levels:
            us = [r["u"] for r in solo if r["type_id"] == tid and r["f_mhz"] == f]
            if us:
                u[str(f)] = sum(us) / len(us)
        types.append({"type_id": tid, "family": fam, "exit": ex, "klass": None,
                      "ckpts": C.default_checkpoints(fam, ex), "tau_s": tau_s,
                      "ckpt_frac": {str(r["ckpt"]): r["frac_p50"] for r in rows},
                      "head_frac": rows[-1]["head_frac_p50"], "u": u,
                      "act_mb": float(mem_by[(fam, ex)]["act_mb"])})
    med = statistics.median(tau_mean_max.values())
    for t in types:
        t["klass"] = "short" if tau_mean_max[t["type_id"]] <= med else "long"
    families = {}
    for t in types:
        mb = float(mem_by[(t["family"], t["exit"])]["resident_mb"])
        families[t["family"]] = {"resident_mb": max(mb, families.get(t["family"], {}).get("resident_mb", 0.0))}
    return {"schema": C.SCHEMAS["types"], "alpha": float(alpha), "f_levels_mhz": f_levels,
            "corun_levels_mhz": corun_levels, "heldout_level_mhz": heldout_level_mhz, "f_max_mhz": f_max,
            "llama_tokens": C.LLAMA_TOKENS, "llama_prompt_tokens": C.LLAMA_PROMPT_TOKENS,
            "families": families, "types": types,
            "provenance": {"klass_rule": KLASS_RULE, "sources": list(SOURCES),
                           "sha256": {s: C.sha256_file(rd + "/" + s) for s in SOURCES}}}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exp2.model.build_types")
    ap.add_argument("--run1-csv", required=True)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 2
    table = build_types(args.run1_csv, args.alpha)
    C.validate_types(table)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(table, fh, indent=2)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
