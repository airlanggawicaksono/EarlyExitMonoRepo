"""exp2.model.coexit_pilot: the CoExit pilot runner (PLAN PL1 f). Simulator only: run it on the board, not the laptop.

    python -m exp2.model.coexit_pilot --types T --slowdown M1.json --power P.json --deadlines bert=S,yolo=S,llama=S
        [--deadline-scales 1.0] [--mixes A,B] [--rhos 0.3,0.6,0.9,1.2,1.5] [--gamma-mix B] [--gamma-rhos 0.9,1.2]
        [--gamma-cv 2.0] [--methods all_final,all_early,edgeserving,greedy,coexit] [--reps 10] [--n-requests 550]
        [--n-warmup 50] [--seed 20261022] [--f-mhz 1020] [--no-sample-slowdown] [--sim-slowdown m1|cells]
        [--policy-slowdown m1|cells] [--run1-dir exp2/frozen/run2/run1_csv] [--smoke] [--summarize-only] --out DIR
                                                                                            exit 0 ok, 2 usage/input

Grid     deadline scale x cell x rep x method. Cells: every --mixes mix with Poisson arrivals at every --rhos, then
         --gamma-mix with Gamma arrivals (cv --gamma-cv) at every --gamma-rhos (--gamma-mix "" adds none).
         MIXES A = bert:23 0.5, yolo:5 0.5; B = llama:15 0.2, bert:23 0.4, yolo:5 0.4.
         rate = contracts.rate_for_load(rho, mix, types). Deadline of a request = t_arrival + deadline_s[family] x scale.
CRN      rep r: exp2.arrivals.schedule(mix, n, seed + r) and SimConfig.seed = seed + r, identical for every method.
Sim      one simulate() per (method, cell, rep), replay mode, n_reps 1, family_exclusive. Policy =
         exits.ExitPolicy(exits.pilot_specs(f_mhz)[method], types, model). The simulator truth is M1 (--slowdown),
         or with --sim-slowdown cells exp2.model.cell_slowdown.CellSlowdown.from_run1_dir(run1_dir, m1) (imported
         only then). The policy model is M1, or with --policy-slowdown cells (needs --sim-slowdown cells) that same
         CellSlowdown object: greedy and coexit then predict with the truth (oracle; the gap to m1 is what slowdown
         prediction error costs). CellSlowdown.predict gives 1.0 alone, else the median s_p50 of the measured key
         (A, f, type), else M1 (fallback; its coverage() then also counts the policy's calls). Run 1 measured every
         cross-family set of the pilot menu at 612, 816 and 1020 MHz, so a family-exclusive policy falls back to M1
         only at another --f-mhz. The policy predicts the median; with sampling the simulator draws one per-request
         sample of the same key.
Metrics  contracts "Pilot metrics" on the trace (metrics()): one row per family of the mix, then one for "all".
         deadline_s is empty on the "all" row. A family with no measured request has n 0 and empty rates.
Outputs  DIR must be outside the repo. --smoke writes DIR/smoke and always starts it fresh.
         reps.csv     C.COEXIT_PILOT_COLS. A key (method, mix, arrival, cv_a, rho, deadline_scale, rep, seed) with a
                      complete "all" row is skipped on resume; the rows of an unfinished run are dropped. Appended and
                      flushed after every simulation.
         summary.csv  C.COEXIT_SUMMARY_COLS over the complete runs in reps.csv. viol and depth: stats.mean_ci over
                      reps. d_viol_vs_edgeserving: stats.paired_diff_ci(viol of the method, viol of edgeserving),
                      paired by (rep, seed). A CI needs 2 values, else it is empty.
         meta.json    args, config, git, sha256 of the inputs, python, start/end/wall. A resume whose config (sizes,
                      seed, f, slowdown truth, policy model, deadlines, input hashes) differs from meta.json exits 2.
--smoke  n 60, warm-up 10, 2 reps, first and last rho of each list; prints the time per simulation and a projected
         full-run time (scaled linearly in n_requests, rough).
Warning  (stderr and meta.json, the run continues) when the solo p95 of a floor exit at f is above its deadline.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

from exp2 import arrivals
from exp2 import contracts as C
from exp2 import exits
from exp2 import stats
from exp2.model import slowdown as SD
from exp2.model.power import PowerModel

__all__ = ["MIXES", "Inputs", "metrics", "parse_args", "load_inputs", "run_pilot", "summarize", "main"]

REPO_ROOT = Path(__file__).resolve().parents[2]
MIXES = {"A": (("bert:23", 0.5), ("yolo:5", 0.5)),
         "B": (("llama:15", 0.2), ("bert:23", 0.4), ("yolo:5", 0.4))}
SMOKE_N, SMOKE_WARMUP, SMOKE_REPS = 60, 10, 2
KEY_COLS = ("method", "mix", "arrival", "cv_a", "rho", "deadline_scale", "rep", "seed")
GROUP_COLS = ("method", "mix", "arrival", "cv_a", "rho", "deadline_scale", "family")
BASELINE = "edgeserving"
_FLOOR = {fam: "%s:%d" % (fam, ex[0]) for fam, ex in C.EXIT_MENU.items()}
_NS = 1e9

Inputs = namedtuple("Inputs", "types m1 power")


# ------------------------------------------------------------------------------------------------ metrics
def _block(rows):
    """Pilot metrics of one group of measured trace rows (contracts "Pilot metrics")."""
    out = {"n": len(rows), "n_success": 0, "slo_violation_ratio": None, "p95_R_s": None, "p99_R_s": None,
           "mean_R_s": None, "mean_norm_depth": None, "share_floor": None, "n_admitted": 0, "n_complete_late": 0,
           "n_truncated": 0, "n_miss": 0, "n_expired": 0}
    if not rows:
        return out
    R, depth, n_floor = [], 0.0, 0
    for r in rows:
        oc, dl = r["outcome"], r["t_deadline_ns"]
        R.append(((dl if oc == "expired_in_queue" else r["t_done_ns"]) - r["t_arrival_ns"]) / _NS)
        if oc == "expired_in_queue":
            out["n_expired"] += 1
        elif oc == "truncated":
            out["n_truncated"] += 1
        elif oc == "miss":
            out["n_miss"] += 1
        run = r["run_type_id"]
        if run is not None:
            out["n_admitted"] += 1
            n_floor += run == _FLOOR[r["family"]]
        if oc == "complete":
            if dl is None or r["t_done_ns"] <= dl:
                out["n_success"] += 1
                depth += C.depth(*C.parse_type_id(run)) / C.depth(*C.parse_type_id(r["type_id"]))
            else:
                out["n_complete_late"] += 1
    n = len(rows)
    srt = sorted(R)
    out.update(slo_violation_ratio=1.0 - out["n_success"] / n, p95_R_s=stats.p_nearest_rank(srt, 0.95),
               p99_R_s=stats.p_nearest_rank(srt, 0.99), mean_R_s=math.fsum(R) / n, mean_norm_depth=depth / n,
               share_floor=n_floor / out["n_admitted"] if out["n_admitted"] else None)
    return out


def metrics(rows, n_warmup, families):
    """{family: metrics for each of `families`, "all": metrics} of simulator trace rows. The first n_warmup rows in
    arrival order are warm-up. Every row must carry run_type_id (null when it never ran)."""
    rows = sorted(rows, key=lambda r: r["t_arrival_ns"])
    missing = [r.get("req_id") for r in rows if "run_type_id" not in r]
    if missing:
        raise C.ContractError("%d trace rows have no run_type_id (first %r): the simulator must report the run type"
                              % (len(missing), missing[0]))
    meas = rows[n_warmup:]
    out = {fam: _block([r for r in meas if r["family"] == fam]) for fam in families}
    out["all"] = _block(meas)
    return out


# ------------------------------------------------------------------------------------------------ CLI
def _floats(s):
    try:
        v = [float(x) for x in s.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("expected comma-separated numbers, got %r" % s) from None
    if not v or not all(math.isfinite(x) and x > 0 for x in v):
        raise argparse.ArgumentTypeError("expected numbers > 0, got %r" % s)
    return v


def _names(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def _deadlines(s):
    out = {}
    for part in s.split(","):
        fam, eq, val = (x.strip() for x in part.partition("="))
        if not eq or fam not in C.EXIT_MENU or fam in out:
            raise argparse.ArgumentTypeError("expected family=seconds, once per family in %s; got %r"
                                             % (",".join(C.EXIT_MENU), part))
        try:
            x = float(val)
        except ValueError:
            x = math.nan
        if not (math.isfinite(x) and x > 0):
            raise argparse.ArgumentTypeError("deadline of %s must be seconds > 0, got %r" % (fam, val))
        out[fam] = x
    return out


def _parser():
    ap = argparse.ArgumentParser(prog="exp2.model.coexit_pilot", description="CoExit pilot runner (simulator).")
    for name in ("types", "slowdown", "power", "out"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--deadlines", type=_deadlines, required=True, help="bert=S,yolo=S,llama=S (seconds, no default)")
    ap.add_argument("--deadline-scales", type=_floats, default="1.0")
    ap.add_argument("--mixes", type=_names, default="A,B")
    ap.add_argument("--rhos", type=_floats, default="0.3,0.6,0.9,1.2,1.5")
    ap.add_argument("--gamma-mix", default="B")
    ap.add_argument("--gamma-rhos", type=_floats, default="0.9,1.2")
    ap.add_argument("--gamma-cv", type=float, default=2.0)
    ap.add_argument("--methods", type=_names, default=",".join(exits.METHODS))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--n-requests", type=int, default=550)
    ap.add_argument("--n-warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--f-mhz", type=int, default=1020)
    ap.add_argument("--no-sample-slowdown", action="store_true")
    ap.add_argument("--sim-slowdown", choices=("m1", "cells"), default="m1")
    ap.add_argument("--policy-slowdown", choices=("m1", "cells"), default="m1")
    ap.add_argument("--run1-dir", default="exp2/frozen/run2/run1_csv")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    return ap


def parse_args(argv=None):
    return _parser().parse_args(argv)


def _families(mix):
    return [C.parse_type_id(t)[0] for t, _ in MIXES[mix]]


def _ends(xs):
    return list(xs) if len(xs) < 2 else [xs[0], xs[-1]]


def _cells(a, smoke):
    """[(mix, arrival, cv_a, rho)] in run order."""
    pick = _ends if smoke else list
    out = [(m, "poisson", 1.0, rho) for m in a.mixes for rho in pick(a.rhos)]
    if a.gamma_mix:
        out += [(a.gamma_mix, "gamma", float(a.gamma_cv), rho) for rho in pick(a.gamma_rhos)]
    return out


def _inside_repo(path):
    root = os.path.normcase(os.path.realpath(REPO_ROOT))
    p = os.path.normcase(os.path.realpath(path))
    try:
        return os.path.commonpath([root, p]) == root
    except ValueError:                                      # another drive on Windows
        return False


def _check(a):
    """Cross-field checks after parsing; ContractError -> exit 2. Nothing is written before this passes."""
    bad = [m for m in a.methods if m not in exits.METHODS]
    if not a.methods or bad:
        raise C.ContractError("--methods: unknown %s (known: %s)" % (bad, ",".join(exits.METHODS)))
    mixes = list(a.mixes) + ([a.gamma_mix] if a.gamma_mix else [])
    bad = [m for m in mixes if m not in MIXES]
    if bad:
        raise C.ContractError("unknown mix %s (known: %s)" % (bad, ",".join(MIXES)))
    missing = sorted({f for m in mixes for f in _families(m)} - set(a.deadlines))
    if missing:
        raise C.ContractError("--deadlines has no deadline for %s" % ",".join(missing))
    if a.reps < 1 or a.n_warmup < 0 or a.n_requests <= a.n_warmup:
        raise C.ContractError("need --reps >= 1 and 0 <= --n-warmup < --n-requests")
    if not (math.isfinite(a.gamma_cv) and a.gamma_cv > 0):
        raise C.ContractError("--gamma-cv must be > 0")
    if a.policy_slowdown == "cells" and a.sim_slowdown != "cells":
        raise C.ContractError("--policy-slowdown cells needs --sim-slowdown cells (the policy predicts with the "
                              "simulator truth)")
    if not _cells(a, smoke=False):
        raise C.ContractError("no cell to run: give --mixes or --gamma-mix")
    if _inside_repo(a.out):
        raise C.ContractError("--out %s is inside the repo %s; write the outputs outside it" % (a.out, REPO_ROOT))


def load_inputs(a):
    """Type table, the slowdown model of --slowdown (M1: the policy's prediction and the default truth), power."""
    types = C.load_types(a.types)
    with open(a.slowdown, encoding="utf-8") as fh:
        m1 = SD.from_json(json.load(fh), types)
    with open(a.power, encoding="utf-8") as fh:
        power = PowerModel.from_json(json.load(fh), types)
    return Inputs(types, m1, power)


# ------------------------------------------------------------------------------------------------ files
def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(float(v))                               # float() also strips a numpy scalar type
    return str(v)


def _key(row):
    return tuple(row[c] for c in KEY_COLS)


def _read_reps(path):
    """-> (rows of complete runs as str dicts, their keys, clean). A run is complete when its "all" row is whole.
    clean is False when a line was dropped or the file does not end with a newline (it must be rewritten)."""
    if not path.exists() or path.stat().st_size == 0:
        return [], set(), True
    with open(path, newline="", encoding="utf-8") as fh:
        text = fh.read()
    lines = text.splitlines()
    if tuple(next(csv.reader(lines[:1]), ())) != C.COEXIT_PILOT_COLS:
        raise C.ContractError("%s: header differs from contracts.COEXIT_PILOT_COLS" % path)
    data = [ln for ln in lines[1:] if ln]
    width = len(C.COEXIT_PILOT_COLS)
    rows = [dict(zip(C.COEXIT_PILOT_COLS, r)) for r in csv.reader(data) if len(r) == width]
    done = {_key(r) for r in rows if r["family"] == "all" and r["sim_runtime_s"] != ""}
    kept = [r for r in rows if _key(r) in done]
    return kept, done, len(kept) == len(data) and text.endswith("\n")


def _rewrite(path, rows):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(C.COEXIT_PILOT_COLS)
        wr.writerows([r[c] for c in C.COEXIT_PILOT_COLS] for r in rows)
    os.replace(tmp, path)


def _write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def _nums(rows, col):
    return [float(r[col]) for r in rows if r[col] != ""]


def _mean(xs):
    return math.fsum(xs) / len(xs) if xs else None


def _ci(xs):
    return stats.mean_ci(xs) if len(xs) >= 2 else (_mean(xs), None, None)


def summarize(reps_path, out_path):
    """Write summary.csv (C.COEXIT_SUMMARY_COLS) from the complete runs in reps.csv; -> number of rows."""
    rows = _read_reps(Path(reps_path))[0]
    if not rows:
        raise C.ContractError("%s has no complete run to summarize" % reps_path)
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[c] for c in GROUP_COLS), []).append(r)
    viol = {g: {(r["rep"], r["seed"]): float(r["slo_violation_ratio"]) for r in rs if r["slo_violation_ratio"] != ""}
            for g, rs in groups.items()}
    out = []
    for g, rs in groups.items():
        mine, base = viol[g], viol.get((BASELINE,) + g[1:], {})
        pairs = sorted(set(mine) & set(base), key=lambda k: (int(k[0]), int(k[1])))
        a, b = [mine[k] for k in pairs], [base[k] for k in pairs]
        d = stats.paired_diff_ci(a, b) if len(pairs) >= 2 else (a[0] - b[0] if pairs else None, None, None)
        v, dep = _ci(_nums(rs, "slo_violation_ratio")), _ci(_nums(rs, "mean_norm_depth"))
        row = dict(zip(GROUP_COLS, g), n_reps=len(rs), viol_mean=v[0], viol_ci_lo=v[1], viol_ci_hi=v[2],
                   depth_mean=dep[0], depth_ci_lo=dep[1], depth_ci_hi=dep[2],
                   p95_R_s_mean=_mean(_nums(rs, "p95_R_s")), p99_R_s_mean=_mean(_nums(rs, "p99_R_s")),
                   share_floor_mean=_mean(_nums(rs, "share_floor")), d_viol_vs_edgeserving_mean=d[0],
                   d_viol_vs_edgeserving_lo=d[1], d_viol_vs_edgeserving_hi=d[2])
        out.append([_fmt(row[c]) for c in C.COEXIT_SUMMARY_COLS])
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(C.COEXIT_SUMMARY_COLS)
        wr.writerows(out)
    return len(out)


# ------------------------------------------------------------------------------------------------ meta
def _git():
    def git(*args):
        return subprocess.run(["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60)
    try:
        sha = git("rev-parse", "HEAD").stdout.strip() or None
        dirty = bool(git("status", "--porcelain").stdout.strip()) if sha else None
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": sha, "dirty": dirty}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _input_hashes(a):
    out = {k: C.sha256_file(getattr(a, k)) for k in ("types", "slowdown", "power")}
    if a.sim_slowdown == "cells":
        for p in sorted(Path(a.run1_dir).glob("*.csv")):
            out["run1_dir/" + p.name] = C.sha256_file(p)
    return out


def _check_config(meta_path, config):
    """A resume must not mix runs made with another config."""
    if not meta_path.exists():
        return
    with open(meta_path, encoding="utf-8") as fh:
        old = json.load(fh).get("config")
    new = json.loads(json.dumps(config))
    if old is not None and old != new:
        diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
        raise C.ContractError("%s holds runs made with another config (%s differ); use a new --out"
                              % (meta_path.parent, ", ".join(diff)))


def _floor_warnings(types, a, cells):
    out = []
    for fam in sorted({f for c in cells for f in _families(c[0])}):
        tid = _FLOOR[fam]
        p95 = types.tau(tid, a.f_mhz, "p95")
        for s in a.deadline_scales:
            d = a.deadlines[fam] * s
            if p95 > d:
                out.append("warning: solo p95 of the floor exit %s at %d MHz is %.4f s > deadline %.4f s (scale %s): "
                           "it misses even alone" % (tid, a.f_mhz, p95, d, s))
    return out


def _projection(a, times):
    per = {m: _mean(ts) for m, ts in times.items() if ts}
    full = len(a.deadline_scales) * len(_cells(a, smoke=False)) * a.reps
    lines = ["smoke: %s %.3f s per simulation (n %d)" % (m, s, SMOKE_N) for m, s in per.items()]
    total = sum(per.values()) * full * a.n_requests / SMOKE_N
    lines.append("smoke: projected full run %.0f s (%.2f h) for %d simulations (time scaled linearly from n %d to "
                 "n %d; rough)" % (total, total / 3600.0, full * len(per), SMOKE_N, a.n_requests))
    return lines


# ------------------------------------------------------------------------------------------------ runner
def _workload(mix, arrival, cv, rho, types):
    wl = {"name": "mix" + mix, "types": [t for t, _ in MIXES[mix]], "probs": [p for _, p in MIXES[mix]],
          "rho": rho, "arrival": arrival, "cv_a": cv}
    wl["rate_per_s"] = C.rate_for_load(rho, wl, types)
    return wl


def _schedule(wl, n, seed, deadlines, scale):
    items = arrivals.schedule(wl, n, seed)
    for it in items:
        d = deadlines[C.parse_type_id(it["type_id"])[0]] * scale
        it["t_deadline_ns"] = it["t_arrival_ns"] + int(round(d * _NS))
    return tuple(items)


def run_pilot(a, inputs, *, simulate_fn=None, policy_factory=None):
    """Run the grid of the parsed and checked args `a`; write reps.csv, summary.csv, meta.json. -> output dir.
    simulate_fn defaults to exp2.model.simulator.simulate, policy_factory(spec, types, model) to exits.ExitPolicy;
    model is M1, or the simulator truth with --policy-slowdown cells."""
    out = Path(a.out) / "smoke" if a.smoke else Path(a.out)
    reps_path, summary_path, meta_path = out / "reps.csv", out / "summary.csv", out / "meta.json"
    if a.summarize_only:
        summarize(reps_path, summary_path)
        print("wrote %s" % summary_path)
        return out
    t_start, start = time.perf_counter(), _now()
    if simulate_fn is None:
        from exp2.model.simulator import simulate as simulate_fn
    if policy_factory is None:
        policy_factory = exits.ExitPolicy
    types, m1, power = inputs
    n, w, n_reps = (SMOKE_N, SMOKE_WARMUP, SMOKE_REPS) if a.smoke else (a.n_requests, a.n_warmup, a.reps)
    truth = m1
    if a.sim_slowdown == "cells":
        from exp2.model.cell_slowdown import CellSlowdown      # phase-2 module (S5), loaded only when asked for
        truth = CellSlowdown.from_run1_dir(a.run1_dir, m1)
    model = truth if a.policy_slowdown == "cells" else m1   # the policy's slowdown prediction
    specs = exits.pilot_specs(a.f_mhz)
    cells = _cells(a, a.smoke)
    warnings = _floor_warnings(types, a, cells)
    for msg in warnings:
        print(msg, file=sys.stderr)
    hashes = _input_hashes(a)
    config = {"n_requests": n, "n_warmup": w, "seed": a.seed, "f_mhz": a.f_mhz,
              "sample_slowdown": not a.no_sample_slowdown, "sim_slowdown": a.sim_slowdown,
              "policy_slowdown": a.policy_slowdown, "deadlines_s": dict(sorted(a.deadlines.items())),
              "inputs_sha256": hashes}
    out.mkdir(parents=True, exist_ok=True)
    if a.smoke and reps_path.exists():
        reps_path.unlink()                                  # a smoke run always measures again
    rows, done, clean = _read_reps(reps_path)
    if done:
        _check_config(meta_path, config)
    if not clean:
        _rewrite(reps_path, rows)
    total = len(a.deadline_scales) * len(cells) * n_reps * len(a.methods)
    meta = {"args": vars(a), "config": config, "git": _git(), "inputs_sha256": hashes, "python": sys.version,
            "platform": sys.platform, "policy_slowdown": getattr(model, "name", type(model).__name__),
            "smoke": a.smoke, "warnings": warnings, "n_sims_total": total, "n_sims_run": 0, "n_sims_skipped": 0,
            "start": start, "end": None, "wall_s": None}
    _write_json(meta_path, meta)

    k = n_run = n_skip = 0
    times = {m: [] for m in a.methods}
    new_file = not reps_path.exists() or reps_path.stat().st_size == 0
    with open(reps_path, "a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        if new_file:
            wr.writerow(C.COEXIT_PILOT_COLS)
            fh.flush()
        for scale in a.deadline_scales:
            for mix, arrival, cv, rho in cells:
                wl = _workload(mix, arrival, cv, rho, types)
                fams = _families(mix)
                for r in range(n_reps):
                    seed = a.seed + r
                    items = None                            # one schedule per (cell, rep), shared by every method
                    for m in a.methods:
                        k += 1
                        if tuple(_fmt(v) for v in (m, mix, arrival, cv, rho, scale, r, seed)) in done:
                            n_skip += 1
                            continue
                        if items is None:
                            items = _schedule(wl, n, seed, a.deadlines, scale)
                        cfg = C.SimConfig(n_requests=n, n_warmup=w, n_reps=1, seed=seed, family_exclusive=True,
                                          sample_slowdown=not a.no_sample_slowdown, alpha=None, schedule=items)
                        policy = policy_factory(specs[m], types, model)
                        t0 = time.perf_counter()
                        res = simulate_fn(wl, specs[m], types, truth, power, cfg, policy=policy)
                        dt = time.perf_counter() - t0
                        blocks = metrics(res.trace, w, fams)
                        base = {"method": m, "mix": mix, "arrival": arrival, "cv_a": cv, "rho": rho,
                                "rate_per_s": wl["rate_per_s"], "deadline_scale": scale, "rep": r, "seed": seed,
                                "sim_runtime_s": dt}
                        for fam in fams + ["all"]:
                            row = dict(base, family=fam, deadline_s=None if fam == "all" else a.deadlines[fam] * scale,
                                       **blocks[fam])
                            wr.writerow([_fmt(row[c]) for c in C.COEXIT_PILOT_COLS])
                        fh.flush()
                        n_run += 1
                        times[m].append(dt)
                        print("[%d/%d] %s mix=%s %s cv=%s rho=%s scale=%s rep=%d: %.3f s"
                              % (k, total, m, mix, arrival, cv, rho, scale, r, dt), flush=True)
    if a.smoke:
        for line in _projection(a, times):
            print(line)
    summarize(reps_path, summary_path)
    meta.update(n_sims_run=n_run, n_sims_skipped=n_skip, end=_now(), wall_s=round(time.perf_counter() - t_start, 3))
    _write_json(meta_path, meta)
    print("ran %d, skipped %d; wrote %s, %s, %s" % (n_run, n_skip, reps_path, summary_path, meta_path))
    return out


def main(argv=None, *, inputs=None, simulate_fn=None, policy_factory=None):
    try:
        a = parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    try:
        _check(a)
        if inputs is None and not a.summarize_only:
            inputs = load_inputs(a)
        run_pilot(a, inputs, simulate_fn=simulate_fn, policy_factory=policy_factory)
    except C.ContractError as e:
        print("coexit_pilot: %s" % e, file=sys.stderr)
        return 2
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as e:
        traceback.print_exc()
        print("coexit_pilot: %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
