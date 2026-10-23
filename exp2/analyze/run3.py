"""exp2.analyze.run3: the Run 3 board analyzer (PLAN PL2 d). contracts "Pilot metrics" on board request rows.

    python -m exp2.analyze.run3 --sessions S [S ..] --out DIR [--sim-dir PILOT_DIR]      exit 0 ok, 2 usage or data

Runs     every subdir of a session dir with manifest.json; sessions in the given order, subdirs by name. session.json
         is optional: session_id = its session_id, else the dir name. valid false -> kept 0, reason
         invalid:<invalid_reason>. A condition_id kept in an earlier run -> kept 0, duplicate. Screen-flagged runs
         are kept and reported (D8).
Trace    to_trace(rows): type_id = req_type_id (the arrival type) when present, else type_id; run_type_id = type_id
         (the RUN type) when admitted, else null. coexit_pilot.metrics(trace, manifest n_warmup, workload families).
         Error rows count as failures. Checks (ContractError, exit 2): arrival types are workload types, is_warmup
         marks the first n_warmup rows in arrival order, one relative deadline per family.
reps.csv     C.COEXIT_PILOT_COLS, numbers by coexit_pilot._fmt. method = policy_key, mix = workload name without
             "mix", deadline_scale 1.0, rep = manifest repeat, seed = manifest seed, deadline_s = (t_deadline -
             t_arrival) / 1e9 of the family (empty on "all"), sim_runtime_s = board wall time t_mono_end - start.
summary.csv  coexit_pilot.summarize(reps.csv).
runs.csv     C.RUN3_RUNS_COLS. screen.screen_window over [t_mono_start + settle_s, t_mono_end] with gpu 1020 MHz and
             the manifest emc_mhz. p99 arrival lag = t_arrival_actual - t_arrival, p99 dispatch = t_srv_recv -
             t_admit: ms, nearest rank, every row with both times. n_rows, n_error and the p99s only for kept runs.
compare.csv  (--sim-dir) C.RUN3_COMPARE_COLS: each board summary row against the pilot summary.csv row with the same
             (method, mix, arrival, cv_a, rho, family) at deadline scale 1.0 (sim cells empty when there is none).
             sim_viol_same_seeds = mean viol of the pilot reps with the seeds of the board reps of that row (seeds
             20261022..20261024 for reps 0-2); empty unless the pilot has every one. same_sign = 1 when the signs of
             board and sim d_viol_vs_edgeserving agree (0 counts as a sign), empty when one is missing.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

from exp2 import contracts as C
from exp2 import stats
from exp2.analyze import screen as SCREEN
from exp2.model import coexit_pilot as P

__all__ = ["to_trace", "analyze", "compare", "main"]

GPU_MHZ = 1020                 # Run 3 runs every method pinned at 1020 MHz, the pilot's f
DEADLINE_SCALE = 1.0
_NS = 1e9
_SCREEN_COLS = ("screen_ok", "gpu_lock_frac", "emc_lock_frac", "max_tj_c", "min_mem_available_mb")


def to_trace(rows):
    """Pilot trace rows (coexit_pilot.metrics input) of board request rows."""
    return [{"req_id": r["req_id"], "type_id": r.get("req_type_id", r["type_id"]),
             "run_type_id": r["type_id"] if r["t_admit_ns"] is not None else None, "family": r["family"],
             "outcome": r["outcome"], "t_arrival_ns": r["t_arrival_ns"], "t_deadline_ns": r["t_deadline_ns"],
             "t_done_ns": r["t_done_ns"]} for r in rows]


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _check(rows, trace, man):
    cid, wl, w = man["condition_id"], man["workload"], man["n_warmup"]
    bad = sorted({t["type_id"] for t in trace} - set(wl["types"]))
    if bad:
        raise C.ContractError("%s: arrival types %s are not workload types %s" % (cid, bad, wl["types"]))
    warm = [r["is_warmup"] for r in sorted(rows, key=lambda r: r["t_arrival_ns"])]
    if warm != [i < w for i in range(len(warm))]:
        raise C.ContractError("%s: is_warmup does not mark the first n_warmup=%d rows in arrival order" % (cid, w))


def _deadlines(rows, families, cid):
    """{family: relative deadline s}; exactly one value per family."""
    out = {}
    for fam in families:
        d = {None if r["t_deadline_ns"] is None else r["t_deadline_ns"] - r["t_arrival_ns"]
             for r in rows if r["family"] == fam}
        if len(d) != 1 or None in d:
            raise C.ContractError("%s: family %s needs one relative deadline, got %s ns"
                                  % (cid, fam, sorted(d, key=str)))
        out[fam] = d.pop() / _NS
    return out


def _p99_ms(rows, k1, k0):
    xs = [r[k1] - r[k0] for r in rows if r[k1] is not None and r[k0] is not None]
    return stats.p_nearest_rank(xs, 0.99) / 1e6 if xs else None


def _run(session_id, man, cdir, clocks, reason):
    """(runs.csv row, reps.csv rows) of one open condition; reps rows only when reason is None (kept)."""
    wl, t0, t1 = man["workload"], man["t_mono_start_ns"], man["t_mono_end_ns"]
    base = {"method": man["policy_key"], "mix": wl["name"].removeprefix("mix"), "arrival": wl["arrival"],
            "cv_a": float(wl["cv_a"]), "rho": float(wl["rho"]), "rep": man["repeat"], "seed": man["seed"]}
    run = dict(base, session_id=session_id, condition_id=man["condition_id"], kept=int(reason is None),
               reason=reason, n_rows=None, n_error=None, p99_arrival_lag_ms=None, p99_dispatch_ms=None,
               wall_s=None if t1 is None else (t1 - t0) / _NS, **dict.fromkeys(_SCREEN_COLS))
    if t1 is not None:
        scr = SCREEN.screen_window(clocks, t0 + round(man["settle_s"] * _NS), t1, gpu_mhz=GPU_MHZ,
                                   emc_mhz=man["emc_mhz"])
        run.update({k: scr[k] for k in _SCREEN_COLS})
    if reason is not None:
        return run, []
    rows = C.read_requests(cdir / "requests.jsonl")
    fams = [C.parse_type_id(t)[0] for t in wl["types"]]
    trace = to_trace(rows)
    _check(rows, trace, man)
    dls = _deadlines(rows, fams, man["condition_id"])
    blocks = P.metrics(trace, man["n_warmup"], fams)
    rep = dict(base, rate_per_s=float(wl["rate_per_s"]), deadline_scale=DEADLINE_SCALE, sim_runtime_s=run["wall_s"])
    reps = [dict(rep, family=f, deadline_s=dls.get(f), **blocks[f]) for f in fams + ["all"]]
    run.update(n_rows=len(rows), n_error=sum(r["outcome"] == "error" for r in rows),
               p99_arrival_lag_ms=_p99_ms(rows, "t_arrival_actual_ns", "t_arrival_ns"),
               p99_dispatch_ms=_p99_ms(rows, "t_srv_recv_ns", "t_admit_ns"))
    return run, reps


def _reason(man, seen):
    if not man["valid"]:
        return "invalid:%s" % man["invalid_reason"]
    if man["t_mono_end_ns"] is None:                  # never written by run_session; a kept run needs its wall time
        return "invalid:no t_mono_end_ns"
    return "duplicate" if man["condition_id"] in seen else None


def _write(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(cols)
        wr.writerows([P._fmt(r[c]) for c in cols] for r in rows)


def _read_csv(path, cols):
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        if tuple(rd.fieldnames or ()) != tuple(cols):
            raise C.ContractError("%s: header %s != %s" % (path, rd.fieldnames, list(cols)))
        return list(rd)


def _gkey(r):
    return r["method"], r["mix"], r["arrival"], float(r["cv_a"]), float(r["rho"]), r["family"]


def _sign(x):
    x = float(x)
    return (x > 0) - (x < 0)


def compare(out_dir, sim_dir):
    """compare.csv rows (C.RUN3_COMPARE_COLS) of out_dir/{summary,reps}.csv against sim_dir/{summary,reps}.csv."""
    out_dir, sim_dir = Path(out_dir), Path(sim_dir)
    seeds = {}
    for r in P._read_reps(out_dir / "reps.csv")[0]:
        seeds.setdefault(_gkey(r), set()).add(int(r["seed"]))
    sim = {_gkey(r): r for r in _read_csv(sim_dir / "summary.csv", C.COEXIT_SUMMARY_COLS)
           if float(r["deadline_scale"]) == DEADLINE_SCALE}
    sim_viol = {(_gkey(r), int(r["seed"])): float(r["slo_violation_ratio"])
                for r in P._read_reps(sim_dir / "reps.csv")[0]
                if float(r["deadline_scale"]) == DEADLINE_SCALE and r["slo_violation_ratio"] != ""}
    out, n_missing = [], 0
    for b in _read_csv(out_dir / "summary.csv", C.COEXIT_SUMMARY_COLS):
        k = _gkey(b)
        s = sim.get(k) or dict.fromkeys(C.COEXIT_SUMMARY_COLS, "")
        n_missing += k not in sim
        same = [sim_viol.get((k, sd)) for sd in sorted(seeds.get(k, ()))]
        bd, sdv = b["d_viol_vs_edgeserving_mean"], s["d_viol_vs_edgeserving_mean"]
        out.append({"method": b["method"], "mix": b["mix"], "arrival": b["arrival"], "cv_a": b["cv_a"],
                    "rho": b["rho"], "family": b["family"], "board_n_reps": b["n_reps"],
                    "board_viol_mean": b["viol_mean"], "board_viol_ci_lo": b["viol_ci_lo"],
                    "board_viol_ci_hi": b["viol_ci_hi"], "sim_viol_mean": s["viol_mean"],
                    "sim_viol_ci_lo": s["viol_ci_lo"], "sim_viol_ci_hi": s["viol_ci_hi"],
                    "sim_viol_same_seeds": math.fsum(same) / len(same) if same and None not in same else None,
                    "board_depth_mean": b["depth_mean"], "sim_depth_mean": s["depth_mean"],
                    "board_d_viol_vs_es_mean": bd, "board_d_viol_vs_es_lo": b["d_viol_vs_edgeserving_lo"],
                    "board_d_viol_vs_es_hi": b["d_viol_vs_edgeserving_hi"], "sim_d_viol_vs_es_mean": sdv,
                    "sim_d_viol_vs_es_lo": s["d_viol_vs_edgeserving_lo"],
                    "sim_d_viol_vs_es_hi": s["d_viol_vs_edgeserving_hi"],
                    "same_sign": int(_sign(bd) == _sign(sdv)) if bd != "" and sdv != "" else None})
    if n_missing:
        print("warning: %d board summary rows have no pilot summary row at deadline scale %s in %s"
              % (n_missing, DEADLINE_SCALE, sim_dir), file=sys.stderr)
    return out


def analyze(session_dirs, out_dir, sim_dir=None):
    """Write runs.csv, reps.csv, summary.csv (and compare.csv with sim_dir) into out_dir. -> runs.csv rows."""
    runs, reps, seen = [], [], set()
    for sdir in map(Path, session_dirs):
        sess = sdir / "session.json"
        session_id = _load(sess)["session_id"] if sess.is_file() else sdir.name
        cpath = sdir / "telemetry" / "clocks.csv"
        clocks = C.read_csv(cpath, "clocks") if cpath.is_file() else []
        for cdir in sorted(p for p in sdir.iterdir() if (p / "manifest.json").is_file()):
            man = C.validate_manifest(_load(cdir / "manifest.json"))
            if man["mode"] != "open":
                continue
            reason = _reason(man, seen)
            if reason is None:
                seen.add(man["condition_id"])
            run, rr = _run(session_id, man, cdir, clocks, reason)
            runs.append(run)
            reps += rr
    if not runs:
        raise C.ContractError("no open manifest.json under %s" % [str(s) for s in session_dirs])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "runs.csv", C.RUN3_RUNS_COLS, runs)
    _write(out / "reps.csv", C.COEXIT_PILOT_COLS, reps)
    n_summary = P.summarize(out / "reps.csv", out / "summary.csv")
    if sim_dir is not None:
        _write(out / "compare.csv", C.RUN3_COMPARE_COLS, compare(out, sim_dir))
    kept = [r for r in runs if r["kept"]]
    print("runs %d, kept %d, flagged %d (screen_ok 0, kept); reps %d rows, summary %d rows; wrote %s"
          % (len(runs), len(kept), sum(r["screen_ok"] != 1 for r in kept), len(reps), n_summary, out))
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m exp2.analyze.run3", description="Run 3 board analyzer.")
    ap.add_argument("--sessions", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim-dir", default=None)
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    try:
        analyze(a.sessions, a.out, a.sim_dir)
    except (ValueError, KeyError, OSError) as e:                 # ContractError is a ValueError
        print("run3: %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
