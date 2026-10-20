"""F2: open-loop Run 2 analyzer (BRIEF_v1 K8, K10, K13-K16, K18; rules in exp2/tests/fixtures/README.md D13, D14; C2b).

    python -m exp2.analyze.open_loop --sessions S [S ..] --freeze freeze.json [--plans P [P ..]] --out DIR

Writes open_conditions.csv, screen.csv and open_summary.json. Exit codes: 0 ok, 2 usage or data that cannot be analysed,
3 refused (freeze file tampered, a session started before the freeze was created, or a hash does not match).
Several sessions (a resumed Run 2) are merged; a condition_id seen in an earlier session is a duplicate and skipped.
A session whose git.commit differs from freeze.git.commit only gives a warning on stderr.

    R            = (t_done - t_arrival) / 1e9 (scheduled arrival, so no coordinated omission); p95 nearest-rank
    energy/req   = stats.energy_per_request: window [first arrival, last t_done or deadline of expired rows]
                   includes idle gaps; stats.energy_j raises ValueError when the power trace does not cover it
    keys         = measured keys (freeze.measured_key_map): an alias name in a manifest is folded into its key
    status       = Wilson 95% lower bound of the pooled per-type complete rate: feasible if the minimum over types is
                   >= COMPLETE_TARGET, borderline if only the point estimate is, else infeasible. The bool
                   `feasible` stays the point-estimate rule (feasible or borderline) and feeds regret and oracle
    regret       = (E_F - E_best) / E_best, best = cheapest feasible key among F + every grid point (through the key
                   map); grid_spread = (max - min) / min over those
    energy_diff  = stats.paired_diff_ci of E_F,r - E_key,r over the repeats both keys have (paired arrivals)
    overshoot    = t_done - t_deadline of non-warmup complete/truncated/miss rows with t_done > t_deadline;
                   late_rate = their share of the type rows (A17)
    several freeze conditions: regret, grid_spread and poa_measured are the mean over conditions that define them

open_summary/3 (R32, D32-1, D32-4 to D32-6)
    status       = bootstrap rule: attain_boot_lo_min is the minimum over types of the 2.5% quantile of the pooled
                   complete rate, resampling runs (repeats) and moving blocks of C.BOOT_BLOCK_REQUESTS consecutive
                   requests inside each run (C.BOOT_RESAMPLES, seed C.BOOT_SEED mixed with the key). feasible if it is
                   >= COMPLETE_TARGET, borderline if only the point estimate is, else infeasible. wilson_lo_min stays.
                   Without per-request sequences (synthetic input) the sequence is rebuilt from the counts, evenly spread.
    primary      = paired log(E_F / E_ref) per (freeze condition, repeat) for ref in C.PRIMARY_REFS, pooled over
                   conditions, t interval, two-sided t p, Holm within the family. secondary: the same for the refs of
                   C.SECONDARY_REFS the freeze measures with >= 2 pairs. A freeze without a ref key gives no pairs for it.
    energy_per_work_j = run window energy / sum of the rows' completion over the measured rows, mean over repeats
    sim_accuracy = per freeze condition over its measured keys, the governor default excluded
    regret       = pooled repeats of a key include its Run 2b confirm runs (same policy_key, repeats 1 and 2). A measured
                   F that misses the complete target makes regret.F null (H4 fails; derive the flag from per_condition)
"""
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

from exp2 import contracts as C
from exp2 import stats as S
from exp2.analyze import screen as SCREEN
from exp2.model import freeze as FZ

_DONE_OUTCOMES = ("complete", "truncated", "miss")


class Refused(RuntimeError):
    """The pre-registration checks failed (exit 3)."""


def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return format(x, ".12g")
    return str(x)


def check_preregistration(session, freeze_path, plan=None):
    """Return (freeze, checks). Raise Refused (message names both clocks) when any check fails."""
    try:
        freeze = C.verify_freeze_file(freeze_path)
    except C.FreezeError as e:
        raise Refused(f"freeze file rejected: {e}") from e
    t_start, t_frz = int(session["t_wall_start_ns"]), int(freeze["created_wall_ns"])
    if t_start <= t_frz:
        raise Refused(f"wall-clock order wrong: session.t_wall_start_ns={t_start} <= "
                      f"freeze.created_wall_ns={t_frz}")
    if session.get("freeze_content_sha256") != freeze["content_sha256"]:
        raise Refused(f"session.freeze_content_sha256 {session.get('freeze_content_sha256')} != "
                      f"freeze content_sha256 {freeze['content_sha256']} (session.t_wall_start_ns={t_start}, "
                      f"freeze.created_wall_ns={t_frz})")
    if plan is not None:
        calc = C.plan_sha256(plan)
        if plan.get("sha256") != calc or session.get("plan_sha256") != calc:
            raise Refused(f"plan hash mismatch: recomputed {calc}, plan.sha256 {plan.get('sha256')}, "
                          f"session.plan_sha256 {session.get('plan_sha256')} (session.t_wall_start_ns={t_start}, "
                          f"freeze.created_wall_ns={t_frz})")
        if (plan.get("freeze") or {}).get("content_sha256") != freeze["content_sha256"]:
            raise Refused("plan.freeze.content_sha256 != freeze content_sha256")
    return freeze, {"hash_ok": True, "wall_order_ok": True, "t_wall_start_ns": t_start,
                    "freeze_created_wall_ns": t_frz}


def _pick_plan(session, plans):
    """The plan whose sha256 the session names. None when no plans were given; Refused when none matches."""
    if not plans:
        return None
    for p in plans:
        if p.get("sha256") == session.get("plan_sha256"):
            return p
    raise Refused(f"plan hash mismatch: no given plan has sha256 {session.get('plan_sha256')} "
                  f"(session.t_wall_start_ns={session.get('t_wall_start_ns')})")


def _rates(rows):
    n = len(rows)
    cnt = {o: sum(1 for r in rows if r["outcome"] == o) for o in C.OUTCOMES}
    return {"n": n, "complete_rate": cnt["complete"] / n, "truncation_rate": cnt["truncated"] / n,
            "miss_rate": cnt["miss"] / n, "expired_rate": cnt["expired_in_queue"] / n,
            "error_rate": cnt["error"] / n, "mean_completion": sum(r["completion"] for r in rows) / n}


def _condition(cdir, power, clocks):
    """(manifest, csv rows, energy_per_req, per-type counts, extra) of one open condition folder. extra = {seq:
    [(type_id, complete?)] in arrival order, energy_per_work_j: window energy / sum of completion, or None}."""
    man = _load_json(cdir / "manifest.json")
    rows = C.read_requests(cdir / "requests.jsonl")
    live = [r for r in rows if not r["is_warmup"]]
    t0, t1, n = S.request_window(rows)
    e_req = S.energy_j(power, t0, t1) / n
    m0, m1 = man["t_mono_start_ns"], man["t_mono_end_ns"]
    e_sess = S.energy_j(power, m0, m1) / n
    mem = [c["mem_available_kb"] / 1024.0 for c in clocks if m0 <= c["t_mono_ns"] <= m1]
    wl = man["workload"]
    order = list(wl["types"]) + sorted({r["type_id"] for r in live} - set(wl["types"]))
    f = man["gpu_mhz"] if man["arm"] == "pinned" else "governor"
    out = []
    for tid in order:
        tr = [r for r in live if r["type_id"] == tid]
        if not tr:
            continue
        resp = [(r["t_done_ns"] - r["t_arrival_ns"]) / 1e9 for r in tr if r["t_done_ns"] is not None]
        over = [(r["t_done_ns"] - r["t_deadline_ns"]) / 1e9 for r in tr
                if r["outcome"] in _DONE_OUTCOMES and r["t_done_ns"] is not None and r["t_deadline_ns"] is not None
                and r["t_done_ns"] > r["t_deadline_ns"]]
        d = {"condition_id": man["condition_id"], "policy_key": man["policy_key"], "workload": wl["name"],
             "rho": wl["rho"], "arrival": wl["arrival"], "alpha": man["alpha"], "f_mhz": f,
             "repeat": man["repeat"], "type_id": tid}
        d.update(_rates(tr))
        d.update({"p95_R_s": S.p_nearest_rank(resp, 0.95) if resp else None, "energy_per_req_j": e_req,
                  "energy_sess_per_req_j": e_sess, "throughput_rps": n / ((t1 - t0) / 1e9),
                  "min_mem_available_mb": min(mem) if mem else None, "late_rate": len(over) / len(tr),
                  "overshoot_p95_s": S.p_nearest_rank(over, 0.95) if over else None,
                  "overshoot_max_s": max(over) if over else None})
        out.append(d)
    seq = [(r["type_id"], r["outcome"] == "complete") for r in sorted(live, key=lambda r: r["t_arrival_ns"])]
    work = sum(r["completion"] for r in live)
    extra = {"seq": seq, "energy_per_work_j": e_req * n / work if work > 0 else None}
    return man, out, e_req, {t["type_id"]: (t["n"], round(t["complete_rate"] * t["n"])) for t in out}, extra


def _mean(xs):
    return sum(xs) / len(xs)


def _median(xs):
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0



def _seq_from_counts(cnt):
    """Evenly spread 0/1 sequence with the given per-type (n, k) counts (used when no request order is known)."""
    seq = []
    for tid, (n, k) in cnt.items():
        done = 0
        for i in range(n):
            ok = (i + 1) * k // n > done
            done += ok
            seq.append((i / n, tid, bool(ok)))
    return [(tid, ok) for _, tid, ok in sorted(seq, key=lambda x: x[0])]


def _prefix(seq, types):
    """Per type: prefix sums of 'is this type' and 'is this type and complete' along the request order."""
    pn, pk = {t: [0] for t in types}, {t: [0] for t in types}
    for tid, ok in seq:
        for t in types:
            hit = tid == t
            pn[t].append(pn[t][-1] + hit)
            pk[t].append(pk[t][-1] + (hit and ok))
    return len(seq), pn, pk


def _boot_attain_lo(runs, seed, resamples=None, block=None):
    """Minimum over types of the 2.5% quantile (nearest rank) of the pooled complete rate. runs: list of request
    sequences [(type_id, complete?)]; each resample draws len(runs) runs with replacement and, inside each, moving
    blocks of `block` consecutive requests (the last block cut to the run length)."""
    resamples = C.BOOT_RESAMPLES if resamples is None else resamples
    block = C.BOOT_BLOCK_REQUESTS if block is None else block
    types = sorted({t for r in runs for t, _ in r})
    pre = [_prefix(r, types) for r in runs]
    rng = random.Random(seed)
    rates = {t: [] for t in types}
    for _ in range(resamples):
        cn, ck = dict.fromkeys(types, 0), dict.fromkeys(types, 0)
        for _r in range(len(pre)):
            n, pn, pk = pre[rng.randrange(len(pre))]
            if n == 0:
                continue
            b = min(block, n)
            rem = n
            while rem > 0:
                st = rng.randrange(n - b + 1)
                ln = min(b, rem)
                for t in types:
                    cn[t] += pn[t][st + ln] - pn[t][st]
                    ck[t] += pk[t][st + ln] - pk[t][st]
                rem -= ln
        for t in types:
            if cn[t] > 0:
                rates[t].append(ck[t] / cn[t])
    return min(S.p_nearest_rank(v, 0.025) for v in rates.values() if v)


def _ref_stat_inputs(fconds, kmaps, meas, ref):
    """Pooled paired log(E_F / E_ref) over freeze conditions and repeats. Returns (log_F, log_ref). A freeze without
    the reference key contributes no pairs."""
    la, lb = [], []
    for fcid in fconds:
        km = kmaps[fcid]
        if "F" not in km or ref not in km:
            continue
        f, g = meas.get((fcid, km["F"])), meas.get((fcid, km[ref]))
        if not f or not g:
            continue
        for r in sorted(set(f["by_rep"]) & set(g["by_rep"])):
            la.append(math.log(f["by_rep"][r]))
            lb.append(math.log(g["by_rep"][r]))
    return la, lb


def _family(fconds, kmaps, meas, refs):
    """{ref: stat} for refs, Holm-adjusted inside the family. A ref with < 2 pairs is left out."""
    got, ps = {}, []
    for ref in refs:
        la, lb = _ref_stat_inputs(fconds, kmaps, meas, ref)
        if len(la) < 2:
            continue
        m, lo, hi, p = S.paired_t_test(la, lb)
        got[ref] = {"log_ratio_mean": m, "ci_lo": lo, "ci_hi": hi, "p": p, "p_holm": None, "n_pairs": len(la)}
        ps.append(ref)
    for ref, adj in zip(ps, S.holm([got[r]["p"] for r in ps])):
        got[ref]["p_holm"] = adj
    return got


def _sim_accuracy(fcid, fc, meas, specs):
    """Simulator accuracy of one freeze condition over its measured, non-governor keys (D32-6), or None."""
    pp, apes, pe, me, agree = [], [], [], [], []
    for (c, key), v in meas.items():
        if c != fcid or specs.get(key, {}).get("f_rule") == "governor" or key not in fc["predictions"]:
            continue
        pred = fc["predictions"][key]
        shared = [t for t in v["rates"] if t in pred["per_type"]]
        if not shared:
            continue
        pp += [abs(pred["per_type"][t]["complete_rate"] - v["rates"][t]) * 100.0 for t in shared]
        agree.append((min(pred["per_type"][t]["complete_rate"] for t in shared) >= C.COMPLETE_TARGET) == v["ok"])
        pe.append(pred["energy_per_req_j"])
        me.append(v["e"])
        apes.append(S.mape([pred["energy_per_req_j"]], [v["e"]]))
    if not pp:
        return None
    return {"abs_err_pp_max": max(pp), "abs_err_pp_median": _median(pp), "class_agreement": sum(agree) / len(agree),
            "spearman_energy": S.spearman(pe, me), "energy_ape_median": _median(apes)}


def _summarize(freeze, conds, checks, counts, resamples=None):
    """conds: list of (manifest, energy_per_req, {type: (n, n_complete)}[, extra]) with extra as in _condition.
    Returns the open_summary/3 dict (primary may miss a ref the data cannot support; validation rejects that)."""
    fconds = {c["condition_id"]: c for c in freeze["conditions"]}
    kmaps = {cid: FZ.measured_key_map(c) for cid, c in fconds.items()}
    specs = {cid: {k: s for k, s, _ in FZ.measured_runs(c)} for cid, c in fconds.items()}
    by = {}  # (freeze cid, measured key) -> list of (repeat, energy, counts, extra)
    for item in conds:
        man, e, cnt = item[:3]
        extra = item[3] if len(item) > 3 else {}
        fcid = man["freeze_condition_id"]
        by.setdefault((fcid, kmaps[fcid][man["policy_key"]]), []).append((man["repeat"], e, cnt, extra))
    per, apes, comp_err, meas = {}, [], [], {}
    for (fcid, key), reps in by.items():
        e_list = [x[1] for x in reps]
        pooled = {}
        for _, _, cnt, _ in reps:
            for tid, (n, k) in cnt.items():
                a = pooled.setdefault(tid, [0, 0])
                a[0] += n
                a[1] += k
        rates = {tid: k / n for tid, (n, k) in pooled.items()}
        wilson_lo = min(S.wilson_ci(k, n)[0] for n, k in pooled.values())
        runs = [x[3].get("seq") or _seq_from_counts(x[2]) for x in reps]
        boot_lo = _boot_attain_lo(runs, f"{C.BOOT_SEED}:{fcid}:{key}", resamples)
        works = [x[3].get("energy_per_work_j") for x in reps]
        epw = _mean(works) if all(w is not None for w in works) else None
        pred = fconds[fcid]["predictions"][key]
        e = _mean(e_list)
        ci = None
        if len(e_list) >= 2:
            _, lo, hi = S.mean_ci(e_list)
            ci = [lo, hi]
        ape = S.mape([pred["energy_per_req_j"]], [e])
        for tid, r in rates.items():
            comp_err.append(S.mape([pred["per_type"][tid]["complete_rate"]], [r]))
        cmin = min(rates.values())
        status = ("feasible" if boot_lo >= C.COMPLETE_TARGET
                  else "borderline" if cmin >= C.COMPLETE_TARGET else "infeasible")
        per[f"{fcid}_{key}"] = {"energy": e, "energy_ci": ci, "complete_rate_min": cmin,
                                "feasible": cmin >= C.COMPLETE_TARGET, "pred": pred["energy_per_req_j"],
                                "ape": ape, "status": status, "wilson_lo_min": wilson_lo, "n_repeats": len(reps),
                                "measured_key": key, "attain_boot_lo_min": boot_lo, "energy_per_work_j": epw}
        apes.append(ape)
        meas[(fcid, key)] = {"e": e, "by_rep": {x[0]: x[1] for x in reps}, "ok": cmin >= C.COMPLETE_TARGET,
                             "cmin": cmin, "rates": rates}
    regrets, spreads, best_key, oracle, diffs, poas, detail, f_bad, sim = [], [], None, {}, {}, [], {}, [], {}
    for fc in freeze["conditions"]:
        fcid, kmap = fc["condition_id"], kmaps[fc["condition_id"]]
        keys = {k: v for (c, k), v in meas.items() if c == fcid}
        feas = {k: v["e"] for k, v in keys.items() if v["ok"]}
        oracle[fcid] = ({"policy_key": min(feas, key=feas.get), "energy_per_req_j": min(feas.values())}
                        if feas else None)
        fkey = kmap.get("F")
        names = ["F"] + [g["name"] for g in fc["grid"]]
        cand = {k: feas[k] for k in dict.fromkeys(kmap[n] for n in names if n in kmap) if k in feas}
        if fkey in keys and not keys[fkey]["ok"]:
            f_bad.append(fcid)                   # measured F misses the target: H4 fails, regret is null
        if cand:
            b = min(cand, key=cand.get)
            best_key = best_key or b
            if fkey in cand:
                regrets.append((cand[fkey] - cand[b]) / cand[b])
            spreads.append((max(cand.values()) - cand[b]) / cand[b])
        if fkey in keys:
            for k, v in keys.items():
                reps = sorted(set(v["by_rep"]) & set(keys[fkey]["by_rep"]))
                if k != fkey and len(reps) >= 2:
                    d, lo, hi = S.paired_diff_ci([keys[fkey]["by_rep"][r] for r in reps],
                                                 [v["by_rep"][r] for r in reps])
                    diffs[f"{fcid}_{k}"] = {"diff_j": d, "ci": [lo, hi], "n_pairs": len(reps), "method": "paired_t"}
        sa = _sim_accuracy(fcid, fc, meas, specs[fcid])
        if sa:
            sim[fcid] = sa
        g = fc["game"]
        wk, ok = (g.get("worst") or {}).get("policy_key"), (g.get("optimum") or {}).get("policy_key")
        if wk and ok:
            wk, ok = kmap.get(wk, wk), kmap.get(ok, ok)
            w, o = keys.get(wk), keys.get(ok)
            ratio = w["e"] / o["e"] if w and o else None
            if ratio is not None:
                poas.append(ratio)
            detail[fcid] = {"worst_key": wk, "optimum_key": ok, "ratio": ratio,
                            "complete_min_worst": w["cmin"] if w else None,
                            "complete_min_optimum": o["cmin"] if o else None}
    return {"schema": C.SCHEMAS["open_summary"], "freeze_content_sha256": freeze["content_sha256"], "checks": checks,
            "counts": counts, "per_condition": per,
            "mape": {"completion": _mean(comp_err), "energy": _mean(apes), "median": _median(apes), "max": max(apes)},
            "regret": {"F": _mean(regrets) if regrets and not f_bad else None,
                       "grid_spread": _mean(spreads) if spreads else None, "best_key": best_key},
            "oracle": oracle, "energy_diff": diffs, "poa_detail": detail,
            "poa_measured": _mean(poas) if poas else None,
            "primary": _family(fconds, kmaps, meas, C.PRIMARY_REFS),
            "secondary": _family(fconds, kmaps, meas, C.SECONDARY_REFS),
            "sim_accuracy": sim}


def _screen_row(session, man, clocks, kept, reason):
    """screen.csv row of one open condition (same columns as the closed analyzer)."""
    t1 = man["t_mono_end_ns"]
    if t1 is None:
        m = {c: None for c in C.SCREEN_CSV_COLS[7:]}
    else:
        lo = man["t_mono_start_ns"] + round(man["settle_s"] * 1e9)
        m = SCREEN.screen_window(clocks, lo, t1, gpu_mhz=man["gpu_mhz"] if man["arm"] == "pinned" else None,
                                 emc_mhz=man["emc_mhz"])
    row = {"session_id": session["session_id"], "condition_id": man["condition_id"], "mode": man["mode"],
           "arm": man["arm"], "gpu_mhz": man["gpu_mhz"], "kept": kept, "reason": reason}
    row.update(m)
    return row


def _write_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([_fmt(r[c]) for c in cols])


def analyze(session_dirs, freeze_path, out_dir, plan_paths=None):
    """Run the checks on every session, then write open_conditions.csv, screen.csv and open_summary.json into
    out_dir. session_dirs and plan_paths may be one path or a list. Returns the summary."""
    if isinstance(session_dirs, (str, Path)):
        session_dirs = [session_dirs]
    if isinstance(plan_paths, (str, Path)):
        plan_paths = [plan_paths]
    plans = [_load_json(p) for p in plan_paths] if plan_paths else None
    sessions = []
    for sd in session_dirs:
        sdir = Path(sd)
        session = _load_json(sdir / "session.json")
        freeze, checks = check_preregistration(session, freeze_path, _pick_plan(session, plans))
        fcommit = (freeze.get("git") or {}).get("commit")
        scommit = (session.get("git") or {}).get("commit")
        if scommit != fcommit:
            print(f"warning: session {session.get('session_id')} git.commit {scommit} != freeze git.commit {fcommit}",
                  file=sys.stderr)
        sessions.append((sdir, session, checks))
    csv_rows, conds, screen, seen = [], [], [], set()
    counts = {"runs_valid": 0, "runs_invalid": 0, "runs_duplicate": 0, "runs_screen_flagged": 0}
    for sdir, session, _ in sessions:
        power = C.read_csv(sdir / "telemetry" / "power.csv", "power")
        clocks = C.read_csv(sdir / "telemetry" / "clocks.csv", "clocks")
        for cid in session["conditions"]:
            man = _load_json(sdir / cid / "manifest.json")
            if man["mode"] != "open":
                continue
            if not man["valid"]:
                print(f"skip invalid condition {cid}", file=sys.stderr)
                counts["runs_invalid"] += 1
                screen.append(_screen_row(session, man, clocks, 0, f"invalid:{man.get('invalid_reason')}"))
                continue
            if man["condition_id"] in seen:
                print(f"skip duplicate condition {cid} in session {session['session_id']}", file=sys.stderr)
                counts["runs_duplicate"] += 1
                screen.append(_screen_row(session, man, clocks, 0, "duplicate"))
                continue
            seen.add(man["condition_id"])
            man, rows, e_req, cnt, extra = _condition(sdir / cid, power, clocks)
            srow = _screen_row(session, man, clocks, 1, None)
            counts["runs_valid"] += 1
            counts["runs_screen_flagged"] += 1 if srow["screen_ok"] == 0 else 0
            screen.append(srow)
            csv_rows += rows
            conds.append((man, e_req, cnt, extra))
    if not conds:
        raise ValueError("no valid open condition in the sessions")
    summary = _summarize(freeze, conds, sessions[0][2], counts)
    C.validate_open_summary(summary)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "open_conditions.csv", C.OPEN_CONDITIONS_COLS, csv_rows)
    _write_csv(out / "screen.csv", C.SCREEN_CSV_COLS, screen)
    with open(out / "open_summary.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2, allow_nan=False)
        fh.write("\n")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m exp2.analyze.open_loop")
    ap.add_argument("--sessions", required=True, nargs="+")
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--plans", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        analyze(a.sessions, a.freeze, a.out, a.plans)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    except (ValueError, OSError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
