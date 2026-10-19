"""F2: open-loop Run 2 analyzer (BRIEF_v1 K8, K10, K13-K16, K18; rules in exp2/tests/fixtures/README.md D13, D14).

    python -m exp2.analyze.open_loop --session DIR --freeze freeze.json [--plan plan.json] --out DIR

Writes open_conditions.csv and open_summary.json. Exit codes: 0 ok, 2 usage or data that cannot be analysed,
3 refused (freeze file tampered, session started before the freeze was created, or a hash does not match).

    R            = (t_done - t_arrival) / 1e9 (scheduled arrival, so no coordinated omission); p95 nearest-rank
    energy/req   = stats.energy_per_request: window [first arrival, last t_done or deadline of expired rows]
                   includes idle gaps; stats.energy_j raises ValueError when the power trace does not cover it
    regret       = (E_F - E_best) / E_best, best = cheapest SLO-feasible key among F + grid (feasible: pooled
                   complete rate of every type >= COMPLETE_TARGET); grid_spread = (max - min) / min over those
    several freeze conditions: regret, grid_spread and poa_measured are the mean over conditions that define them
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from exp2 import contracts as C
from exp2 import stats as S


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


def _rates(rows):
    n = len(rows)
    cnt = {o: sum(1 for r in rows if r["outcome"] == o) for o in C.OUTCOMES}
    return {"n": n, "complete_rate": cnt["complete"] / n, "truncation_rate": cnt["truncated"] / n,
            "miss_rate": cnt["miss"] / n, "expired_rate": cnt["expired_in_queue"] / n,
            "error_rate": cnt["error"] / n, "mean_completion": sum(r["completion"] for r in rows) / n}


def _condition(cdir, power, clocks):
    """(manifest, csv rows, energy_per_req, per-type counts) of one open condition folder."""
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
        d = {"condition_id": man["condition_id"], "policy_key": man["policy_key"], "workload": wl["name"],
             "rho": wl["rho"], "arrival": wl["arrival"], "alpha": man["alpha"], "f_mhz": f,
             "repeat": man["repeat"], "type_id": tid}
        d.update(_rates(tr))
        d.update({"p95_R_s": S.p_nearest_rank(resp, 0.95) if resp else None, "energy_per_req_j": e_req,
                  "energy_sess_per_req_j": e_sess, "throughput_rps": n / ((t1 - t0) / 1e9),
                  "min_mem_available_mb": min(mem) if mem else None})
        out.append(d)
    return man, out, e_req, {t["type_id"]: (t["n"], round(t["complete_rate"] * t["n"])) for t in out}


def _mean(xs):
    return sum(xs) / len(xs)


def _summarize(freeze, conds, checks):
    """conds: list of (manifest, energy_per_req, {type: (n, n_complete)}). Returns the open_summary dict."""
    by = {}  # (freeze cid, policy_key) -> list of (energy, counts)
    for man, e, cnt in conds:
        by.setdefault((man["freeze_condition_id"], man["policy_key"]), []).append((e, cnt))
    per, apes, comp_err, meas_e = {}, [], [], {}
    for (fcid, key), reps in by.items():
        e_list = [e for e, _ in reps]
        pooled = {}
        for _, cnt in reps:
            for tid, (n, k) in cnt.items():
                a = pooled.setdefault(tid, [0, 0])
                a[0] += n
                a[1] += k
        rates = {tid: k / n for tid, (n, k) in pooled.items()}
        fc = next(c for c in freeze["conditions"] if c["condition_id"] == fcid)
        pred = fc["predictions"][key]
        e = _mean(e_list)
        ci = None
        if len(e_list) >= 2:
            _, lo, hi = S.mean_ci(e_list)
            ci = [lo, hi]
        ape = S.mape([pred["energy_per_req_j"]], [e])
        for tid, r in rates.items():
            comp_err.append(S.mape([pred["per_type"][tid]["complete_rate"]], [r]))
        cmin = min(rates.values())
        per[f"{fcid}_{key}"] = {"energy": e, "energy_ci": ci, "complete_rate_min": cmin,
                                "feasible": cmin >= C.COMPLETE_TARGET, "pred": pred["energy_per_req_j"],
                                "ape": ape}
        apes.append(ape)
        meas_e[(fcid, key)] = (e, e_list, cmin >= C.COMPLETE_TARGET)
    regrets, spreads, best_key, oracle, diffs, poas = [], [], None, {}, {}, []
    for fc in freeze["conditions"]:
        fcid = fc["condition_id"]
        keys = {k: v for (c, k), v in meas_e.items() if c == fcid}
        feas = {k: v[0] for k, v in keys.items() if v[2]}
        oracle[fcid] = ({"policy_key": min(feas, key=feas.get), "energy_per_req_j": min(feas.values())}
                        if feas else None)
        cand = {k: feas[k] for k in ["F"] + [g["name"] for g in fc["grid"]] if k in feas}
        if cand:
            b = min(cand, key=cand.get)
            best_key = best_key or b
            if "F" in cand:
                regrets.append((cand["F"] - cand[b]) / cand[b])
            spreads.append((max(cand.values()) - cand[b]) / cand[b])
        if "F" in keys:
            for k, (_, el, _) in keys.items():
                if k != "F" and len(el) >= 2 and len(keys["F"][1]) >= 2:
                    d, lo, hi = S.welch_diff_ci(keys["F"][1], el)
                    diffs[k] = {"diff_j": d, "ci": [lo, hi]}
        g = fc["game"]
        w, o = (keys.get(g["worst"]["policy_key"]), keys.get(g["optimum"]["policy_key"]))
        if w and o:
            poas.append(w[0] / o[0])
    return {"schema": "exp2.open_summary/1", "freeze_content_sha256": freeze["content_sha256"], "checks": checks,
            "per_condition": per,
            "mape": {"completion": _mean(comp_err), "energy": _mean(apes), "median": _median(apes), "max": max(apes)},
            "regret": {"F": _mean(regrets) if regrets else None, "grid_spread": _mean(spreads) if spreads else None,
                       "best_key": best_key},
            "oracle": oracle, "energy_diff": diffs, "poa_measured": _mean(poas) if poas else None}


def _median(xs):
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def analyze(session_dir, freeze_path, out_dir, plan_path=None):
    """Run the checks, then write open_conditions.csv and open_summary.json into out_dir. Returns the summary."""
    sdir = Path(session_dir)
    session = _load_json(sdir / "session.json")
    plan = _load_json(plan_path) if plan_path else None
    freeze, checks = check_preregistration(session, freeze_path, plan)
    power = C.read_csv(sdir / "telemetry" / "power.csv", "power")
    clocks = C.read_csv(sdir / "telemetry" / "clocks.csv", "clocks")
    csv_rows, conds = [], []
    for cid in session["conditions"]:
        man_path = sdir / cid / "manifest.json"
        if _load_json(man_path)["mode"] != "open":
            continue
        if not _load_json(man_path)["valid"]:
            print(f"skip invalid condition {cid}", file=sys.stderr)
            continue
        man, rows, e_req, cnt = _condition(sdir / cid, power, clocks)
        csv_rows += rows
        conds.append((man, e_req, cnt))
    if not conds:
        raise ValueError("no valid open condition in the session")
    summary = _summarize(freeze, conds, checks)
    C.validate_open_summary(summary)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "open_conditions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(C.OPEN_CONDITIONS_COLS)
        for r in csv_rows:
            w.writerow([_fmt(r[c]) for c in C.OPEN_CONDITIONS_COLS])
    with open(out / "open_summary.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2, allow_nan=False)
        fh.write("\n")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m exp2.analyze.open_loop")
    ap.add_argument("--session", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        analyze(a.session, a.freeze, a.out, a.plan)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    except (ValueError, OSError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
