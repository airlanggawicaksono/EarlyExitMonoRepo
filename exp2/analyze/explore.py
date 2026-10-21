"""Exploratory checks E1 to E9 of exp2/frozen/run2/ANALYSIS_PLAN_ADDENDUM_1.md (exploratory, never tuned).

    python -I -m exp2.analyze.explore --sessions S [S ..] --frozen DIR --final DIR --out DIR

Streams each session's telemetry/power.csv once (csv module, one pass, per-run window integrals only), builds the
piecewise-constant set in service per run from the request service intervals [t_start_ns, t_end_ns], and writes
explore_runs.csv and explore_summary.json into DIR. Formulas and choices: idea/exp2_queue_or_run/EXPLORE_PROVENANCE.md.

Reading choices the addendum does not fix (pre-set before any Run 2 number was computed):
    "about 1" / "equal within noise" = the median of the per-run ratio lies in [1 - TOL, 1 + TOL], TOL = 0.10.
    headline population = Run 2 session only (first session given); Run 2b runs are flagged and reported in "incl_2b".
    only pinned runs (arm == "pinned") enter checks that need a fixed level f.
    serial policy = k_short == 1 and k_long == 1 (at most one request in service); every other pinned policy co-runs.
"""
import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path

TOL = 0.10
_NS = 1e9
RUN_COLS = ("session", "condition_id", "freeze_condition_id", "policy_key", "workload", "rho", "f_mhz", "repeat",
            "serial", "n", "window_s", "energy_j", "energy_per_req_j", "power_covered", "idle_s", "busy_frac",
            "corun_frac", "triple_frac", "max_in_service", "uncovered_set_s", "work_cr", "work_wasted", "work_rate",
            "cr_rate", "R_c", "R_P", "P_pred_w", "cap_denominator_s", "mean_shortfall", "bert_trunc_rate")


# ------------------------------------------------------------------------------------------------ helpers
def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _med(xs):
    return statistics.median(xs) if xs else None


def _quant(xs, q):
    s = sorted(xs)
    if not s:
        return None
    i = q * (len(s) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def _stats(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "median": _med(xs), "mean": sum(xs) / len(xs), "q25": _quant(xs, 0.25),
            "q75": _quant(xs, 0.75), "min": min(xs), "max": max(xs)}


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None


def perm_p(x, y, B=2000, seed=0):
    """One-sided permutation p of rho > 0 (fixed seed)."""
    obs = spearman(x, y)
    if obs is None:
        return None
    rng, y2, hit = random.Random(seed), list(y), 0
    for _ in range(B):
        rng.shuffle(y2)
        r = spearman(x, y2)
        hit += 1 if r is not None and r >= obs - 1e-12 else 0
    return (hit + 1) / (B + 1)


def _fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return format(x, ".10g")
    return str(x)


def _close(x, target=1.0, tol=TOL):
    return x is not None and abs(x - target) <= tol


# ------------------------------------------------------------------------------------------------ Run 1 tables
def _setkey(a):
    return tuple(sorted(a.split("|"))) if isinstance(a, str) else tuple(sorted(a))


def load_run1(frozen):
    """-> dict with capacity c[(A,f)], power P[(A,f)] (board W), excess e[(A,f)], dP[(type,f)], p_idle[f], sets.

    corun.csv: per (cell, type) row; s = median over rows of s_p50 per (A, f, type); g_i = 1/s; c = sum_i g_i.
    P(A,f) = median over distinct cells of p_board_w. Solo P({i},f) = median over repeats of solo.csv p_board_w."""
    frozen = Path(frozen)
    p_idle = {int(k): v for k, v in _load_json(frozen / "power.json")["p_idle_w"].items()}
    s_rows, p_cell = {}, {}
    with open(frozen / "run1_csv" / "corun.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            a, f = _setkey(r["A"]), int(r["f_mhz"])
            s_rows.setdefault((a, f, r["type_id"]), []).append(float(r["s_p50"]))
            p_cell[(a, f, r["cell_id"])] = float(r["p_board_w"])
    solo = {}
    with open(frozen / "run1_csv" / "solo.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            solo.setdefault((r["type_id"], int(r["f_mhz"])), []).append(float(r["p_board_w"]))
    p_solo = {k: _med(v) for k, v in solo.items()}
    g, pcell_by = {}, {}
    for (a, f, t), v in s_rows.items():
        g.setdefault((a, f), {})[t] = 1.0 / _med(v)
    for (a, f, _c), v in p_cell.items():
        pcell_by.setdefault((a, f), []).append(v)
    cap, power, excess = {}, {}, {}
    dP = {k: v - p_idle[k[1]] for k, v in p_solo.items() if k[1] in p_idle}
    for (a, f), gi in g.items():
        cap[(a, f)] = sum(gi.values())
        power[(a, f)] = _med(pcell_by[(a, f)])
        if f in p_idle and all((t, f) in dP for t in a):
            excess[(a, f)] = power[(a, f)] - p_idle[f] - sum(dP[(t, f)] * gi[t] for t in a)
    for (t, f), p in p_solo.items():           # a single request: c = 1, P = solo board power, e = 0
        cap[((t,), f)] = 1.0
        power[((t,), f)] = p
        excess[((t,), f)] = 0.0
    return {"p_idle": p_idle, "cap": cap, "power": power, "excess": excess, "dP": dP}


def load_types(frozen):
    d = _load_json(Path(frozen) / "types.json")
    out = {}
    for t in d["types"]:
        out[t["type_id"]] = {"phi": {int(k): v for k, v in t["ckpt_frac"].items()}, "head": t["head_frac"],
                             "tau": {int(f): v["mean"] for f, v in t["tau_s"].items()}}
    return out


# ------------------------------------------------------------------------------------------------ one run
def request_work(row, tinfo, f):
    """(credited, wasted) solo seconds of one request, Eq. (work): tau = mean solo time of its type at f."""
    ti = tinfo[row["type_id"]]
    tau = ti["tau"][f]
    phi, head = ti["phi"], ti["head"]
    out, ck = row["outcome"], row.get("ckpt") or []
    if out == "complete":
        return tau, 0.0
    if out == "truncated" and len(ck) >= 2:
        c_late, c_ok = ck[-1][0], ck[-2][0]
        return tau * (phi[c_ok] + head), tau * (phi[c_late] - phi[c_ok])
    if out == "truncated" and len(ck) == 1:      # one late checkpoint recorded, reached is the last on-time one
        c_late = ck[-1][0]
        c_ok = row.get("reached")
        return (tau * (phi[c_ok] + head), tau * (phi[c_late] - phi[c_ok])) if c_ok in phi else (0.0, 0.0)
    if out == "miss" and ck:
        return 0.0, tau * phi[ck[0][0]]
    return 0.0, 0.0


def set_occupancy(rows, w0, w1):
    """{set tuple: seconds} over [w0, w1] from the service intervals [t_start, t_end] of all rows, plus max |A|."""
    ev = []
    for r in rows:
        a, b = r.get("t_start_ns"), r.get("t_end_ns")
        if a is None or b is None or b <= a:
            continue
        a, b = max(a, w0), min(b, w1)
        if b > a:
            ev.append((a, 1, r["type_id"]))
            ev.append((b, 0, r["type_id"]))
    ev.sort(key=lambda e: (e[0], e[1]))          # ends (0) before starts (1) at equal time
    cur, t_prev, occ, mx = [], w0, {}, 0
    for t, kind, tid in ev:
        if t > t_prev and cur:
            k = tuple(sorted(cur))
            occ[k] = occ.get(k, 0.0) + (t - t_prev) / _NS
        t_prev = t
        if kind == 1:
            cur.append(tid)
            mx = max(mx, len(cur))
        else:
            cur.remove(tid)
    return occ, mx


def analyse_run(session, man, rows, r1, tinfo):
    """Per-run record (without the energy, filled in after the power pass)."""
    live = [r for r in rows if not r.get("is_warmup")]
    starts = [r["t_arrival_ns"] for r in live]
    ends = [r["t_done_ns"] if r.get("t_done_ns") is not None else r["t_deadline_ns"] for r in live
            if r.get("t_done_ns") is not None or r.get("t_deadline_ns") is not None]
    w0, w1 = min(starts), max(ends)
    W = (w1 - w0) / _NS
    pinned = man["arm"] == "pinned"
    f = man["gpu_mhz"] if pinned else None
    sp = man["policy_spec"]
    rec = {"session": session, "condition_id": man["condition_id"], "freeze_condition_id": man["freeze_condition_id"],
           "policy_key": man["policy_key"], "workload": man["workload"]["name"], "rho": man["workload"]["rho"],
           "f_mhz": f if pinned else "governor", "repeat": man["repeat"],
           "serial": int(pinned and sp.get("k_short") == 1 and sp.get("k_long") == 1), "n": len(live),
           "window_s": W, "_w0": w0, "_w1": w1, "_lambda": man["workload"]["rate_per_s"], "_pinned": pinned,
           "_f": f, "_nth": len(rows)}
    occ, mx = set_occupancy(rows, w0, w1)
    busy = sum(occ.values())
    rec.update({"idle_s": W - busy, "busy_frac": busy / W, "max_in_service": mx,
                "corun_frac": sum(v for k, v in occ.items() if len(k) >= 2) / W,
                "triple_frac": sum(v for k, v in occ.items() if len(k) == 3) / W})
    rec["_occ"] = occ
    n_shortfall = sum(1.0 - r["completion"] for r in live) / len(live)
    rec["mean_shortfall"] = n_shortfall
    bert = [r for r in live if r["family"] == "bert"]
    rec["bert_trunc_rate"] = (sum(1 for r in bert if r["outcome"] == "truncated") / len(bert)) if bert else None
    if pinned:
        cr = wa = 0.0
        v_by = {}
        for r in live:
            c, x = request_work(r, tinfo, f)
            cr += c
            wa += x
            v_by[r["type_id"]] = v_by.get(r["type_id"], 0.0) + c + x
        rec.update({"work_cr": cr, "work_wasted": wa, "work_rate": (cr + wa) / W, "cr_rate": cr / W})
        rec["_v_by"] = v_by
        den, unc, pw = 0.0, 0.0, r1["p_idle"][f] * rec["idle_s"]
        for a, t in occ.items():
            if (a, f) in r1["cap"]:
                den += t * r1["cap"][(a, f)]
                pw += t * r1["power"][(a, f)]
            else:
                unc += t
        rec.update({"cap_denominator_s": den, "uncovered_set_s": unc, "P_pred_w": pw / W})
        rec["R_c"] = (cr + wa) / den if den > 0 and unc == 0 else None
    return rec


# ------------------------------------------------------------------------------------------------ power pass
def integrate_windows(power_path, windows):
    """Stream power.csv once. windows: list of (w0_ns, w1_ns). Returns list of (energy_j, covered?) per window.

    Piecewise-linear trace between samples, ends interpolated; mW * s / 1000 = J. Windows may be in any order."""
    order = sorted(range(len(windows)), key=lambda i: windows[i][0])
    res = [0.0] * len(windows)
    cov = [0.0] * len(windows)
    j0 = 0
    prev = None
    with open(power_path, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        head = next(rd)
        it, ip = head.index("t_mono_ns"), head.index("vdd_in_mw")
        for cells in rd:
            if not cells:
                continue
            t, p = int(float(cells[it])), float(cells[ip])
            if prev is not None:
                ta, pa = prev
                if t > ta:
                    while j0 < len(order) and windows[order[j0]][1] <= ta:
                        j0 += 1
                    k = j0
                    while k < len(order) and windows[order[k]][0] < t:
                        wi = order[k]
                        a, b = max(ta, windows[wi][0]), min(t, windows[wi][1])
                        if b > a:
                            sl = (p - pa) / (t - ta)
                            pa2, pb2 = pa + sl * (a - ta), pa + sl * (b - ta)
                            res[wi] += (b - a) / _NS * (pa2 + pb2) / 2.0 / 1000.0
                            cov[wi] += (b - a) / _NS
                        k += 1
            prev = (t, p)
    return [(res[i], cov[i]) for i in range(len(windows))]


# ------------------------------------------------------------------------------------------------ sessions
def process_session(sdir, label, r1, tinfo, seen):
    sdir = Path(sdir)
    session = _load_json(sdir / "session.json")
    recs = []
    for cid in session["conditions"]:
        man = _load_json(sdir / cid / "manifest.json")
        if man["mode"] != "open" or not man["valid"] or man["condition_id"] in seen:
            continue
        seen.add(man["condition_id"])
        rows = []
        with open(sdir / cid / "requests.jsonl", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        if not any(not r.get("is_warmup") for r in rows):
            continue
        recs.append(analyse_run(label, man, rows, r1, tinfo))
        del rows
    en = integrate_windows(sdir / "telemetry" / "power.csv", [(r["_w0"], r["_w1"]) for r in recs])
    for r, (e, c) in zip(recs, en):
        r["energy_j"] = e
        r["energy_per_req_j"] = e / r["n"]
        r["power_covered"] = c / r["window_s"]
        if r["_pinned"]:
            den = r["P_pred_w"] * r["window_s"]
            r["R_P"] = e / den if den > 0 and r["uncovered_set_s"] == 0 and r["power_covered"] > 0.999 else None
    return recs


# ------------------------------------------------------------------------------------------------ checks
def _grp(recs, **kw):
    return [r for r in recs if all(r.get(k) == v for k, v in kw.items())]


def _ratio_check(recs, key):
    xs = [r[key] for r in recs if r.get(key) is not None]
    st = _stats(xs)
    st["share_within_tol"] = (sum(1 for x in xs if _close(x)) / len(xs)) if xs else None
    return st


def check_e1(runs, runs_all):
    p = [r for r in runs if r["_pinned"]]
    by_f = {str(f): _stats([r["R_c"] for r in p if r["f_mhz"] == f]) for f in sorted({r["f_mhz"] for r in p})}
    by_c = {c: _stats([r["R_c"] for r in p if r["freeze_condition_id"] == c]) for c in sorted({r["freeze_condition_id"] for r in p})}
    st = _ratio_check(p, "R_c")
    return {"formula": "R_c = (credited + wasted work) / sum_A T_A c(A,f)", "expected": "about 1 (median within 0.90-1.10)",
            "run2": st, "by_f": by_f, "by_freeze_condition": by_c,
            "serial_runs": _ratio_check([r for r in p if r["serial"]], "R_c"),
            "corun_runs": _ratio_check([r for r in p if not r["serial"]], "R_c"),
            "incl_2b": _ratio_check([r for r in runs_all if r["_pinned"]], "R_c"),
            "outcome": "as_expected" if _close(st.get("median")) else "not_as_expected"}


def check_e2(runs, runs_all):
    p = [r for r in runs if r["_pinned"]]
    st = _ratio_check(p, "R_P")
    by_f = {str(f): _stats([r["R_P"] for r in p if r["f_mhz"] == f]) for f in sorted({r["f_mhz"] for r in p})}
    by_c = {c: _stats([r["R_P"] for r in p if r["freeze_condition_id"] == c]) for c in sorted({r["freeze_condition_id"] for r in p})}
    return {"formula": "R_P = E_window / (P_idle T_0 + sum_A T_A P_Run1(A,f))", "expected": "about 1 (median within 0.90-1.10)",
            "run2": st, "by_f": by_f, "by_freeze_condition": by_c,
            "serial_runs": _ratio_check([r for r in p if r["serial"]], "R_P"),
            "corun_runs": _ratio_check([r for r in p if not r["serial"]], "R_P"),
            "incl_2b": _ratio_check([r for r in runs_all if r["_pinned"]], "R_P"),
            "outcome": "as_expected" if _close(st.get("median")) else "not_as_expected"}


def check_e3(runs, runs_all):
    s = [r for r in runs if r["_pinned"] and r["serial"]]
    for r in s:
        r["_e3"] = r["work_rate"] / r["busy_frac"] if r["busy_frac"] > 0 else None
    st = _ratio_check(s, "_e3")
    diff = _stats([r["work_rate"] - r["busy_frac"] for r in s])
    return {"formula": "ratio = processed work rate / busy fraction, serial runs (capacity 1)",
            "expected": "equal within noise (median ratio within 0.90-1.10)", "run2_serial": st,
            "difference_work_rate_minus_busy": diff, "max_in_service_max": max((r["max_in_service"] for r in s), default=None),
            "by_f": {str(f): _stats([r["_e3"] for r in s if r["f_mhz"] == f]) for f in sorted({r["f_mhz"] for r in s})},
            "outcome": "as_expected" if _close(st.get("median")) else "not_as_expected"}


def _mean(xs):
    return sum(xs) / len(xs)


def check_e4(runs, r1):
    p = [r for r in runs if r["_pinned"]]
    ser = {}
    for r in p:
        if r["serial"]:
            ser.setdefault((r["freeze_condition_id"], r["f_mhz"]), []).append(r)
    rows_out = []
    for r in p:
        if r["serial"]:
            continue
        refs = ser.get((r["freeze_condition_id"], r["f_mhz"]))
        f = r["f_mhz"]
        if not refs or not r.get("uncovered_set_s") == 0.0:
            continue
        types = sorted(set(r["_v_by"]) | {t for q in refs for t in q["_v_by"]})
        if any((t, f) not in r1["dP"] for t in types):
            continue
        p_r = r["energy_j"] / r["window_s"]
        p_s = _mean([q["energy_j"] / q["window_s"] for q in refs])
        dv = sum(r1["dP"][(t, f)] * (r["_v_by"].get(t, 0.0) / r["window_s"]
                                     - _mean([q["_v_by"].get(t, 0.0) / q["window_s"] for q in refs])) for t in types)
        lam = r["_lambda"]
        delta = ((p_r - p_s) - dv) / lam
        eps_s = _mean([q["energy_per_req_j"] for q in refs])
        e_max = max(v for (a, ff), v in r1["excess"].items() if ff == f and len(a) >= 2)
        bound = r["corun_frac"] * e_max / lam
        out = {"condition_id": r["condition_id"], "freeze_condition_id": r["freeze_condition_id"], "f_mhz": f,
               "delta_j_per_req": delta, "rel_to_eps_serial": delta / eps_s, "bound_j_per_req": bound,
               "ge_0": delta >= 0, "le_bound": delta <= bound, "n_serial_refs": len(refs)}
        r["_e4"] = delta
        rows_out.append(out)
    d = [x["delta_j_per_req"] for x in rows_out]
    rel = [x["rel_to_eps_serial"] for x in rows_out]
    bd = [x["bound_j_per_req"] for x in rows_out]
    ok = bool(d) and _med(d) >= 0 and _med(d) <= _med(bd) and abs(_med(rel)) < 0.01
    return {"formula": "eps(pol) - eps(serial) - (1/lambda) sum_i dP_i (v_i(pol) - v_i(serial)); serial = mean of the serial "
            "runs of the same freeze condition and level", "expected": ">= 0, <= corun fraction x max e / lambda, below 1% of eps",
            "n_comparisons": len(rows_out), "delta_j_per_req": _stats(d), "rel_to_eps_serial": _stats(rel),
            "bound_j_per_req": _stats(bd), "share_ge_0": (sum(x["ge_0"] for x in rows_out) / len(rows_out)) if rows_out else None,
            "share_le_bound": (sum(x["le_bound"] for x in rows_out) / len(rows_out)) if rows_out else None,
            "share_below_1pct_abs": (sum(1 for x in rel if abs(x) < 0.01) / len(rel)) if rel else None,
            "outcome": "as_expected" if ok else "not_as_expected",
            "outcome_rule": "median delta >= 0, median delta <= median bound, median |delta/eps| < 1%",
            "per_comparison": rows_out}


def check_e5(runs, freeze):
    """Measured co-run minus serial energy at equal f against the frozen predicted difference, per measured key."""
    from exp2.model import freeze as FZ
    fconds = {c["condition_id"]: c for c in freeze["conditions"]}
    kmaps = {cid: FZ.measured_key_map(c) for cid, c in fconds.items()}
    key_e = {}
    for r in runs:
        if not r["_pinned"]:
            continue
        k = kmaps[r["freeze_condition_id"]][r["policy_key"]]
        key_e.setdefault((r["freeze_condition_id"], k), {"E": [], "serial": r["serial"], "f": r["f_mhz"]})["E"].append(
            r["energy_per_req_j"])
    out = []
    for cid, fc in fconds.items():
        keys = {k: v for (c, k), v in key_e.items() if c == cid}
        for f in sorted({v["f"] for v in keys.values()}):
            sk = [k for k, v in keys.items() if v["f"] == f and v["serial"] and k in fc["predictions"]]
            if not sk:
                continue
            em_s = _mean([_mean(keys[k]["E"]) for k in sk])
            ep_s = _mean([fc["predictions"][k]["energy_per_req_j"] for k in sk])
            for k, v in keys.items():
                if v["f"] != f or v["serial"] or k not in fc["predictions"]:
                    continue
                dm = _mean(v["E"]) - em_s
                dp = fc["predictions"][k]["energy_per_req_j"] - ep_s
                out.append({"freeze_condition_id": cid, "mix": cid[0], "f_mhz": f, "key": k, "measured_diff_j": dm,
                            "predicted_diff_j": dp, "bias_j": dm - dp, "bias_rel_serial": (dm - dp) / em_s,
                            "serial_keys": sk})
    pos = [x for x in out if x["bias_j"] > 0]
    res = {"formula": "bias = (E_meas(key) - mean E_meas(serial keys)) - (E_pred(key) - mean E_pred(serial keys)), same freeze "
           "condition and f, mean over Run 2 repeats", "expected": "positive sign, below 1% of serial energy in mix A",
           "n_comparisons": len(out), "share_positive": (len(pos) / len(out)) if out else None}
    for lab, sel in (("all", out), ("mix_A", [x for x in out if x["mix"] == "A"]), ("mix_B", [x for x in out if x["mix"] == "B"])):
        res[lab] = {"bias_j": _stats([x["bias_j"] for x in sel]), "bias_rel_serial": _stats([x["bias_rel_serial"] for x in sel]),
                    "share_positive": (sum(1 for x in sel if x["bias_j"] > 0) / len(sel)) if sel else None}
    ma = res["mix_A"]["bias_rel_serial"]
    ok = bool(out) and res["all"]["bias_j"]["median"] > 0 and ma["n"] > 0 and abs(ma["median"]) < 0.01
    res["outcome"] = "as_expected" if ok else "not_as_expected"
    res["outcome_rule"] = "median bias over all comparisons > 0 and |median relative bias| in mix A < 1%"
    res["per_comparison"] = out
    return res


def check_e6(runs):
    sel = [r for r in runs if r["_pinned"] and r["freeze_condition_id"] == "A_r060" and r["f_mhz"] == 612]
    ser = [r["mean_shortfall"] for r in sel if r["serial"]]
    cor = [r["mean_shortfall"] for r in sel if not r["serial"]]
    ok = bool(ser) and bool(cor) and _mean(cor) > _mean(ser) and _med(cor) > _med(ser)
    return {"formula": "mean depth shortfall = 1 - mean(completion) over all non-warm-up requests of the run (expired = 0)",
            "expected": "co-run policies exceed serial policies", "serial": _stats(ser), "corun": _stats(cor),
            "difference_of_means": (_mean(cor) - _mean(ser)) if ser and cor else None,
            "share_corun_above_max_serial": (sum(1 for x in cor if x > max(ser)) / len(cor)) if ser and cor else None,
            "outcome": "as_expected" if ok else "not_as_expected",
            "outcome_rule": "mean and median of the co-run runs both exceed those of the serial runs"}


def check_e7(runs):
    sel = [r for r in runs if r["_pinned"] and r["freeze_condition_id"] == "A_r060" and r["f_mhz"] == 612]
    ratio = [(r["cr_rate"] / r["busy_frac"]) if r["busy_frac"] > 0 else None for r in sel]
    viol = [r["condition_id"] for r, x in zip(sel, ratio) if x is not None and x > 1.0]
    return {"formula": "credited work rate <= busy fraction x c*, c* = 1 (every pair has capacity < 1 here); ratio = cr_rate / busy_frac",
            "expected": "always (ratio <= 1)", "n_runs": len(sel), "ratio": _stats(ratio), "n_violations": len(viol),
            "violations": viol[:20], "serial": _stats([x for r, x in zip(sel, ratio) if r["serial"]]),
            "corun": _stats([x for r, x in zip(sel, ratio) if not r["serial"]]),
            "outcome": "as_expected" if sel and not viol else "not_as_expected",
            "outcome_rule": "no run with ratio > 1 (strict)"}


def check_e8():
    return {"expected": "rank fidelity of energy improves in mix A", "outcome": "not_computable",
            "reason": "freeze.json stores only per-type outcome rates and energy_per_req_j per measured key, no predicted "
                      "occupancy per set. Changing P(A,f) in the energy prediction needs a new simulator run for every "
                      "measured key (exp2.model). The brief forbids simulating on the laptop, so this needs the board."}


def check_e9(runs, runs_all):
    def one(rs):
        sel = [r for r in rs if r["_pinned"] and r["workload"] == "B" and r["f_mhz"] == 612
               and r["bert_trunc_rate"] is not None]
        x = [r["triple_frac"] for r in sel]
        y = [r["bert_trunc_rate"] for r in sel]
        return {"n_runs": len(sel), "spearman_rho": spearman(x, y), "perm_p_one_sided": perm_p(x, y),
                "n_runs_with_triples": sum(1 for v in x if v > 0), "triple_frac": _stats(x), "bert_trunc_rate": _stats(y)}
    res = one(runs)
    return {"formula": "Spearman rho between triple fraction (time with 3 requests in service / window) and bert truncation "
            "rate (truncated / bert requests), mix B runs pinned at 612 MHz", "expected": "positive association",
            "run2": res, "incl_2b": one(runs_all),
            "outcome": "as_expected" if (res["spearman_rho"] or 0) > 0 else "not_as_expected",
            "outcome_rule": "rho > 0 (permutation p given for reference)"}


# ------------------------------------------------------------------------------------------------ driver
def run(sessions, frozen, final, out):
    frozen, out = Path(frozen), Path(out)
    r1, tinfo = load_run1(frozen), load_types(frozen)
    freeze = _load_json(frozen / "freeze.json")
    seen, all_recs = set(), []
    for i, sd in enumerate(sessions):
        all_recs += process_session(sd, "run2" if i == 0 else "run2b", r1, tinfo, seen)
    r2 = [r for r in all_recs if r["session"] == "run2"]
    checks = {"E1": check_e1(r2, all_recs), "E2": check_e2(r2, all_recs), "E3": check_e3(r2, all_recs),
              "E4": check_e4(r2, r1), "E5": check_e5(r2, freeze), "E6": check_e6(r2), "E7": check_e7(r2),
              "E8": check_e8(), "E9": check_e9(r2, all_recs)}
    sanity = None
    oc = Path(final) / "open_conditions.csv"
    if oc.exists():                              # energy per request must equal the confirmatory analysis
        ref = {}
        with open(oc, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                ref.setdefault(r["condition_id"], float(r["energy_per_req_j"]))
        d = [abs(r["energy_per_req_j"] / ref[r["condition_id"]] - 1.0) for r in all_recs if r["condition_id"] in ref]
        sanity = {"n_matched": len(d), "max_rel_diff_vs_open_conditions": max(d) if d else None}
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "explore_runs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(RUN_COLS)
        for r in all_recs:
            w.writerow([_fmt(r.get(c)) for c in RUN_COLS])
    summary = {"schema": "exp2.explore/1", "tolerance_about_one": TOL, "n_runs": len(all_recs),
               "n_runs_run2": len(r2), "n_runs_run2b": len(all_recs) - len(r2),
               "energy_sanity": sanity, "checks": checks,
               "outcomes": {k: v["outcome"] for k, v in checks.items()}}
    with open(out / "explore_summary.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2, allow_nan=False)
        fh.write("\n")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -I -m exp2.analyze.explore")
    ap.add_argument("--sessions", required=True, nargs="+")
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    s = run(a.sessions, a.frozen, a.final, a.out)
    print(json.dumps(s["outcomes"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
