"""F1: closed-loop Run 1 analyzer (BRIEF_v1 K8, K10, K11; rules in exp2/tests/fixtures/README.md D2-D8).

    python -m exp2.analyze.closed_loop --sessions S [S ..] --out DIR

Reads closed-plan session directories (session.json, telemetry/, <condition>/{manifest.json,requests.jsonl}) and writes
the eight Run 1 CSVs of K11 plus screen.csv (one row per condition: kept, reason, clock-lock screening). Several
sessions (Run 1, 1b, 1d) are merged: later repeats are shifted (D4). Conditions with manifest valid=false, an error row,
fewer than MIN_MEASURED_REQ_PER_TENANT measured requests per tenant (AM1) or no solo baseline are left out, written to
screen.csv with kept=0 and reported on stderr. Anything else that cannot be computed honestly raises ValueError (exit 2).

    tau        = (t_done - t_admit) / 1e9, non-warmup rows, time order (dispatcher view)
    solo       window [first t_admit, last t_done]; u = mean gpu_load_permille / 1000 over clock rows in the window
    co-run     window [max over tenants of first t_admit, min over tenants of last t_done]; only rows fully inside
               count; s = tau / tau_solo_p50 (solo of the same session at the same type and f, else pooled)
    AM1        solo: all measured rows of the tenant; co-run: rows fully inside the window; fewer than the minimum
               -> skip with C.few_requests_reason, before the repeat shift
    CI         stats.block_bootstrap_ci(xs, nearest-rank p50, B=1000, seed=0); ess without clamp
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from exp2 import contracts as C
from exp2 import stats as S
from exp2.analyze import screen as SCREEN

_B, _SEED = 1000, 0
_SOLO_SPLITS = ("solo", "solo_heldout_level")
_CORUN_SPLITS = ("train", "heldout")


def _p50(a):
    return float(S.p_nearest_rank(a, 0.5))


def _tkey(type_id):
    fam, ex = C.parse_type_id(type_id)
    return fam, ex


def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return format(x, ".12g")
    return str(x)


def _boot(xs):
    ci = S.block_bootstrap_ci(xs, _p50, B=_B, seed=_SEED)
    return ci.lo, ci.hi


def _window_energy(power, lo, hi):
    """(energy_j, window_s) over [lo, hi] ns."""
    return S.energy_j(power, lo, hi), (hi - lo) / 1e9


class _Session:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = C.validate_session(_load_json(self.path / "session.json"))
        if self.meta["plan_kind"] != "closed":
            raise ValueError("%s: plan_kind %r is not 'closed'" % (self.path, self.meta["plan_kind"]))
        self.power = C.read_csv(self.path / "telemetry" / "power.csv", "power")
        self.clocks = C.read_csv(self.path / "telemetry" / "clocks.csv", "clocks")
        self.conditions = []        # (manifest, non-warmup rows sorted by t_admit, skip reason or None)
        for cid in self.meta["conditions"]:
            man = C.validate_manifest(_load_json(self.path / cid / "manifest.json"))
            # run_session writes no requests.jsonl for idle windows (board Run 1, 2026-10-20)
            rows = [] if man["mode"] == "idle" else C.read_requests(self.path / cid / "requests.jsonl")
            real = [r for r in rows if not r["is_warmup"]]
            bad = [r for r in real if r["outcome"] != "complete"]
            why = None
            if not man["valid"]:
                why = "manifest invalid: %s" % man["invalid_reason"]
            elif bad:
                why = "%d non-complete row(s), first: %s" % (len(bad), bad[0]["err"])
            real.sort(key=lambda r: r["t_admit_ns"])
            self.conditions.append((man, real, why))

    def u_mean(self, lo, hi):
        load = [c["gpu_load_permille"] / 1000 for c in self.clocks if lo <= c["t_mono_ns"] <= hi]
        if not load:
            raise ValueError("no clock samples in window [%d, %d] ns of %s" % (lo, hi, self.path))
        return sum(load) / len(load)

    def screen_row(self, man):
        """screen.csv row of one condition (kept=1, reason=None until a later stage skips it)."""
        t1 = man["t_mono_end_ns"]
        if t1 is None:
            m = {c: None for c in C.SCREEN_CSV_COLS[7:]}
        else:
            lo = man["t_mono_start_ns"] + round(man["settle_s"] * 1e9)
            m = SCREEN.screen_window(self.clocks, lo, t1, gpu_mhz=man["gpu_mhz"], emc_mhz=man["emc_mhz"])
        row = {"session_id": self.meta["session_id"], "condition_id": man["condition_id"], "mode": man["mode"],
               "arm": man["arm"], "gpu_mhz": man["gpu_mhz"], "kept": 1, "reason": None}
        row.update(m)
        return row


def _tau(rows):
    return [(r["t_done_ns"] - r["t_admit_ns"]) / 1e9 for r in rows]


def _corun_window(per):
    """(lo, hi, {type: rows fully inside}). A tenant without rows gives lo = hi = None (counts stay honest: its own
    count is 0); an empty window gives no rows inside."""
    if not all(per.values()):
        return None, None, per
    lo = max(v[0]["t_admit_ns"] for v in per.values())
    hi = min(max(r["t_done_ns"] for r in v) for v in per.values())
    if hi <= lo:
        return lo, hi, {t: [] for t in per}
    return lo, hi, {t: [r for r in v if r["t_admit_ns"] >= lo and r["t_done_ns"] <= hi] for t, v in per.items()}


def _category(why):
    for tag in ("few_requests", "no_solo_baseline", "manifest invalid"):
        if why.startswith(tag):
            return tag
    return "non_complete"


def analyze_sessions(session_dirs):
    """Return ({kind: [row dict]} for the K11 CSVs plus "screen", [(session_id, condition_id, reason)])."""
    solo, solo_samples, corun_raw, idle_acc, ckpt_acc, warm, mem = [], [], [], {}, {}, [], {}
    skipped, screen = [], []
    seen_solo, seen_corun = {}, {}      # (type, f) / (cell_id, f) -> repeats taken by earlier sessions
    for si, sd in enumerate(session_dirs):
        ses = _Session(sd)
        sid = ses.meta["session_id"]
        s_solo, s_samples, s_corun, scr = [], [], [], {}
        pre = ses.meta["idle"]["pre"]
        e_pre, w_pre = _window_energy(ses.power, pre["t0_ns"], pre["t1_ns"])
        p_idle_pre = e_pre / w_pre
        acc = idle_acc.setdefault((pre["gpu_mhz"], "pre"), [0.0, 0.0])
        acc[0] += e_pre
        acc[1] += w_pre
        for fam, sv in ses.meta["servers"].items():
            e, w = _window_energy(ses.power, sv["t_spawn_ns"], sv["t_ready_ns"])
            rd = sv["ready"]
            warm.append({"session_id": sid, "family": fam, "t_load_s": rd["warmup"]["load_s"],
                         "first_infer_s_max": max(rd["warmup"]["first_infer_s"].values()),
                         "warm_energy_j": e - p_idle_pre * w,
                         "cache": "cold" if rd["warmup"]["inductor_cache_files_at_start"] == 0 else "warm",
                         "_t": sv["t_spawn_ns"]})
            for ex in rd["exits_warmed"]:
                key = (fam, ex)
                old = mem.get(key, (0.0, 0.0))
                mem[key] = (max(old[0], rd["mem"]["torch_reserved_mb"]), max(old[1], rd["warmup"]["act_mb"][str(ex)]))
        for man, rows, why in ses.conditions:
            cid = man["condition_id"]
            scr[cid] = ses.screen_row(man)
            screen.append(scr[cid])
            if why is None and man["mode"] == "closed":
                cell = list(man["cell"])
                if man["split"] in _SOLO_SPLITS and len(cell) == 1:
                    why = C.few_requests_reason({cell[0]: sum(1 for r in rows if r["type_id"] == cell[0])})
                elif man["split"] in _CORUN_SPLITS:
                    per = {t: [r for r in rows if r["type_id"] == t] for t in cell}
                    why = C.few_requests_reason({t: len(v) for t, v in _corun_window(per)[2].items()})
            if why is not None:
                skipped.append((sid, cid, why))
                scr[cid].update(kept=0, reason=why)
                continue
            if man["mode"] == "idle":
                lo = man["t_mono_start_ns"] + round(man["settle_s"] * 1e9)
                e, w = _window_energy(ses.power, lo, man["t_mono_end_ns"])
                acc = idle_acc.setdefault((man["gpu_mhz"], "resident"), [0.0, 0.0])
                acc[0] += e
                acc[1] += w
                continue
            if man["mode"] != "closed":
                raise ValueError("%s: unexpected mode %r in a closed session" % (cid, man["mode"]))
            f, rep, cell = man["gpu_mhz"], man["repeat"], list(man["cell"])
            if man["split"] in _SOLO_SPLITS:
                if len(cell) != 1:
                    raise ValueError("%s: solo condition with cell %s" % (cid, cell))
                tid = cell[0]
                tau = _tau(rows)
                lo, hi = rows[0]["t_admit_ns"], max(r["t_done_ns"] for r in rows)
                e, w = _window_energy(ses.power, lo, hi)
                clo, chi = _boot(tau)
                s_solo.append({"type_id": tid, "f_mhz": f, "repeat": rep, "n": len(tau), "tau_p50": _p50(tau),
                               "tau_mean": sum(tau) / len(tau), "tau_p95": float(S.p_nearest_rank(tau, 0.95)),
                               "tau_p50_lo": clo, "tau_p50_hi": chi, "ess": S.ess(tau), "u": ses.u_mean(lo, hi),
                               "p_board_w": e / w, "window_s": w})
                s_samples += [{"type_id": tid, "f_mhz": f, "repeat": rep, "tau_s": t} for t in tau]
                for r in rows:
                    span = r["t_end_ns"] - r["t_start_ns"]
                    if span <= 0:
                        raise ValueError("%s %s: t_end <= t_start" % (cid, r["req_id"]))
                    cks = r["ckpt"]
                    head = (r["t_end_ns"] - cks[-1][1]) / span
                    for c, t in cks:
                        a = ckpt_acc.setdefault((tid, f, c), ([], []))
                        a[0].append((t - r["t_start_ns"]) / span)
                        a[1].append(head)
            elif man["split"] in _CORUN_SPLITS:
                per = {t: [r for r in rows if r["type_id"] == t] for t in cell}
                lo, hi, inside = _corun_window(per)
                if lo is None or hi <= lo:
                    raise ValueError("%s: empty co-run window" % cid)
                e, w = _window_energy(ses.power, lo, hi)
                s_corun.append({"cell_id": cid.rsplit("_f", 1)[0], "cid": cid, "split": man["split"], "f": f,
                                "rep": rep, "cell": cell, "per": per, "inside": inside, "e": e, "w": w, "si": si})
            else:
                raise ValueError("%s: unknown split %r" % (cid, man["split"]))
        # D4: repeats of a later session continue after the repeats already present (AM1 drops came first)
        shift = {k: seen_solo.get(k, 0) for k in {(r["type_id"], r["f_mhz"]) for r in s_solo}}
        for r in s_solo + s_samples:
            r["repeat"] += shift[(r["type_id"], r["f_mhz"])]
            r["_si"] = si
        for r in s_solo:
            seen_solo[(r["type_id"], r["f_mhz"])] = seen_solo.get((r["type_id"], r["f_mhz"]), 0) + 1
        shift_c = {(c["cell_id"], c["f"]): seen_corun.get((c["cell_id"], c["f"]), 0) for c in s_corun}
        for c in s_corun:
            c["rep"] += shift_c[(c["cell_id"], c["f"])]
        for c in s_corun:
            seen_corun[(c["cell_id"], c["f"])] = seen_corun.get((c["cell_id"], c["f"]), 0) + 1
        solo += s_solo
        solo_samples += s_samples
        corun_raw += [dict(c, scr=scr[c["cid"]], sid=sid) for c in s_corun]

    keys = [(r["type_id"], r["f_mhz"], r["repeat"]) for r in solo]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate solo (type, f, repeat) across the given sessions")
    solo.sort(key=lambda r: (_tkey(r["type_id"]), r["f_mhz"], r["repeat"]))
    solo_samples = [r for _, r in sorted(
        enumerate(solo_samples), key=lambda p: (_tkey(p[1]["type_id"]), p[1]["f_mhz"], p[1]["repeat"], p[0]))]
    pooled, own, u_by = {}, {}, {}
    for r in solo_samples:
        pooled.setdefault((r["type_id"], r["f_mhz"]), []).append(r["tau_s"])
        own.setdefault((r.pop("_si"), r["type_id"], r["f_mhz"]), []).append(r["tau_s"])
    for r in solo:
        u_by.setdefault((r["type_id"], r["f_mhz"]), []).append(r["u"])
        del r["_si"]
    solo_p50 = {k: _p50(v) for k, v in pooled.items()}
    own_p50 = {k: _p50(v) for k, v in own.items()}
    u_f = {k: sum(v) / len(v) for k, v in u_by.items()}

    corun, corun_samples = [], []
    corun_raw.sort(key=lambda c: (c["cell_id"], c["f"], c["rep"]))
    for c in corun_raw:
        miss = [t for t in c["cell"] if (t, c["f"]) not in solo_p50]
        if miss:
            why = "no_solo_baseline:%s@%d" % (miss[0], c["f"])
            skipped.append((c["sid"], c["cid"], why))
            c["scr"].update(kept=0, reason=why)
            continue
        u_a = sum(u_f[(t, c["f"])] for t in c["cell"])
        for t in c["cell"]:
            inside = c["inside"][t]
            base = own_p50.get((c["si"], t, c["f"]), solo_p50[(t, c["f"])])
            s = [x / base for x in _tau(inside)]
            lo_ci, hi_ci = _boot(s)
            corun.append({"cell_id": c["cell_id"], "split": c["split"], "f_mhz": c["f"], "repeat": c["rep"],
                          "type_id": t, "family": C.parse_type_id(t)[0], "A": "|".join(c["cell"]),
                          "n": len(c["per"][t]), "n_window": len(inside), "s_p50": _p50(s),
                          "s_mean": sum(s) / len(s), "s_p95": float(S.p_nearest_rank(s, 0.95)),
                          "s_p50_lo": lo_ci, "s_p50_hi": hi_ci, "ess": S.ess(s), "tau_solo_p50": base, "U_A": u_a,
                          "window_s": c["w"], "energy_j": c["e"], "p_board_w": c["e"] / c["w"]})
            corun_samples += [{"cell_id": c["cell_id"], "f_mhz": c["f"], "repeat": c["rep"], "type_id": t, "s": x}
                              for x in s]

    idle = [{"f_mhz": f, "kind": k, "p_idle_w": a[0] / a[1]}
            for (f, k), a in sorted(idle_acc.items(), key=lambda kv: (kv[0][1] != "pre", kv[0][0]))]
    ckpt = []
    for (tid, f, c) in sorted(ckpt_acc, key=lambda k: (_tkey(k[0]), k[1], k[2])):
        fr, hd = ckpt_acc[(tid, f, c)]
        ckpt.append({"type_id": tid, "f_mhz": f, "ckpt": c, "frac_p50": _p50(fr), "head_frac_p50": _p50(hd)})
    warm.sort(key=lambda r: r["_t"])
    for r in warm:
        del r["_t"]
    mem_rows = [{"family": fam, "exit": ex, "resident_mb": v[0], "act_mb": v[1]} for (fam, ex), v in sorted(mem.items())]
    return ({"solo": solo, "solo_samples": solo_samples, "corun": corun, "corun_samples": corun_samples,
             "idle": idle, "ckpt": ckpt, "warmup": warm, "mem": mem_rows, "screen": screen}, skipped)


def write_run1_csv(tables, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols_of = dict(C.RUN1_CSV_COLS)
    if "screen" in tables:
        cols_of["screen"] = C.SCREEN_CSV_COLS
    for kind, cols in cols_of.items():
        with open(out / (kind + ".csv"), "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(cols)
            for r in tables[kind]:
                w.writerow([_fmt(r[c]) for c in cols])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m exp2.analyze.closed_loop")
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 2
    try:
        tables, skipped = analyze_sessions(args.sessions)
    except (ValueError, OSError, KeyError) as e:
        print("closed_loop: %s" % e, file=sys.stderr)
        return 2
    for sid, cid, why in skipped:
        print("closed_loop: SKIPPED %s/%s: %s" % (sid, cid, why), file=sys.stderr)
    write_run1_csv(tables, args.out)
    by = {}
    for _, _, why in skipped:
        by[_category(why)] = by.get(_category(why), 0) + 1
    n = len(tables["screen"])
    flagged = sum(1 for r in tables["screen"] if r["kept"] and r["screen_ok"] == 0)
    print("closed_loop: kept %d of %d conditions; skipped %d (%s); screen_ok=0 on %d kept" % (
        n - len(skipped), n, len(skipped), ", ".join("%s=%d" % kv for kv in sorted(by.items())) or "none", flagged),
        file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
