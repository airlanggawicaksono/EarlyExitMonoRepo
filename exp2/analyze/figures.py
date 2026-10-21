"""Figures of the Exp2 paper from the MAXN_SUPER profiling data and the final Run 2 analysis.

numpy, matplotlib and the standard library only. No torch, no model run. One function per figure.
CLI: python -m exp2.analyze.figures --run1-csv DIR --frozen DIR --run1c DIR [DIR ...] --run2 DIR --out DIR [--only NAME ...]
  --run1-csv  exp2/frozen/run2/run1_csv (corun.csv solo.csv idle.csv)
  --frozen    exp2/frozen/run2 (slowdown.json power.json freeze.json)
  --run1c     Run 1c session folders (session.json + telemetry/power.csv); the cold/warm state comes from the folder name
  --run2      analysis_final folder (open_conditions.csv)
A figure whose input is missing is skipped with a printed reason. Output: NAME.pdf and NAME.png (200 dpi).
"""
import argparse
import collections
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ------------------------------------------------------------------------------------------------ style
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7"}
FAMILY_COLOR = {"bert": OI["blue"], "llama": OI["verm"], "yolo": OI["green"]}
POLICY_COLOR = {"F": OI["black"], "default": OI["orange"], "default_pinned": OI["sky"], "serial_stf": OI["purple"],
                "serial": OI["yellow"], "solo_profile_freq": OI["blue"]}
REFS = ("default", "default_pinned", "serial_stf", "serial", "solo_profile_freq")
TYPES = ("bert:5", "bert:23", "llama:3", "llama:15", "yolo:1", "yolo:5")
LEVELS = (612, 816, 1020)
COND_ORDER = ("A_r030", "A_r060", "B_r030", "B_r060", "B_r060_b")
COND_LABEL = {"A_r030": "A, 0.3, 1", "A_r060": "A, 0.6, 1", "B_r030": "B, 0.3, 1", "B_r060": "B, 0.6, 1",
              "B_r060_b": "B, 0.6, 2"}
COND_COLOR = dict(zip(COND_ORDER, (OI["blue"], OI["sky"], OI["verm"], OI["orange"], OI["purple"])))
LEVEL_COLOR = {612: OI["sky"], 816: OI["orange"], 1020: OI["verm"]}
CM = 1 / 2.54
W = 12 * CM          # 12 cm


class Skip(Exception):
    """Raised by a figure function when its input is missing."""


def _style():
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "axes.titlesize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
                         "legend.frameon": False})


def _save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [f"{name}.pdf", f"{name}.png"]


# ------------------------------------------------------------------------------------------------ loaders
def _csv(path):
    path = Path(path)
    if not path.is_file():
        raise Skip(f"missing {path}")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _json(path):
    path = Path(path)
    if not path.is_file():
        raise Skip(f"missing {path}")
    with open(path) as fh:
        return json.load(fh)


def _need(ctx, key):
    if not ctx.get(key):
        raise Skip(f"argument --{key.replace('_', '-')} not given")
    return ctx[key]


def _short(A):
    return " + ".join(A.split("|"))


def set_group(A):
    if "llama:3" in A:
        return "with llama:3"
    if "llama:15" in A:
        return "with llama:15"
    return "encoder and detector"


def _corun_cells(rows):
    """(A, f) -> {'c': [rows per repeat], 'd': [...]}: Run 1d rows have a cell_id prefix 'd_'."""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["A"], int(float(r["f_mhz"])), "d" if r["cell_id"].startswith("d_") else "c", r["repeat"])].append(r)
    return by


def capacity_table(rows):
    """{(A, f): median over repeats of sum 1/s_p50}, Run 1d rows where Run 1d measured the same set and level."""
    cap = collections.defaultdict(dict)
    for (A, f, src, _rep), rs in _corun_cells(rows).items():
        cap[(A, f)].setdefault(src, []).append(sum(1.0 / float(x["s_p50"]) for x in rs))
    return {k: st.median(v.get("d", v.get("c"))) for k, v in cap.items()}


def gamma_table(rows, solo_rows, p_idle):
    """{(A, f): median over repeats of sum_i (P_solo,i - P_idle) / s_p50,i / (P(A,f) - P_idle)}."""
    ps = collections.defaultdict(list)
    for r in solo_rows:
        ps[(r["type_id"], int(float(r["f_mhz"])))].append(float(r["p_board_w"]))
    ps = {k: st.median(v) for k, v in ps.items()}
    gm = collections.defaultdict(dict)
    for (A, f, src, _rep), rs in _corun_cells(rows).items():
        pa = float(rs[0]["p_board_w"])
        num = sum((ps[(x["type_id"], f)] - p_idle[f]) / float(x["s_p50"]) for x in rs)
        gm[(A, f)].setdefault(src, []).append(num / (pa - p_idle[f]))
    return {k: st.median(v.get("d", v.get("c"))) for k, v in gm.items()}


def _ordered_sets(keys):
    sets = sorted({A for A, _ in keys}, key=lambda A: (["encoder and detector", "with llama:3", "with llama:15"].index(
        set_group(A)), A))
    return sets


def _p_idle(ctx):
    pw = _json(Path(_need(ctx, "frozen")) / "power.json")
    return {int(k): float(v) for k, v in pw["p_idle_w"].items()}


# ------------------------------------------------------------------------------------------------ 2D: profiling
def fig_capacity(ctx, out):
    rows = _csv(Path(_need(ctx, "run1_csv")) / "corun.csv")
    cap = capacity_table(rows)
    sets = _ordered_sets(cap)
    M = np.array([[cap.get((A, f), np.nan) for f in LEVELS] for A in sets])
    fig, ax = plt.subplots(figsize=(W, 0.24 * len(sets) + 0.9))
    norm = TwoSlopeNorm(vcenter=1.0, vmin=min(0.85, np.nanmin(M)), vmax=max(1.35, np.nanmax(M)))
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(3), [str(f) for f in LEVELS])
    ax.set_yticks(range(len(sets)), [_short(A) for A in sets])
    ax.set_xlabel("GPU frequency (MHz)")
    ax.set_ylabel("Co-run set")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Capacity $c(A,f)$ (1)")
    fig.tight_layout()
    return _save(fig, out, "fig_capacity")


def fig_gamma(ctx, out):
    rows = _csv(Path(_need(ctx, "run1_csv")) / "corun.csv")
    solo = _csv(Path(_need(ctx, "run1_csv")) / "solo.csv")
    g = gamma_table(rows, solo, _p_idle(ctx))
    sets = _ordered_sets(g)
    fig, ax = plt.subplots(figsize=(W, 0.24 * len(sets) + 1.0))
    ys = np.arange(len(sets))
    for f in LEVELS:
        ax.scatter([g.get((A, f), np.nan) for A in sets], ys, s=22, color=LEVEL_COLOR[f], label=f"{f} MHz",
                   edgecolor="black", linewidth=0.3, zorder=3)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_yticks(ys, [_short(A) for A in sets])
    ax.invert_yaxis()
    ax.set_xlabel(r"Co-run energy ratio $\gamma(A,f)$ (1)")
    ax.set_ylabel("Co-run set")
    ax.grid(axis="x", linewidth=0.3)
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0.0, 1.0), handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout()
    return _save(fig, out, "fig_gamma")


def fig_solo_time(ctx, out):
    rows = _csv(Path(_need(ctx, "run1_csv")) / "solo.csv")
    fig, ax = plt.subplots(figsize=(W, 3.2))
    marks = {"5": "o", "23": "s", "3": "o", "15": "s", "1": "o"}
    for t in TYPES:
        fam, ex = t.split(":")
        pts = collections.defaultdict(list)
        for r in rows:
            if r["type_id"] == t:
                pts[int(float(r["f_mhz"]))].append(float(r["tau_p50"]))
        if not pts:
            continue
        fs = sorted(pts)
        ls = "-" if t in ("bert:5", "llama:3", "yolo:1") else "--"
        ax.plot(fs, [st.median(pts[f]) for f in fs], ls, color=FAMILY_COLOR[fam], linewidth=1.0)
        for f in fs:
            ax.scatter([f] * len(pts[f]), pts[f], s=9, color=FAMILY_COLOR[fam], marker=marks[ex], alpha=0.7,
                       linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("GPU frequency (MHz)")
    ax.set_ylabel("Solo median service time (s)")
    ax.set_xticks([408, 612, 714, 816, 1020])
    ax.grid(which="both", linewidth=0.2)
    handles = [Line2D([], [], color=FAMILY_COLOR[t.split(":")[0]], linestyle="-" if t in ("bert:5", "llama:3", "yolo:1")
                      else "--", marker="o" if t in ("bert:5", "llama:3", "yolo:1") else "s", markersize=3, label=t)
               for t in TYPES]
    ax.legend(handles=handles, ncol=2, loc="upper right", handlelength=2.2)
    fig.tight_layout()
    return _save(fig, out, "fig_solo_time")


def fig_power(ctx, out):
    d = Path(_need(ctx, "run1_csv"))
    solo, corun, idle = _csv(d / "solo.csv"), _csv(d / "corun.csv"), _csv(d / "idle.csv")
    fig, ax = plt.subplots(figsize=(W, 3.4))
    by = collections.defaultdict(list)
    for r in corun:
        by[(r["A"], int(float(r["f_mhz"])))].append(float(r["p_board_w"]))
    sets = sorted({A for A, _ in by})
    for k, A in enumerate(sets):
        fs = sorted(f for a, f in by if a == A)
        ax.plot(fs, [st.median(by[(A, f)]) for f in fs], color="#999999", linewidth=0.6, alpha=0.6,
                label="Co-run sets (20)" if k == 0 else None)
    for fam in FAMILY_COLOR:
        pts = collections.defaultdict(list)
        for r in solo:
            if r["type_id"].startswith(fam + ":"):
                pts[int(float(r["f_mhz"]))].append(float(r["p_board_w"]))
        fs = sorted(pts)
        ax.plot(fs, [st.median(pts[f]) for f in fs], "-o", color=FAMILY_COLOR[fam], markersize=3, linewidth=1.3,
                label=f"Solo {fam}")
    for kind, style, lab in (("resident", "-", "Warm idle (models resident)"), ("pre", "", "Idle before start")):
        pts = sorted((int(float(r["f_mhz"])), float(r["p_idle_w"])) for r in idle if r["kind"] == kind)
        if pts:
            ax.plot(*zip(*pts), style + "s" if style else "s", color="black", markersize=4,
                    markerfacecolor="black" if kind == "resident" else "white", label=lab, linewidth=1.3)
    ax.set_xlabel("GPU frequency (MHz)")
    ax.set_ylabel("Board power (W)")
    ax.set_xticks([408, 612, 714, 816, 1020])
    ax.grid(linewidth=0.2)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return _save(fig, out, "fig_power")


def _beta_fit(rows):
    """M1 beta per (family, f) from rows (type_id, family, A, f, s), as in exp2.model.slowdown."""
    num, den = {}, {}
    for _, fam, A, f, s in rows:
        if len(A) > 1:
            k = (fam, f)
            num[k] = num.get(k, 0.0) + (s - 1.0) * (len(A) - 1)
            den[k] = den.get(k, 0.0) + (len(A) - 1) ** 2
    return {k: num[k] / den[k] for k in num}


def loco_predictions(corun_rows):
    """Leave-one-cell-out predictions of M0 and M1 over the split != heldout cells.
    -> list of (cell_id, type_id, f, measured s_p50, pred M0, pred M1)."""
    train = [r for r in corun_rows if r.get("split") != "heldout"]

    def rows_of(rs):
        return [(r["type_id"], r["type_id"].split(":")[0], tuple(r["A"].split("|")), int(float(r["f_mhz"])),
                 float(r["s_p50"])) for r in rs]

    out = []
    for cell in sorted({r["cell_id"] for r in train}):
        fit = rows_of([r for r in train if r["cell_id"] != cell])
        held = rows_of([r for r in train if r["cell_id"] == cell])
        c0 = collections.defaultdict(list)
        for _, _, A, f, s in fit:
            c0[f].append(len(A) / s)
        c0 = {f: float(np.mean(v)) for f, v in c0.items()}
        beta = _beta_fit(fit)
        dflt = float(np.mean(list(beta.values())))
        for tid, fam, A, f, y in held:
            near0 = c0[f] if f in c0 else c0[min(c0, key=lambda k: abs(k - f))]
            ks = [k for k in beta if k[0] == fam]
            b = dflt if not ks else beta[min(ks, key=lambda k: abs(k[1] - f))]
            out.append((cell, tid, f, y, max(1.0, len(A) / near0), max(1.0, 1.0 + b * (len(A) - 1))))
    return out


def mdape(pred, meas):
    return float(np.median(100.0 * np.abs(np.array(pred) - np.array(meas)) / np.array(meas)))


def fig_slowdown_fit(ctx, out):
    rows = _csv(Path(_need(ctx, "run1_csv")) / "corun.csv")
    sd = _json(Path(_need(ctx, "frozen")) / "slowdown.json")
    lp = loco_predictions(rows)
    meas = [x[3] for x in lp]
    fig, axes = plt.subplots(1, 2, figsize=(W * 1.45, 2.9), sharex=True, sharey=True)
    lim = (0.9 * min(meas), 1.05 * max(max(meas), max(x[4] for x in lp), max(x[5] for x in lp)))
    ref = sd.get("selection", {}).get("loco_mdape", {})
    for ax, (idx, name) in zip(axes, ((4, "M0"), (5, "M1"))):
        for fam, col in FAMILY_COLOR.items():
            sel = [x for x in lp if x[1].startswith(fam + ":")]
            ax.scatter([x[3] for x in sel], [x[idx] for x in sel], s=10, color=col, label=fam, linewidth=0, alpha=0.8)
        ax.plot(lim, lim, color="black", linewidth=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("Measured median slowdown (1)")
        md = mdape([x[idx] for x in lp], meas)
        ax.text(0.04, 0.95, f"{name}, LOCO MdAPE {md:.1f}%", transform=ax.transAxes, va="top")
        ax.grid(linewidth=0.2)
        ctx.setdefault("_notes", {})[f"slowdown_{name}_mdape"] = (md, ref.get(name))
    axes[0].set_ylabel("Predicted median slowdown (1)")
    axes[1].legend(loc="lower right", handletextpad=0.1)
    fig.tight_layout()
    return _save(fig, out, "fig_slowdown_fit")


# ------------------------------------------------------------------------------------------------ Run 1c
def load_run1c(dirs):
    """-> {(family, state): dict(spawn_to_ready_s, first_infer_s, energy_j or None, note)}."""
    res = {}
    for d in dirs or []:
        d = Path(d)
        sj = d / "session.json"
        if not sj.is_file():
            continue
        state = "cold" if d.name.endswith("_cold") else "warm" if d.name.endswith("_warm") else None
        sess = json.load(open(sj))
        for fam, srv in (sess.get("servers") or {}).items():
            if state is None or "t_spawn_ns" not in srv or "t_ready_ns" not in srv:
                continue
            t0, t1 = srv["t_spawn_ns"], srv["t_ready_ns"]
            fi = (srv.get("ready", {}).get("warmup", {}).get("first_infer_s") or {})
            e = _energy_j(d / "telemetry" / "power.csv", t0, t1)
            res[(fam, state)] = {"ready_s": (t1 - t0) / 1e9, "first_s": max(fi.values()) if fi else None,
                                 "energy_j": e}
    return res


def _energy_j(path, t0, t1):
    """Integral of VDD_IN power (trapezoid, mW -> J) over t0..t1 on the t_mono_ns clock, or None."""
    if not Path(path).is_file():
        return None
    t, p = [], []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            t.append(int(r["t_mono_ns"]))
            p.append(float(r["vdd_in_mw"]))
    t, p = np.array(t, dtype=np.int64), np.array(p)
    if len(t) < 2 or t[0] > t0 or t[-1] < t1:
        return None
    ts = np.concatenate(([t0], t[(t > t0) & (t < t1)], [t1]))
    ps = np.interp(ts, t, p)
    return float(np.sum((ps[1:] + ps[:-1]) / 2 * np.diff(ts)) / 1e12)   # mW * ns -> J


def _run1c(ctx):
    res = load_run1c(ctx.get("run1c"))
    if not res:
        raise Skip("no Run 1c session with spawn and ready times (--run1c)")
    return res


def fig_warmup(ctx, out):
    res = _run1c(ctx)
    fams = [f for f in ("bert", "llama", "yolo") if (f, "cold") in res and (f, "warm") in res]
    if not fams:
        raise Skip("no family with both a cold and a warm Run 1c session")
    have_e = all(res[(f, s)]["energy_j"] is not None for f in fams for s in ("cold", "warm"))
    panels = [("ready_s", "Start-up, spawn to ready (s)"), ("first_s", "First inference (s)")]
    if have_e:
        panels.append(("energy_j", "Start-up energy (J)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(W * len(panels) / 2.2, 2.6))
    x = np.arange(len(fams))
    for ax, (key, lab) in zip(np.atleast_1d(axes), panels):
        for k, (state, hatch) in enumerate((("cold", ""), ("warm", "//"))):
            v = [res[(f, state)][key] for f in fams]
            ax.bar(x + (k - 0.5) * 0.38, v, 0.38, color=[FAMILY_COLOR[f] for f in fams], hatch=hatch,
                   edgecolor="white" if not hatch else "black", linewidth=0.4)
        ax.set_xticks(x, fams)
        ax.set_ylabel(lab)
    fig.legend(handles=[matplotlib.patches.Patch(facecolor="#bbbbbb", label="Cold caches"),
                        matplotlib.patches.Patch(facecolor="#bbbbbb", hatch="//", edgecolor="black",
                                                 label="Warm caches")], loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out, "fig_warmup")


# ------------------------------------------------------------------------------------------------ Run 2
def load_run2(run2_dir):
    """-> {cond: {policy: dict(min_c [pooled over repeats, lowest type], energy [mean over repeats], n)}}."""
    rows = _csv(Path(run2_dir) / "open_conditions.csv")
    runs = collections.defaultdict(list)
    for r in rows:
        runs[r["condition_id"]].append(r)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for cid, rs in runs.items():
        r0 = rs[0]
        cond = f"{r0['workload']}_r{int(round(float(r0['rho']) * 100)):03d}" + ("_b" if r0["arrival"] == "gamma" else "")
        by[cond][r0["policy_key"]].append(rs)
    out = {}
    for c, ps in by.items():
        out[c] = {}
        for p, rep in ps.items():
            tot, ok = collections.defaultdict(float), collections.defaultdict(float)
            for rs in rep:
                for x in rs:
                    n = float(x.get("n") or 1)
                    tot[x["type_id"]] += n
                    ok[x["type_id"]] += n * float(x["complete_rate"])
            out[c][p] = {"min_c": min(ok[t] / tot[t] for t in tot),   # pooled over the repeats, lowest over the types
                         "energy": float(np.mean([float(rs[0]["energy_per_req_j"]) for rs in rep])), "n": len(rep)}
    return out


def _run2(ctx):
    return load_run2(_need(ctx, "run2"))


def _hl(p):
    return p == "F" or p in REFS


def fig_run2_energy_attain(ctx, out):
    data = _run2(ctx)
    conds = [c for c in COND_ORDER if c in data]
    fig, axes = plt.subplots(2, 3, figsize=(W * 1.7, 4.6))
    axes = axes.ravel()
    for ax, c in zip(axes, conds):
        for p, v in data[c].items():
            if not _hl(p):
                ax.scatter(v["min_c"], v["energy"], s=8, color="#bbbbbb", linewidth=0, zorder=2)
        for p in REFS:
            if p in data[c]:
                v = data[c][p]
                ax.scatter(v["min_c"], v["energy"], s=30, color=POLICY_COLOR[p], edgecolor="black", linewidth=0.5,
                           zorder=3)
        if "F" in data[c]:
            v = data[c]["F"]
            ax.scatter(v["min_c"], v["energy"], s=90, marker="*", color=POLICY_COLOR["F"], zorder=4)
        ax.axvline(0.95, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Lowest per-type complete rate (1)")
        ax.set_ylabel("Energy per request (J)")
        ax.text(0.03, 0.97, COND_LABEL[c], transform=ax.transAxes, va="top")
    for ax in axes[len(conds):]:
        ax.axis("off")
    handles = [Line2D([], [], marker="*", color="none", markerfacecolor="black", markersize=10, label="F")]
    handles += [Line2D([], [], marker="o", color="none", markerfacecolor=POLICY_COLOR[p], markeredgecolor="black",
                       markersize=6, label=p) for p in REFS]
    handles.append(Line2D([], [], marker="o", color="none", markerfacecolor="#bbbbbb", markersize=4, label="grid point"))
    handles.append(Line2D([], [], color="black", linestyle="--", label="0.95"))
    axes[-1].legend(handles=handles, loc="center", ncol=1)
    fig.tight_layout()
    return _save(fig, out, "fig_run2_energy_attain")


def fig_sim_accuracy(ctx, out):
    data = _run2(ctx)
    fz = _json(Path(_need(ctx, "frozen")) / "freeze.json")
    fig, axes = plt.subplots(1, 2, figsize=(W * 1.45, 2.9))
    pts = {"c": [], "e": []}
    for c in fz["conditions"]:
        cid = c["condition_id"]
        if cid not in data:
            continue
        pr = c["predictions"]
        for p, v in data[cid].items():
            if p not in pr or "per_type" not in pr[p]:
                continue
            pc = min(t["complete_rate"] for t in pr[p]["per_type"].values())
            pts["c"].append((cid, p, pc, v["min_c"]))
            pts["e"].append((cid, p, pr[p]["energy_per_req_j"], v["energy"]))
    if not pts["c"]:
        raise Skip("no measured policy with a frozen prediction")
    for ax, key, lab in ((axes[0], "c", "complete rate (1)"), (axes[1], "e", "energy per request (J)")):
        for cid in COND_ORDER:
            sel = [x for x in pts[key] if x[0] == cid]
            ax.scatter([x[2] for x in sel], [x[3] for x in sel], s=10, color=COND_COLOR[cid], linewidth=0,
                       label=COND_LABEL[cid], alpha=0.85)
            f = [x for x in sel if x[1] == "F"]
            ax.scatter([x[2] for x in f], [x[3] for x in f], s=60, marker="*", color=COND_COLOR[cid],
                       edgecolor="black", linewidth=0.5, zorder=4)
        allv = [x[2] for x in pts[key]] + [x[3] for x in pts[key]]
        lo, hi = min(allv), max(allv)
        pad = 0.04 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=0.8)
        ax.set_xlabel("Predicted " + ("lowest per-type " if key == "c" else "") + lab)
        ax.set_ylabel("Measured " + ("lowest per-type " if key == "c" else "") + lab)
        ax.grid(linewidth=0.2)
    axes[0].legend(loc="upper left", fontsize=8, handletextpad=0.1,
                   title="Mix, load, CV (star = F)", title_fontsize=8)
    fig.tight_layout()
    return _save(fig, out, "fig_sim_accuracy")


# ------------------------------------------------------------------------------------------------ 3D
def _bars3d(ax, xs, ys, zs, colors, dx=0.6, dy=0.6):
    ax.bar3d(xs, ys, np.zeros(len(zs)), dx, dy, zs, color=colors, shade=True, edgecolor="black", linewidth=0.2)


def fig3d_warmup(ctx, out):
    res = _run1c(ctx)
    fams = [f for f in ("bert", "llama", "yolo") if (f, "cold") in res and (f, "warm") in res]
    if not fams:
        raise Skip("no family with both a cold and a warm Run 1c session")
    have_e = all(res[(f, s)]["energy_j"] is not None for f in fams for s in ("cold", "warm"))
    panels = [("ready_s", "Start-up (s)")] + ([("energy_j", "Energy (J)")] if have_e else [])
    fig = plt.figure(figsize=(W * len(panels) / 1.4, 3.4))
    for k, (key, lab) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), k + 1, projection="3d")
        xs, ys, zs, cols = [], [], [], []
        for i, f in enumerate(fams):
            for j, s in enumerate(("cold", "warm")):
                xs.append(i)
                ys.append(1 - j)
                zs.append(res[(f, s)][key])
                cols.append(FAMILY_COLOR[f])
        _bars3d(ax, np.array(xs) - 0.3, np.array(ys) - 0.3, zs, cols)
        ax.set_xticks(range(len(fams)), fams)
        ax.set_yticks([0, 1], ["warm", "cold"])
        ax.set_xlabel("Family", labelpad=4)
        ax.set_ylabel("Caches", labelpad=4)
        ax.set_zlabel(lab, labelpad=2)
        ax.view_init(elev=24, azim=-58)
    return _save(fig, out, "fig3d_warmup")


def fig3d_capacity(ctx, out):
    cap = capacity_table(_csv(Path(_need(ctx, "run1_csv")) / "corun.csv"))
    sets = _ordered_sets(cap)
    fig = plt.figure(figsize=(W * 1.45, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    xs, ys, zs, cols = [], [], [], []
    floor = 0.8
    for i, A in enumerate(sets):
        for j, f in enumerate(LEVELS):
            if (A, f) in cap:
                xs.append(j)
                ys.append(i)
                zs.append(cap[(A, f)])
                cols.append(plt.get_cmap("RdBu_r")(0.5 + (cap[(A, f)] - 1.0) / 0.8))
    zs = np.array(zs)
    ax.bar3d(np.array(xs) - 0.3, np.array(ys) - 0.3, np.full(len(zs), floor), 0.6, 0.6, zs - floor, color=cols,
             shade=True, edgecolor="black", linewidth=0.2)
    ax.set_zlim(floor, 1.4)
    ax.set_xticks(range(3), [str(f) for f in LEVELS])
    ax.set_yticks([0, 4, 9, 14, 19], ["1", "5", "10", "15", "20"])
    ax.set_xlabel("GPU frequency (MHz)", labelpad=4)
    ax.set_ylabel("Co-run set (number)", labelpad=4)
    ax.set_zlabel("Capacity (1)", labelpad=2)
    ax.view_init(elev=26, azim=-62)
    ctx.setdefault("_notes", {})["fig3d_capacity_sets"] = [f"{i + 1}: {_short(A)}" for i, A in enumerate(sets)]
    return _save(fig, out, "fig3d_capacity")


def fig3d_energy(ctx, out):
    rows = _csv(Path(_need(ctx, "run2")) / "open_conditions.csv")
    fz = _json(Path(_need(ctx, "frozen")) / "freeze.json")
    spec = {}
    for c in fz["conditions"]:
        spec[c["condition_id"]] = {g["name"]: g for g in c["grid"]}
    en = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        cond = f"{r['workload']}_r{int(round(float(r['rho']) * 100)):03d}" + ("_b" if r["arrival"] == "gamma" else "")
        if r["type_id"] == "bert:5":    # one row per run: energy_per_req_j is the same on every type row
            en[cond][r["policy_key"]].append(float(r["energy_per_req_j"]))
    conds = [c for c in ("A_r060", "B_r060") if c in en and c in spec]
    if not conds:
        raise Skip("no A_r060 / B_r060 grid in the Run 2 data")
    lim_pos = {1: 0, 2: 1, None: 2}
    fig = plt.figure(figsize=(W * len(conds) / 1.1, 3.8))
    for k, c in enumerate(conds):
        ax = fig.add_subplot(1, len(conds), k + 1, projection="3d")
        pts = []
        for name, vs in en[c].items():
            g = spec[c].get(name)
            if g is None or g["order"] != "stf":
                continue
            pts.append((lim_pos[g["k_short"]], lim_pos[g["k_long"]], g["f_mhz"], float(np.mean(vs))))
        if not pts:
            continue
        floor = 0.8 * min(p[3] for p in pts)
        for f, off in ((612, -0.36), (1020, 0.0)):
            sel = [p for p in pts if p[2] == f]
            ax.bar3d([p[0] + off - 0.02 for p in sel], [p[1] - 0.3 for p in sel], np.full(len(sel), floor), 0.34, 0.6,
                     [p[3] - floor for p in sel], color=LEVEL_COLOR[f], shade=True, edgecolor="black", linewidth=0.2)
        ax.set_zlim(floor, max(p[3] for p in pts) * 1.02)
        lab = ["1", "2", "inf"]
        ax.set_xticks(range(3), lab)
        ax.set_yticks(range(3), lab)
        ax.set_xlabel(r"$k_{\mathrm{short}}$", labelpad=2)
        ax.set_ylabel(r"$k_{\mathrm{long}}$", labelpad=2)
        ax.set_zlabel("Energy per request (J)", labelpad=2)
        ax.view_init(elev=26, azim=-55)
        ax.text2D(0.02, 0.95, COND_LABEL[c], transform=ax.transAxes)
        ax.legend(handles=[matplotlib.patches.Patch(color=LEVEL_COLOR[f], label=f"{f} MHz") for f in (612, 1020)],
                  loc="upper right")
    return _save(fig, out, "fig3d_energy")


FIGS = collections.OrderedDict((f.__name__, f) for f in (
    fig_capacity, fig_gamma, fig_solo_time, fig_power, fig_slowdown_fit, fig_warmup, fig_run2_energy_attain,
    fig_sim_accuracy, fig3d_warmup, fig3d_capacity, fig3d_energy))


def run(ctx, out, only=None):
    """Make the figures; -> ({name: [files]}, {name: skip reason})."""
    _style()
    made, skipped = {}, {}
    for name, fn in FIGS.items():
        if only and name not in only:
            continue
        try:
            made[name] = fn(ctx, out)
            print(f"[figures] {name}: wrote {', '.join(made[name])}")
        except Skip as e:
            skipped[name] = str(e)
            print(f"[figures] {name}: SKIPPED ({e})")
    return made, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exp2.analyze.figures")
    ap.add_argument("--run1-csv")
    ap.add_argument("--frozen")
    ap.add_argument("--run1c", nargs="*", default=[])
    ap.add_argument("--run2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None, choices=list(FIGS))
    a = ap.parse_args(argv)
    ctx = {"run1_csv": a.run1_csv, "frozen": a.frozen, "run1c": a.run1c, "run2": a.run2}
    made, skipped = run(ctx, a.out, a.only)
    for k, v in sorted(ctx.get("_notes", {}).items()):
        print(f"[note] {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
