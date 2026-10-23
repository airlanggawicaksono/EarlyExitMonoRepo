"""Figures of the Exp2 paper from the MAXN_SUPER profiling data and the final Run 2 analysis.

numpy, matplotlib and the standard library only. No torch, no model run. One function per figure.
CLI: python -m exp2.analyze.figures --run1-csv DIR --frozen DIR --run1c DIR [DIR ...] --run2 DIR --out DIR
         [--only NAME ...] [--no-titles]
  --run1-csv  exp2/frozen/run2/run1_csv (corun.csv solo.csv idle.csv)
  --frozen    exp2/frozen/run2 (slowdown.json power.json freeze.json)
  --run1c     Run 1c session folders (session.json + telemetry/power.csv); the cold/warm state comes from the folder name
  --run2      analysis_final folder (open_conditions.csv)
  --no-titles paper version: no title and no takeaway line (captions are in LaTeX). Default: titles on.
A figure whose input is missing is skipped with a printed reason. Output: NAME.pdf and NAME.png (200 dpi).
"""
import argparse
import collections
import csv
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ------------------------------------------------------------------------------------------------ names and colours
TYPES = ("bert:5", "bert:23", "llama:3", "llama:15", "yolo:1", "yolo:5")
TYPE_NAME = {"bert:5": "BERT exit 5", "bert:23": "BERT exit 23", "llama:3": "LLaMA exit 3",
             "llama:15": "LLaMA exit 15", "yolo:1": "YOLO exit 1", "yolo:5": "YOLO exit 5"}
TYPE_SHORT = {"bert:5": "BERT-5", "bert:23": "BERT-23", "llama:3": "LLaMA-3", "llama:15": "LLaMA-15",
              "yolo:1": "YOLO-1", "yolo:5": "YOLO-5"}
# One hue per family; the early exit is light with a circle, the late exit is dark with a square.
TYPE_STYLE = {"bert:5": ("#56B4E9", "o"), "bert:23": ("#00508A", "s"),
              "llama:3": ("#F5A25D", "o"), "llama:15": ("#A83F00", "s"),
              "yolo:1": ("#5FD0A5", "o"), "yolo:5": ("#00694D", "s")}
FAMILY_COLOR = {"bert": "#0072B2", "llama": "#D55E00", "yolo": "#009E73"}
FAMILY_NAME = {"bert": "BERT", "llama": "LLaMA", "yolo": "YOLO"}
# Names of RESTRUCTURE_ADDENDUM_1 sections A and B: F is CoSwarm's policy, the other keys are the baselines (legend
# order). Colours: Okabe-Ito. A colour used for two methods always comes with another marker. Filled marker: three
# runs, hollow marker: one run (fig_run2_energy_attain).
POLICY_NAME = {"F": "CoSwarm (this work)", "default_pinned": "Unrestricted co-running", "default": "Governor default",
               "serial": "Serial", "serial_stf": "Serial, short first", "solo_profile_freq": "Solo-profile frequency",
               "serial_edgeserving": "Serial, deadline-aware", "priority_serial": "Priority serial",
               "capacity_one": "Capacity-one admission", "probe_margin": "Probe-margin frequency",
               "greenllm_lanes": "Queue-aware dynamic frequency", "occupancy_f": "Occupancy-based frequency"}
POLICY_COLOR = {"F": "#000000", "default_pinned": "#56B4E9", "default": "#E69F00", "serial": "#F0E442",
                "serial_stf": "#CC79A7", "solo_profile_freq": "#0072B2", "serial_edgeserving": "#D55E00",
                "priority_serial": "#009E73", "capacity_one": "#0072B2", "probe_margin": "#56B4E9",
                "greenllm_lanes": "#009E73", "occupancy_f": "#E69F00"}
POLICY_MARKER = {"F": "*", "default_pinned": "o", "default": "o", "serial": "s", "serial_stf": "s",
                 "solo_profile_freq": "D", "serial_edgeserving": "s", "priority_serial": "p", "capacity_one": ">",
                 "probe_margin": "D", "greenllm_lanes": "^", "occupancy_f": "v"}
GRID_GREY = "#BDBDBD"
LEVELS = (612, 816, 1020)
LEVEL_COLOR = {612: "#56B4E9", 816: "#E69F00", 1020: "#D55E00"}
COND_ORDER = ("A_r030", "A_r060", "B_r030", "B_r060", "B_r060_b")
COND_LABEL = {"A_r030": "Mix A: BERT + YOLO, load 0.3, Poisson", "A_r060": "Mix A: BERT + YOLO, load 0.6, Poisson",
              "B_r030": "Mix B: all three, load 0.3, Poisson", "B_r060": "Mix B: all three, load 0.6, Poisson",
              "B_r060_b": "Mix B: all three, load 0.6, bursty"}
COND_COLOR = dict(zip(COND_ORDER, ("#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#CC79A7")))
GROUPS = ("BERT + YOLO pairs", "Sets with LLaMA exit 3", "Sets with LLaMA exit 15")
GROUP_LABEL2 = {GROUPS[0]: "BERT + YOLO\npairs", GROUPS[1]: "With LLaMA\nexit 3", GROUPS[2]: "With LLaMA\nexit 15"}
CM = 1 / 2.54
LIGHT_RED = "#F6D5D0"
LIGHT_GREEN = "#D6EEDD"


class Skip(Exception):
    """Raised by a figure function when its input is missing."""


def _style():
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10, "axes.titlesize": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
                         "legend.frameon": False})


def _finish(fig, ctx, name, out, title, take=None, top=0.92, bottom=0.06, adjust=None):
    """Title (what the figure answers) and one takeaway line, unless --no-titles. Then tight layout and save."""
    on = ctx.get("titles", True)
    if on:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
        if take:
            fig.text(0.5, 0.004, take, ha="center", va="bottom", fontsize=10, style="italic")
    if adjust:
        fig.subplots_adjust(**adjust)
    else:
        fig.tight_layout(rect=(0, bottom if on and take else 0, 1, top if on else 1))
    return _save(fig, out, name)


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
    return " + ".join(TYPE_SHORT.get(t, t) for t in A.split("|"))


def set_group(A):
    if "llama:3" in A:
        return GROUPS[1]
    if "llama:15" in A:
        return GROUPS[2]
    return GROUPS[0]


def _corun_cells(rows):
    """(A, f, source, repeat) -> rows: Run 1d rows have a cell_id prefix 'd_'."""
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
    return sorted({A for A, _ in keys}, key=lambda A: (GROUPS.index(set_group(A)), A))


def _group_spans(sets):
    spans = []
    for g in GROUPS:
        idx = [i for i, A in enumerate(sets) if set_group(A) == g]
        if idx:
            spans.append((g, idx[0], idx[-1]))
    return spans


def _p_idle(ctx):
    pw = _json(Path(_need(ctx, "frozen")) / "power.json")
    return {int(k): float(v) for k, v in pw["p_idle_w"].items()}


def _handles_types():
    return [Line2D([], [], color=TYPE_STYLE[t][0], marker=TYPE_STYLE[t][1], markersize=6, linestyle="none",
                   markeredgecolor="black", markeredgewidth=0.3, label=TYPE_NAME[t]) for t in TYPES]


# ------------------------------------------------------------------------------------------------ 2D: profiling
def fig_capacity(ctx, out):
    rows = _csv(Path(_need(ctx, "run1_csv")) / "corun.csv")
    cap = capacity_table(rows)
    sets = _ordered_sets(cap)
    M = np.array([[cap.get((A, f), np.nan) for f in LEVELS] for A in sets])
    fig, ax = plt.subplots(figsize=(7.6, 0.33 * len(sets) + 1.4))
    norm = TwoSlopeNorm(vcenter=1.0, vmin=min(0.85, np.nanmin(M)), vmax=max(1.35, np.nanmax(M)))
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_xticks(range(3), [f"{f} MHz" for f in LEVELS])
    ax.set_yticks(range(len(sets)), [_short(A) for A in sets])
    ax.set_xlabel("GPU frequency")
    ax.set_ylabel("Requests running together (BERT-5 = BERT exit 5, ...)")
    for g, a, b in _group_spans(sets):
        if a > 0:
            ax.axhline(a - 0.5, color="white", linewidth=4)
        ax.text(2.62, (a + b) / 2, GROUP_LABEL2[g], rotation=270, ha="left", va="center", fontsize=10,
                fontweight="bold", clip_on=False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.2, fraction=0.05)
    cb.set_label("Co-run capacity (serial execution = 1.0)")
    cb.ax.text(0.5, 1.02, "faster\nthan serial", transform=cb.ax.transAxes, ha="center", va="bottom", fontsize=10)
    cb.ax.text(0.5, -0.02, "slower\nthan serial", transform=cb.ax.transAxes, ha="center", va="top", fontsize=10)
    pair = [cap[(A, f)] for A in sets for f in LEVELS if set_group(A) == GROUPS[0]]
    take = (f"BERT + YOLO pairs stay between {min(pair):.2f} and {max(pair):.2f}; "
            f"sets with LLaMA reach up to {np.nanmax(M):.2f}.")
    return _finish(fig, ctx, "fig_capacity", out,
                   "Running requests together: is it faster than running them one after another?", take,
                   adjust={"left": 0.3, "right": 0.82, "top": 0.93, "bottom": 0.1})


def fig_gamma(ctx, out):
    rows = _csv(Path(_need(ctx, "run1_csv")) / "corun.csv")
    solo = _csv(Path(_need(ctx, "run1_csv")) / "solo.csv")
    g = gamma_table(rows, solo, _p_idle(ctx))
    sets = _ordered_sets(g)
    fig, ax = plt.subplots(figsize=(8.2, 0.33 * len(sets) + 1.6))
    ys = np.arange(len(sets))
    vals = [v for v in g.values()]
    hi = max(1.02, max(vals) + 0.01)
    lo = min(vals) - 0.01
    ax.axvspan(1.0, hi, color=LIGHT_GREEN, zorder=0)
    ax.axvspan(lo, 1.0, color=LIGHT_RED, zorder=0, alpha=0.6)
    for f in LEVELS:
        ax.scatter([g.get((A, f), np.nan) for A in sets], ys, s=40, color=LEVEL_COLOR[f], label=f"{f} MHz",
                   edgecolor="black", linewidth=0.4, zorder=3)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2, zorder=2)
    ax.set_xlim(lo, hi)
    ax.text(1.0, -1.9, "serial execution (1.0)", ha="center", va="bottom", fontsize=10, fontweight="bold",
            clip_on=False)
    ax.text(0.9985, -0.9, "co-run saves no energy", ha="right", va="bottom", fontsize=10, clip_on=False)
    ax.text(1.0015, -0.9, "co-run saves energy", ha="left", va="bottom", fontsize=10, clip_on=False)
    ax.set_yticks(ys, [_short(A) for A in sets])
    ax.set_ylim(len(sets) - 0.4, -0.6)
    for a, b in [(a - 0.5, b) for _g, a, b in _group_spans(sets)][1:]:
        ax.axhline(a, color="#888888", linewidth=0.6)
    ax.set_xlabel("Co-run energy ratio (1.0 = same energy per work as serial; below 1 = no saving)")
    ax.set_ylabel("Requests running together (BERT-5 = BERT exit 5, ...)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="GPU frequency", title_fontsize=10)
    take = f"All {len(vals)} ratios are below 1.0 (from {min(vals):.3f} to {max(vals):.3f}): running together never saves energy."
    return _finish(fig, ctx, "fig_gamma", out,
                   "Running requests together: does it save energy compared with running them one after another?",
                   take, adjust={"left": 0.3, "right": 0.82, "top": 0.9, "bottom": 0.1})


def fig_solo_time(ctx, out):
    rows = _csv(Path(_need(ctx, "run1_csv")) / "solo.csv")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ratios = []
    for t in TYPES:
        pts = collections.defaultdict(list)
        for r in rows:
            if r["type_id"] == t:
                pts[int(float(r["f_mhz"]))].append(float(r["tau_p50"]))
        if not pts:
            continue
        col, mk = TYPE_STYLE[t]
        fs = sorted(pts)
        ls = "-" if mk == "o" else "--"
        ax.plot(fs, [st.median(pts[f]) for f in fs], ls, color=col, linewidth=1.6)
        for f in fs:
            ax.scatter([f] * len(pts[f]), pts[f], s=22, color=col, marker=mk, edgecolor="black", linewidth=0.3,
                       zorder=3)
        if 408 in pts and 1020 in pts:
            ratios.append(st.median(pts[408]) / st.median(pts[1020]))
    ax.set_yscale("log")
    ax.set_xlabel("GPU frequency (MHz)")
    ax.set_ylabel("Time to serve one request, running alone (s)")
    ax.set_xticks([408, 612, 714, 816, 1020])
    ax.grid(which="both", linewidth=0.3, alpha=0.6)
    ax.legend(handles=_handles_types(), loc="center left", bbox_to_anchor=(1.01, 0.5))
    take = None
    if ratios:
        take = (f"Raising the GPU from 408 to 1020 MHz makes every request type {min(ratios):.1f} to "
                f"{max(ratios):.1f} times faster.")
    return _finish(fig, ctx, "fig_solo_time", out, "Running alone: how long does each request take at each GPU speed?",
                   take)


def fig_power(ctx, out):
    d = Path(_need(ctx, "run1_csv"))
    solo, corun, idle = _csv(d / "solo.csv"), _csv(d / "corun.csv"), _csv(d / "idle.csv")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    by = collections.defaultdict(list)
    for r in corun:
        by[(r["A"], int(float(r["f_mhz"])))].append(float(r["p_board_w"]))
    sets = sorted({A for A, _ in by})
    for k, A in enumerate(sets):
        fs = sorted(f for a, f in by if a == A)
        ax.plot(fs, [st.median(by[(A, f)]) for f in fs], color="#9A9A9A", linewidth=0.8, alpha=0.7,
                label=f"{len(sets)} co-run sets" if k == 0 else None)
    for fam in FAMILY_COLOR:
        pts = collections.defaultdict(list)
        for r in solo:
            if r["type_id"].startswith(fam + ":"):
                pts[int(float(r["f_mhz"]))].append(float(r["p_board_w"]))
        fs = sorted(pts)
        ax.plot(fs, [st.median(pts[f]) for f in fs], "-o", color=FAMILY_COLOR[fam], markersize=5, linewidth=2.0,
                label=f"{FAMILY_NAME[fam]} running alone")
    res = sorted((int(float(r["f_mhz"])), float(r["p_idle_w"])) for r in idle if r["kind"] == "resident")
    ax.plot(*zip(*res), "-s", color="black", markersize=5, linewidth=2.0, label="Idle, models loaded")
    ax.set_xlabel("GPU frequency (MHz)")
    ax.set_ylabel("Board power (W)")
    ax.set_xticks([408, 612, 714, 816, 1020])
    ax.grid(linewidth=0.3, alpha=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    top = [st.median(by[(A, 1020)]) for A in sets if (A, 1020) in by]
    take = (f"At 1020 MHz, co-run sets draw {min(top):.1f} to {max(top):.1f} W; the idle board draws "
            f"{res[-1][1]:.1f} W.") if top else None
    return _finish(fig, ctx, "fig_power", out, "How much power does the board draw at each GPU speed?", take)


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
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharex=True, sharey=True)
    lim = (0.9 * min(meas), 1.05 * max(max(meas), max(x[4] for x in lp), max(x[5] for x in lp)))
    ref = sd.get("selection", {}).get("loco_mdape", {})
    names = {"M0": "Slowdown predictor M0", "M1": "Slowdown predictor M1"}
    err = {}
    for ax, (idx, name) in zip(axes, ((4, "M0"), (5, "M1"))):
        for t in TYPES:
            sel = [x for x in lp if x[1] == t]
            col, mk = TYPE_STYLE[t]
            ax.scatter([x[3] for x in sel], [x[idx] for x in sel], s=26, color=col, marker=mk, edgecolor="black",
                       linewidth=0.3, alpha=0.9, label=TYPE_NAME[t])
        ax.plot(lim, lim, color="black", linewidth=1.0)
        ax.text(lim[1] * 0.97, lim[1] * 0.93, "perfect prediction", ha="right", va="top", rotation=40, fontsize=10)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("Measured slowdown (x, running alone = 1)")
        err[name] = mdape([x[idx] for x in lp], meas)
        ax.set_title(f"{names[name]}: MdAPE {err[name]:.1f}%", fontsize=11)
        ax.grid(linewidth=0.3, alpha=0.6)
        ctx.setdefault("_notes", {})[f"slowdown_{name}_mdape"] = (err[name], ref.get(name))
    axes[0].set_ylabel("Predicted slowdown (x, running alone = 1)")
    axes[1].legend(handles=_handles_types(), loc="center left", bbox_to_anchor=(1.01, 0.5))
    take = (f"Predictions for left-out sets: MdAPE {err['M0']:.1f}% for predictor M0, "
            f"{err['M1']:.1f}% for predictor M1.")
    return _finish(fig, ctx, "fig_slowdown_fit", out,
                   "Running two or three requests together: can we predict how much slower each one gets?", take)


# ------------------------------------------------------------------------------------------------ Run 1c
def load_run1c(dirs):
    """-> {(family, state): dict(ready_s, first_s, energy_j or None)}."""
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
    fams = [f for f in ("bert", "llama", "yolo") if (f, "cold") in res and (f, "warm") in res]
    if not fams:
        raise Skip("no family with both a cold and a warm Run 1c session")
    return res, fams


def _warm_take(res, fams):
    r = {f: res[(f, "cold")]["ready_s"] / res[(f, "warm")]["ready_s"] for f in fams}
    lo, hi = min(r, key=r.get), max(r, key=r.get)
    return f"Filled caches shorten the start-up by {r[lo]:.1f}x ({FAMILY_NAME[lo]}) to {r[hi]:.1f}x ({FAMILY_NAME[hi]})."


def _state_legend(ax=None, fig=None, loc="upper right", **kw):
    h = [mpatches.Patch(facecolor="#BBBBBB", edgecolor="black", label="Empty caches (cold)"),
         mpatches.Patch(facecolor="#BBBBBB", hatch="//", edgecolor="black", label="Filled caches (warm)")]
    return (fig or ax).legend(handles=h, loc=loc, **kw)


def _grouped_family_bars(ax, res, fams, key, fmt):
    x = np.arange(len(fams))
    for k, (state, hatch) in enumerate((("cold", ""), ("warm", "//"))):
        v = [res[(f, state)][key] for f in fams]
        bars = ax.bar(x + (k - 0.5) * 0.38, v, 0.38, color=[FAMILY_COLOR[f] for f in fams], hatch=hatch,
                      edgecolor="black", linewidth=0.5)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val, fmt.format(val), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, [FAMILY_NAME[f] for f in fams])
    ax.margins(y=0.12)


def fig_warmup(ctx, out):
    res, fams = _run1c(ctx)
    have_e = all(res[(f, s)]["energy_j"] is not None for f in fams for s in ("cold", "warm"))
    panels = [("ready_s", "Time from process start to ready (s)", "{:.0f}"),
              ("first_s", "First request after start (s)", "{:.1f}")]
    if have_e:
        panels.append(("energy_j", "Energy from start to ready (J)", "{:.0f}"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.6))
    for ax, (key, lab, fmt) in zip(np.atleast_1d(axes), panels):
        _grouped_family_bars(ax, res, fams, key, fmt)
        ax.set_ylabel(lab)
    _state_legend(fig=fig, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.93))
    return _finish(fig, ctx, "fig_warmup", out, "Starting a model: how long does it take, and what does it cost?",
                   _warm_take(res, fams), top=0.86)


# ------------------------------------------------------------------------------------------------ Run 2
def _cond_key(r):
    return f"{r['workload']}_r{int(round(float(r['rho']) * 100)):03d}" + ("_b" if r["arrival"] == "gamma" else "")


def load_run2(run2_dir):
    """-> {cond: {policy: dict(min_c [pooled over repeats, lowest type], per_type [pooled over repeats],
    energy [mean over repeats], n)}}. Pooled: sum over the repeats of complete_rate x n, divided by the sum of n."""
    rows = _csv(Path(run2_dir) / "open_conditions.csv")
    runs = collections.defaultdict(list)
    for r in rows:
        runs[r["condition_id"]].append(r)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for cid, rs in runs.items():
        by[_cond_key(rs[0])][rs[0]["policy_key"]].append(rs)
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
            per = {t: ok[t] / tot[t] for t in tot}   # pooled over the repeats
            out[c][p] = {"min_c": min(per.values()), "per_type": per,   # min_c: lowest over the types
                         "energy": float(np.mean([float(rs[0]["energy_per_req_j"]) for rs in rep])), "n": len(rep)}
    return out


def _run2(ctx):
    return load_run2(_need(ctx, "run2"))


# Figure 2 of the paper (RESTRUCTURE_ADDENDUM_1 section D): mix A holds the existing models, mix B adds the new one.
ADMIT_CONDS = ("A_r030", "A_r060", "B_r030", "B_r060")
ADMIT_POLICIES = ("serial", "default_pinned", "F")
ADMIT_LABEL = {"A_r030": "Mix A (existing models), load 0.3", "A_r060": "Mix A (existing models), load 0.6",
               "B_r030": "Mix B (new model added), load 0.3", "B_r060": "Mix B (new model added), load 0.6"}
# Panel (row, column) of fig_admit: load 0.3 on top, mix A left. Each column is as wide as its bar groups, so every
# bar has the same width (mix B has six types: two mix B panels side by side leave no room for the numbers in the bars).
ADMIT_CELL = {"A_r030": (0, 0), "B_r030": (0, 1), "A_r060": (1, 0), "B_r060": (1, 1)}
TARGET = 0.95
# Paper figures (fig_admit, fig_run2_energy_attain, fig_sim_accuracy): every text at PRINT_FONT.
PRINT_FONT = 11   # 11 pt in a figure of at most 7 in prints at 8 pt or more at \textwidth (31pc, 5.15 in)
PRINT_RC = {k: PRINT_FONT for k in ("font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize",
                                    "ytick.labelsize", "legend.fontsize")}


def admit_table(run2_dir):
    """{cond: {policy: {type: on-time rate}}} of fig_admit: mixes A and B at loads 0.3 and 0.6 (Poisson), keys serial,
    default_pinned and F. Per type: sum over the repeats of complete_rate x n, divided by the sum of n."""
    data = load_run2(run2_dir)
    return {c: {p: dict(data[c][p]["per_type"]) for p in ADMIT_POLICIES if p in data[c]}
            for c in ADMIT_CONDS if c in data}


@plt.rc_context(PRINT_RC)
def fig_admit(ctx, out):
    tab = admit_table(_need(ctx, "run2"))
    if not any(tab.values()):
        raise Skip("no serial, default_pinned or F run in mix A or B at load 0.3 or 0.6")
    on = ctx.get("titles", True)
    kinds = {}   # request types per panel, in the order of TYPES
    for c in ADMIT_CONDS:
        seen = {t for v in tab.get(c, {}).values() for t in v}
        kinds[c] = [t for t in TYPES if t in seen] + sorted(seen - set(TYPES))
    ratios = [max([len(kinds[c]) for c in ADMIT_CONDS if ADMIT_CELL[c][1] == j] + [1]) + 0.1 for j in (0, 1)]
    # Saved with 0.1 in padding: at most 7 in wide. The titled version adds height, the panels keep their size.
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 6.5 if on else 5.9), sharey=True, gridspec_kw={"width_ratios": ratios})
    w = 0.3
    for c in ADMIT_CONDS:
        ax, ps, types = axes[ADMIT_CELL[c]], tab.get(c, {}), kinds[c]
        for k, p in enumerate(ADMIT_POLICIES):
            for i, t in enumerate(types):
                if t not in ps.get(p, {}):
                    continue
                x, v = i + (k - 1) * w, 100 * ps[p][t]
                ax.bar(x, v, w, color=POLICY_COLOR[p], edgecolor="black", linewidth=0.4, zorder=3)
                if v < 100 * TARGET:   # the rates below the target are printed in the bar, white in the black bar
                    ax.text(x, v - 2, f"{v:.1f}", rotation=90, ha="center", va="top", zorder=5,
                            color="white" if p == "F" else "black")
        ax.axhline(100 * TARGET, color="black", linestyle="--", linewidth=1.2, zorder=2)   # behind the bars
        # Two-level axis: the exit number under each bar group, "<family> exit" once under its exits.
        ax.set_xticks(range(len(types)), [t.split(":")[1] if t in TYPE_NAME else t for t in types])
        fams = [(t.split(":")[0], i) for i, t in enumerate(types) if t in TYPE_NAME]
        for fam in dict.fromkeys(f for f, _ in fams):
            xs = [i for f, i in fams if f == fam]
            ax.annotate(f"{FAMILY_NAME[fam]} exit", xy=((xs[0] + xs[-1]) / 2, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -22), textcoords="offset points", ha="center", va="top", fontweight="bold")
        ax.set_xlim(-0.55, max(len(types), 1) - 0.45)
        ax.set_ylim(0, 102)
        ax.set_yticks(range(0, 101, 20))
        ax.set_title(ADMIT_LABEL[c])
        if not types:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
    for ax in axes[:, 0]:
        ax.set_ylabel("Requests on time (%)")
    h = [mpatches.Patch(facecolor=POLICY_COLOR[p], edgecolor="black", linewidth=0.4, label=POLICY_NAME[p])
         for p in ADMIT_POLICIES]
    h.append(Line2D([], [], color="black", linestyle="--", linewidth=1.2, label="95% on-time target"))
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.955 if on else 1.0))   # above the panels
    met = {p: sum(1 for c in tab if p in tab[c] and min(tab[c][p].values()) >= TARGET) for p in ADMIT_POLICIES}
    take = ("Panels where every type meets the target: "
            + ", ".join(f"{POLICY_NAME[p].replace(' (this work)', '')} {met[p]}" for p in ADMIT_POLICIES)
            + f" of {len(tab)}.")
    ctx.setdefault("_notes", {})["fig_admit_per_type_pct"] = {
        c: {POLICY_NAME[p]: {TYPE_SHORT.get(t, t): round(100 * v, 1) for t, v in sorted(tab[c][p].items())}
            for p in tab[c]}
        for c in tab}
    return _finish(fig, ctx, "fig_admit", out, "Adding a new model: is every request type still on time?", take,
                   adjust={"left": 0.1, "right": 0.99, "top": 0.82 if on else 0.861, "bottom": 0.115 if on else 0.085,
                           "hspace": 0.47, "wspace": 0.06})


# Run 2 energy figure: mix A in the top row, mix B in the bottom row, the legend of the methods in the sixth cell.
RUN2_CELL = {"A_r030": (0, 0), "A_r060": (0, 1), "B_r030": (1, 0), "B_r060": (1, 1), "B_r060_b": (1, 2)}
RUN2_SHORT = {"A_r030": "Mix A, load 0.3", "A_r060": "Mix A, load 0.6", "B_r030": "Mix B, load 0.3",
              "B_r060": "Mix B, load 0.6", "B_r060_b": "Mix B, load 0.6, bursty"}
ONE_RUN = "One run each:"
LEGEND_WRAP = {"capacity_one": "Capacity-one\nadmission", "probe_margin": "Probe-margin\nfrequency",
               "greenllm_lanes": "Queue-aware dynamic\nfrequency", "occupancy_f": "Occupancy-based\nfrequency"}


def _method_marker(p, hollow):
    """Marker style of a named method: filled with a black edge (three runs) or hollow (one run)."""
    col = POLICY_COLOR[p]
    return {"marker": POLICY_MARKER[p], "facecolor": "white" if hollow else col,
            "edgecolor": col if hollow else "black", "edgewidth": 1.4 if hollow else 0.6}


def _method_handle(p, hollow):
    m = _method_marker(p, hollow)
    return Line2D([], [], linestyle="none", marker=m["marker"], markersize=7, markerfacecolor=m["facecolor"],
                  markeredgecolor=m["edgecolor"], markeredgewidth=m["edgewidth"],
                  label=LEGEND_WRAP.get(p, POLICY_NAME[p]))


def fig_run2_energy_attain(ctx, out):
    data = _run2(ctx)
    conds = [c for c in COND_ORDER if c in data and c in RUN2_CELL]
    on = ctx.get("titles", True)
    named = [p for p in POLICY_NAME if p != "F" and any(p in data[c] for c in conds)]
    one = {p: max(data[c][p]["n"] for c in conds if p in data[c]) == 1 for p in named}
    with plt.rc_context(PRINT_RC):
        # Saved with 0.1 in padding: at most 7 in wide. The titled version adds height, the panels keep their size.
        fig, axes = plt.subplots(2, 3, figsize=(6.7, 8.3 if on else 7.0))
        for c in conds:
            ax = axes[RUN2_CELL[c]]
            lo = min(100 * v["min_c"] for v in data[c].values()) - 4
            ax.axvspan(lo - 10, 95, color=LIGHT_RED, linewidth=0, zorder=0)
            ax.axvline(95, color="black", linestyle="--", linewidth=1.2, zorder=1)
            ax.set_xlim(lo, 102)
            for p, v in data[c].items():
                x, y = 100 * v["min_c"], v["energy"]
                if p == "F":
                    ax.scatter(x, y, s=260, marker="*", color="black", edgecolors="white", linewidths=0.6, zorder=6)
                elif p not in POLICY_NAME:   # the other grid policies and the ablations
                    ax.scatter(x, y, s=12, color=GRID_GREY, linewidths=0, zorder=2)
                else:
                    m = _method_marker(p, v["n"] == 1)
                    ax.scatter(x, y, s=46, marker=m["marker"], facecolors=m["facecolor"], edgecolors=m["edgecolor"],
                               linewidths=m["edgewidth"], zorder=4 if v["n"] == 1 else 5)
            ax.set_title(RUN2_SHORT[c])
        used = {RUN2_CELL[c] for c in conds}
        for cell in np.ndindex(axes.shape):
            if cell not in used:
                axes[cell].axis("off")
        h = [Line2D([], [], linestyle="none", marker="*", markersize=15, markerfacecolor="black",
                    markeredgecolor="white", markeredgewidth=0.6, label=POLICY_NAME["F"])]
        h += [_method_handle(p, False) for p in named if not one[p]]
        if any(one.values()):
            h.append(Line2D([], [], linestyle="none", label=ONE_RUN))
            h += [_method_handle(p, True) for p in named if one[p]]
        leg = axes[0, 2].legend(handles=h, loc="upper left", bbox_to_anchor=(-0.31, 1.12), borderaxespad=0,
                                handlelength=1.2, handletextpad=0.5, labelspacing=0.15)
        for t in leg.get_texts():
            if t.get_text() == ONE_RUN:
                t.set_fontstyle("italic")
        ref = [Line2D([], [], linestyle="none", marker="o", markersize=4.5, markerfacecolor=GRID_GREY,
                      markeredgecolor="none", label="Other grid policies and ablations"),
               Line2D([], [], color="black", linestyle="--", linewidth=1.2, label="95% on-time target"),
               mpatches.Patch(facecolor=LIGHT_RED, edgecolor="none", label="Misses the target")]
        fig.legend(handles=ref, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.055 if on else 0.0),
                   handlelength=1.5, handletextpad=0.5, columnspacing=1.0)
        fig.supxlabel("Requests completing on time (worst type, %)", y=0.1 if on else 0.045, fontsize=PRINT_FONT)
        fig.supylabel("Energy per request (J)", x=0.005, fontsize=PRINT_FONT)
        f_rates = ", ".join(f"{100 * data[c]['F']['min_c']:.1f}" for c in conds if "F" in data[c])
        take = (f"Worst-type on-time rate of CoSwarm in the five panels: {f_rates} %.\n"
                "Cheaper policies mostly end left of the target.")
        return _finish(fig, ctx, "fig_run2_energy_attain", out,
                       "Run 2 on the Jetson: how much energy does each policy use,\nand does it keep requests on time?",
                       take, adjust={"left": 0.1, "right": 0.975, "top": 0.882 if on else 0.955,
                                     "bottom": 0.181 if on else 0.13, "hspace": 0.32, "wspace": 0.34})


def _sim_points(data, fz):
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
            pts["c"].append((cid, p, 100 * pc, 100 * v["min_c"]))
            pts["e"].append((cid, p, pr[p]["energy_per_req_j"], v["energy"]))
    return pts


@plt.rc_context(PRINT_RC)
def fig_sim_accuracy(ctx, out):
    data = _run2(ctx)
    fz = _json(Path(_need(ctx, "frozen")) / "freeze.json")
    pts = _sim_points(data, fz)
    if not pts["c"]:
        raise Skip("no measured policy with a frozen prediction")
    on = ctx.get("titles", True)
    # Two square panels side by side, the legend below them. Saved with 0.1 in padding: at most 7 in wide.
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 5.2 if on else 4.48))
    for ax, key, lab in ((axes[0], "c", "Requests on time, worst type (%)"), (axes[1], "e", "Energy per request (J)")):
        for cid in COND_ORDER:
            sel = [x for x in pts[key] if x[0] == cid]
            ax.scatter([x[2] for x in sel], [x[3] for x in sel], s=18, color=COND_COLOR[cid], linewidth=0, alpha=0.85)
            f = [x for x in sel if x[1] == "F"]
            ax.scatter([x[2] for x in f], [x[3] for x in f], s=170, marker="*", color=COND_COLOR[cid],
                       edgecolor="black", linewidth=0.7, zorder=4)
        allv = [x[2] for x in pts[key]] + [x[3] for x in pts[key]]
        lo, hi = min(allv), max(allv)
        pad = 0.04 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1.0)
        ax.set_aspect("equal")   # same scale and ticks on both axes: perfect prediction is the diagonal
        ax.locator_params(nbins=5)
        ax.set_title(lab)
        ax.grid(linewidth=0.3, alpha=0.6)
    fig.supxlabel("Predicted by the simulator", y=0.248 if on else 0.194, fontsize=PRINT_FONT)
    fig.supylabel("Measured on the Jetson", x=0.005, fontsize=PRINT_FONT)
    dot = {c: Line2D([], [], linestyle="none", marker="o", markersize=7, markerfacecolor=COND_COLOR[c],
                     markeredgecolor="none", label=RUN2_SHORT[c]) for c in COND_ORDER}
    star = Line2D([], [], linestyle="none", marker="*", markersize=12, markerfacecolor="white",
                  markeredgecolor="black", label=POLICY_NAME["F"])
    line = Line2D([], [], color="black", linewidth=1.0, label="Perfect prediction")
    gap = [Line2D([], [], linestyle="none", label=" ") for _ in range(2)]
    # Legend columns (filled column by column): mix A, mix B, the star and the line.
    h = [dot["A_r030"], dot["A_r060"], gap[0], dot["B_r030"], dot["B_r060"], dot["B_r060_b"], star, line, gap[1]]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.081 if on else 0.0),
               handlelength=1.5, handletextpad=0.5, columnspacing=1.0)
    ape = float(np.median([100 * abs(m - p) / m for _c, _p, p, m in pts["e"]]))
    pp = float(np.median([abs(m - p) for _c, _p, p, m in pts["c"]]))
    take = (f"Over {len(pts['e'])} measured policies: median energy error {ape:.1f}%,\n"
            f"median on-time error {pp:.1f} percentage points.")
    return _finish(fig, ctx, "fig_sim_accuracy", out, "Does the simulator predict what the Jetson measured?", take,
                   adjust={"left": 0.1015, "right": 0.99, "top": 0.885 if on else 0.933,
                           "bottom": 0.35 if on else 0.3125, "wspace": 0.144})


# ------------------------------------------------------------------------------------------------ 3D and 2D twins
def _top_label(ax, x, y, z, text, fs=9):
    ax.text(x, y, z, text, ha="center", va="bottom", fontsize=fs, zorder=10)


def fig3d_warmup(ctx, out):
    res, fams = _run1c(ctx)
    have_e = all(res[(f, s)]["energy_j"] is not None for f in fams for s in ("cold", "warm"))
    panels = [("ready_s", "Time from process start to ready (s)", "{:.0f}")]
    if have_e:
        panels.append(("energy_j", "Energy from start to ready (J)", "{:.0f}"))
    fig = plt.figure(figsize=(7.4 * len(panels), 6.0))
    for k, (key, lab, fmt) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), k + 1, projection="3d")
        for i, f in enumerate(fams):
            for j, s in enumerate(("cold", "warm")):
                z = res[(f, s)][key]
                ax.bar3d(i - 0.3, (1 - j) - 0.3, 0, 0.6, 0.6, z, color=FAMILY_COLOR[f], shade=True,
                         edgecolor="black", linewidth=0.3, alpha=1.0 if s == "cold" else 0.55)
                _top_label(ax, i, 1 - j, z * 1.02, fmt.format(z))
        ax.set_xticks(range(len(fams)), [FAMILY_NAME[f] for f in fams])
        ax.set_yticks([0, 1], ["filled\n(warm)", "empty\n(cold)"])
        ax.set_xlabel("Model family", labelpad=8)
        ax.set_ylabel("Compilation caches", labelpad=10)
        ax.set_zlabel(lab, labelpad=6)
        ax.view_init(elev=26, azim=-52)
    fig.subplots_adjust(wspace=0.05)
    return _finish(fig, ctx, "fig3d_warmup", out, "Starting a model: how long does it take, and what does it cost?",
                   _warm_take(res, fams), top=0.95)


def fig2d_warmup(ctx, out):
    res, fams = _run1c(ctx)
    have_e = all(res[(f, s)]["energy_j"] is not None for f in fams for s in ("cold", "warm"))
    panels = [("ready_s", "Time from process start to ready (s)", "{:.0f}")]
    if have_e:
        panels.append(("energy_j", "Energy from start to ready (J)", "{:.0f}"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.8))
    for ax, (key, lab, fmt) in zip(np.atleast_1d(axes), panels):
        _grouped_family_bars(ax, res, fams, key, fmt)
        ax.set_ylabel(lab)
    _state_legend(fig=fig, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.93))
    return _finish(fig, ctx, "fig2d_warmup", out, "Starting a model: how long does it take, and what does it cost?",
                   _warm_take(res, fams), top=0.86)


def _cap_setup(ctx):
    cap = capacity_table(_csv(Path(_need(ctx, "run1_csv")) / "corun.csv"))
    sets = _ordered_sets(cap)
    ctx.setdefault("_notes", {})["fig3d_capacity_sets"] = [f"{i + 1}: {_short(A)}" for i, A in enumerate(sets)]
    return cap, sets


def _cap_take(cap, sets):
    pair = [cap[(A, f)] for A in sets for f in LEVELS if set_group(A) == GROUPS[0] and (A, f) in cap]
    return (f"BERT + YOLO pairs stay between {min(pair):.2f} and {max(pair):.2f}; "
            f"sets with LLaMA reach up to {max(cap.values()):.2f}.")


def fig3d_capacity(ctx, out):
    cap, sets = _cap_setup(ctx)
    fig = plt.figure(figsize=(17, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    floor = 0.8
    for i, A in enumerate(sets):
        for j, f in enumerate(LEVELS):
            if (A, f) in cap:
                z = cap[(A, f)]
                ax.bar3d(i - 0.35, j - 0.3, floor, 0.7, 0.6, z - floor, color=plt.get_cmap("RdBu_r")(0.5 + (z - 1) / 0.8),
                         shade=True, edgecolor="black", linewidth=0.2)
                _top_label(ax, i, j, z + 0.015, f"{z:.2f}", fs=8)
    ax.set_zlim(floor, 1.45)
    spans = _group_spans(sets)
    ax.set_xticks([(a + b) / 2 for _g, a, b in spans],
                  [GROUP_LABEL2[g].replace(chr(10), " ") + f"\n(sets {a + 1}-{b + 1})" for g, a, b in spans])
    ax.set_yticks(range(3), [f"{f} MHz" for f in LEVELS])
    ax.set_xlabel("Requests running together (numbered as in the heatmap)", labelpad=14)
    ax.set_ylabel("GPU frequency", labelpad=8)
    ax.set_zlabel("Co-run capacity (serial execution = 1.0)", labelpad=6)
    ax.view_init(elev=55, azim=-88)
    return _finish(fig, ctx, "fig3d_capacity", out,
                   "Running requests together: is it faster than running them one after another?",
                   _cap_take(cap, sets), top=0.95)


def fig2d_capacity(ctx, out):
    cap, sets = _cap_setup(ctx)
    fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(sets) + 1.8))
    ypos, y = {}, 0.0
    spans = {}
    for g in GROUPS:
        first = y
        for A in [A for A in sets if set_group(A) == g]:
            ypos[A] = y
            y += 1.0
        spans[g] = (first, y - 1.0)
        y += 0.6
    lo = min(cap.values())
    hi = max(cap.values())
    ax.axvspan(0.85 * min(lo, 1), 1.0, color=LIGHT_RED, zorder=0, alpha=0.6)
    ax.axvspan(1.0, hi + 0.12, color=LIGHT_GREEN, zorder=0)
    for k, f in enumerate(LEVELS):
        for A in sets:
            if (A, f) not in cap:
                continue
            v = cap[(A, f)]
            yy = ypos[A] + (k - 1) * 0.27
            ax.barh(yy, v - 1.0, left=1.0, height=0.26, color=LEVEL_COLOR[f], edgecolor="black", linewidth=0.3,
                    zorder=3)
            ax.text(v + (0.004 if v >= 1 else -0.004), yy, f"{v:.2f}", ha="left" if v >= 1 else "right", va="center",
                    fontsize=8)
    ax.axvline(1.0, color="black", linewidth=1.2, zorder=4)
    ax.set_xlim(0.85 * min(lo, 1), hi + 0.12)
    ax.set_yticks([ypos[A] for A in sets], [_short(A) for A in sets])
    ax.invert_yaxis()
    ax.set_xlabel("Co-run capacity (serial execution = 1.0); bars start at 1.0")
    ax.set_ylabel("Requests running together (BERT-5 = BERT exit 5, ...)")
    ax.text(0.99, 1.0, "slower than serial", ha="right", va="top", transform=ax.get_xaxis_transform(), fontsize=10,
            style="italic", color="#8A2B20")
    ax.text(1.01, 1.0, "faster than serial", ha="left", va="top", transform=ax.get_xaxis_transform(), fontsize=10,
            style="italic", color="#1D6B3A")
    ax.legend(handles=[mpatches.Patch(facecolor=LEVEL_COLOR[f], edgecolor="black", label=f"{f} MHz") for f in LEVELS],
              loc="center left", bbox_to_anchor=(1.01, 0.5), title="GPU frequency", title_fontsize=10)
    return _finish(fig, ctx, "fig2d_capacity", out,
                   "Running requests together: is it faster than running them one after another?",
                   _cap_take(cap, sets))


def _energy_data(ctx):
    rows = _csv(Path(_need(ctx, "run2")) / "open_conditions.csv")
    fz = _json(Path(_need(ctx, "frozen")) / "freeze.json")
    spec = {c["condition_id"]: {g["name"]: g for g in c["grid"]} for c in fz["conditions"]}
    en = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["type_id"] == "bert:5":    # one row per run: energy_per_req_j is the same on every type row
            en[_cond_key(r)][r["policy_key"]].append(float(r["energy_per_req_j"]))
    conds = [c for c in ("A_r060", "B_r060") if c in en and c in spec]
    if not conds:
        raise Skip("no A_r060 / B_r060 grid in the Run 2 data")
    lim_pos = {1: 0, 2: 1, None: 2}
    pts = {}
    for c in conds:
        pts[c] = []
        for name, vs in en[c].items():
            g = spec[c].get(name)
            if g is not None and g["order"] == "stf":
                pts[c].append((lim_pos[g["k_short"]], lim_pos[g["k_long"]], g["f_mhz"], float(np.mean(vs))))
    return conds, pts


LIM = ["1", "2", "no limit"]
ENERGY_TITLE = "Run 2 grid: how does the limit on requests in service and the GPU speed change energy per request?"


def fig3d_energy(ctx, out):
    conds, pts = _energy_data(ctx)
    fig = plt.figure(figsize=(8.6 * len(conds), 6.8))
    for k, c in enumerate(conds):
        ax = fig.add_subplot(1, len(conds), k + 1, projection="3d")
        if not pts[c]:
            continue
        floor = 0.8 * min(p[3] for p in pts[c])
        for f, off in ((612, -0.38), (1020, 0.0)):
            for a, b, ff, z in [p for p in pts[c] if p[2] == f]:
                ax.bar3d(a + off - 0.02, b - 0.3, floor, 0.36, 0.6, z - floor, color=LEVEL_COLOR[f], shade=True,
                         edgecolor="black", linewidth=0.2)
                _top_label(ax, a + off + 0.16, b, z + 0.01 * z, f"{z:.2f}", fs=9)
        ax.set_zlim(floor, max(p[3] for p in pts[c]) * 1.08)
        ax.set_xticks(range(3), LIM)
        ax.set_yticks(range(3), LIM)
        ax.set_xlabel("Most short requests at once", labelpad=8)
        ax.set_ylabel("Most long requests at once", labelpad=8)
        ax.set_zlabel("Energy per request (J)", labelpad=6)
        ax.set_title(COND_LABEL[c], fontsize=10, fontweight="bold")
        ax.view_init(elev=34, azim=-48)
        ax.legend(handles=[mpatches.Patch(color=LEVEL_COLOR[f], label=f"{f} MHz") for f in (612, 1020)],
                  loc="upper left")
    return _finish(fig, ctx, "fig3d_energy", out, ENERGY_TITLE, "Shortest-job-first ordering. Bars start above zero.",
                   top=0.94)


def fig2d_energy(ctx, out):
    conds, pts = _energy_data(ctx)
    fig, axes = plt.subplots(len(conds), 1, figsize=(11, 4.4 * len(conds)), squeeze=False)
    labs = [f"{a} / {b}" for a in LIM for b in LIM]
    for ax, c in zip(axes[:, 0], conds):
        for k, f in enumerate((612, 1020)):
            for a, b, ff, z in [p for p in pts[c] if p[2] == f]:
                x = a * 3 + b + (k - 0.5) * 0.38
                ax.bar(x, z, 0.38, color=LEVEL_COLOR[f], edgecolor="black", linewidth=0.4)
                ax.text(x, z, f"{z:.2f}", ha="center", va="bottom", fontsize=9, rotation=90)
        ax.set_xticks(range(9), labs)
        ax.set_ylabel("Energy per request (J)")
        ax.set_title(COND_LABEL[c], fontsize=10, fontweight="bold")
        ax.margins(y=0.2)
    axes[-1, 0].set_xlabel("Most short requests / most long requests at once")
    axes[0, 0].legend(handles=[mpatches.Patch(color=LEVEL_COLOR[f], label=f"{f} MHz") for f in (612, 1020)],
                      loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _finish(fig, ctx, "fig2d_energy", out, ENERGY_TITLE, "Shortest-job-first ordering.", top=0.95)


FIGS = collections.OrderedDict((f.__name__, f) for f in (
    fig_capacity, fig_gamma, fig_solo_time, fig_power, fig_slowdown_fit, fig_warmup, fig_admit, fig_run2_energy_attain,
    fig_sim_accuracy, fig3d_warmup, fig3d_capacity, fig3d_energy, fig2d_warmup, fig2d_capacity, fig2d_energy))


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
    ap.add_argument("--no-titles", action="store_true", help="paper version: no title, no takeaway line")
    a = ap.parse_args(argv)
    ctx = {"run1_csv": a.run1_csv, "frozen": a.frozen, "run1c": a.run1c, "run2": a.run2, "titles": not a.no_titles}
    made, skipped = run(ctx, a.out, a.only)
    for k, v in sorted(ctx.get("_notes", {}).items()):
        print(f"[note] {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
