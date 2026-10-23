"""Run 3 figures of the CoExit paper from the board analysis folder (analysis_run3_all).

numpy, matplotlib and the standard library only. No torch, no model run. One function per figure.
CLI: python -m exp2.analyze.fig_run3 --analysis DIR --out DIR [--pilot-m1 DIR] [--pilot-oracle DIR] [--no-titles]
  --analysis     analysis_run3_all folder: summary.csv (contracts.COEXIT_SUMMARY_COLS), compare.csv (RUN3_COMPARE_COLS)
  --pilot-m1     simulator pilot folder (summary.csv, same columns) with the M1 slowdown prediction (CoExit (sim.))
  --pilot-oracle simulator pilot folder whose CoExit predicts with the measured slowdown (CoExit-Oracle (sim.))
  --no-titles    paper version: no title and no takeaway line (captions are in LaTeX). Default: titles on.
Rows: family "all" (every DNN) at deadline scale 1.0 (compare.csv has no deadline scale column: Run 3 is nominal only).
Exceptions: fig_run3_family plots the families bert, yolo and llama; fig_sim_deadline plots the deadline scales.
A figure whose input is missing is skipped with a printed reason, a missing series or cell too.
Output: NAME.pdf and NAME.png (200 dpi).
"""
import argparse
import collections
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from exp2.analyze.figures import GRID_GREY, PRINT_RC, Skip, _csv, _finish, _need, _style  # noqa: E402

# Paper names: PAPER_CONTRACT_COEXIT.md sec. 6. Colours: Okabe-Ito. CoExit-NoGate is the ablation: dashed, hollow.
METHODS = ("all_final", "all_early", "edgeserving", "greedy", "coexit", "coexit_nogate")
METHOD_NAME = {"all_final": "All-Final", "all_early": "All-Early", "edgeserving": "EdgeServing",
               "greedy": "Deadline-Greedy", "coexit": "CoExit", "coexit_nogate": "CoExit-NoGate"}
# (colour, marker, line style, marker face colour; None = filled with the line colour)
METHOD_STYLE = {"all_final": ("#E69F00", "s", "-", None), "all_early": ("#56B4E9", "o", "-", None),
                "edgeserving": ("#D55E00", "D", "-", None), "greedy": ("#009E73", "^", "-", None),
                "coexit": ("#000000", "*", "-", None), "coexit_nogate": ("#666666", "*", "--", "white")}
MIX_TITLE = {"A": "Mix A (BERT + YOLO)", "B": "Mix B (BERT + YOLO + LLaMA)"}
SIMBOARD_METHODS = ("greedy", "coexit")
CELL_MARKER = {("A", "poisson"): ("o", "Mix A, Poisson"), ("B", "poisson"): ("s", "Mix B, Poisson"),
               ("B", "gamma"): ("D", "Mix B, Gamma")}
STEP = 0.3    # x distance of a cell with another arrival process, to the right of the Poisson loads
DODGE = 0.022  # horizontal offset between methods so that the error bars do not overlap
SUMMARY_NUM = ("viol_mean", "viol_ci_lo", "viol_ci_hi", "depth_mean", "depth_ci_lo", "depth_ci_hi")
SUMMARY_KEY = ("method", "family", "mix", "arrival", "rho", "deadline_scale")
# Simulator series of fig_run3_ablation: thin dotted lines, smaller markers (PAPER_CONTRACT_COEXIT.md sec. 6)
SIM_STYLE = {"coexit_sim": ("#000000", "x", ":", None), "oracle_sim": ("#CC79A7", "P", ":", None)}
SIM_NAME = {"coexit_sim": "CoExit (sim.)", "oracle_sim": "CoExit-Oracle (sim.)"}
STYLE = {**METHOD_STYLE, **SIM_STYLE}
NAME = {**METHOD_NAME, **SIM_NAME}
FAMILY_NAME = {"bert": "BERT", "yolo": "YOLO", "llama": "LLaMA"}
# fig_run3_family: (mix, family) per panel, row 0 = mix A, row 1 = mix B. None = the legend slot.
FAMILY_PANELS = ((("A", "bert"), ("A", "yolo"), None), (("B", "bert"), ("B", "yolo"), ("B", "llama")))
ABLATION_KEYS = ("edgeserving", "greedy", "coexit_nogate", "coexit", "coexit_sim", "oracle_sim")
SIM_METHODS = ("all_final", "all_early", "edgeserving", "greedy", "coexit")   # the pilot has no CoExit-NoGate
SCALES = (0.75, 1.0, 1.5)
SIM_RHO = 0.9


# ------------------------------------------------------------------------------------------------ loaders
def summary_table(analysis):
    """{(mix, arrival, rho): {method: {column: float}}} of summary.csv, family "all", deadline scale 1.0."""
    tab = collections.defaultdict(dict)
    for r in _csv(Path(analysis) / "summary.csv"):
        if r["family"] == "all" and float(r["deadline_scale"]) == 1.0:
            tab[(r["mix"], r["arrival"], float(r["rho"]))][r["method"]] = {k: float(r[k]) for k in SUMMARY_NUM}
    return dict(tab)


def simboard_points(analysis):
    """{method: [point]} of compare.csv, family "all", Deadline-Greedy and CoExit, rows with a board and a simulator
    value. point = {"cell": (mix, arrival, rho), "board", "board_lo", "board_hi", "sim", "sim_lo", "sim_hi"}: the
    difference of the violation ratio to EdgeServing and its 95 % confidence interval."""
    pts = collections.defaultdict(list)
    for r in _csv(Path(analysis) / "compare.csv"):
        if r["family"] != "all" or r["method"] not in SIMBOARD_METHODS:
            continue
        if not r["board_d_viol_vs_es_mean"] or not r["sim_d_viol_vs_es_mean"]:
            continue
        p = {"cell": (r["mix"], r["arrival"], float(r["rho"]))}
        for side in ("board", "sim"):
            p[side] = float(r[f"{side}_d_viol_vs_es_mean"])
            p[f"{side}_lo"] = float(r[f"{side}_d_viol_vs_es_lo"])
            p[f"{side}_hi"] = float(r[f"{side}_d_viol_vs_es_hi"])
        pts[r["method"]].append(p)
    return dict(pts)


def summary_cells(folder):
    """{(family, mix, arrival, rho, deadline_scale): {method: {column: float}}} of FOLDER/summary.csv. The board summary
    and the simulator pilot summary have the same columns (contracts.COEXIT_SUMMARY_COLS)."""
    path = Path(folder) / "summary.csv"
    rows = _csv(path)
    lack = [c for c in SUMMARY_KEY + SUMMARY_NUM if rows and c not in rows[0]]
    if lack:
        raise Skip(f"{path} has no column {', '.join(lack)}")
    tab = collections.defaultdict(dict)
    for r in rows:
        key = (r["family"], r["mix"], r["arrival"], float(r["rho"]), float(r["deadline_scale"]))
        tab[key][r["method"]] = {k: float(r[k]) for k in SUMMARY_NUM}
    return dict(tab)


# ------------------------------------------------------------------------------------------------ figures
def _say(name, text):
    print(f"[fig_run3] {name}: {text}")


def _series(ax, pos, vals, metric, key, dx, ms=8, lw=1.4):
    """One series (STYLE key) on one panel: mean of `metric` with its 95 % CI error bars, one line over the Poisson
    cells and single markers for the other cells. vals = {(arrival, x): {column: float}}, pos = {(arrival, x): (x, _)}.
    ms, lw: marker size and line width (smaller for the simulator series and the small panels)."""
    col, mk, ls, mfc = STYLE[key]
    have = sorted(vals, key=lambda c: pos[c][0])
    x = np.array([pos[c][0] + dx for c in have])
    y = np.array([vals[c][f"{metric}_mean"] for c in have])
    err = np.clip([y - [vals[c][f"{metric}_ci_lo"] for c in have], [vals[c][f"{metric}_ci_hi"] for c in have] - y],
                  0, None)
    ax.errorbar(x, y, yerr=err, fmt="none", ecolor=col, elinewidth=1.0 if lw >= 1.4 else 0.8, capsize=2, zorder=2)
    pois = np.array([c[0] == "poisson" for c in have])
    kw = {"color": col, "marker": mk, "markerfacecolor": mfc or col, "markersize": ms, "zorder": 3}
    ax.plot(x[pois], y[pois], linestyle=ls, linewidth=lw, label=NAME[key], **kw)
    ax.plot(x[~pois], y[~pois], linestyle="none", **kw)


def _handles(keys):
    return [Line2D([], [], color=STYLE[k][0], marker=STYLE[k][1], linestyle=STYLE[k][2],
                   markerfacecolor=STYLE[k][3] or STYLE[k][0], markersize=6 if k in SIM_STYLE else 8,
                   linewidth=1.1 if k in SIM_STYLE else 1.4, label=NAME[k]) for k in keys]


def _load_axis(ax, pos, sep):
    """x axis of the load: one tick per cell, the dotted separator before the cells of another arrival process."""
    if sep is not None:
        ax.axvline(sep, color=GRID_GREY, linestyle=":", linewidth=1.2, zorder=1)
    xs, labels = zip(*sorted(pos.values()))
    ax.set_xticks(xs, labels)
    ax.set_xlim(xs[0] - STEP / 2, xs[-1] + STEP / 2)
    ax.set_xlabel(r"Load $\rho$")
    ax.grid(axis="y", color=GRID_GREY, linewidth=0.5, alpha=0.6)


def _dodge(k, n):
    return (k - (n - 1) / 2) * DODGE
def _positions(cells):
    """{(arrival, rho): (x, tick label)} of one mix and the x of the dotted separator (None without one). Poisson
    cells sit at their load. Cells with another arrival process sit to the right of the separator."""
    pois = sorted(rho for a, rho in cells if a == "poisson")
    other = sorted((a, rho) for a, rho in cells if a != "poisson")
    right = pois[-1] if pois else 0.0
    pos = {("poisson", rho): (rho, f"{rho:g}") for rho in pois}
    for k, (a, rho) in enumerate(other):
        pos[(a, rho)] = (right + STEP * (k + 1), f"{rho:g}\n{a.capitalize()}")
    return pos, (right + STEP / 2 if other and pois else None)


def _method_handles(methods):
    return [Line2D([], [], color=METHOD_STYLE[m][0], marker=METHOD_STYLE[m][1], linestyle=METHOD_STYLE[m][2],
                   markerfacecolor=METHOD_STYLE[m][3] or METHOD_STYLE[m][0], markersize=8, label=METHOD_NAME[m])
            for m in methods]


def _metric_fig(ctx, out, name, metric, ylabel, title, take):
    """Mean of `metric` (viol or depth) with its 95 % confidence interval versus the load, one panel per mix."""
    tab = summary_table(_need(ctx, "analysis"))
    if not tab:
        raise Skip("no family 'all' row at deadline scale 1.0 in summary.csv")
    on = ctx.get("titles", True)
    mixes = sorted({k[0] for k in tab})
    cells = {mix: [(a, rho) for (m, a, rho) in tab if m == mix] for mix in mixes}
    fig, axes = plt.subplots(1, len(mixes), figsize=(6.9, 3.9 if on else 3.6), sharey=True, squeeze=False,
                             gridspec_kw={"width_ratios": [len(cells[mix]) for mix in mixes]})
    drawn = []
    for ax, mix in zip(axes[0], mixes):
        pos, sep = _positions(cells[mix])
        order = sorted(pos, key=lambda c: pos[c][0])
        for k, meth in enumerate(METHODS):
            col, mk, ls, mfc = METHOD_STYLE[meth]
            have = [c for c in order if meth in tab[(mix,) + c]]
            if not have:
                continue
            drawn.append(meth)
            dx = (k - (len(METHODS) - 1) / 2) * DODGE
            x = np.array([pos[c][0] + dx for c in have])
            v = [tab[(mix,) + c][meth] for c in have]
            y = np.array([r[f"{metric}_mean"] for r in v])
            err = np.clip([y - [r[f"{metric}_ci_lo"] for r in v], [r[f"{metric}_ci_hi"] for r in v] - y], 0, None)
            ax.errorbar(x, y, yerr=err, fmt="none", ecolor=col, elinewidth=1.0, capsize=2, zorder=2)
            pois = np.array([c[0] == "poisson" for c in have])
            kw = {"color": col, "marker": mk, "markerfacecolor": mfc or col, "markersize": 8, "zorder": 3}
            ax.plot(x[pois], y[pois], linestyle=ls, linewidth=1.4, label=METHOD_NAME[meth], **kw)
            ax.plot(x[~pois], y[~pois], linestyle="none", **kw)
        if sep is not None:
            ax.axvline(sep, color=GRID_GREY, linestyle=":", linewidth=1.2, zorder=1)
        xs = [pos[c][0] for c in order]
        ax.set_xticks(xs, [pos[c][1] for c in order])
        ax.set_xlim(xs[0] - STEP / 2, xs[-1] + STEP / 2)
        ax.set_xlabel(r"Load $\rho$")
        ax.set_title(MIX_TITLE.get(mix, f"Mix {mix}"), fontweight="bold")
        ax.grid(axis="y", color=GRID_GREY, linewidth=0.5, alpha=0.6)
    axes[0, 0].set_ylabel(ylabel)
    fig.align_xlabels(axes[0])   # the two-line tick of a Gamma cell would push one x label lower
    meths = [m for m in METHODS if m in drawn]
    fig.legend(handles=_method_handles(meths), loc="upper center", bbox_to_anchor=(0.5, 0.92 if on else 1.0),
               ncol=3, handlelength=2.6, columnspacing=1.2)
    adjust = {"left": 0.1, "right": 0.99, "bottom": 0.25 if on else 0.19, "top": 0.63 if on else 0.72, "wspace": 0.08}
    return _finish(fig, ctx, name, out, title, take, adjust=adjust)


@plt.rc_context(PRINT_RC)
def fig_run3_viol(ctx, out):
    return _metric_fig(ctx, out, "fig_run3_viol", "viol", "Violation ratio",
                       "Run 3: violation ratio of every method versus the load (all DNNs, nominal deadlines)",
                       "Error bars: 95 % confidence interval over the repeats. Dashed: CoExit without the capacity gate.")


@plt.rc_context(PRINT_RC)
def fig_run3_depth(ctx, out):
    return _metric_fig(ctx, out, "fig_run3_depth", "depth", "Normalized depth",
                       "Run 3: normalized depth of every method versus the load (all DNNs, nominal deadlines)",
                       "Error bars: 95 % confidence interval over the repeats. Dashed: CoExit without the capacity gate.")


@plt.rc_context(PRINT_RC)
def fig_run3_simboard(ctx, out):
    pts = simboard_points(_need(ctx, "analysis"))
    if not pts:
        raise Skip("no Deadline-Greedy or CoExit row of family 'all' with a board and a simulator value in compare.csv")
    on = ctx.get("titles", True)
    fig, ax = plt.subplots(figsize=(5.4, 5.6 if on else 5.2))
    every = [v for ps in pts.values() for p in ps for v in (p["board_lo"], p["board_hi"], p["sim_lo"], p["sim_hi"])]
    pad = 0.05 * (max(every) - min(every) or 1.0)
    lim = (min(every) - pad, max(every) + pad)
    ax.plot(lim, lim, color="black", linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(0, color=GRID_GREY, linewidth=0.8, zorder=0)
    ax.axvline(0, color=GRID_GREY, linewidth=0.8, zorder=0)
    groups = []
    for meth in SIMBOARD_METHODS:
        col = METHOD_STYLE[meth][0]
        for p in pts.get(meth, []):
            mk = CELL_MARKER.get(p["cell"][:2], ("^", ""))[0]
            if p["cell"][:2] not in groups:
                groups.append(p["cell"][:2])
            ax.errorbar(p["sim"], p["board"], xerr=[[max(p["sim"] - p["sim_lo"], 0)], [max(p["sim_hi"] - p["sim"], 0)]],
                        yerr=[[max(p["board"] - p["board_lo"], 0)], [max(p["board_hi"] - p["board"], 0)]], fmt="none",
                        ecolor=col, elinewidth=0.9, capsize=2, alpha=0.7, zorder=2)
            ax.plot(p["sim"], p["board"], linestyle="none", marker=mk, color=col, markersize=8, zorder=3,
                    label=METHOD_NAME[meth])
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Violation ratio minus EdgeServing, simulator")
    ax.set_ylabel("Violation ratio minus EdgeServing, board")
    h = [mpatches.Patch(color=METHOD_STYLE[m][0], label=METHOD_NAME[m]) for m in SIMBOARD_METHODS if m in pts]
    groups = [g for g in CELL_MARKER if g in groups] + [g for g in groups if g not in CELL_MARKER]
    h += [Line2D([], [], color="#7F7F7F", marker=CELL_MARKER.get(g, ("^", ""))[0], linestyle="none", markersize=8,
                 label=CELL_MARKER.get(g, ("", f"Mix {g[0]}, {g[1].capitalize()}"))[1]) for g in groups]
    h.append(Line2D([], [], color="black", linestyle="--", linewidth=1.0, label="y = x"))
    ax.legend(handles=h, loc="upper left")
    return _finish(fig, ctx, "fig_run3_simboard", out,
                   "Run 3: does the simulator reproduce the board difference to EdgeServing?",
                   "One point per cell (all DNNs). Error bars: 95 % confidence interval.", top=0.93, bottom=0.05)


@plt.rc_context(PRINT_RC)
def fig_run3_family(ctx, out):
    """Violation ratio of every DNN versus the load: mix A (BERT, YOLO) on top, mix B (BERT, YOLO, LLaMA) below."""
    name = "fig_run3_family"
    tab = {}   # (mix, family) -> {(arrival, rho): {method: row}}
    for (fam, mix, a, rho, s), v in summary_cells(_need(ctx, "analysis")).items():
        if fam in FAMILY_NAME and s == 1.0:
            tab.setdefault((mix, fam), {})[(a, rho)] = v
    if not any(p in tab for row in FAMILY_PANELS for p in row if p):
        raise Skip("no BERT, YOLO or LLaMA row at deadline scale 1.0 in summary.csv")
    on = ctx.get("titles", True)
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 6.3 if on else 5.9), sharey=True, squeeze=False)
    drawn = set()
    for i, row in enumerate(FAMILY_PANELS):
        for j, panel in enumerate(row):
            ax = axes[i, j]
            if panel not in tab:
                if panel:
                    _say(name, f"panel mix {panel[0]} {FAMILY_NAME[panel[1]]} skipped (no row in summary.csv)")
                ax.axis("off")
                continue
            data = tab[panel]
            pos, sep = _positions(list(data))
            for k, meth in enumerate(METHODS):
                vals = {c: data[c][meth] for c in data if meth in data[c]}
                if vals:
                    drawn.add(meth)
                    _series(ax, pos, vals, "viol", meth, 1.5 * _dodge(k, len(METHODS)), ms=6)
            _load_axis(ax, pos, sep)
            ax.set_title(f"Mix {panel[0]}, {FAMILY_NAME[panel[1]]}", fontweight="bold")
        axes[i, 0].set_ylabel("Violation ratio")
    fig.align_xlabels(axes[1])
    axes[0, 2].legend(handles=_handles([m for m in METHODS if m in drawn]), loc="center", handlelength=2.6)
    adjust = {"left": 0.1, "right": 0.99, "bottom": 0.12 if on else 0.09, "top": 0.88 if on else 0.95,
              "wspace": 0.1, "hspace": 0.55}
    return _finish(fig, ctx, name, out, "Run 3: violation ratio of every DNN versus the load (nominal deadlines)",
                   "Error bars: 95 % confidence interval over the repeats. Dashed: CoExit without the capacity gate.",
                   adjust=adjust)


@plt.rc_context(PRINT_RC)
def fig_run3_ablation(ctx, out):
    """Normalized depth versus the load: CoExit, its ablation and two baselines on the board, plus CoExit with the M1
    slowdown and with the measured slowdown (oracle) in the simulator on the same cells."""
    name = "fig_run3_ablation"
    board = summary_cells(_need(ctx, "analysis"))
    sims = {"coexit_sim": summary_cells(_need(ctx, "pilot_m1")),
            "oracle_sim": summary_cells(_need(ctx, "pilot_oracle"))}
    tab = {}   # mix -> {(arrival, rho): {key: row}}
    for (fam, mix, a, rho, s), v in board.items():
        if fam == "all" and s == 1.0:
            tab.setdefault(mix, {})[(a, rho)] = {m: v[m] for m in ABLATION_KEYS if m in v}
    if not tab:
        raise Skip("no family 'all' row at deadline scale 1.0 in summary.csv")
    for key, cells in sims.items():
        miss = []
        for mix in sorted(tab):
            for (a, rho), v in sorted(tab[mix].items()):
                row = cells.get(("all", mix, a, rho, 1.0), {}).get("coexit")
                if row is None:
                    miss.append(f"{mix} {a} {rho:g}")
                else:
                    v[key] = row
        if miss:
            _say(name, f"{NAME[key]} skipped at {', '.join(miss)} (no coexit row, family all, deadline scale 1.0)")
    on = ctx.get("titles", True)
    mixes = sorted(tab)
    fig, axes = plt.subplots(1, len(mixes), figsize=(6.9, 4.1 if on else 3.8), sharey=True, squeeze=False,
                             gridspec_kw={"width_ratios": [len(tab[mix]) for mix in mixes]})
    drawn = set()
    for ax, mix in zip(axes[0], mixes):
        data = tab[mix]
        pos, sep = _positions(list(data))
        for k, key in enumerate(ABLATION_KEYS):
            vals = {c: data[c][key] for c in data if key in data[c]}
            if vals:
                drawn.add(key)
                sim = key in SIM_STYLE
                _series(ax, pos, vals, "depth", key, _dodge(k, len(ABLATION_KEYS)), ms=6 if sim else 8,
                        lw=1.1 if sim else 1.4)
        _load_axis(ax, pos, sep)
        ax.set_title(MIX_TITLE.get(mix, f"Mix {mix}"), fontweight="bold")
    axes[0, 0].set_ylabel("Normalized depth")
    fig.align_xlabels(axes[0])
    fig.legend(handles=_handles([k for k in ABLATION_KEYS if k in drawn]), loc="upper center",
               bbox_to_anchor=(0.5, 0.92 if on else 1.0), ncol=3, handlelength=2.6, columnspacing=1.2)
    adjust = {"left": 0.1, "right": 0.99, "bottom": 0.22 if on else 0.17, "top": 0.65 if on else 0.75, "wspace": 0.08}
    return _finish(fig, ctx, name, out,
                   "Run 3 ablation: normalized depth of CoExit without the gate and with the measured slowdown",
                   "Solid and dashed: board. Dotted: simulator on the same cells. Error bars: 95 % CI.",
                   adjust=adjust)


def _scale_label(s):
    return f"{s:g}" if s % 1 else f"{s:.1f}"


@plt.rc_context(PRINT_RC)
def fig_sim_deadline(ctx, out):
    """Simulator: normalized depth (top) and violation ratio (bottom) versus the deadline scale at load 0.9, Poisson
    arrivals, mix A left and mix B right."""
    name = "fig_sim_deadline"
    tab = {}   # mix -> {("poisson", scale): {method: row}}
    for (fam, mix, a, rho, s), v in summary_cells(_need(ctx, "pilot_m1")).items():
        if fam == "all" and a == "poisson" and rho == SIM_RHO and s in SCALES:
            tab.setdefault(mix, {})[("poisson", s)] = v
    mixes = [m for m in ("A", "B") if m in tab]
    if not mixes:
        raise Skip(f"no family 'all' Poisson row at rho {SIM_RHO:g} and deadline scale {SCALES} in the pilot summary")
    on = ctx.get("titles", True)
    fig, axes = plt.subplots(2, len(mixes), figsize=(6.7, 5.6 if on else 5.2), sharex=True, sharey="row",
                             squeeze=False)
    drawn = set()
    for j, mix in enumerate(mixes):
        data = tab[mix]
        pos = {c: (c[1], _scale_label(c[1])) for c in data}
        miss = [_scale_label(s) for s in SCALES if ("poisson", s) not in data]
        if miss:
            _say(name, f"mix {mix} has no deadline scale {', '.join(miss)} in the pilot summary")
        for i, (metric, ylabel) in enumerate((("depth", "Normalized depth"), ("viol", "Violation ratio"))):
            ax = axes[i, j]
            for k, meth in enumerate(SIM_METHODS):
                vals = {c: data[c][meth] for c in data if meth in data[c]}
                if not vals:
                    if i == 0:
                        _say(name, f"{METHOD_NAME[meth]} skipped in mix {mix} (no row in the pilot summary)")
                    continue
                drawn.add(meth)
                _series(ax, pos, vals, metric, meth, 0.4 * _dodge(k, len(SIM_METHODS)))   # CIs are narrow here
            ax.grid(axis="y", color=GRID_GREY, linewidth=0.5, alpha=0.6)
            if j == 0:
                ax.set_ylabel(ylabel)
        axes[0, j].set_title(f"{MIX_TITLE.get(mix, f'Mix {mix}')}", fontweight="bold")
        axes[1, j].set_xticks(SCALES, [_scale_label(s) for s in SCALES])
        axes[1, j].set_xlim(SCALES[0] - 0.12, SCALES[-1] + 0.12)
        axes[1, j].set_xlabel("Deadline scale")
    fig.legend(handles=_handles([m for m in SIM_METHODS if m in drawn]), loc="upper center",
               bbox_to_anchor=(0.5, 0.94 if on else 1.0), ncol=3, handlelength=2.6, columnspacing=1.2)
    adjust = {"left": 0.12, "right": 0.99, "bottom": 0.14 if on else 0.1, "top": 0.78 if on else 0.84,
              "wspace": 0.08, "hspace": 0.12}
    return _finish(fig, ctx, name, out,
                   f"Simulator: depth and violation ratio versus the deadline scale (load {SIM_RHO:g}, Poisson)",
                   "Error bars: 95 % confidence interval over the repeats.", adjust=adjust)


FIGS = collections.OrderedDict((f.__name__, f) for f in (fig_run3_viol, fig_run3_depth, fig_run3_simboard,
                                                         fig_run3_family, fig_run3_ablation, fig_sim_deadline))


def run(ctx, out):
    """Make the figures; -> ({name: [files]}, {name: skip reason})."""
    _style()
    made, skipped = {}, {}
    for name, fn in FIGS.items():
        try:
            made[name] = fn(ctx, out)
            print(f"[fig_run3] {name}: wrote {', '.join(made[name])}")
        except Skip as e:
            skipped[name] = str(e)
            print(f"[fig_run3] {name}: SKIPPED ({e})")
    return made, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exp2.analyze.fig_run3")
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot-m1", help="simulator pilot folder, M1 slowdown (fig_run3_ablation, fig_sim_deadline)")
    ap.add_argument("--pilot-oracle", help="simulator pilot folder, measured slowdown (fig_run3_ablation)")
    ap.add_argument("--no-titles", action="store_true", help="paper version: no title, no takeaway line")
    a = ap.parse_args(argv)
    run({"analysis": a.analysis, "pilot_m1": a.pilot_m1, "pilot_oracle": a.pilot_oracle, "titles": not a.no_titles},
        a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
