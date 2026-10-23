"""Run 3 figures of the CoExit paper from the board analysis folder (analysis_run3_all).

numpy, matplotlib and the standard library only. No torch, no model run. One function per figure.
CLI: python -m exp2.analyze.fig_run3 --analysis DIR --out DIR [--no-titles]
  --analysis  analysis_run3_all folder: summary.csv (contracts.COEXIT_SUMMARY_COLS), compare.csv (RUN3_COMPARE_COLS)
  --no-titles paper version: no title and no takeaway line (captions are in LaTeX). Default: titles on.
Rows: family "all" (every DNN) at deadline scale 1.0 (compare.csv has no deadline scale column: Run 3 is nominal only).
A figure whose input is missing is skipped with a printed reason. Output: NAME.pdf and NAME.png (200 dpi).
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


# ------------------------------------------------------------------------------------------------ figures
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


FIGS = collections.OrderedDict((f.__name__, f) for f in (fig_run3_viol, fig_run3_depth, fig_run3_simboard))


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
    ap.add_argument("--no-titles", action="store_true", help="paper version: no title, no takeaway line")
    a = ap.parse_args(argv)
    run({"analysis": a.analysis, "titles": not a.no_titles}, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
