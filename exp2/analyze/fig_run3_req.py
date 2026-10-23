"""Run 3 figures of the CoExit paper from the per-request rows (requests.jsonl) of the board sessions.

numpy, matplotlib and the standard library only. No torch, no model run. One function per figure.
CLI: python -m exp2.analyze.fig_run3_req --sessions DIR [DIR ...] --out DIR [--no-titles]
  --sessions  Run 3 session folders (session_run3, session_run3x, session_run3_nogate): every subdir with manifest.json.
              Runs: open, valid, first condition_id kept (as exp2.analyze.run3). Rows: contracts.read_requests.
  --no-titles paper version: no title and no takeaway line (captions are in LaTeX). Default: titles on.
Rules (contracts "Pilot metrics"): measured = is_warmup false. success = outcome complete and t_done <= t_deadline.
R = t_done - t_arrival (expired_in_queue: t_done = t_deadline, so R = D); D = t_deadline - t_arrival.
fig_run3_cdf       empirical CDF of R / D over the measured rows of every DNN, reps 0-2 pooled, mix A and mix B
                   (Poisson, rho 0.9), one line per method, linear x axis, dotted line at R / D = 1.
fig_run3_timeline  service intervals t_start..t_end per DNN in one WINDOW_S window, EdgeServing and CoExit, mix B
                   Poisson rho 0.9 rep 0. Time 0 = the first arrival of the run (both runs share the arrival
                   schedule). Window: starts LEAD_S before the first measured LLaMA request that starts within
                   ON_ARRIVAL_S of its arrival in both runs. Floor exit = the run type differs from the arrival type
                   (req_type_id). Miss marker at t_done of every failed row (expired: at its deadline).
A figure whose input is missing is skipped with a printed reason. Output: NAME.pdf and NAME.png (200 dpi).
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from exp2 import contracts as C  # noqa: E402
from exp2.analyze.fig_run3 import METHOD_NAME, METHOD_STYLE, METHODS, MIX_TITLE  # noqa: E402
from exp2.analyze.figures import FAMILY_COLOR, FAMILY_NAME, GRID_GREY, PRINT_RC, Skip, _finish, _style  # noqa: E402

CDF_CELLS = (("A", "poisson", 0.9), ("B", "poisson", 0.9))
REPS = (0, 1, 2)
TIMELINE_CELL, TIMELINE_REP, TIMELINE_METHODS = ("B", "poisson", 0.9), 0, ("edgeserving", "coexit")
FAMILIES = ("bert", "yolo", "llama")     # timeline rows, top to bottom
WINDOW_S, LEAD_S, ON_ARRIVAL_S = 3.0, 0.25, 0.05
BAR_H = 0.5
DEADLINE_GREY = "#7F7F7F"
_NS = 1e9


# ------------------------------------------------------------------------------------------------ loaders
def load_runs(session_dirs, cells=None):
    """[{"method", "cell": (mix, arrival, rho), "rep", "rows"}] of the open, valid runs (first condition_id kept).
    cells: keep only these cells (None: all)."""
    runs, seen = [], set()
    for sdir in map(Path, session_dirs):
        for cdir in sorted(p for p in sdir.iterdir() if (p / "manifest.json").is_file()):
            with open(cdir / "manifest.json", encoding="utf-8") as fh:
                man = C.validate_manifest(json.load(fh))
            wl = man["workload"]
            if man["mode"] != "open" or not man["valid"] or man["condition_id"] in seen:
                continue
            cell = (wl["name"].removeprefix("mix"), wl["arrival"], float(wl["rho"]))
            if cells is not None and cell not in cells:
                continue
            seen.add(man["condition_id"])
            runs.append({"method": man["policy_key"], "cell": cell, "rep": man["repeat"],
                         "rows": C.read_requests(cdir / "requests.jsonl")})
    return runs


def success(row):
    return row["outcome"] == "complete" and row["t_done_ns"] is not None and row["t_done_ns"] <= row["t_deadline_ns"]


def r_over_d(row):
    """R / D of one row; a row without t_done (error) -> inf."""
    if row["t_done_ns"] is None:
        return float("inf")
    return (row["t_done_ns"] - row["t_arrival_ns"]) / (row["t_deadline_ns"] - row["t_arrival_ns"])


def cdf_data(runs, cell, reps=REPS):
    """{method: sorted R / D of the measured rows of `cell`, reps pooled}."""
    out = collections.defaultdict(list)
    for run in runs:
        if run["cell"] == cell and run["rep"] in reps:
            out[run["method"]] += [r_over_d(r) for r in run["rows"] if not r["is_warmup"]]
    return {m: np.sort(np.array(v, dtype=float)) for m, v in out.items()}


def ecdf(values):
    """(x sorted, y = i / n) of the empirical CDF."""
    x = np.sort(np.asarray(values, dtype=float))
    return x, np.arange(1, len(x) + 1) / len(x)


def _origin(run):
    return min(r["t_arrival_ns"] for r in run["rows"])


def pick_window(run_a, run_b):
    """Window start (s since the first arrival of run_a): LEAD_S before the first measured LLaMA request (arrival
    order) that starts within ON_ARRIVAL_S of its arrival in both runs."""
    rows_b = {r["req_id"]: r for r in run_b["rows"]}
    o = _origin(run_a)
    for r in sorted(run_a["rows"], key=lambda r: r["t_arrival_ns"]):
        q = rows_b.get(r["req_id"])
        if r["family"] != "llama" or r["is_warmup"] or q is None:
            continue
        if all(x["t_start_ns"] is not None and x["t_start_ns"] - x["t_arrival_ns"] <= ON_ARRIVAL_S * _NS
               for x in (r, q)):
            return (r["t_arrival_ns"] - o) / _NS - LEAD_S
    raise Skip("no measured LLaMA request starts on arrival in both runs")


def timeline(run, t0, t1):
    """{family: {"bars": [(start s, duration s, floor)], "miss": [t_done s]}} inside [t0, t1], s since the first
    arrival. Every row counts (warm-up rows also occupy the GPU)."""
    o = _origin(run)
    out = {f: {"bars": [], "miss": []} for f in FAMILIES}
    for r in sorted(run["rows"], key=lambda r: r["t_arrival_ns"]):
        fam = out.setdefault(r["family"], {"bars": [], "miss": []})
        if r["t_start_ns"] is not None and r["t_end_ns"] is not None:
            s, e = (r["t_start_ns"] - o) / _NS, (r["t_end_ns"] - o) / _NS
            if s < t1 and e > t0:
                fam["bars"].append((s, e - s, r["type_id"] != r.get("req_type_id", r["type_id"])))
        if not success(r) and r["t_done_ns"] is not None:
            t = (r["t_done_ns"] - o) / _NS
            if t0 <= t <= t1:
                fam["miss"].append(t)
    return out


# ------------------------------------------------------------------------------------------------ figures
def _light(colour, k=0.55):
    return tuple(c + (1 - c) * k for c in mcolors.to_rgb(colour))


def _need_runs(ctx):
    if not ctx.get("runs"):
        raise Skip("no open, valid run of mix A or B, Poisson, rho 0.9 under --sessions")
    return ctx["runs"]


@plt.rc_context(PRINT_RC)
def fig_run3_cdf(ctx, out):
    runs = _need_runs(ctx)
    data = {c: d for c in CDF_CELLS if (d := cdf_data(runs, c))}
    if not data:
        raise Skip("no measured row of mix A or B, Poisson, rho 0.9, reps 0-2")
    on = ctx.get("titles", True)
    fin = [v[np.isfinite(v)] for d in data.values() for v in d.values()]
    xmax = np.ceil(10 * max(float(v.max()) for v in fin if v.size)) / 10 + 0.05
    fig, axes = plt.subplots(1, len(data), figsize=(6.9, 3.7 if on else 3.4), sharey=True, squeeze=False)
    drawn = set()
    for ax, (cell, d) in zip(axes[0], data.items()):
        ax.axvline(1.0, color=DEADLINE_GREY, linestyle=":", linewidth=1.4, zorder=1)
        for m in METHODS:
            if m not in d:
                continue
            col, _, ls, _ = METHOD_STYLE[m]
            x, y = ecdf(d[m])
            ok = np.isfinite(x)
            ax.step(np.r_[0.0, x[ok]], np.r_[0.0, y[ok]], where="post", color=col, linestyle=ls,
                    linewidth=2.0 if m == "coexit" else 1.5, label=METHOD_NAME[m], zorder=4 if m == "coexit" else 3)
            drawn.add(m)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Response time / deadline")
        ax.set_title(MIX_TITLE.get(cell[0], f"Mix {cell[0]}"), fontweight="bold")
        ax.grid(color=GRID_GREY, linewidth=0.5, alpha=0.6)
    axes[0, 0].set_ylabel("Share of requests")
    h = [Line2D([], [], color=METHOD_STYLE[m][0], linestyle=METHOD_STYLE[m][2], linewidth=2.0 if m == "coexit" else 1.5,
                label=METHOD_NAME[m]) for m in METHODS if m in drawn]
    h.append(Line2D([], [], color=DEADLINE_GREY, linestyle=":", linewidth=1.4, label="Deadline"))
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.92 if on else 1.0), ncol=4, handlelength=2.4,
               columnspacing=1.2)
    adjust = {"left": 0.1, "right": 0.98, "bottom": 0.2 if on else 0.14, "top": 0.68 if on else 0.76, "wspace": 0.08}
    return _finish(fig, ctx, "fig_run3_cdf", out,
                   "Run 3: response time over the deadline, every request (rho 0.9, reps 0-2 pooled)",
                   "Left of the dotted line: on time. The step at 1: requests dropped at their deadline.",
                   adjust=adjust)


@plt.rc_context(PRINT_RC)
def fig_run3_timeline(ctx, out):
    runs = {r["method"]: r for r in _need_runs(ctx) if r["cell"] == TIMELINE_CELL and r["rep"] == TIMELINE_REP}
    miss = [m for m in TIMELINE_METHODS if m not in runs]
    if miss:
        raise Skip(f"no mix B, Poisson, rho 0.9, rep {TIMELINE_REP} run of {miss}")
    t0 = pick_window(runs["coexit"], runs["edgeserving"])
    t1 = t0 + WINDOW_S
    on = ctx.get("titles", True)
    ys = {f: len(FAMILIES) - 1 - k for k, f in enumerate(FAMILIES)}
    fig, axes = plt.subplots(len(TIMELINE_METHODS), 1, figsize=(6.9, 4.6 if on else 4.2), sharex=True)
    for ax, m in zip(axes, TIMELINE_METHODS):
        tl = timeline(runs[m], t0, t1)
        for s, w, _ in tl["llama"]["bars"]:
            ax.axvspan(s, s + w, color=FAMILY_COLOR["llama"], alpha=0.1, linewidth=0, zorder=0)
        for fam in FAMILIES:
            col, y = FAMILY_COLOR[fam], ys[fam]
            for s, w, floor in tl[fam]["bars"]:
                ax.broken_barh([(s, w)], (y - BAR_H / 2, BAR_H), facecolors=_light(col) if floor else col,
                               edgecolor=col, hatch="////" if floor else None, linewidth=0.8, zorder=2)
            if tl[fam]["miss"]:
                ax.plot(tl[fam]["miss"], [y + BAR_H / 2 + 0.14] * len(tl[fam]["miss"]), linestyle="none",
                        marker="x", color="black", markersize=7, markeredgewidth=1.6, zorder=4)
        ax.set_yticks([ys[f] for f in FAMILIES], [FAMILY_NAME[f] for f in FAMILIES])
        ax.set_ylim(-0.5, len(FAMILIES) - 0.2)
        ax.tick_params(axis="y", length=0)
        ax.set_title(METHOD_NAME[m], loc="left", fontweight="bold")
        ax.grid(axis="x", color=GRID_GREY, linewidth=0.5, alpha=0.6)
    axes[-1].set_xlim(t0, t1)
    axes[-1].set_xlabel("Time since the first arrival of the run (s)")
    h = [mpatches.Patch(facecolor=DEADLINE_GREY, edgecolor=DEADLINE_GREY, label="Target exit"),
         mpatches.Patch(facecolor=_light(DEADLINE_GREY), edgecolor=DEADLINE_GREY, hatch="////", label="Floor exit"),
         Line2D([], [], color="black", marker="x", linestyle="none", markersize=7, markeredgewidth=1.6,
                label="Missed deadline"),
         mpatches.Patch(color=FAMILY_COLOR["llama"], alpha=0.15, linewidth=0, label="LLaMA running")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.93 if on else 1.0), ncol=4, handlelength=1.8,
               columnspacing=1.2)
    adjust = {"left": 0.1, "right": 0.98, "bottom": 0.15 if on else 0.11, "top": 0.8 if on else 0.86, "hspace": 0.35}
    return _finish(fig, ctx, "fig_run3_timeline", out,
                   "Run 3: what runs on the GPU while LLaMA runs (mix B, rho 0.9, rep 0)",
                   "CoExit runs BERT and YOLO next to LLaMA. EdgeServing runs one request at a time.", adjust=adjust)


FIGS = collections.OrderedDict((f.__name__, f) for f in (fig_run3_cdf, fig_run3_timeline))


def run(ctx, out):
    """Make the figures; -> ({name: [files]}, {name: skip reason})."""
    _style()
    ctx = dict(ctx, runs=load_runs(ctx["sessions"], CDF_CELLS))
    made, skipped = {}, {}
    for name, fn in FIGS.items():
        try:
            made[name] = fn(ctx, out)
            print(f"[fig_run3_req] {name}: wrote {', '.join(made[name])}")
        except Skip as e:
            skipped[name] = str(e)
            print(f"[fig_run3_req] {name}: SKIPPED ({e})")
    return made, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exp2.analyze.fig_run3_req")
    ap.add_argument("--sessions", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-titles", action="store_true", help="paper version: no title, no takeaway line")
    a = ap.parse_args(argv)
    run({"sessions": a.sessions, "titles": not a.no_titles}, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
