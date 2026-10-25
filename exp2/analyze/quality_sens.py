"""exp2.analyze.quality_sens: floor-quality sensitivity of Run 3 (reviewer W1). No board run, no model.

    python -m exp2.analyze.quality_sens --sessions DIR [DIR ...] --reps REPS_CSV --out-csv FILE --out-fig DIR
        [--no-titles]                                                                exit 0 ok, 2 usage or data

Score    contracts "Pilot metrics" success z_r = outcome complete and t_done_ns <= t_deadline_ns, over the measured
         (non-warm-up) requests in arrival order. Run exit = exit of the RUN type (to_trace run_type_id). Floor and
         target = contracts.EXIT_MENU[family]. Q(theta) = mean of z_r * (1 at the target exit, theta at the floor
         exit) = (n_target + theta * n_floor) / n. Q(1) = 1 - violation ratio, Q(0) = on-time target share.
Runs     every open manifest.json subdir of the sessions, read as exp2.analyze.run3 reads them (validate_manifest,
         read_requests, to_trace, the warm-up check); invalid and duplicate runs are skipped as run3 skips them.
Check    Q(1) of every run = 1 - slo_violation_ratio of its family "all" row in REPS_CSV within 1e-9, and the same n.
         The run sets must be equal. Any mismatch raises ContractError (exit 2).
CSV      COLS, long format. kind "run": Q of one run (n = measured requests). kind "mean": mean and Student-t 95 % CI
         over the repeats (n = repeats). kind "diff": Q(CoExit) - Q(other), paired by (rep, seed), stats.paired_diff_ci
         (n = pairs). kind "theta_star": theta where the mean paired difference is 0, by linear interpolation over
         THETAS ("none" without a sign change). method = policy_key, mix = workload name without "mix".
Figure   fig_run3_theta: Q versus theta in FIG_CELLS, one line per method with its 95 % CI band, theta* of CoExit
         versus All-Early dotted. Names and colours of exp2.analyze.fig_run3. NAME.pdf and NAME.png (200 dpi).
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from exp2 import contracts as C  # noqa: E402
from exp2 import stats  # noqa: E402
from exp2.analyze import run3 as R  # noqa: E402
from exp2.analyze.fig_run3 import METHOD_NAME, METHOD_STYLE, METHODS  # noqa: E402
from exp2.analyze.figures import GRID_GREY, PRINT_RC, _finish, _style  # noqa: E402
from exp2.model import coexit_pilot as P  # noqa: E402

__all__ = ["THETAS", "counts", "q_score", "break_even", "load_runs", "check_q1", "table", "fig_theta", "main"]

THETAS = tuple(i / 10 for i in range(11))
OURS = "coexit"
OTHERS = ("edgeserving", "all_early", "greedy", "coexit_nogate", "all_final")
TOL = 1e-9
COLS = ("kind", "mix", "arrival", "cv_a", "rho", "method", "other", "rep", "seed", "n", "theta", "value", "ci_lo",
        "ci_hi")
FIG_CELLS = (("A", "poisson", 0.9), ("B", "poisson", 0.9), ("B", "gamma", 0.9))
FIG_NAME = "fig_run3_theta"


# ------------------------------------------------------------------------------------------------ score
def counts(trace, n_warmup):
    """(n, n_target, n_floor) of the measured rows of a pilot trace: on-time requests at the target / floor exit."""
    meas = sorted(trace, key=lambda r: r["t_arrival_ns"])[n_warmup:]
    nt = nf = 0
    for r in meas:
        dl = r["t_deadline_ns"]
        if r["outcome"] != "complete" or (dl is not None and r["t_done_ns"] > dl):
            continue
        fam, ex = C.parse_type_id(r["run_type_id"])
        floor, target = C.EXIT_MENU[fam]
        if ex not in (floor, target):
            raise C.ContractError("%s: run exit %d is neither floor %d nor target %d" % (r["req_id"], ex, floor, target))
        nt += ex == target
        nf += ex == floor
    if not meas:
        raise C.ContractError("no measured request after %d warm-up rows" % n_warmup)
    return len(meas), nt, nf


def q_score(c, theta):
    """Q(theta) of counts c = (n, n_target, n_floor) or a run dict."""
    n, nt, nf = (c["n"], c["n_target"], c["n_floor"]) if isinstance(c, dict) else c
    return (nt + theta * nf) / n


def break_even(thetas, diffs):
    """theta where the linear interpolation of diffs over thetas is 0 (first crossing); None without one."""
    for i, (t0, d0) in enumerate(zip(thetas, diffs)):
        if d0 == 0:
            return t0
        if i + 1 < len(thetas) and d0 * diffs[i + 1] < 0:
            t1, d1 = thetas[i + 1], diffs[i + 1]
            return t0 + (t1 - t0) * d0 / (d0 - d1)
    return None


# ------------------------------------------------------------------------------------------------ data
def load_runs(session_dirs):
    """One dict per kept open run: method, mix, arrival, cv_a, rho, rep, seed, condition_id, n, n_target, n_floor."""
    runs, seen, skipped = [], set(), 0
    for sdir in map(Path, session_dirs):
        for cdir in sorted(p for p in sdir.iterdir() if (p / "manifest.json").is_file()):
            man = C.validate_manifest(R._load(cdir / "manifest.json"))
            if man["mode"] != "open":
                continue
            if R._reason(man, seen) is not None:
                skipped += 1
                continue
            seen.add(man["condition_id"])
            rows = C.read_requests(cdir / "requests.jsonl")
            trace = R.to_trace(rows)
            R._check(rows, trace, man)
            n, nt, nf = counts(trace, man["n_warmup"])
            wl = man["workload"]
            runs.append({"method": man["policy_key"], "mix": wl["name"].removeprefix("mix"), "arrival": wl["arrival"],
                         "cv_a": float(wl["cv_a"]), "rho": float(wl["rho"]), "rep": man["repeat"], "seed": man["seed"],
                         "condition_id": man["condition_id"], "n": n, "n_target": nt, "n_floor": nf})
    if not runs:
        raise C.ContractError("no open manifest.json under %s" % [str(s) for s in session_dirs])
    print("runs kept %d, skipped %d (invalid or duplicate)" % (len(runs), skipped))
    return runs


def _key(method, mix, arrival, cv_a, rho, rep, seed):
    return method, mix, arrival, float(cv_a), float(rho), int(rep), int(seed)


def check_q1(runs, reps_path):
    """Max |Q(1) - (1 - slo_violation_ratio)| over the runs against the family "all" rows of reps_path.
    ContractError when the run sets differ, an n differs, or a difference exceeds TOL."""
    want = {_key(*(r[c] for c in ("method", "mix", "arrival", "cv_a", "rho", "rep", "seed"))): r
            for r in P._read_reps(Path(reps_path))[0] if r["family"] == "all"}
    got = {_key(*(r[c] for c in ("method", "mix", "arrival", "cv_a", "rho", "rep", "seed"))): r for r in runs}
    if set(want) != set(got):
        raise C.ContractError("run sets differ: only in sessions %s, only in %s %s"
                              % (sorted(set(got) - set(want)), reps_path, sorted(set(want) - set(got))))
    worst = 0.0
    for k, run in got.items():
        rep = want[k]
        err = abs(q_score(run, 1.0) - (1.0 - float(rep["slo_violation_ratio"])))
        if int(rep["n"]) != run["n"] or err > TOL:
            raise C.ContractError("%s: Q(1) %r vs 1 - viol %r (n %d vs %s)" % (run["condition_id"], q_score(run, 1.0),
                                  1.0 - float(rep["slo_violation_ratio"]), run["n"], rep["n"]))
        worst = max(worst, err)
    return worst


def _ci(xs):
    return stats.mean_ci(xs) if len(xs) >= 2 else (xs[0], None, None)


def table(runs):
    """CSV rows (dicts over COLS) of the runs: kinds run, mean, diff, theta_star."""
    cells = {}
    for r in runs:
        cells.setdefault((r["mix"], r["arrival"], r["cv_a"], r["rho"]), {}).setdefault(r["method"], {})[
            (int(r["rep"]), int(r["seed"]))] = r
    out = []

    def add(kind, cell, method, **kw):
        out.append(dict(dict.fromkeys(COLS), kind=kind, mix=cell[0], arrival=cell[1], cv_a=cell[2], rho=cell[3],
                        method=method, **kw))

    for cell in sorted(cells):
        by = cells[cell]
        for meth in sorted(by):
            for (rep, seed), r in sorted(by[meth].items()):
                for th in THETAS:
                    add("run", cell, meth, rep=rep, seed=seed, n=r["n"], theta=th, value=q_score(r, th))
            for th in THETAS:
                m, lo, hi = _ci([q_score(r, th) for r in by[meth].values()])
                add("mean", cell, meth, n=len(by[meth]), theta=th, value=m, ci_lo=lo, ci_hi=hi)
        mine = by.get(OURS, {})
        for other in OTHERS:
            pairs = sorted(set(mine) & set(by.get(other, {})))
            if not pairs:
                continue
            diffs = []
            for th in THETAS:
                a, b = [q_score(mine[k], th) for k in pairs], [q_score(by[other][k], th) for k in pairs]
                d = stats.paired_diff_ci(a, b) if len(pairs) >= 2 else (a[0] - b[0], None, None)
                diffs.append(d[0])
                add("diff", cell, OURS, other=other, n=len(pairs), theta=th, value=d[0], ci_lo=d[1], ci_hi=d[2])
            ts = break_even(THETAS, diffs)
            add("theta_star", cell, OURS, other=other, n=len(pairs), value="none" if ts is None else ts)
    return out


def _write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(COLS)
        wr.writerows([P._fmt(r[c]) for c in COLS] for r in rows)


# ------------------------------------------------------------------------------------------------ figure
@plt.rc_context(PRINT_RC)
def fig_theta(rows, out, titles=True):
    """fig_run3_theta from table rows; -> written file names ([] when no FIG_CELLS cell is in the rows)."""
    mean, star = {}, {}
    for r in rows:
        spec = (r["mix"], r["arrival"], r["rho"])
        if r["kind"] == "mean":
            mean.setdefault(spec, {}).setdefault(r["method"], []).append((r["theta"], r["value"], r["ci_lo"],
                                                                         r["ci_hi"]))
        elif r["kind"] == "theta_star" and r["other"] == "all_early" and r["value"] != "none":
            star[spec] = r["value"]
    panels = [s for s in FIG_CELLS if s in mean]
    if not panels:
        print("[quality_sens] %s: SKIPPED (no cell of %s)" % (FIG_NAME, FIG_CELLS))
        return []
    _style()
    fig, axes = plt.subplots(1, len(panels), figsize=(6.9, 3.6 if titles else 3.3), sharey=True, squeeze=False)
    drawn = []
    for ax, spec in zip(axes[0], panels):
        for meth in METHODS:
            pts = sorted(mean[spec].get(meth, []))
            if not pts:
                continue
            drawn += [meth] if meth not in drawn else []
            col, mk, ls, mfc = METHOD_STYLE[meth]
            th, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            if all(p[2] is not None for p in pts):
                ax.fill_between(th, [p[2] for p in pts], [p[3] for p in pts], color=col, alpha=0.15, linewidth=0,
                                zorder=1)
            ax.plot(th, y, color=col, linestyle=ls, linewidth=1.3, marker=mk, markevery=2,
                    markersize=7 if meth.startswith(OURS) else 5, markerfacecolor=mfc or col, zorder=3)
        if spec in star:
            ax.axvline(star[spec], color="black", linestyle=":", linewidth=1.1, zorder=2)
            ax.text(star[spec] - 0.03, 0.03, r"$\theta^*$ = %.2f" % star[spec], ha="right", va="bottom",
                    fontsize=9, transform=ax.get_xaxis_transform())
        mix, arrival, rho = spec
        ax.set_title(r"Mix %s, %s, $\rho$ = %g" % (mix, arrival.capitalize(), rho), fontweight="bold")
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks(THETAS[::2], ["%g" % t for t in THETAS[::2]])
        ax.set_xlabel(r"Floor quality $\theta$")
        ax.grid(axis="y", color=GRID_GREY, linewidth=0.5, alpha=0.6)
    axes[0, 0].set_ylabel(r"Quality score $Q(\theta)$")
    handles = [Line2D([], [], color=METHOD_STYLE[m][0], marker=METHOD_STYLE[m][1], linestyle=METHOD_STYLE[m][2],
                      markerfacecolor=METHOD_STYLE[m][3] or METHOD_STYLE[m][0],
                      markersize=7 if m.startswith(OURS) else 5, label=METHOD_NAME[m]) for m in drawn]
    handles.append(Line2D([], [], color="black", linestyle=":", linewidth=1.1,
                          label=r"$\theta^*$: CoExit = All-Early"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.92 if titles else 1.0), ncol=4,
               handlelength=2.4, columnspacing=1.0)
    adjust = {"left": 0.1, "right": 0.99, "bottom": 0.2 if titles else 0.15, "top": 0.62 if titles else 0.72,
              "wspace": 0.08}
    return _finish(fig, {"titles": titles}, FIG_NAME, out,
                   "Run 3: quality score versus the quality of the floor exit (all DNNs)",
                   "Bands: 95 % confidence interval over the repeats. Dotted: break-even of CoExit and All-Early.",
                   adjust=adjust)


# ------------------------------------------------------------------------------------------------ CLI
def _report(rows):
    """Print Q(CoExit) - Q(other) at theta 0, 0.5, 1 and theta* per cell."""
    for r in rows:
        if r["kind"] == "diff" and r["theta"] in (0.0, 0.5, 1.0):
            ci = "" if r["ci_lo"] is None else " [%+.3f, %+.3f]" % (r["ci_lo"], r["ci_hi"])
            print("%s %-7s rho %.1f  vs %-13s theta %.1f  %+.3f%s  (n %d)" % (r["mix"], r["arrival"], r["rho"],
                  r["other"], r["theta"], r["value"], ci, r["n"]))
        elif r["kind"] == "theta_star":
            v = r["value"] if r["value"] == "none" else "%.3f" % r["value"]
            print("%s %-7s rho %.1f  vs %-13s theta* %s" % (r["mix"], r["arrival"], r["rho"], r["other"], v))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m exp2.analyze.quality_sens", description=__doc__.splitlines()[0])
    ap.add_argument("--sessions", required=True, nargs="+")
    ap.add_argument("--reps", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-fig", required=True)
    ap.add_argument("--no-titles", action="store_true", help="paper version: no title, no takeaway line")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    try:
        runs = load_runs(a.sessions)
        worst = check_q1(runs, a.reps)
        print("check Q(1) = 1 - slo_violation_ratio: %d runs, max abs diff %.3g <= %g" % (len(runs), worst, TOL))
        rows = table(runs)
        _write(a.out_csv, rows)
        print("wrote %s (%d rows)" % (a.out_csv, len(rows)))
        _report(rows)
        files = fig_theta(rows, a.out_fig, titles=not a.no_titles)
        print("wrote %s" % ", ".join(str(Path(a.out_fig) / f) for f in files))
    except (ValueError, KeyError, OSError) as e:                 # ContractError is a ValueError
        print("quality_sens: %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
