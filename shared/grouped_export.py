"""Per-TASK grouped benchmark export (the 'group it correctly' view).

The flat exporter (csv_export.write_benchmark_csvs) writes one CSV set per
(task, mode) with rows = exits — so comparing modes on a task means opening N
folders, and the quality sheet jams every metric column together.

This module groups by TASK and PIVOTS:
    rows  = exit (cost ladder)
    cols  = series  (joint / pairwise / segd / pretrained baseline)
    value = that task's OWN main_metric (acc / f1 / mcc / top1 / mAP / acc_norm /
            perplexity), read from quality_results.json

Comparability rule: results are comparable only within (backend, task, metric).
Each task's metric direction (higher- vs lower-better) is recorded so plots /
"best exit" logic never blend incomparable metrics (perplexity is lower-better).

Layout written:
    {results_root}/{backend}/{task}/
        quality.csv   rows=exit, cols=series, value=main_metric
        latency.csv   end_to_end_sec
        energy.csv    power x e2e (J)
        memory.csv    vram_allocated_mb (full model — constant across exits)
        metric.json   {main_metric, direction, family, series}
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from .csv_export import _load_run_dir, _write, _f


# ---- metric registry: name -> (direction, family) ---------------------------
# direction "up" = higher-is-better; "down" = lower-is-better.
METRIC_REGISTRY = {
    "acc":         ("up",   "accuracy"),
    "accuracy":    ("up",   "accuracy"),
    "top1":        ("up",   "accuracy"),
    "top5":        ("up",   "accuracy"),
    "acc_norm":    ("up",   "accuracy"),
    "f1":          ("up",   "glue"),
    "mcc":         ("up",   "glue"),
    "map":         ("up",   "detection"),
    "map50":       ("up",   "detection"),
    "map50_95":    ("up",   "detection"),
    "mAP":         ("up",   "detection"),
    "rouge_l":     ("up",   "generation"),
    "rougeL":      ("up",   "generation"),
    "exact_match": ("up",   "generation"),
    "perplexity":  ("down", "perplexity"),
}


def metric_direction(name) -> str:
    return METRIC_REGISTRY.get(str(name), ("up", "other"))[0]


def metric_family(name) -> str:
    return METRIC_REGISTRY.get(str(name), ("up", "other"))[1]


def higher_is_better(name) -> bool:
    return metric_direction(name) == "up"


# ---- path / exit parsing ----------------------------------------------------
def _exit_key(name: str):
    """Sort key for 'exit_3' / 'exit_3_P4' -> (3, 'P4'). Non-exit -> (inf, '')."""
    m = re.search(r"exit_(\d+)(?:_(.+))?", str(name))
    if not m:
        return (10 ** 9, "")
    return (int(m.group(1)), m.group(2) or "")


def _parse_run(rel_parts):
    """rel path parts {task}/{series}/exit_k[_sub] -> (task, series, exit_name).
    series = training mode (joint/pairwise/segd) or weight_source (pretrained)."""
    exit_name = rel_parts[-1]
    series = rel_parts[-2] if len(rel_parts) >= 2 else "default"
    task = "/".join(rel_parts[:-2]) if len(rel_parts) >= 3 else (
        rel_parts[0] if len(rel_parts) >= 2 else "root")
    return task, series, exit_name


def _pivot_write(path: Path, cell_map: Dict, exits_sorted, series_sorted) -> None:
    """cell_map: {(series, exit_name): value} -> rows=exit, cols=series."""
    if not cell_map:
        return
    header = ["exit"] + series_sorted
    rows = []
    for ex in exits_sorted:
        row = {"exit": ex}
        for s in series_sorted:
            v = cell_map.get((s, ex))
            row[s] = _f(v, 6) if v is not None else ""
        rows.append(row)
    _write(path, header, rows)


# ---- main entry -------------------------------------------------------------
def write_grouped_csvs(
    backend_out_dir: Union[str, Path],
    results_root: Union[str, Path],
    backend_name: str,
) -> None:
    """backend_out_dir = logs/benchmark/{backend}. Writes per-task pivoted CSVs
    under {results_root}/{backend_name}/{task}/."""
    backend_out_dir = Path(backend_out_dir)
    run_dirs = {p.parent for p in backend_out_dir.rglob("hw_results.json")} | \
               {p.parent for p in backend_out_dir.rglob("quality_results.json")}
    if not run_dirs:
        print(f"[grouped] {backend_name}: no runs under {backend_out_dir}")
        return

    # task -> aggregated cells per panel + the task's main_metric
    tasks: Dict[str, Dict] = defaultdict(lambda: {
        "series": set(), "exits": set(), "metric": None,
        "quality": {}, "latency": {}, "energy": {}, "memory": {},
    })

    for rd in run_dirs:
        rel = rd.relative_to(backend_out_dir).parts
        task, series, exit_name = _parse_run(rel)
        merged = _load_run_dir(rd)

        # main_metric is a string -> read straight from the quality json
        mm = None
        qf = rd / "quality_results.json"
        if qf.exists():
            try:
                mm = json.loads(qf.read_text(encoding="utf-8")).get("main_metric")
            except Exception:
                mm = None

        t = tasks[task]
        t["series"].add(series)
        t["exits"].add(exit_name)
        if mm and t["metric"] is None:
            t["metric"] = mm

        key = (series, exit_name)
        if mm and mm in merged:
            t["quality"][key] = merged[mm]

        e2e = merged.get("end_to_end_sec_mean")
        if e2e is not None:
            t["latency"][key] = e2e
            power = merged.get("avg_power_w")
            if power not in (None, ""):
                t["energy"][key] = float(power) * float(e2e)

        vram = merged.get("avg_vram_allocated_mb", merged.get("vram_allocated_mb"))
        if vram not in (None, ""):
            t["memory"][key] = vram

    for task, t in sorted(tasks.items()):
        out = Path(results_root) / backend_name / task.replace("/", "_")
        out.mkdir(parents=True, exist_ok=True)
        series_sorted = sorted(t["series"])
        exits_sorted = sorted(t["exits"], key=_exit_key)
        metric = t["metric"] or "unknown"

        _pivot_write(out / "quality.csv", t["quality"], exits_sorted, series_sorted)
        _pivot_write(out / "latency.csv", t["latency"], exits_sorted, series_sorted)
        _pivot_write(out / "energy.csv", t["energy"], exits_sorted, series_sorted)
        _pivot_write(out / "memory.csv", t["memory"], exits_sorted, series_sorted)

        (out / "metric.json").write_text(json.dumps({
            "task": task,
            "main_metric": metric,
            "direction": metric_direction(metric),   # up = higher better
            "family": metric_family(metric),
            "series": series_sorted,
        }, indent=2), encoding="utf-8")
        print(f"[grouped] {backend_name}/{task}: metric={metric}"
              f"({metric_direction(metric)}) series={series_sorted} exits={len(exits_sorted)}")


# ---- per-task plotting (one figure per task, series = lines) -----------------
def _ordered(df):
    """Sort a pivoted grouped CSV by (exit_num, sub_tag)."""
    df = df.copy()
    df["_kn"] = df["exit"].map(lambda s: _exit_key(s)[0])
    df["_ks"] = df["exit"].map(lambda s: _exit_key(s)[1])
    return df.sort_values(["_kn", "_ks"]).reset_index(drop=True)


def _series_cols(df):
    return [c for c in df.columns if c not in ("exit", "_kn", "_ks")]


def _plot_vs_exit(csv_path, title, ylabel, save_path, *, log=False):
    """One line per series, value vs exit (x-tick = exit name, handles YOLO subs)."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty or "exit" not in df.columns:
        return
    df = _ordered(df)
    x = list(range(len(df)))
    fig, ax = plt.subplots(figsize=(max(7, len(df) * 0.45), 4.2))
    plotted = False
    for c in _series_cols(df):
        y = pd.to_numeric(df[c], errors="coerce")
        if y.notna().any():
            ax.plot(x, y, marker="o", linewidth=2, label=c)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    if log:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(df["exit"], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Exit")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, title="mode / source")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {save_path}")


def _plot_pareto(q_csv, lat_csv, title, xlabel, ylabel, save_path):
    """The early-exit money plot: metric (y) vs latency (x), one line per series,
    points ordered along the latency axis (the accuracy/cost trade-off curve)."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not (q_csv.exists() and lat_csv.exists()):
        return
    q = _ordered(pd.read_csv(q_csv))
    lat = pd.read_csv(lat_csv).set_index("exit")
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for s in _series_cols(q):
        if s not in lat.columns:
            continue
        yv = pd.to_numeric(q[s], errors="coerce")
        xv = pd.to_numeric(lat[s].reindex(q["exit"]).values, errors="coerce")
        m = pd.DataFrame({"x": xv, "y": yv.values}).dropna().sort_values("x")
        if not m.empty:
            ax.plot(m["x"], m["y"], marker="o", linewidth=1.8, label=s)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, title="mode / source")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {save_path}")


def plot_grouped_csvs(results_root: Union[str, Path], backend_name: str) -> None:
    """One set of figures PER TASK (titled with backend + dataset + metric):
    quality / latency / energy vs exit + the metric-vs-latency Pareto.
    Reads {results_root}/{backend_name}/{task}/ written by write_grouped_csvs."""
    base = Path(results_root) / backend_name
    if not base.is_dir():
        return
    for task_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        task = task_dir.name
        meta = {}
        mj = task_dir / "metric.json"
        if mj.exists():
            try:
                meta = json.loads(mj.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        metric = meta.get("main_metric", "metric")
        lower_better = meta.get("direction", "up") == "down"
        better = "lower=better" if lower_better else "higher=better"
        head = f"{backend_name} · {task}"   # task == the dataset used

        _plot_vs_exit(
            task_dir / "quality.csv",
            title=f"{head} — {metric} vs exit",
            ylabel=f"{metric} ({better})",
            save_path=task_dir / "plot_quality.png",
            log=lower_better,   # perplexity spans orders of magnitude
        )
        _plot_vs_exit(
            task_dir / "latency.csv",
            title=f"{head} — latency vs exit",
            ylabel="latency e2e / sample (s)",
            save_path=task_dir / "plot_latency.png",
        )
        _plot_vs_exit(
            task_dir / "energy.csv",
            title=f"{head} — energy vs exit",
            ylabel="energy / sample (J)",
            save_path=task_dir / "plot_energy.png",
        )
        _plot_pareto(
            task_dir / "quality.csv",
            task_dir / "latency.csv",
            title=f"{head} — {metric} vs latency (Pareto)",
            xlabel="latency e2e / sample (s)",
            ylabel=f"{metric} ({better})",
            save_path=task_dir / "plot_pareto.png",
        )


# ---- per-METRIC-FAMILY grouping (bucket a backend's tasks by metric category) -
# Tasks that share a metric category live on the same scale + direction (accuracy
# 0..1 up, generation 0..1 up, detection mAP up, perplexity down), and within ONE
# backend they share the exit/latency ladder -> a clean cross-task overlay.
def _read_pivot(csv_path: Path):
    """Load a per-task pivoted CSV (rows=exit, cols=series), exit-ordered. Pandas
    (used by the overlay plot only — keeps the plotting path consistent with the
    rest of the module)."""
    import pandas as pd

    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty or "exit" not in df.columns:
        return None
    return _ordered(df)


def _read_pivot_rows(csv_path: Path):
    """Stdlib reader for a pivoted CSV -> (series_cols, [(exit_name, {series: float|None})])
    sorted by exit. Pandas-free so the summary table works anywhere."""
    import csv

    if not csv_path.exists():
        return None
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or not rows[0] or rows[0][0] != "exit":
        return None
    series = rows[0][1:]
    out = []
    for line in rows[1:]:
        if not line:
            continue
        vals = {}
        for s, cell in zip(series, line[1:]):
            try:
                vals[s] = float(cell) if cell != "" else None
            except ValueError:
                vals[s] = None
        out.append((line[0], vals))
    out.sort(key=lambda t: _exit_key(t[0]))
    return series, out


def _write_family_summary(path: Path, items) -> None:
    """One row per (task, series): its best exit on the task's own metric, plus
    the latency at that exit (the operating point you'd actually ship)."""
    rows = []
    for task_dir, meta in items:
        lower = meta.get("direction", "up") == "down"
        metric = meta.get("main_metric", "metric")
        q = _read_pivot_rows(task_dir / "quality.csv")
        if q is None:
            continue
        qseries, qrows = q
        lat = _read_pivot_rows(task_dir / "latency.csv")
        lat_map = {ex: vals for ex, vals in lat[1]} if lat is not None else {}
        for s in qseries:
            best_ex = best_val = None
            for ex, vals in qrows:
                v = vals.get(s)
                if v is None:
                    continue
                if best_val is None or (v < best_val if lower else v > best_val):
                    best_val, best_ex = v, ex
            if best_ex is None:
                continue
            lv = lat_map.get(best_ex, {}).get(s)
            rows.append({
                "task": task_dir.name, "metric": metric, "series": s,
                "best_exit": best_ex, "value": _f(best_val, 6),
                "latency_at_best": _f(lv, 6) if lv is not None else "",
            })
    _write(path, ["task", "metric", "series", "best_exit", "value", "latency_at_best"], rows)


def _plot_family_overlay(save_path: Path, backend: str, family: str, items) -> None:
    """Metric-vs-latency, one line per (task, series) across the whole family."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lower = items[0][1].get("direction", "up") == "down" if items else False
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plotted = False
    for task_dir, _meta in items:
        q = _read_pivot(task_dir / "quality.csv")
        lat = _read_pivot(task_dir / "latency.csv")
        if q is None or lat is None:
            continue
        lat_idx = lat.set_index("exit")
        for s in _series_cols(q):
            if s not in lat_idx.columns:
                continue
            yv = pd.to_numeric(q[s], errors="coerce")
            xv = pd.to_numeric(lat_idx[s].reindex(q["exit"]).values, errors="coerce")
            m = pd.DataFrame({"x": xv, "y": yv.values}).dropna().sort_values("x")
            if m.empty:
                continue
            ax.plot(m["x"], m["y"], marker="o", linewidth=1.6, label=f"{task_dir.name}/{s}")
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    if lower:
        ax.set_yscale("log")
    better = "lower=better" if lower else "higher=better"
    ax.set_xlabel("latency e2e / sample (s)")
    ax.set_ylabel(f"{family} metric ({better})")
    ax.set_title(f"{backend} · {family} — metric vs latency (all tasks)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, title="task / mode")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {save_path}")


def group_by_metric(results_root: Union[str, Path], backend_name: str) -> None:
    """Bucket a backend's per-task results by metric FAMILY and write, per family,
    a summary.csv (best exit per task+mode) + overlay_pareto.png. Reads the
    per-task dirs written by write_grouped_csvs, so run it AFTER that.
    Output: {results_root}/_by_metric/{backend}/{family}/."""
    base = Path(results_root) / backend_name
    if not base.is_dir():
        return
    fam_tasks: Dict[str, list] = defaultdict(list)
    for task_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        mj = task_dir / "metric.json"
        if not mj.exists():
            continue
        try:
            meta = json.loads(mj.read_text(encoding="utf-8"))
        except Exception:
            continue
        fam_tasks[meta.get("family", "other")].append((task_dir, meta))

    for family, items in sorted(fam_tasks.items()):
        out = Path(results_root) / "_by_metric" / backend_name / family
        out.mkdir(parents=True, exist_ok=True)
        _write_family_summary(out / "summary.csv", items)
        try:
            _plot_family_overlay(out / "overlay_pareto.png", backend_name, family, items)
        except Exception as e:  # plotting deps optional; summary.csv is the data product
            print(f"[by_metric] {backend_name}/{family}: plot skipped ({e})")
        print(f"[by_metric] {backend_name}/{family}: {len(items)} task(s)")
