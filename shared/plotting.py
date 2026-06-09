"""Aggregate per-task CSVs into per-metric plots vs exit layer.

Layout: `results/{model}/{task}/{latency,energy,quality,hardware}.csv`
Outputs: one PNG per metric to `results/{model}/plots/`
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

HIGHER_BETTER = {"acc", "accuracy", "f1", "mcc", "glue_score", "top1_acc", "top5_acc",
                 "top1", "top5", "map", "map50", "mAP"}
LOWER_BETTER = {"perplexity", "ece", "nll_sum"}

CSV_TYPES = ("latency", "energy", "quality", "hardware")
EXIT_TICK_START = 1

# (csv_type, col, y_label, file_stem)
DEFAULT_PANELS: List[Tuple[str, str, str, str]] = [
    ("hardware", "gpu_vram_total_mb", "Total GPU Memory (MB)", "gpu_vram_total_mb"),
    ("hardware", "avg_vram_allocated_mb", "GPU Allocated Memory (MB)", "gpu_vram_allocated_mb"),
    ("hardware", "avg_vram_reserved_mb", "GPU Reserved Memory (MB)", "gpu_vram_reserved_mb"),
    ("hardware", "avg_gpu_mem_used_mb", "GPU Used Memory (MB)", "gpu_mem_used_mb"),
    ("hardware", "avg_gpu_clock_mhz", "GPU Clock Speed (MHz)", "gpu_clock_mhz"),
    ("hardware", "avg_cpu_clock_mhz", "CPU Clock Speed (MHz)", "cpu_clock_mhz"),
    ("hardware", "avg_cpu_cores_used", "CPU Cores Used", "cpu_cores"),
    ("hardware", "avg_ram_used_mb", "RAM Used (MB)", "ram_mb"),
    ("hardware", "avg_unified_mem_used_mb", "Unified Memory Used (MB) [Jetson]", "unified_mem_mb"),
    ("latency", "ttft_sec_mean", "Time to First Token (s)", "ttft"),
    ("latency", "end_to_end_sec_mean", "Latency e2e / Sample (s)", "latency_per_sample"),
    ("latency", "throughput_samples_per_sec", "Throughput (samples / sec)", "throughput"),
    ("energy", "avg_power_w", "Power / Sample (W)", "power_w"),
    ("energy", "avg_energy_j", "Energy / Sample (J)", "energy_per_sample"),
]


def _exit_num(method: str) -> int:
    m = re.search(r"\d+", str(method))
    return (int(m.group()) + 1) if m else -1


# YOLO multi-scale sub-exits (P3/P4/P5) -> distinct colors on the same plot.
_SUB_COLORS = {"P3": "#2563eb", "P4": "#16a34a", "P5": "#dc2626"}
# Small x-nudge per scale so overlapping (equal-value) lines stay visible as a
# 3-dot triplet at each exit layer.
_SUB_DX = {"P3": -0.15, "P4": 0.0, "P5": 0.15}

# Detection head scales. Feature-map sizes are for the 640px bench input
# (IMG_SIZE=640): P3=stride8->80x80 (small objects), P4=stride16->40x40,
# P5=stride32->20x20 (large objects).
_SUB_SCALE = {
    "P3": ("small scale", "80x80"),
    "P4": ("medium scale", "40x40"),
    "P5": ("large scale", "20x20"),
}


def _sub_label(sub: str) -> str:
    """Legend label for a YOLO sub-exit: 'small scale (80x80)' etc."""
    name, dims = _SUB_SCALE.get(sub, (sub, ""))
    return f"{name} ({dims})" if dims else name


def _sub_tag(method: str) -> str:
    """Extract YOLO sub-exit scale tag (P3/P4/P5) from method name, else ''."""
    m = re.search(r"_(P\d)\b", str(method))
    return m.group(1) if m else ""


def load_model_csvs(model_dir: Path) -> Dict[str, pd.DataFrame]:
    """Walk `results/{model}/{task}/*.csv` (skip `average/` subdir).
    Returns long DataFrame per CSV type: task, method, exit, + metric cols.
    """
    model_dir = Path(model_dir)
    avg_dir = model_dir / "average"
    out: Dict[str, pd.DataFrame] = {}
    for csv_name in CSV_TYPES:
        frames = []
        for f in model_dir.rglob(f"{csv_name}.csv"):
            if avg_dir in f.parents:
                continue
            df = pd.read_csv(f)
            df["task"] = f.parent.name
            df["exit"] = df["method"].map(_exit_num)
            df["sub"] = df["method"].map(_sub_tag)
            frames.append(df)
        out[csv_name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out


def agg_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Group by exit -> mean+std+count across tasks."""
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    sub = df[["exit", metric]].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby("exit")[metric].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g.sort_values("exit").reset_index(drop=True)


def _normalize_one_task(sub: pd.DataFrame, main_metric: str) -> Optional[pd.DataFrame]:
    if main_metric not in sub.columns:
        return None
    vals = sub[["exit", main_metric]].apply(pd.to_numeric, errors="coerce").dropna().copy()
    if vals.empty:
        return None
    vals.columns = ["exit", "value"]
    vmin, vmax = vals["value"].min(), vals["value"].max()
    rng = vmax - vmin
    if rng <= 0:
        vals["normalized"] = 0.5
    elif main_metric in LOWER_BETTER:
        vals["normalized"] = (vmax - vals["value"]) / rng
    else:
        vals["normalized"] = (vals["value"] - vmin) / rng
    return vals[["exit", "normalized"]]


def normalize_quality(quality_df: pd.DataFrame) -> pd.DataFrame:
    """Per-task min-max normalize main_metric -> [0,1]. Returns: task, exit, normalized, main_metric."""
    if quality_df.empty or "main_metric" not in quality_df.columns:
        return pd.DataFrame()
    frames = []
    for task, sub in quality_df.groupby("task"):
        mm = sub["main_metric"].dropna().astype(str).iloc[0] if sub["main_metric"].notna().any() else ""
        norm = _normalize_one_task(sub, mm)
        if norm is None:
            continue
        norm = norm.copy()
        norm["task"] = task
        norm["main_metric"] = mm
        frames.append(norm)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _drop_warmup_outliers(agg: pd.DataFrame) -> pd.DataFrame:
    """Drop cold-start spike rows so they don't appear as runaway dots.

    The first measured (exit, sub) run absorbs torch.compile / cudnn warmup, so
    its latency/energy can be ~5x every other point. Remove rows whose mean is
    above 4x the series median (only fires when a real spike exists)."""
    import numpy as np

    if agg.empty or "mean" not in agg.columns or len(agg) < 4:
        return agg
    med = float(np.median(agg["mean"].values))
    if med <= 0:
        return agg
    keep = agg[agg["mean"] <= 4.0 * med]
    return keep.reset_index(drop=True) if not keep.empty else agg


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {path}")


def plot_metric_separate(
    agg: pd.DataFrame,
    ylabel: str,
    title: str,
    model_name: str,
    save_path: Optional[Path] = None,
    log_scale: bool = False,
) -> plt.Figure:
    """One figure: mean±std band vs exit layer for a single HW metric."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if agg.empty:
        ax.set_title(f"{model_name} — {title} (no data)")
        if save_path:
            _save_fig(fig, save_path)
        return fig
    agg = _drop_warmup_outliers(agg)
    x = agg["exit"].values
    y = agg["mean"].values
    s = agg["std"].values
    n = int(agg["count"].iloc[0]) if "count" in agg else 0
    max_exit = int(x.max())
    xticks = list(range(EXIT_TICK_START, max_exit + 1))
    ax.plot(x, y, marker="o", linewidth=2, color="#2563eb", label="mean")
    ax.fill_between(x, y - s, y + s, alpha=0.2, color="#2563eb", label=f"±std (n tasks={n})")
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Exit layer")
    ax.set_ylabel(ylabel + (" [log]" if log_scale else ""))
    ax.set_title(f"{model_name} — {title}" + (" (log scale)" if log_scale else ""))
    ax.set_xticks(xticks)
    ax.set_xlim(EXIT_TICK_START - 0.5, max_exit + 0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


def _distinct_subs(long_df: pd.DataFrame) -> List[str]:
    if long_df.empty or "sub" not in long_df.columns:
        return []
    return sorted(s for s in long_df["sub"].fillna("").unique() if s)


def _agg_by_sub(long_df: pd.DataFrame, metric: str) -> Dict[str, pd.DataFrame]:
    """{sub_tag: per-exit agg} for each YOLO scale (P3/P4/P5)."""
    out: Dict[str, pd.DataFrame] = {}
    for s in _distinct_subs(long_df):
        a = agg_metric(long_df[long_df["sub"] == s], metric)
        if not a.empty:
            out[s] = a
    return out


def plot_metric_subexit(
    sub_aggs: Dict[str, pd.DataFrame],
    ylabel: str,
    title: str,
    model_name: str,
    save_path: Optional[Path] = None,
    log_scale: bool = False,
    drop_outliers: bool = True,
) -> plt.Figure:
    """One figure: a colored mean±std line per sub-exit scale (P3/P4/P5) vs exit.

    drop_outliers: strip warmup spikes (HW metrics). Off for quality — a high
    mAP at the native exit is real signal, not a spike to discard."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if not sub_aggs:
        ax.set_title(f"{model_name} — {title} (no data)")
        if save_path:
            _save_fig(fig, save_path)
        return fig
    max_exit = EXIT_TICK_START
    for sub, agg in sub_aggs.items():
        if drop_outliers:
            agg = _drop_warmup_outliers(agg)
        x, y, s = agg["exit"].values, agg["mean"].values, agg["std"].values
        max_exit = max(max_exit, int(x.max()))
        # Nudge each scale on x so identical values (memory/clock are the same
        # across P3/P4/P5 — all weights resident) stay visible as 3 colored dots
        # per exit instead of overlapping into one line.
        xj = x + _SUB_DX.get(sub, 0.0)
        color = _SUB_COLORS.get(sub, None)
        ax.plot(xj, y, marker="o", linewidth=2, color=color, label=_sub_label(sub))
        ax.fill_between(xj, y - s, y + s, alpha=0.15, color=color)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Exit layer")
    ax.set_ylabel(ylabel + (" [log]" if log_scale else ""))
    ax.set_title(f"{model_name} — {title} (sub-exit scales)")
    ax.set_xticks(list(range(EXIT_TICK_START, max_exit + 1)))
    ax.set_xlim(EXIT_TICK_START - 0.5, max_exit + 0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, title="scale")
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


def plot_panel(
    long_df: pd.DataFrame,
    col: str,
    ylabel: str,
    model_name: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Dispatch: multi-sub colored lines if YOLO sub-exits present, else single line."""
    if len(_distinct_subs(long_df)) > 1:
        return plot_metric_subexit(_agg_by_sub(long_df, col), ylabel, ylabel, model_name, save_path)
    return plot_metric_separate(agg_metric(long_df, col), ylabel, ylabel, model_name, save_path)


def _common_higher_metric(quality_df: pd.DataFrame) -> Optional[str]:
    """Most frequent higher-better metric (acc/f1/mAP/...) present as a column."""
    if "main_metric" not in quality_df.columns:
        return None
    all_m = quality_df["main_metric"].dropna().astype(str)
    higher = all_m[all_m.isin(HIGHER_BETTER)]
    if higher.empty:
        return None
    metric = higher.value_counts().index[0]
    return metric if metric in quality_df.columns else None


def _draw_quality(agg, model_name, save_path, *, ylabel, title, color, log, band):
    """Shared single-line quality figure (optional log axis + std band)."""
    if agg.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    x, y, s = agg["exit"].values, agg["mean"].values, agg["std"].values
    n = int(agg["count"].max()) if "count" in agg else 0
    max_exit = int(x.max())
    ax.plot(x, y, marker="o", linewidth=2, color=color, label=f"mean (n tasks={n})")
    if band:
        ax.fill_between(x, y - s, y + s, alpha=0.2, color=color, label="±std")
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("Exit layer")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{model_name} — {title}")
    ax.set_xticks(list(range(EXIT_TICK_START, max_exit + 1)))
    ax.set_xlim(EXIT_TICK_START - 0.5, max_exit + 0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


def plot_quality(
    quality_df: pd.DataFrame,
    model_name: str,
    save_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """One quality plot per model. No min-max normalization.

    Perplexity models (llama): log-scale perplexity over EVERY task that reports
    perplexity (all 5, not just generation tasks) — line only, std band omitted
    because early-exit perplexity spans many orders of magnitude. Other models:
    raw most-common higher-better metric (acc / f1 / mAP), linear, with std band.
    """
    if quality_df.empty:
        return None
    if "perplexity" in quality_df.columns and quality_df["perplexity"].notna().any():
        agg = agg_metric(quality_df, "perplexity")
        return _draw_quality(
            agg, model_name, save_path,
            ylabel="Perplexity (log scale, lower=better)",
            title="Perplexity vs exit layer (log scale)",
            color="#2563eb", log=True, band=False,
        )
    metric = _common_higher_metric(quality_df)
    if metric is None:
        return None
    # YOLO: mAP differs per scale (P3 small / P4 med / P5 large objects) -> one
    # colored line per sub-exit, same as the HW panels.
    if len(_distinct_subs(quality_df)) > 1:
        return plot_metric_subexit(
            _agg_by_sub(quality_df, metric),
            f"{metric} (higher=better)", f"{metric} vs exit layer",
            model_name, save_path, drop_outliers=False,
        )
    agg = agg_metric(quality_df, metric)
    return _draw_quality(
        agg, model_name, save_path,
        ylabel=f"{metric} (higher=better)",
        title=f"{metric} vs exit layer",
        color="#16a34a", log=False, band=True,
    )


def plot_model_all(
    model_dir: Path,
    panels: List[Tuple[str, str, str, str]] = DEFAULT_PANELS,
) -> None:
    """Generate one PNG per metric + quality plots for a model. Saves to `{model_dir}/plots/`."""
    model_dir = Path(model_dir)
    plot_dir = model_dir / "plots"
    model_name = model_dir.name
    data = load_model_csvs(model_dir)
    if all(df.empty for df in data.values()):
        print(f"[plot] no CSVs under {model_dir}")
        return
    for csv_t, col, ylabel, stem in panels:
        long_df = data.get(csv_t, pd.DataFrame())
        fig = plot_panel(long_df, col, ylabel, model_name, save_path=plot_dir / f"{stem}.png")
        plt.close(fig)
    q_fig = plot_quality(data.get("quality", pd.DataFrame()), model_name, save_path=plot_dir / "quality.png")
    if q_fig:
        plt.close(q_fig)


# keep old name as alias so existing notebook cells still work
def plot_model_panel(model_dir, save_path=None, **kwargs):
    plot_model_all(Path(model_dir))
