"""A100 vs Jetson comparison built on the SAME pipeline as benchmark.ipynb.

Instead of inventing keys, this reuses the canonical exporters
(`shared.write_benchmark_csvs` + `shared.write_average_csvs`) — exactly what
produces `results/{model}/average/*.csv` for the A100 — and just points them at
a second log root:

    logs/benchmark        -> results/{model}/average/*.csv          (A100)
    logs.jetson/benchmark -> results.jetson/{model}/average/*.csv   (Jetson)

Both `average/` sets are averaged-over-all-tasks per model (mean+std). From the
two already-averaged sources it then builds ONE double plot per model (A100 vs
Jetson, labelled) + an xlsx for copy-paste.

Usage (or just run the cell in benchmark.ipynb):
    python compare_devices.py            # export both + plot + xlsx
    python compare_devices.py --force    # re-export even if csvs exist
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DEVICES = [
    ("A100",   REPO_ROOT / "logs" / "benchmark",        REPO_ROOT / "results"),
    ("Jetson", REPO_ROOT / "logs.jetson" / "benchmark", REPO_ROOT / "results.jetson"),
]
OUT_ROOT = REPO_ROOT / "results_compare"
MODELS = ["bert", "vision", "yolo", "llama"]

# (avg csv file, mean column, std column, axis label). quality col is auto-picked.
PANELS = [
    ("latency.csv",  "end_to_end_sec_mean_mean",   "end_to_end_sec_mean_std",   "latency (s)"),
    ("energy.csv",   "avg_energy_j_mean",          "avg_energy_j_std",          "energy (J)"),
    ("hardware.csv", "avg_vram_allocated_mb_mean", "avg_vram_allocated_mb_std", "VRAM alloc (MB)"),
    ("quality.csv",  None,                         None,                        "quality"),
]
QUALITY_PREF = ["acc_mean", "map_mean", "map50_mean", "glue_score_mean",
                "f1_mean", "exact_match_mean", "rougeL_mean", "perplexity_mean"]


def _exit_sort_key(method: str):
    nums = [int(x) for x in re.findall(r"\d+", str(method))]
    return nums or [10 ** 9]


# ---- export: same as benchmark.ipynb, parametrised by log root --------------
def export_device(bench_root: Path, csv_root: Path, force: bool):
    """Mirror the notebook's export for one log root -> csv_root/{model}/...
    + csv_root/{model}/average/*.csv (averaged across tasks)."""
    from shared import write_benchmark_csvs, write_average_csvs
    if not bench_root.exists():
        print(f"[export] {bench_root} missing; skip")
        return
    for model in MODELS:
        out_dir = bench_root / model
        if not out_dir.exists():
            continue
        avg_dir = csv_root / model / "average"
        if avg_dir.exists() and not force:
            print(f"[export] {csv_root.name}/{model}: exists, skip (use --force)")
            continue
        run_dirs = {p.parent for p in out_dir.rglob("hw_results.json")} | \
                   {p.parent for p in out_dir.rglob("quality_results.json")}
        if not run_dirs:
            continue
        groups = defaultdict(dict)
        for rd in run_dirs:
            gk = rd.parent.relative_to(out_dir).as_posix().replace("/", "_") or "root"
            groups[gk][rd.name] = rd
        for gk, runs in sorted(groups.items()):
            order = sorted(runs.keys(), key=_exit_sort_key)
            write_benchmark_csvs(results_files=runs, out_dir=csv_root / model / gk,
                                 baseline_key=None, method_order=order)
        write_average_csvs(csv_root / model)
        print(f"[export] {csv_root.name}/{model}: {len(groups)} tasks -> average/")


# ---- double plot from the two averaged csv sets -----------------------------
def _quality_col(df):
    for c in QUALITY_PREF:
        if c in df.columns:
            return c
    cand = [c for c in df.columns if c.endswith("_mean") and c not in ("exit_mean",)]
    return cand[0] if cand else None


def _read_avg(csv_root: Path, model: str, fname: str):
    import pandas as pd
    p = csv_root / model / "average" / fname
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df.sort_values("exit") if "exit" in df.columns else df


def double_plot(model: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable: {e}")
        return False
    # need at least one device with data
    have = {dev: csv for dev, _, csv in DEVICES if (csv / model / "average").exists()}
    if not have:
        return False
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.3 * len(PANELS), 3.8))
    if len(PANELS) == 1:
        axes = [axes]
    colors = {"A100": "#1f77b4", "Jetson": "#d62728"}
    for ax, (fname, mcol, scol, ylabel) in zip(axes, PANELS):
        plotted = False
        for dev, _, csv_root in DEVICES:
            df = _read_avg(csv_root, model, fname)
            if df is None or "exit" not in df.columns:
                continue
            col = mcol or _quality_col(df)
            if col is None or col not in df.columns:
                continue
            x, y = df["exit"], df[col]
            ax.plot(x, y, marker="o", label=dev, color=colors.get(dev))
            scol_eff = scol if (scol and scol in df.columns) else None
            if scol_eff is None and not mcol:  # quality std col follows the picked mean
                guess = col.replace("_mean", "_std")
                scol_eff = guess if guess in df.columns else None
            if scol_eff is not None:
                ax.fill_between(x, y - df[scol_eff], y + df[scol_eff],
                                alpha=0.15, color=colors.get(dev))
            plotted = True
        ax.set_title(ylabel)
        ax.set_xlabel("exit")
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize=8)
    fig.suptitle(f"{model} — A100 vs Jetson (averaged across all tasks)")
    fig.tight_layout()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_ROOT / f"{model}.png", dpi=120)
    plt.close(fig)
    print(f"[plot] {OUT_ROOT.name}/{model}.png ({'+'.join(have)})")
    return True


# ---- xlsx: averaged-per-model, both devices side by side --------------------
def write_xlsx():
    import pandas as pd
    try:
        import openpyxl  # noqa: F401
    except Exception as e:
        print(f"[xlsx] openpyxl missing ({e}); pip install openpyxl"); return
    path = OUT_ROOT / "average_by_model.xlsx"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        wrote = 0
        for model in MODELS:
            merged = None
            for dev, _, csv_root in DEVICES:
                for fname, mcol, _scol, _yl in PANELS:
                    df = _read_avg(csv_root, model, fname)
                    if df is None or "exit" not in df.columns:
                        continue
                    col = mcol or _quality_col(df)
                    if col is None or col not in df.columns:
                        continue
                    sub = df[["exit", col]].rename(columns={col: f"{dev}_{col}"})
                    merged = sub if merged is None else merged.merge(sub, on="exit", how="outer")
            if merged is not None:
                merged.sort_values("exit").to_excel(xl, sheet_name=model[:31], index=False)
                wrote += 1
        if wrote == 0:
            pd.DataFrame({"note": ["no data"]}).to_excel(xl, sheet_name="empty", index=False)
    print(f"[xlsx] wrote {path}")


def run(force: bool = False):
    for _dev, bench_root, csv_root in DEVICES:
        export_device(bench_root, csv_root, force)
    for model in MODELS:
        double_plot(model)
    write_xlsx()
    print(f"[done] plots + xlsx under {OUT_ROOT}")


def main():
    ap = argparse.ArgumentParser(description="A100 vs Jetson comparison (reuses benchmark export).")
    ap.add_argument("--force", action="store_true", help="re-export csvs even if they exist")
    run(force=ap.parse_args().force)


if __name__ == "__main__":
    main()
