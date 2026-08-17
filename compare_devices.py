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

# Archived data lives under datasource/ (logs/ is the empty on-device capture dir).
# One entry per device/power-mode; export_device/double_plot/write_xlsx are all
# generic over this list, so adding modes needs nothing else.
_DS = REPO_ROOT / "datasource"
_JET = _DS / "logs.jetson-orin"
OUT_ROOT = REPO_ROOT / "result"          # everything lands here, one folder
_EXP = OUT_ROOT / "_export"              # intermediate per-device averaged csvs
DEVICES = [
    ("A100",  _DS / "logs.a100",             _EXP / "a100"),
    ("MAXN",  _JET / "benchmark.maxn_super", _EXP / "maxn"),
    ("25W",   _JET / "benchmark",            _EXP / "j25w"),
    ("15W",   _JET / "benchmark.15w",        _EXP / "j15w"),
]
MODELS = ["bert", "vision", "yolo", "llama"]

# (avg csv file, mean column, std column, axis label). quality col is auto-picked.
PANELS = [
    ("latency.csv",  "end_to_end_sec_mean_mean",   "end_to_end_sec_mean_std",   "latency (s)"),
    ("energy.csv",   "avg_energy_j_mean",          "avg_energy_j_std",          "energy (J)"),
    ("energy.csv",   "avg_power_w_mean",           "avg_power_w_std",           "power (W)"),
    ("hardware.csv", "avg_gpu_mem_used_mb_mean", "avg_gpu_mem_used_mb_std", "GPU mem (MB)"),
    ("hardware.csv", "avg_ram_used_mb_mean",      "avg_ram_used_mb_std",     "System RAM (MB)"),
    ("quality.csv",  None,                         None,                        "quality"),
]
QUALITY_PREF = ["acc_mean", "map_mean", "map50_mean", "glue_score_mean",
                "f1_mean", "exact_match_mean", "rougeL_mean", "perplexity_mean"]


def _exit_sort_key(method: str):
    nums = [int(x) for x in re.findall(r"\d+", str(method))]
    return nums or [10 ** 9]


def _num(series):
    """Parse the exporter's pretty-printed numbers, e.g. '2.225x10^-4' (with a
    unicode times sign), back to float. Plain floats pass through unchanged."""
    import pandas as pd
    s = (series.astype(str)
         .str.replace("×", "x", regex=False)
         .str.replace("x10^", "e", regex=False)
         .str.replace(" ", "", regex=False))
    return pd.to_numeric(s, errors="coerce")


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
    # "method" is the unique row key (yolo has 3 sub-exits per exit number:
    # exit_0_P3/P4/P5 all carry exit=0 — sorting/joining on "exit" alone
    # collapses or cross-joins them). Sort by method's numeric parts.
    if "method" in df.columns:
        return df.sort_values(
            "method", key=lambda s: s.map(lambda m: tuple(_exit_sort_key(m)))
        ).reset_index(drop=True)
    return df.sort_values("exit") if "exit" in df.columns else df


def _method_labels(dfs) -> list:
    """Union of row keys across devices, exit-order sorted. Row key = method
    (keeps yolo sub-exits P3/P4/P5 distinct); falls back to exit numbers."""
    methods = []
    for df in dfs:
        if df is None:
            continue
        keys = df["method"] if "method" in df.columns else df.get("exit", [])
        for m in keys:
            if str(m) not in methods:
                methods.append(str(m))
    return sorted(methods, key=lambda m: tuple(_exit_sort_key(m)))


def double_plot(model: str):
    """One separate figure per metric -> result/{model}/{metric}.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable: {e}")
        return False
    have = {dev: csv for dev, _, csv in DEVICES if (csv / model / "average").exists()}
    if not have:
        return False
    colors = {"A100": "#8172b3", "MAXN": "#55a868", "25W": "#4c72b0", "15W": "#c44e52"}
    out_dir = OUT_ROOT / model
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for fname, mcol, scol, ylabel in PANELS:
        # slug: "GPU mem (MB)" -> "gpu_mem", "latency (s)" -> "latency"
        slug = re.sub(r"\s*\(.*?\)", "", ylabel).strip().lower().replace(" ", "_")
        dev_dfs = {dev: _read_avg(csv_root, model, fname) for dev, _, csv_root in DEVICES}
        labels = _method_labels(dev_dfs.values())
        if not labels:
            continue
        xpos = {m: i for i, m in enumerate(labels)}
        # every metric gets two versions: with A100 (true cross-device scale) +
        # without (A100 dwarfs Jetson, so the without-view makes the Jetson
        # funnel readable). quality kept single (A100 is the reference point).
        variants = ([(f"{slug}_with_a100", ()), (f"{slug}_without_a100", ("A100",))]
                    if slug != "quality" else [(slug, ())])
        for out_slug, exclude in variants:
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            plotted = False
            for dev, _, csv_root in DEVICES:
                if dev in exclude:
                    continue
                df = dev_dfs.get(dev)
                if df is None:
                    continue
                col = mcol or _quality_col(df)
                if col is None or col not in df.columns:
                    continue
                keys = (df["method"] if "method" in df.columns else df["exit"]).astype(str)
                x = keys.map(xpos)
                y = _num(df[col])
                ax.plot(x, y, marker="o", label=dev, color=colors.get(dev))
                scol_eff = scol if (scol and scol in df.columns) else None
                if scol_eff is None and not mcol:  # quality std col follows the picked mean
                    guess = col.replace("_mean", "_std")
                    scol_eff = guess if guess in df.columns else None
                if scol_eff is not None:
                    sd = _num(df[scol_eff])       # variance funnel (cross-task +/-1 std)
                    ax.fill_between(x, y - sd, y + sd, alpha=0.22, color=colors.get(dev),
                                    linewidth=0)
                plotted = True
            if not plotted:
                plt.close(fig)
                continue
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([m.replace("exit_", "") for m in labels],
                               rotation=45 if len(labels) > 8 else 0, fontsize=7)
            ax.set_xlabel("exit")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{model} — {ylabel}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"{out_slug}.png", dpi=120)
            plt.close(fig)
        n += 1
    print(f"[plot] {OUT_ROOT.name}/{model}/*.png ({n} metrics, {'+'.join(have)})")
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
                    if df is None:
                        continue
                    col = mcol or _quality_col(df)
                    if col is None or col not in df.columns:
                        continue
                    # join on "method" — the unique row key. Joining on "exit"
                    # cross-multiplies yolo's 3 sub-exit rows per exit (P3×P5 etc).
                    if "method" not in df.columns:
                        if "exit" not in df.columns:
                            continue
                        df = df.assign(method=df["exit"].astype(str))
                    sub = df[["method", col]].rename(columns={col: f"{dev}_{col}"})
                    merged = sub if merged is None else merged.merge(sub, on="method", how="outer")
            if merged is not None:
                merged = merged.sort_values(
                    "method", key=lambda s: s.map(lambda m: tuple(_exit_sort_key(m)))
                )
                # split method into exit / sub_exit columns for readable copy-paste
                merged.insert(1, "exit", merged["method"].map(
                    lambda m: (_exit_sort_key(m) or [-1])[0]))
                merged.insert(2, "sub_exit", merged["method"].map(
                    lambda m: (_re_sub_exit(m) or "")))
                merged.to_excel(xl, sheet_name=model[:31], index=False)
                wrote += 1
        if wrote == 0:
            pd.DataFrame({"note": ["no data"]}).to_excel(xl, sheet_name="empty", index=False)
    print(f"[xlsx] wrote {path}")


def _re_sub_exit(method: str):
    """'exit_0_P3' -> 'P3'; None when the method has no sub-exit suffix."""
    m = re.search(r"_(P\d)$", str(method))
    return m.group(1) if m else None


# ---- per-model report (yolo.xlsx layout): one xlsx per MODEL, metric sheets,
# each sub-metric a column group with one column per device/mode version --------
# sheet -> averaged csv it reads. Each entry: (display name, averaged mean column).
REPORT_SHEETS = {
    "latency": ("latency.csv", [
        ("E2E latency / sample (s)", "end_to_end_sec_mean_mean"),
        ("Throughput (samples/s)",   "throughput_samples_per_sec_mean")]),
    "energy": ("energy.csv", [
        ("Energy / sample (J)", "avg_energy_j_mean"),
        ("Average power (W)",   "avg_power_w_mean")]),
    "power": ("energy.csv", [
        ("Average power (W)",   "avg_power_w_mean")]),
    "memory": ("hardware.csv", [
        ("GPU memory used (MB)", "avg_gpu_mem_used_mb_mean"),
        ("System RAM used (MB)", "avg_ram_used_mb_mean"),
        ("CPU cores used",       "avg_cpu_cores_used_mean")]),
}


def _legend_df():
    import pandas as pd
    rows = [("Layer", "Exit layer; YOLO uses P3/P4/P5 sub-exits.")]
    rows += [(dev, f"{dev} measurement (mean over all tasks).") for dev, _, _ in DEVICES]
    rows += [("Lower-is-better", "Latency, energy, power, memory, RAM, CPU cores."),
             ("Higher-is-better", "Throughput.")]
    return pd.DataFrame(rows, columns=["Column", "Meaning"])


# device -> (header fill, light body tint), matched to the plot colours.
_DEVCLR = {"A100": ("8172B3", "ECE8F4"), "MAXN": ("55A868", "E6F1EA"),
           "25W": ("4C72B0", "E4EAF3"), "15W": ("C44E52", "F6E5E7")}


def _style_sheet(ws, submetrics, devnames):
    """Merge sub-metric headers; colour baseline, mode-value and Δ% columns;
    borders; freeze panes. Group layout = A100, then (mode, Δ%) per other mode."""
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    base_dev, other_devs = devnames[0], devnames[1:]
    gw = 1 + 2 * len(other_devs)
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")
    white = Font(bold=True, color="FFFFFF")
    grey_hd, grey_bd = "A6A6A6", "ECECEC"      # Δ% header / body tint
    ws.merge_cells("A1:A2")
    a = ws["A1"]; a.value = "Layer"; a.font = white; a.alignment = center; a.border = border
    a.fill = PatternFill("solid", fgColor="404040")
    col = 2
    for disp, _c in submetrics:
        c0, c1 = col, col + gw - 1
        ws.merge_cells(start_row=1, start_column=c0, end_row=1, end_column=c1)
        h = ws.cell(row=1, column=c0)
        h.value = disp; h.font = white; h.alignment = center; h.border = border
        h.fill = PatternFill("solid", fgColor="595959")
        b = ws.cell(row=2, column=c0)          # baseline header
        b.font = white; b.alignment = center; b.border = border
        b.fill = PatternFill("solid", fgColor=_DEVCLR[base_dev][0])
        cc = c0 + 1
        for dev in other_devs:
            v = ws.cell(row=2, column=cc)       # mode header
            v.font = white; v.alignment = center; v.border = border
            v.fill = PatternFill("solid", fgColor=_DEVCLR[dev][0])
            g = ws.cell(row=2, column=cc + 1)   # Δ% header
            g.font = Font(bold=True, color="404040"); g.alignment = center; g.border = border
            g.fill = PatternFill("solid", fgColor=grey_hd)
            cc += 2
        col = c1 + 1
    for r in range(3, ws.max_row + 1):
        lc = ws.cell(row=r, column=1)
        lc.font = Font(bold=True); lc.alignment = center; lc.border = border
        col = 2
        for _disp, _c in submetrics:
            bc = ws.cell(row=r, column=col)     # baseline value
            bc.border = border; bc.alignment = center
            bc.fill = PatternFill("solid", fgColor=_DEVCLR[base_dev][1])
            if isinstance(bc.value, (int, float)):
                bc.number_format = "0.####"
            cc = col + 1
            for dev in other_devs:
                vc = ws.cell(row=r, column=cc)  # mode value
                vc.border = border; vc.alignment = center
                vc.fill = PatternFill("solid", fgColor=_DEVCLR[dev][1])
                if isinstance(vc.value, (int, float)):
                    vc.number_format = "0.####"
                dc = ws.cell(row=r, column=cc + 1)  # Δ%
                dc.border = border; dc.alignment = center
                dc.fill = PatternFill("solid", fgColor=grey_bd)
                if isinstance(dc.value, (int, float)):
                    dc.number_format = '+0.0"%";-0.0"%"'
                cc += 2
            col += gw
    ws.column_dimensions["A"].width = 9
    for cc in range(2, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(cc)].width = 9
    ws.freeze_panes = "B3"


def _style_legend(ws):
    from openpyxl.styles import PatternFill, Font, Alignment
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="404040")
        c.alignment = Alignment(horizontal="left", vertical="center")
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 60


def write_report():
    """One xlsx PER MODEL under result.report/, laid out like yolo.xlsx: metric
    sheets (latency/energy/memory) + legend, each sub-metric a column group with
    one column per device/mode version (A100, MAXN, 25W, 15W)."""
    import pandas as pd
    try:
        import openpyxl  # noqa: F401
    except Exception as e:
        print(f"[report] openpyxl missing ({e}); pip install openpyxl"); return
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    devnames = [dev for dev, _, _ in DEVICES]
    for model in MODELS:
        sheets = {}
        for sheet, (fname, submetrics) in REPORT_SHEETS.items():
            devdf = {dev: _read_avg(csv_root, model, fname) for dev, _, csv_root in DEVICES}
            if all(v is None for v in devdf.values()):
                continue
            labels = _method_labels(devdf.values())
            if not labels:
                continue
            # yolo.xlsx layout = 2-row header, then data. Per sub-metric the columns
            # are: A100 (baseline), then (value, delta%) for each other mode, where
            # delta% = (mode - A100)/A100 * 100. Written raw (header=False) to dodge
            # pandas' MultiIndex-columns limitation.
            base_dev, other_devs = devnames[0], devnames[1:]
            gw = 1 + 2 * len(other_devs)   # baseline + (value, delta) per other mode
            row1, row2, lut = ["Layer"], [""], {}
            for disp, col in submetrics:
                row1 += [disp] + [""] * (gw - 1)
                row2.append(base_dev)
                for dev in other_devs:
                    row2 += [dev, "Δ%"]
                for dev in devnames:
                    df = devdf.get(dev)
                    if df is not None and col in df.columns:
                        keys = (df["method"] if "method" in df.columns else df["exit"]).astype(str)
                        lut[(disp, dev)] = dict(zip(keys, _num(df[col])))
                    else:
                        lut[(disp, dev)] = {}
            rows = [row1, row2]
            for lab in labels:
                r = [lab.replace("exit_", "")]
                for disp, _col in submetrics:
                    b = lut[(disp, base_dev)].get(str(lab))
                    r.append(b)
                    for dev in other_devs:
                        v = lut[(disp, dev)].get(str(lab))
                        r.append(v)
                        ok = (b is not None and v is not None and pd.notna(b)
                              and pd.notna(v) and b != 0)
                        r.append((v - b) / b * 100.0 if ok else None)
                rows.append(r)
            sheets[sheet] = rows
        if not sheets:
            continue
        path = OUT_ROOT / f"{model}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as xl:
            _legend_df().to_excel(xl, sheet_name="legend", index=False)
            for sheet, rows in sheets.items():
                pd.DataFrame(rows).to_excel(xl, sheet_name=sheet, header=False, index=False)
                _style_sheet(xl.sheets[sheet], REPORT_SHEETS[sheet][1], devnames)
            _style_legend(xl.sheets["legend"])
        print(f"[report] wrote {path}")


# ---- summary_table.png: Δ% vs A100 for EVERY exit layer --------------------
# (metric, csv, baseline mean col, baseline header, baseline value fmt)
_SUMMARY_METRICS = [
    ("Latency", "latency.csv", "end_to_end_sec_mean_mean", "Lat A100(s)", "{:.4g}"),
    ("Energy",  "energy.csv",  "avg_energy_j_mean",        "Eng A100(J)", "{:.3g}"),
    ("Power",   "energy.csv",  "avg_power_w_mean",         "Pwr A100(W)", "{:.1f}"),
]


def summary_table(model: str):
    """One table image per model -> result/{model}/summary_table.png. Rows = ALL
    exit layers; per metric: A100 baseline + Δ% for MAXN/25W/15W (every mode)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[summary] matplotlib unavailable: {e}"); return
    import pandas as pd
    base_dev = DEVICES[0][0]
    other = [d for d, _, _ in DEVICES[1:]]
    # per (metric, dev) -> {method: value}
    lut, labels = {}, []
    for _mname, fname, col, _hd, _fmt in _SUMMARY_METRICS:
        for dev, _, csv_root in DEVICES:
            df = _read_avg(csv_root, model, fname)
            if df is None or col not in df.columns:
                lut[(col, dev)] = {}; continue
            keys = (df["method"] if "method" in df.columns else df["exit"]).astype(str)
            lut[(col, dev)] = dict(zip(keys, _num(df[col])))
            for k in keys:
                if k not in labels:
                    labels.append(k)
    if not labels:
        return
    labels = sorted(labels, key=lambda m: tuple(_exit_sort_key(m)))
    header = ["Exit"]
    for _mn, _fn, _col, hd, _fmt in _SUMMARY_METRICS:
        header += [hd] + [f"{d} Δ" for d in other]
    rows = []
    for lab in labels:
        r = [lab.replace("exit_", "")]
        for _mn, _fn, col, _hd, fmt in _SUMMARY_METRICS:
            b = lut[(col, base_dev)].get(lab)
            r.append(fmt.format(b) if b is not None and pd.notna(b) else "-")
            for dev in other:
                v = lut[(col, dev)].get(lab)
                ok = (b is not None and v is not None and pd.notna(b)
                      and pd.notna(v) and b != 0)
                r.append(f"{(v - b) / b * 100:+.0f}%" if ok else "-")
        rows.append(r)
    fig, ax = plt.subplots(figsize=(1.05 * len(header), 0.42 * (len(rows) + 1) + 0.6))
    ax.axis("off")
    ax.set_title(f"{model.upper()} — Δ% vs A100 baseline (all exit layers)",
                 fontweight="bold", fontsize=12, pad=12)
    tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.3)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#BFBFBF")
        if r == 0:
            cell.set_facecolor("#404040"); cell.set_text_props(color="w", fontweight="bold")
    fig.tight_layout()
    out_dir = OUT_ROOT / model
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "summary_table.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[summary] result/{model}/summary_table.png ({len(rows)} layers)")


def run(force: bool = False):
    for _dev, bench_root, csv_root in DEVICES:
        export_device(bench_root, csv_root, force)
    for model in MODELS:
        double_plot(model)
        summary_table(model)
    write_xlsx()
    write_report()
    print(f"[done] plots + per-metric report under {OUT_ROOT}")


def main():
    ap = argparse.ArgumentParser(description="A100 vs Jetson comparison (reuses benchmark export).")
    ap.add_argument("--force", action="store_true", help="re-export csvs even if they exist")
    run(force=ap.parse_args().force)


if __name__ == "__main__":
    main()
