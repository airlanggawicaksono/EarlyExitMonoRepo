"""Generic CSV exporter. Reads benchmark_results.json (multiple files) → 4 CSVs.

Columns adapt per modality:
- LLaMa: TTFT, per-token latency, ROUGE
- BERT/Vision: per-sample latency, accuracy/F1
- All: VRAM, power, GPU clocks, energy

Usage:
    write_benchmark_csvs(
        results_files={"exit_8": "logs/exit_8/benchmark_results.json", ...},
        baseline_key="baseline",
        out_dir="results/",
    )
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Union


def _f(val, decimals=6):
    try:
        return f"{float(val):.{decimals}f}"
    except Exception:
        return ""


def _g(val, sig=3):
    """3-sig-fig format. 0.000182 -> 0.000182, 0.1135 -> 0.114, 17.583 -> 17.6."""
    try:
        return f"{float(val):.{sig}g}"
    except Exception:
        return ""


def _f4(val):
    """4-decimal fixed format for averages. 0.113532 -> 0.1135, 17.583 -> 17.5830."""
    try:
        return f"{float(val):.4f}"
    except Exception:
        return ""


def _e3(val):
    """Scientific notation as M×10^E. 0.0364 -> 3.640×10^-2, 0 -> 0."""
    try:
        v = float(val)
        if v == 0.0:
            return "0"
        s = f"{v:.3e}"
        mantissa, exp = s.split("e")
        return f"{mantissa}×10^{int(exp)}"
    except Exception:
        return ""


def _g6(val):
    """6-sig-fig for sub-ms values. 0.001601 -> 0.001601, 0.000182 -> 0.000182."""
    try:
        return f"{float(val):.6g}"
    except Exception:
        return ""


def _write(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[csv_export] wrote {path}")


def _load(file: Union[str, Path]) -> Dict:
    """Load hw_results.json. Returns aggregate dict + std_<key> from samples."""
    data = json.loads(Path(file).read_text(encoding="utf-8"))
    agg = dict(data.get("aggregate", data))
    _inject_sample_stds(agg, data.get("samples", []))
    return agg


# Map aggregate key -> sample-list key for per-sample std computation
_SAMPLE_STD_PAIRS = (
    ("avg_energy_j",          "energy_j"),
    ("joules_per_sample",     "energy_j"),
    ("avg_power_w",           "power_w"),
    ("avg_vram_allocated_mb", "vram_allocated_mb"),
    ("avg_vram_reserved_mb",  "vram_reserved_mb"),
    ("avg_proc_vram_used_mb", "proc_vram_used_mb"),
    ("avg_ram_used_mb",       "ram_used_mb"),
    ("avg_cpu_cores_used",    "cpu_cores_used"),
    ("avg_gpu_sm_clock_mhz",  "gpu_sm_clock_mhz"),
    ("avg_gpu_mem_clock_mhz", "gpu_mem_clock_mhz"),
    ("ttft_sec_mean",         "ttft_sec"),
    ("end_to_end_sec_mean",   "end_to_end_sec"),
    ("per_sample_sec_mean",   "end_to_end_sec"),
)


def _pstd(vals: List) -> float:
    """Population std over numeric values. 0.0 if <2 valid values."""
    nums = [float(v) for v in vals if isinstance(v, (int, float))]
    if len(nums) < 2:
        return 0.0
    mean = sum(nums) / len(nums)
    return (sum((x - mean) ** 2 for x in nums) / len(nums)) ** 0.5


def _inject_sample_stds(agg: Dict, samples: List[Dict]) -> None:
    """Compute std_<agg_key> from per-sample distributions and write into agg."""
    if not samples:
        return
    for agg_k, samp_k in _SAMPLE_STD_PAIRS:
        if agg_k in agg:
            agg[f"std_{agg_k}"] = _pstd([s.get(samp_k) for s in samples])
    if "edp_j_s" in agg:
        agg["std_edp_j_s"] = _pstd(
            [s.get("energy_j", 0) * s.get("end_to_end_sec", 0) for s in samples]
        )


def _load_device_caps(src: Union[str, Path]) -> Dict:
    """Read top-level device_caps from hw_results.json (not aggregate)."""
    src = Path(src)
    hw = src / "hw_results.json" if src.is_dir() else (
        src if src.name == "hw_results.json" else src.parent / "hw_results.json"
    )
    if not hw.exists():
        return {}
    data = json.loads(hw.read_text(encoding="utf-8"))
    return data.get("device_caps", {})


def _load_run_dir(run_dir: Union[str, Path]) -> Dict:
    """Load both hw_results.json + quality_results.json from a run dir, merge.

    Modern split-pass output:
        run_dir/hw_results.json       -> latency + memory + energy
        run_dir/quality_results.json  -> accuracy / F1 / mAP / ROUGE / etc

    Falls back to old single benchmark_results.json if hw_results.json absent.
    """
    run_dir = Path(run_dir)
    merged: Dict = {}

    hw_file = run_dir / "hw_results.json"
    if hw_file.exists():
        merged.update(_load(hw_file))

    q_file = run_dir / "quality_results.json"
    if q_file.exists():
        q = json.loads(q_file.read_text(encoding="utf-8"))
        if "per_exit" in q:
            merged["per_exit"] = q["per_exit"]
        if "metrics" in q:
            merged.update(
                {k: v for k, v in q["metrics"].items() if isinstance(v, (int, float))}
            )
        for k, v in q.items():
            if isinstance(v, (int, float)):
                merged[k] = v

    legacy = run_dir / "benchmark_results.json"
    if not merged and legacy.exists():
        merged.update(_load(legacy))

    return merged


def write_benchmark_csvs(
    results_files: Dict[str, Union[str, Path]],
    out_dir: Union[str, Path],
    baseline_key: Optional[str] = None,
    method_order: Optional[List[str]] = None,
) -> None:
    """Read N run files OR run dirs, write latency/energy/device/quality/hardware CSVs.

    results_files: {method_name: path}
        path can be:
          - benchmark_results.json (legacy, single-pass)
          - hw_results.json (modern split-pass — quality auto-loaded if same dir)
          - run dir (loads both hw_results.json + quality_results.json)
    method_order:  explicit ordering; default = dict insertion order
    """
    out_dir = Path(out_dir)

    def _smart_load(p: Union[str, Path]) -> Dict:
        p = Path(p)
        if p.is_dir():
            return _load_run_dir(p)
        if p.name == "hw_results.json":
            return _load_run_dir(p.parent)
        return _load(p)

    methods = {k: _smart_load(v) for k, v in results_files.items()}
    keys = method_order or list(methods.keys())

    # ------- latency (ttft + e2e + throughput; per_sample_sec dropped = same as e2e) --
    lat_fields = [
        "method",
        "ttft_sec_mean", "std_ttft_sec_mean",
        "end_to_end_sec_mean", "std_end_to_end_sec_mean",
        "throughput_samples_per_sec",
    ]
    lat_rows = []
    for k in keys:
        m = methods[k]
        lat_rows.append({
            "method": k,
            "ttft_sec_mean": _g6(m.get("ttft_sec_mean", 0)),
            "std_ttft_sec_mean": _e3(m.get("std_ttft_sec_mean", 0)),
            "end_to_end_sec_mean": _g6(m.get("end_to_end_sec_mean", 0)),
            "std_end_to_end_sec_mean": _e3(m.get("std_end_to_end_sec_mean", 0)),
            "throughput_samples_per_sec": _f4(m.get("throughput_samples_per_sec", 0)),
        })
    _write(out_dir / "latency.csv", lat_fields, lat_rows)

    # ------- energy (avg_energy_j only; joules_per_sample dropped = same value) --
    eng_fields = [
        "method",
        "avg_energy_j", "std_avg_energy_j",
        "avg_power_w", "std_avg_power_w",
        "edp_j_s", "std_edp_j_s",
    ]
    eng_rows = []
    for k in keys:
        m = methods[k]
        eng_rows.append({
            "method": k,
            "avg_energy_j": _f4(m.get("avg_energy_j", 0)),
            "std_avg_energy_j": _e3(m.get("std_avg_energy_j", 0)),
            "avg_power_w": _f4(m.get("avg_power_w", 0)),
            "std_avg_power_w": _e3(m.get("std_avg_power_w", 0)),
            "edp_j_s": _g6(m.get("edp_j_s", 0)),
            "std_edp_j_s": _e3(m.get("std_edp_j_s", 0)),
        })
    _write(out_dir / "energy.csv", eng_fields, eng_rows)

    # ------- device caps (from top-level of hw_results.json) ------------
    # Pre-load once; also inject computed VRAM fields into methods for hardware CSV
    all_caps = {k: _load_device_caps(results_files[k]) for k in keys}
    for k in keys:
        total = float(all_caps[k].get("gpu_vram_total_mb", 0))
        used = float(methods[k].get("avg_proc_vram_used_mb", 0) or 0)
        methods[k]["gpu_vram_total_mb"] = total
        if total > 0:
            methods[k]["avg_vram_free_mb"] = total - used
            methods[k]["std_avg_vram_free_mb"] = methods[k].get("std_avg_proc_vram_used_mb", 0)

    dev_fields = [
        "method",
        "gpu_name",
        "gpu_vram_total_mb",
        "gpu_cuda_capability",
        "gpu_multi_processor_count",
        "gpu_power_monitoring",
        "cpu_max_cores_physical",
        "cpu_max_cores_logical",
        "cpu_max_freq_mhz",
        "ram_total_mb",
        "papi_available",
    ]
    dev_rows = []
    for k in keys:
        row = {"method": k}
        for f in dev_fields[1:]:
            row[f] = all_caps[k].get(f, "")
        dev_rows.append(row)
    _write(out_dir / "device.csv", dev_fields, dev_rows)

    # ------- quality (auto-detect numeric metric keys per-run) ----------
    QUALITY_SKIP = {"force_exit", "n_samples", "n_tokens", "nll_sum"}

    def _load_quality(src: Union[str, Path]) -> Dict:
        src = Path(src)
        q_file = src / "quality_results.json" if src.is_dir() else src.parent / "quality_results.json"
        if not q_file.exists():
            return {}
        return json.loads(q_file.read_text(encoding="utf-8"))

    quality_raw = {k: _load_quality(v) for k, v in results_files.items()}
    quality_nums = {
        k: {kk: vv for kk, vv in q.items()
            if isinstance(vv, (int, float)) and kk not in QUALITY_SKIP}
        for k, q in quality_raw.items()
    }
    present_q = sorted({f for m in quality_nums.values() for f in m.keys()})

    qual_columns = ["method", "main_metric"] + present_q
    qual_rows = []
    for k in keys:
        m = quality_nums.get(k, {})
        row = {"method": k, "main_metric": quality_raw.get(k, {}).get("main_metric", "")}
        for q in present_q:
            row[q] = _f(m.get(q), 4) if q in m else ""
        qual_rows.append(row)
    _write(out_dir / "quality.csv", qual_columns, qual_rows)

    # ------- hardware — grouped by component with blank separator cols -------
    # gpu_vram_total_mb / avg_vram_free_mb injected into methods above
    hw_groups = [
        ("-- GPU --", [
            ("gpu_vram_total_mb",     "gpu_vram_total_mb"),
            ("avg_gpu_mem_mb",        "avg_proc_vram_used_mb"),
            ("avg_vram_allocated_mb", "avg_vram_allocated_mb"),
            ("avg_vram_reserved_mb",  "avg_vram_reserved_mb"),
            ("avg_vram_free_mb",      "avg_vram_free_mb"),
            ("avg_gpu_clock_mhz",     "avg_gpu_sm_clock_mhz"),
            ("avg_gpu_mem_clock_mhz", "avg_gpu_mem_clock_mhz"),
        ]),
        ("-- CPU --", [
            ("avg_cpu_cores_used", "avg_cpu_cores_used"),
        ]),
        ("-- RAM --", [
            ("avg_ram_used_mb", "avg_ram_used_mb"),
        ]),
    ]
    hw_fields = ["method"]
    for sep, group in hw_groups:
        hw_fields.append(sep)
        for col, _ in group:
            hw_fields.append(col)
            hw_fields.append(f"std_{col}")
    hw_rows = []
    for k in keys:
        m = methods[k]
        row = {"method": k}
        for sep, group in hw_groups:
            row[sep] = ""
            for col, src in group:
                row[col] = _f4(m.get(src, 0))
                row[f"std_{col}"] = _e3(m.get(f"std_{src}", 0))
        hw_rows.append(row)
    _write(out_dir / "hardware.csv", hw_fields, hw_rows)


# ============================================================================
# Cross-task aggregation: results/{model}/{task}/*.csv → results/{model}/average/*.csv
# ============================================================================

import re as _re


def _exit_num(method: str) -> int:
    m = _re.search(r"\d+", str(method))
    return int(m.group()) if m else -1


def _is_skip_col(col: str) -> bool:
    """Drop bookkeeping / unwanted numeric cols from cross-task aggregation."""
    if col in {"exit", "force_exit", "n_samples"}:
        return True
    if col.startswith(("delta_", "max_", "std_")):
        return True
    if col.endswith("_pct"):
        return True
    return False


def _fmt_f4_series(s):
    return s.map(_f4)


def _fmt_e3_series(s):
    return s.map(_e3)


_SUB_MS_COLS = {
    "ttft_sec_mean", "end_to_end_sec_mean", "per_sample_sec_mean", "edp_j_s"
}


def _aggregate_csv_type(
    model_dir: Path, csv_name: str, avg_dir: Path
) -> Optional[Path]:
    """Read all `{task}/{csv_name}.csv` under model_dir → write paired
    `{col}_mean,{col}_std` per method to `{avg_dir}/{csv_name}.csv`.

    Drops: delta_*, max_*, exit, force_exit, n_samples. Adds single `n_tasks` col.
    """
    import pandas as pd

    files = [f for f in model_dir.rglob(f"{csv_name}.csv") if avg_dir not in f.parents]
    if not files:
        return None

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["task"] = f.parent.name
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)
    if "method" not in big.columns:
        return None

    numeric_all = big.select_dtypes(include="number").columns.tolist()
    numeric = [c for c in numeric_all if not _is_skip_col(c)]
    if not numeric:
        return None

    n_tasks = big.groupby("method")["task"].nunique().rename("n_tasks")
    mean_df = big.groupby("method")[numeric].mean()
    std_df = big.groupby("method")[numeric].std().fillna(0.0)

    out_cols = {}
    for c in numeric:
        fmt = (lambda s: s.map(_g6)) if c in _SUB_MS_COLS else _fmt_f4_series
        out_cols[f"{c}_mean"] = fmt(mean_df[c])
        out_cols[f"{c}_std"] = _fmt_e3_series(std_df[c])
    paired = pd.DataFrame(out_cols)
    paired.insert(0, "n_tasks", n_tasks)
    paired = paired.reset_index()
    paired.insert(1, "exit", paired["method"].map(_exit_num))
    paired = paired.sort_values("exit").reset_index(drop=True)

    avg_dir.mkdir(parents=True, exist_ok=True)
    out_path = avg_dir / f"{csv_name}.csv"
    paired.to_csv(out_path, index=False)
    print(f"[csv_export] wrote {out_path}")
    return out_path


def write_average_csvs(model_results_dir: Union[str, Path]) -> None:
    """For each per-task CSV under `results/{model}/`, aggregate per-exit
    (mean+std+count) across tasks → `results/{model}/average/{type}.csv`.

    Operates on already-exported per-task CSVs, not raw JSON.
    """
    model_dir = Path(model_results_dir)
    avg_dir = model_dir / "average"
    for csv_name in ("latency", "energy", "quality", "hardware"):
        _aggregate_csv_type(model_dir, csv_name, avg_dir)
