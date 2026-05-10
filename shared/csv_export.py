"""Generic CSV exporter. Reads benchmark_results.json (multiple files) → 4 CSVs.

Columns adapt per modality:
- LLaMa: TTFT, per-token latency, ROUGE
- BERT/Vision: per-sample latency, accuracy/F1
- All: VRAM, power, GPU%, energy

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


def _pct(val, base):
    try:
        return f"{(float(val) - float(base)) / float(base) * 100:.2f}" if base else "0.00"
    except Exception:
        return ""


def _delta(val, base):
    try:
        return f"{float(val) - float(base):.6f}"
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
    """Load benchmark_results.json. Returns the `aggregate` dict (or whole file)."""
    data = json.loads(Path(file).read_text(encoding="utf-8"))
    return data.get("aggregate", data)


def write_benchmark_csvs(
    results_files: Dict[str, Union[str, Path]],
    out_dir: Union[str, Path],
    baseline_key: Optional[str] = None,
    method_order: Optional[List[str]] = None,
) -> None:
    """Read N benchmark_results.json files, write 4 averaged-vs-baseline CSVs.

    results_files: {method_name: path/to/benchmark_results.json}
    baseline_key:  one of the method names; if set, deltas are computed vs it
    method_order:  optional explicit ordering; default = dict insertion order
    """
    out_dir = Path(out_dir)
    methods = {k: _load(v) for k, v in results_files.items()}
    keys = method_order or list(methods.keys())

    bl = methods.get(baseline_key) if baseline_key else None

    # ------- latency -----------------------------------------------------
    lat_fields = [
        "method",
        "ttft_sec_mean",         "delta_ttft_pct",
        "end_to_end_sec_mean",   "delta_e2e_pct",
        "per_sample_sec_mean",   "delta_per_sample_pct",
        "throughput_samples_per_sec",
    ]
    lat_rows = []
    for k in keys:
        m = methods[k]
        lat_rows.append({
            "method": k,
            "ttft_sec_mean":              _f(m.get("ttft_sec_mean", 0)),
            "delta_ttft_pct":             _pct(m.get("ttft_sec_mean", 0), bl.get("ttft_sec_mean", 0)) if bl else "",
            "end_to_end_sec_mean":        _f(m.get("end_to_end_sec_mean", 0)),
            "delta_e2e_pct":              _pct(m.get("end_to_end_sec_mean", 0), bl.get("end_to_end_sec_mean", 0)) if bl else "",
            "per_sample_sec_mean":        _f(m.get("per_sample_sec_mean", 0)),
            "delta_per_sample_pct":       _pct(m.get("per_sample_sec_mean", 0), bl.get("per_sample_sec_mean", 0)) if bl else "",
            "throughput_samples_per_sec": _f(m.get("throughput_samples_per_sec", 0), 3),
        })
    _write(out_dir / "latency.csv", lat_fields, lat_rows)

    # ------- energy ------------------------------------------------------
    eng_fields = [
        "method",
        "total_energy_j",     "delta_energy_pct",
        "joules_per_sample",  "delta_jps_pct",
        "avg_power_w",        "delta_power_w",
    ]
    eng_rows = []
    for k in keys:
        m = methods[k]
        eng_rows.append({
            "method": k,
            "total_energy_j":     _f(m.get("total_energy_j", 0), 4),
            "delta_energy_pct":   _pct(m.get("total_energy_j", 0), bl.get("total_energy_j", 0)) if bl else "",
            "joules_per_sample":  _f(m.get("joules_per_sample", 0)),
            "delta_jps_pct":      _pct(m.get("joules_per_sample", 0), bl.get("joules_per_sample", 0)) if bl else "",
            "avg_power_w":        _f(m.get("avg_power_w", 0), 2),
            "delta_power_w":      _delta(m.get("avg_power_w", 0), bl.get("avg_power_w", 0)) if bl else "",
        })
    _write(out_dir / "energy.csv", eng_fields, eng_rows)

    # ------- quality (whatever quality fields appear) -------------------
    quality_fields = [
        "accuracy", "f1", "rouge2_f1", "rougeL_f1", "perplexity",
        "top1_acc", "top5_acc", "mcc", "spearman_corr",
    ]
    present_q = sorted({
        f for m in methods.values() for f in m.keys() if f in quality_fields
    })
    qual_rows = []
    for k in keys:
        m = methods[k]
        row = {"method": k}
        for q in present_q:
            row[q] = _f(m.get(q, 0), 4)
            if bl and q in bl:
                row[f"delta_{q}_pct"] = _pct(m.get(q, 0), bl.get(q, 0))
        qual_rows.append(row)
    qual_columns = ["method"] + present_q + [f"delta_{q}_pct" for q in present_q if bl]
    _write(out_dir / "quality.csv", qual_columns, qual_rows)

    # ------- hardware ----------------------------------------------------
    hw_fields = [
        "method",
        "avg_vram_allocated_gb",
        "avg_ram_used_gb",
        "avg_cpu_pct",
        "avg_gpu_util_pct",
        "avg_gpu_mem_util_pct",
        "avg_power_w",
        "avg_gpu_sm_clock_mhz",
        "avg_gpu_mem_clock_mhz",
    ]
    hw_rows = []
    for k in keys:
        m = methods[k]
        hw_rows.append({
            "method": k,
            "avg_vram_allocated_gb": _f(m.get("avg_vram_allocated_gb", 0), 3),
            "avg_ram_used_gb":       _f(m.get("avg_ram_used_gb", 0), 3),
            "avg_cpu_pct":           _f(m.get("avg_cpu_pct", 0), 2),
            "avg_gpu_util_pct":      _f(m.get("avg_gpu_util_pct", 0), 2),
            "avg_gpu_mem_util_pct":  _f(m.get("avg_gpu_mem_util_pct", 0), 2),
            "avg_power_w":           _f(m.get("avg_power_w", 0), 2),
            "avg_gpu_sm_clock_mhz":  _f(m.get("avg_gpu_sm_clock_mhz", 0), 0),
            "avg_gpu_mem_clock_mhz": _f(m.get("avg_gpu_mem_clock_mhz", 0), 0),
        })
    _write(out_dir / "hardware.csv", hw_fields, hw_rows)
