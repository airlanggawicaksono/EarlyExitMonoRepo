"""
Export benchmark_results.json → 3 CSVs with delta vs baseline.

Usage:
    python benchmark_to_csv.py <benchmark_results.json> [output_dir]

Outputs:
    latency.csv   — TTFT, E2E, per-token latency, tokens/s + deltas
    energy.csv    — total energy, J/tok, power, VRAM, RAM, GPU% + deltas
    quality.csv   — ROUGE-2, ROUGE-L, perplexity, accuracy + deltas
"""

import csv
import json
import os
import sys


def _pct(val, base):
    return f"{(val - base) / base * 100:.2f}" if base else "0.00"

def _f(val, decimals=6):
    return f"{val:.{decimals}f}"


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {path}")


def main(results_path, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)

    with open(results_path, encoding="utf-8") as f:
        d = json.load(f)

    bl       = d["baseline_latency"]
    per_exit = d.get("per_exit", {})
    ee       = d.get("ee_latency", {})
    quality  = d.get("quality", {})

    def _exit_sort(item):
        k = item[0]
        if k.startswith("exit_"):
            try: return int(k.split("_")[1])
            except: pass
        return 9999

    # Order: exit_8 → exit_16 → exit_24 → dynamic_ee → baseline
    methods = {}
    for key, val in sorted(per_exit.items(), key=_exit_sort):
        q_key = "base_final" if key == "base" else key
        methods[key] = {"lat": val, "qual": quality.get(q_key)}
    methods["dynamic_ee"] = {"lat": ee, "qual": None}
    methods["baseline"]   = {"lat": bl, "qual": None}

    # Baseline reference values
    bl_ttft    = bl["ttft_sec_mean"]
    bl_per_tok = bl["per_token_latency_sec_mean"]
    bl_e2e     = bl["end_to_end_sec_mean"]
    bl_tok_s   = 1.0 / bl_per_tok if bl_per_tok > 0 else 0.0
    bl_energy  = bl["total_energy_j"]
    bl_jpt     = bl["joules_per_token"]
    bl_power   = bl["avg_power_w"]
    bl_r2      = bl["rouge2_f1"]
    bl_rl      = bl["rougeL_f1"]

    # ------------------------------------------------------------------ #
    # Latency CSV
    # ------------------------------------------------------------------ #
    lat_fields = [
        "method",
        "avg_ttft_s",        "delta_ttft_s",        "delta_ttft_pct",
        "avg_e2e_s",         "delta_e2e_s",          "delta_e2e_pct",
        "avg_per_tok_s",     "delta_per_tok_s",      "delta_per_tok_pct",
        "avg_tok_per_s",     "delta_tok_per_s",      "delta_tok_per_s_pct",
    ]
    lat_rows = []
    for method, data in methods.items():
        lat     = data["lat"]
        ttft    = lat["ttft_sec_mean"]
        e2e     = lat["end_to_end_sec_mean"]
        per_tok = lat["per_token_latency_sec_mean"]
        tok_s   = 1.0 / per_tok if per_tok > 0 else 0.0
        lat_rows.append({
            "method":               method,
            "avg_ttft_s":           _f(ttft),
            "delta_ttft_s":         _f(ttft - bl_ttft),
            "delta_ttft_pct":       _pct(ttft, bl_ttft),
            "avg_e2e_s":            _f(e2e),
            "delta_e2e_s":          _f(e2e - bl_e2e),
            "delta_e2e_pct":        _pct(e2e, bl_e2e),
            "avg_per_tok_s":        _f(per_tok),
            "delta_per_tok_s":      _f(per_tok - bl_per_tok),
            "delta_per_tok_pct":    _pct(per_tok, bl_per_tok),
            "avg_tok_per_s":        _f(tok_s, 3),
            "delta_tok_per_s":      _f(tok_s - bl_tok_s, 3),
            "delta_tok_per_s_pct":  _pct(tok_s, bl_tok_s),
        })
    write_csv(os.path.join(output_dir, "latency.csv"), lat_fields, lat_rows)

    # ------------------------------------------------------------------ #
    # Energy CSV
    # ------------------------------------------------------------------ #
    energy_fields = [
        "method",
        "total_energy_j",    "delta_energy_j",      "delta_energy_pct",
        "joules_per_token",  "delta_jpt",            "delta_jpt_pct",
        "avg_power_w",       "delta_power_w",
        "avg_vram_gb",
        "avg_gpu_mem_util_pct",
        "avg_ram_gb",
        "avg_cpu_pct",
        "avg_gpu_util_pct",
        "avg_gpu_sm_clock_mhz",
        "avg_gpu_mem_clock_mhz",
    ]
    energy_rows = []
    for method, data in methods.items():
        lat      = data["lat"]
        energy_j = lat["total_energy_j"]
        jpt      = lat["joules_per_token"]
        power    = lat["avg_power_w"]
        energy_rows.append({
            "method":              method,
            "total_energy_j":      _f(energy_j, 4),
            "delta_energy_j":      _f(energy_j - bl_energy, 4),
            "delta_energy_pct":    _pct(energy_j, bl_energy),
            "joules_per_token":    _f(jpt),
            "delta_jpt":           _f(jpt - bl_jpt),
            "delta_jpt_pct":       _pct(jpt, bl_jpt),
            "avg_power_w":         _f(power, 2),
            "delta_power_w":       _f(power - bl_power, 2),
            "avg_vram_gb":          _f(lat.get("avg_vram_gb", 0), 3),
            "avg_gpu_mem_util_pct": _f(lat.get("avg_gpu_mem_util_pct", 0), 2),
            "avg_ram_gb":           _f(lat.get("avg_ram_gb", 0), 3),
            "avg_cpu_pct":          _f(lat.get("avg_cpu_pct", 0), 2),
            "avg_gpu_util_pct":     _f(lat.get("avg_gpu_util_pct", 0), 2),
            "avg_gpu_sm_clock_mhz": lat.get("avg_gpu_sm_clock_mhz", ""),
            "avg_gpu_mem_clock_mhz": lat.get("avg_gpu_mem_clock_mhz", ""),
        })
    write_csv(os.path.join(output_dir, "energy.csv"), energy_fields, energy_rows)

    # ------------------------------------------------------------------ #
    # Quality / Accuracy CSV
    # ------------------------------------------------------------------ #
    qual_fields = [
        "method",
        "rouge2_f1",   "delta_rouge2",   "delta_rouge2_pct",
        "rougeL_f1",   "delta_rougeL",   "delta_rougeL_pct",
        "perplexity",
        "accuracy",
    ]
    qual_rows = []
    for method, data in methods.items():
        lat  = data["lat"]
        qual = data["qual"]
        r2   = lat.get("rouge2_f1", 0.0)
        rl   = lat.get("rougeL_f1", 0.0)
        qual_rows.append({
            "method":           method,
            "rouge2_f1":        _f(r2, 4),
            "delta_rouge2":     _f(r2 - bl_r2, 4),
            "delta_rouge2_pct": _pct(r2, bl_r2) if bl_r2 else "",
            "rougeL_f1":        _f(rl, 4),
            "delta_rougeL":     _f(rl - bl_rl, 4),
            "delta_rougeL_pct": _pct(rl, bl_rl) if bl_rl else "",
            "perplexity":       _f(qual["perplexity"], 2) if qual else "",
            "accuracy":         _f(qual["accuracy"], 4)   if qual else "",
        })
    write_csv(os.path.join(output_dir, "quality.csv"), qual_fields, qual_rows)

    # ------------------------------------------------------------------ #
    # Hardware Profile CSV
    # ------------------------------------------------------------------ #
    hw_fields = [
        "method",
        "avg_gpu_util_pct",
        "avg_vram_gb",
        "avg_gpu_mem_util_pct",
        "avg_ram_gb",
        "avg_cpu_pct",
        "avg_power_w",
        "avg_gpu_sm_clock_mhz",
        "avg_gpu_mem_clock_mhz",
    ]
    hw_rows = []
    for method, data in methods.items():
        lat = data["lat"]
        hw_rows.append({
            "method":               method,
            "avg_gpu_util_pct":     _f(lat.get("avg_gpu_util_pct", 0), 2),
            "avg_vram_gb":          _f(lat.get("avg_vram_gb", 0), 3),
            "avg_gpu_mem_util_pct": _f(lat.get("avg_gpu_mem_util_pct", 0), 2),
            "avg_ram_gb":           _f(lat.get("avg_ram_gb", 0), 3),
            "avg_cpu_pct":          _f(lat.get("avg_cpu_pct", 0), 2),
            "avg_power_w":          _f(lat.get("avg_power_w", 0), 2),
            "avg_gpu_sm_clock_mhz": lat.get("avg_gpu_sm_clock_mhz", ""),
            "avg_gpu_mem_clock_mhz": lat.get("avg_gpu_mem_clock_mhz", ""),
        })
    write_csv(os.path.join(output_dir, "hardware.csv"), hw_fields, hw_rows)


if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.json"
    output_dir   = sys.argv[2] if len(sys.argv) > 2 else "."
    main(results_path, output_dir)
