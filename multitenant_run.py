"""Experiment 2: Concurrent Model Executions (CME) of early-exit models.

Measurement scheme (see idea/exp2_scheme.md for the full write-up and the
reasoning behind each phase). Three explicitly separated phases:

  PHASE 1 - SOLO BASELINE
      Each (tenant, exit) runs ALONE. Gives per-sample latency, throughput,
      P95 tail latency, power, energy and memory with no interference. This is
      the denominator for every ratio reported later.

  PHASE 2 - DURATION CALIBRATION
      From the solo per-sample latency, compute how many samples each tenant
      needs to stay busy for the SAME target window T:
          n_samples_i = T / latency_solo_i
      Without this a fast tenant (YOLO) finishes long before a slow one (Llama)
      and most of the "concurrent" window is actually solo, which silently
      understates interference and makes the throughput ratio meaningless.

  PHASE 3 - CONCURRENT (CME)
      All tenants launch together with their calibrated sample counts, so they
      finish at roughly the same time and the overlap window covers the run.
      Reported per tenant: latency, throughput, P95, slowdown. Reported for the
      system: aggregate throughput, throughput gain, pair power/energy/memory.

Metrics follow the edge multi-tenancy literature (Hao/Subedi/Ramaswamy/Kim,
arXiv:2107.12486 and ACM TOIT 2023): THROUGHPUT versus a single-tenancy
baseline is primary, memory is a first-class factor. On top of that we report
per-tenant slowdown (the interference lens) and P95 tail latency (the SLO lens
used by Clockwork / EdgeServing). Power (W, instantaneous) and energy (J, power
integrated over time) are distinct and both reported; on Jetson both are
pair-aggregate only because the board exposes one shared INA3221 rail.

No new inference code: each tenant is the same tested single-exit run as
Experiment 1 (bench_jetson.py), first alone then concurrently.

Usage (Jetson). fam:exit[:sub]  (yolo sub 0/1/2 = P3/P4/P5):
    python multitenant_run.py --scenario llama_yolo        # heterogeneous grid
    python multitenant_run.py --scenario yolo_scale        # tenancy 2/3/4
    python multitenant_run.py --grid bert vision           # any 6x6 pair
    python multitenant_run.py --pair llama:8 yolo:5:2      # one manual cell
    python multitenant_run.py --pair bert:12 vision:12 --duration 60
    python multitenant_run.py --list
Off-device check (reads datasource, no GPU):
    python multitenant_run.py --selftest
"""
import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LOGS = REPO_ROOT / "logs"
OUT_DIR = REPO_ROOT / "result" / "multitenant"

# resident cost per instance, GB (weights + ~0.5 GB CUDA context)
RESIDENT_GB = {"yolo": 0.54, "vision": 1.1, "bert": 1.2, "llama": 3.0, "llama3b": 6.4}
HEADROOM_GB = 1.0

# target concurrent window, seconds (phase 2 calibrates sample counts to this)
DEFAULT_DURATION = 30.0
MIN_SAMPLES, MAX_SAMPLES = 20, 20000

# exit count for the flat-index families (yolo: 6 exits x 3 sub-exits = 18 leaves)
FAM_N = {"bert": 24, "vision": 24, "llama": 16, "llama3b": 28}
YOLO_LEAVES = 18

# heterogeneous scenarios: one model per task type (see idea/exp2_design.md)
SCENARIOS = {
    "llama_yolo": ("llama", "yolo"),      # LLM + detector (headline)
    "yolo_vit":   ("yolo", "vision"),     # detector + classifier
    "bert_yolo":  ("bert", "yolo"),       # encoder + detector
    "llama_vit":  ("llama", "vision"),    # LLM + classifier
    "triple":     ("llama", "yolo", "vision"),
}
# Tenancy-scaling studies: same model repeated, so tenant COUNT is the only
# variable and any slowdown is pure contention rather than a model-mix artifact.
# Start here before the heterogeneous grids: it validates the harness and answers
# "does slowdown grow with n" which a pair-only view cannot.
# Exit anchors are swept as a second axis, producing an S(exit, n_tenants) surface
# instead of a 3-point line.
#   (family, [tenant counts])
# BERT at 1.2 GB/instance fits to n=4 (4.8 GB) with headroom on a headless board.
SCALING = {
    "bert_scale": ("bert", [2, 3, 4]),
    "yolo_scale": ("yolo", [2, 3, 4]),
    "vit_scale":  ("vision", [2, 3, 4]),
}


# ---- anchors ---------------------------------------------------------------
def _k_points(n, k):
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def _anchor(fam, k=6):
    """k evenly-spaced (exit, sub) anchors. yolo maps leaf l -> (l//3, l%3)."""
    if fam == "yolo":
        return [(l // 3, l % 3) for l in _k_points(YOLO_LEAVES, k)]
    return [(e, None) for e in _k_points(FAM_N[fam], k)]


def parse_tenant(spec):
    """'bert:12' -> ('bert', 12, None);  'yolo:5:2' -> ('yolo', 5, 2)."""
    parts = spec.split(":")
    return (parts[0],
            int(parts[1]) if len(parts) > 1 else 0,
            int(parts[2]) if len(parts) > 2 else None)


def preflight(families):
    """Memory gate: refuse to launch rather than risk an OOM kill."""
    import psutil
    need = sum(RESIDENT_GB.get(f, 1.0) for f in families) + HEADROOM_GB
    free = psutil.virtual_memory().available / 1e9
    ok = need <= free
    print(f"[preflight] need ~{need:.1f} GB (+{HEADROOM_GB} headroom), free {free:.1f} GB "
          f"-> {'OK' if ok else 'ABORT'}")
    return ok


# ---- running one tenant ----------------------------------------------------
def bench_cmd(fam, ex, sub, subdir, n_samples=None):
    argv = [sys.executable, str(REPO_ROOT / "bench_jetson.py"), fam,
            "--exit", str(ex), "--no-quality"]
    if sub is not None:
        argv += ["--sub-exit", str(sub)]
    if n_samples:
        argv += ["--n-samples", str(int(n_samples))]
    return argv, {"BENCH_SUBDIR": subdir}


def run_one(fam, ex, sub, subdir, import_os, n_samples=None):
    argv, envextra = bench_cmd(fam, ex, sub, subdir, n_samples)
    env = dict(import_os.environ)
    env.update(envextra)
    env.setdefault("MALLOC_ARENA_MAX", "2")     # 6-core A78AE -> ~48 arenas by default
    return subprocess.Popen(argv, env=env), time.perf_counter()


def _find_hw(root, fam, ex, sub=None):
    """Newest hw_results.json for this family+exit(+sub). yolo dirs are
    exit_{e}_P{sub+3}; flat families are exit_{e}."""
    base = Path(root) / fam
    if fam == "yolo" and sub is not None:
        pats = [f"**/exit_{ex}_P{sub + 3}/hw_results.json"]
    else:
        pats = [f"**/exit_{ex}/hw_results.json", f"**/exit_{ex}_*/hw_results.json"]
    cands = [p for g in pats for p in base.glob(g)]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def _p95(samples):
    """P95 end-to-end latency from the per-sample list (the SLO lens)."""
    lats = sorted(s["end_to_end_sec"] for s in samples
                  if isinstance(s.get("end_to_end_sec"), (int, float)))
    return lats[min(int(len(lats) * 0.95), len(lats) - 1)] if lats else None


def _read_hw(path):
    if path is None or not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    a = d.get("aggregate", {})
    return {"lat": a.get("per_sample_sec_mean"),
            "thru": a.get("throughput_samples_per_sec"),
            "p95": _p95(d.get("samples") or []),
            "power_w": a.get("avg_power_w"),         # Watts, instantaneous draw
            "energy_j": a.get("avg_energy_j"),       # Joules per inference
            # memory splits two ways for multi-tenancy (see idea/exp2_scheme.md):
            #   static  = weights + persistent buffers, resident the whole time,
            #             independent of exit/batch. The ADMISSION cost.
            #   dynamic = activations/KV/workspace, transient, moves with the exit
            #             index. Decides whether co-tenants survive a joint peak.
            # Older runs predate these fields and report None; nothing is invented.
            "gpu_mem_static_mb": a.get("gpu_mem_static_mb"),
            "gpu_mem_dynamic_mb": a.get("gpu_mem_dynamic_mb"),
            "gpu_mem_peak_mb": a.get("peak_vram_allocated_mb"),
            "vram_mb": a.get("avg_vram_allocated_mb"),
            "ram_mb": a.get("avg_ram_used_mb"),
            "n": a.get("n_samples"),
            "total_sec": a.get("total_sec"),
            # Wall-clock timed window. None for runs that predate this field.
            "timed_start_unix": a.get("timed_start_unix"),
            "timed_end_unix": a.get("timed_end_unix")}


# ---- PHASE 1: solo baseline ------------------------------------------------
def measure_solo(fam, ex, sub, tag, import_os):
    sd = f"mt_solo_{tag}_{fam}_{ex}" + (f"_P{sub}" if sub is not None else "")
    proc, _ = run_one(fam, ex, sub, sd, import_os)
    proc.wait()
    return _read_hw(_find_hw(LOGS / sd, fam, ex, sub))


# ---- PHASE 2: duration calibration -----------------------------------------
def calibrate(solo_hw, duration):
    """How many samples keeps this tenant busy for `duration` seconds?"""
    if not solo_hw or not solo_hw.get("lat"):
        return None
    n = int(duration / solo_hw["lat"])
    return max(MIN_SAMPLES, min(MAX_SAMPLES, n))


# ---- PHASE 3: concurrent ---------------------------------------------------
def measure_concurrent(tenants, tag, counts, import_os):
    """Launch every tenant at once with its calibrated sample count."""
    procs, starts = {}, {}
    for i, (fam, ex, sub) in enumerate(tenants):
        sd = f"mt_conc_{tag}_{i}_{fam}_{ex}"
        procs[i], starts[i] = run_one(fam, ex, sub, sd, import_os, counts.get(i))
    ends = {}
    for i, p in procs.items():
        p.wait()
        ends[i] = time.perf_counter()
    # Process-lifetime overlap (kept for backward compat; biased high because it
    # includes model-load time, not just the measurement window).
    span = max(ends.values()) - min(starts.values())
    overlap = max(0.0, min(ends.values()) - max(starts.values()))
    overlap_frac = round(overlap / span, 3) if span > 0 else 0.0
    shared = {i: _read_hw(_find_hw(LOGS / f"mt_conc_{tag}_{i}_{f}_{e}", f, e, s))
              for i, (f, e, s) in enumerate(tenants)}
    # True overlap: only the profiler-timed measurement window (cross-process
    # comparable because both use time.time()).
    t_starts = [hw["timed_start_unix"] for hw in shared.values()
                if hw and hw.get("timed_start_unix") is not None]
    t_ends   = [hw["timed_end_unix"]   for hw in shared.values()
                if hw and hw.get("timed_end_unix")   is not None]
    if len(t_starts) == len(tenants) and len(t_ends) == len(tenants):
        true_overlap = min(t_ends) - max(t_starts)
        true_span    = max(t_ends) - min(t_starts)
        timed_overlap_frac = round(max(0.0, min(1.0, true_overlap / true_span)), 3) if true_span > 0 else 0.0
    else:
        timed_overlap_frac = None
    return shared, overlap_frac, timed_overlap_frac


# ---- metrics ---------------------------------------------------------------
def build_row(tenants, tag, solo, shared, overlap_frac, counts, duration,
              timed_overlap_frac=None):
    """One CSV row: per-tenant metrics + system-level aggregates."""
    row = {"tag": tag, "n_tenants": len(tenants),
           # overlap_frac: process-lifetime overlap (kept for backward compat).
           # timed_overlap_frac: profiler-timed window overlap (the honest figure).
           "overlap_frac": overlap_frac,
           "timed_overlap_frac": timed_overlap_frac,
           "target_window_sec": duration}
    powers, busy, thru_sh, thru_so, vram, rams = [], [], [], [], [], []
    statics, peaks = [], []
    for i, (fam, ex, sub) in enumerate(tenants):
        so, sh = solo.get(i), shared.get(i)
        row[f"t{i}"] = f"{fam}@{ex}" + (f"_P{sub}" if sub is not None else "")
        row[f"t{i}_n_samples"] = counts.get(i)
        # latency + the interference lens
        row[f"t{i}_lat_solo"] = so["lat"] if so else None
        row[f"t{i}_lat_shared"] = sh["lat"] if sh else None
        row[f"t{i}_slowdown"] = (round(sh["lat"] / so["lat"], 3)
                                 if sh and so and sh["lat"] and so["lat"] else None)
        # throughput (primary metric)
        row[f"t{i}_thru_solo"] = so["thru"] if so else None
        row[f"t{i}_thru_shared"] = sh["thru"] if sh else None
        # tail latency (the SLO lens)
        row[f"t{i}_p95_solo"] = round(so["p95"], 6) if so and so.get("p95") else None
        row[f"t{i}_p95_shared"] = round(sh["p95"], 6) if sh and sh.get("p95") else None
        row[f"t{i}_p95_ratio"] = (round(sh["p95"] / so["p95"], 3)
                                  if sh and so and sh.get("p95") and so.get("p95") else None)
        # memory, split static vs dynamic (first-class factor in the prior work)
        row[f"t{i}_gpu_mem_static_mb"] = sh.get("gpu_mem_static_mb") if sh else None
        row[f"t{i}_gpu_mem_dynamic_mb"] = sh.get("gpu_mem_dynamic_mb") if sh else None
        row[f"t{i}_gpu_mem_peak_mb"] = sh.get("gpu_mem_peak_mb") if sh else None
        # how much the co-location inflated the transient part
        row[f"t{i}_dynamic_ratio"] = (
            round(sh["gpu_mem_dynamic_mb"] / so["gpu_mem_dynamic_mb"], 3)
            if sh and so and sh.get("gpu_mem_dynamic_mb") and so.get("gpu_mem_dynamic_mb")
            else None)
        row[f"t{i}_vram_mb"] = sh["vram_mb"] if sh else None
        row[f"t{i}_ram_mb"] = sh["ram_mb"] if sh else None
        row[f"t{i}_busy_sec"] = round(sh["total_sec"], 2) if sh and sh.get("total_sec") else None
        if sh:
            if sh.get("power_w"):
                powers.append(sh["power_w"])
            if sh.get("total_sec"):
                busy.append(sh["total_sec"])
            if sh.get("thru"):
                thru_sh.append(sh["thru"])
            if sh.get("vram_mb"):
                vram.append(sh["vram_mb"])
            if sh.get("ram_mb"):
                rams.append(sh["ram_mb"])
            if sh.get("gpu_mem_static_mb"):
                statics.append(sh["gpu_mem_static_mb"])
            if sh.get("gpu_mem_peak_mb"):
                peaks.append(sh["gpu_mem_peak_mb"])
        if so and so.get("thru"):
            thru_so.append(so["thru"])
    # System level. One shared rail, so the device figure is the max, never a sum.
    row["pair_power_w"] = round(max(powers), 3) if powers else None
    # Energy must NOT be summed across tenants either. Each tenant integrates the
    # SAME shared-rail power over the same wall-clock window, so each one already
    # accounts for roughly the whole device; adding them multiplies by n_tenants.
    # Device energy over the window is simply device power x window length.
    row["pair_window_sec"] = round(max(busy), 2) if busy else None
    row["pair_energy_j"] = (round(row["pair_power_w"] * max(busy), 3)
                            if powers and busy else None)
    row["agg_throughput"] = round(sum(thru_sh), 3) if thru_sh else None
    row["throughput_gain"] = (round(sum(thru_sh) / max(thru_so), 3)
                              if thru_sh and thru_so else None)
    row["pair_vram_mb"] = round(sum(vram), 1) if vram else None
    row["pair_ram_mb"] = round(max(rams), 1) if rams else None
    # Two distinct system-level memory questions:
    #   static sum -> can these tenants coexist on the board at all (admission)
    #   peak sum   -> do they survive if their transient peaks land together
    row["pair_gpu_mem_static_mb"] = round(sum(statics), 1) if statics else None
    row["pair_gpu_mem_peak_mb"] = round(sum(peaks), 1) if peaks else None
    return row


# ---- driver ----------------------------------------------------------------
def run_cells(cells, tag, duration=DEFAULT_DURATION):
    """cells: list of tenant-lists [(fam,ex,sub),...]. Phase 1 solo (cached),
    phase 2 calibrate, phase 3 concurrent. One CSV row per cell."""
    import os
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    worst = max(cells, key=lambda c: sum(RESIDENT_GB.get(f, 1.0) for f, _, _ in c))
    if not preflight([f for f, _, _ in worst]):
        print("[abort] largest cell would not fit; refusing to launch (no OOM).")
        return None

    print(f"[phase 1] solo baselines (target concurrent window {duration:.0f}s)")
    solo = {}
    for cell in cells:
        for key in cell:
            if key not in solo:
                fam, ex, sub = key
                solo[key] = measure_solo(fam, ex, sub, tag, os)
                hw = solo[key]
                print(f"  solo {fam}@{ex}: lat={hw['lat'] if hw else None} "
                      f"thru={hw['thru'] if hw else None}")

    rows = []
    for ci, cell in enumerate(cells):
        ctag = f"{tag}_{ci}"
        # phase 2
        counts = {i: calibrate(solo[k], duration) for i, k in enumerate(cell)}
        print(f"[phase 2] {ctag} calibrated n_samples: "
              + ", ".join(f"{k[0]}@{k[1]}={counts[i]}" for i, k in enumerate(cell)))
        # phase 3
        shared, ovf, timed_ovf = measure_concurrent(cell, ctag, counts, os)
        solo_map = {i: solo[k] for i, k in enumerate(cell)}
        row = build_row(cell, ctag, solo_map, shared, ovf, counts, duration,
                        timed_overlap_frac=timed_ovf)
        _append_csv(row)
        rows.append(row)
        desc = " ".join(f"{f}@{e}:x{row[f't{i}_slowdown']}"
                        for i, (f, e, s) in enumerate(cell))
        ov_display = timed_ovf if timed_ovf is not None else ovf
        ov_label = "timed_ov" if timed_ovf is not None else "ov(lifetime)"
        low = ov_display < 0.9
        print(f"[phase 3] {ctag} {desc}  gain={row['throughput_gain']}  "
              f"{ov_label}={ov_display}"
              + ("   <-- LOW OVERLAP, cell is suspect" if low else ""))
    print(f"[done] {len(rows)} cells -> {OUT_DIR / 'concurrent_slowdown.csv'}")
    return rows


def run_pair(tenants, tag="run", duration=DEFAULT_DURATION):
    return run_cells([tenants], tag, duration)


def run_grid(fam_a, fam_b, tag="grid", k=6, duration=DEFAULT_DURATION):
    anchors_a, anchors_b = _anchor(fam_a, k), _anchor(fam_b, k)
    cells = [[(fam_a, ea, sa), (fam_b, eb, sb)]
             for (ea, sa) in anchors_a for (eb, sb) in anchors_b]
    print(f"[grid] {fam_a} x {fam_b} = {len(cells)} cells")
    return run_cells(cells, tag, duration)


def run_scenario(name, tag=None, duration=DEFAULT_DURATION, k=6):
    tag = tag or name
    if name in SCALING:
        fam, counts = SCALING[name]
        cells = [[(fam, ex, sub)] * c
                 for (ex, sub) in _anchor(fam, k)
                 for c in counts]
        print(f"[scenario {name}] tenancy scaling {counts} of {fam}, "
              f"{k} exit anchors = {len(cells)} cells")
        return run_cells(cells, tag, duration)
    if name not in SCENARIOS:
        print(f"[scenario] unknown '{name}'; see --list")
        return None
    fams = SCENARIOS[name]
    eff_k = k if len(fams) == 2 else min(k, 3)   # triple at k=6 would be 216 cells
    anchors = [_anchor(f, eff_k) for f in fams]
    cells = [[(fams[j], e, s) for j, (e, s) in enumerate(combo)]
             for combo in itertools.product(*anchors)]
    print(f"[scenario {name}] {fams} = {len(cells)} cells (k={eff_k})")
    return run_cells(cells, tag, duration)


def _append_csv(row):
    import csv
    path = OUT_DIR / "concurrent_slowdown.csv"
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    keys = list(row.keys())
    if existing:
        keys = list(dict.fromkeys(existing[0].split(",") + keys))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for ln in existing[1:]:
            if ln.strip():
                w.writerow(dict(zip(existing[0].split(","), ln.split(","))))
        w.writerow(row)


def _selftest():
    """Off-device: parse + calibration + metric math + CSV against datasource."""
    ds = REPO_ROOT / "datasource" / "logs.jetson-orin"
    so = _read_hw(_find_hw(ds / "benchmark", "bert", 0))
    sh = _read_hw(_find_hw(ds / "benchmark.15w", "bert", 0))
    assert so and so["lat"] and so["thru"], "solo parse failed"
    assert so["p95"], "p95 not computed from per-sample list"
    n = calibrate(so, 30.0)
    assert MIN_SAMPLES <= n <= MAX_SAMPLES, f"calibration out of range: {n}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    row = build_row([("bert", 0, None)], "selftest", {0: so}, {0: sh}, 1.0, {0: n}, 30.0,
                   timed_overlap_frac=None)
    assert row["t0_slowdown"] and row["agg_throughput"], "core metrics missing"
    assert row["t0_p95_ratio"], "p95 ratio missing"
    assert row["pair_power_w"] and row["pair_energy_j"], "power/energy missing"
    assert "timed_overlap_frac" in row, "timed_overlap_frac missing from row"
    assert row["timed_overlap_frac"] is None, "timed_overlap_frac should be None (old data)"
    _append_csv(row)
    print(f"[selftest] calibrate(30s)={n} samples | slowdown={row['t0_slowdown']} "
          f"gain={row['throughput_gain']} p95_ratio={row['t0_p95_ratio']} "
          f"power={row['pair_power_w']}W energy={row['pair_energy_j']}J "
          f"timed_overlap_frac={row['timed_overlap_frac']}  OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs="+", metavar="FAM:EXIT[:SUB]")
    ap.add_argument("--grid", nargs=2, metavar=("FAM_A", "FAM_B"))
    ap.add_argument("--scenario", metavar="NAME")
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                    help=f"target concurrent window in seconds (default {DEFAULT_DURATION:.0f})")
    ap.add_argument("--k", type=int, default=6,
                    help="exit anchors sampled per model (minimum 2)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.k < 2:
        print(f"[error] --k must be at least 2 (got {a.k}); "
              f"_k_points divides by k-1 and would raise ZeroDivisionError")
        sys.exit(1)
    if a.selftest:
        _selftest()
    elif a.list:
        print("scenarios:", ", ".join(list(SCENARIOS) + list(SCALING)))
    elif a.scenario:
        run_scenario(a.scenario, a.tag, a.duration, k=a.k)
    elif a.grid:
        run_grid(a.grid[0], a.grid[1], a.tag or "grid", k=a.k, duration=a.duration)
    elif a.pair:
        run_pair([parse_tenant(s) for s in a.pair], a.tag or "run", a.duration)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
