"""Experiment 2: Concurrent Model Executions (CME) of early-exit models.

Measurement scheme (see idea/exp2_scheme.md for the full write-up and the
reasoning behind each phase). Three explicitly separated phases:

  PHASE 1 - SOLO BASELINE
      Each (tenant, exit) runs ALONE. Gives per-sample latency, throughput,
      P95 tail latency, power, energy and memory with no interference. This is
      the denominator for every ratio reported later. A probe run is used to
      obtain the wall latency; the real solo run uses the calibrated count so
      both solo and concurrent cover the same target window.

  PHASE 2 - DURATION CALIBRATION
      From the per-sample WALL time (total_sec / n), compute how many samples
      each tenant needs to stay busy for the SAME target window T:
          n_samples_i = T / lat_wall_i
      Without this a fast tenant (YOLO) finishes long before a slow one (Llama)
      and most of the "concurrent" window is actually solo, which silently
      understates interference and makes the throughput ratio meaningless.
      Wall time is used because forward-only timing misses dataloading, .cuda()
      transfer, and per-sample telemetry, which make actual process time
      12-30% longer than the sum of forward passes.

  PHASE 3 - CONCURRENT (CME)
      All tenants launch together with their calibrated sample counts, so they
      finish at roughly the same time and the overlap window covers the run.
      Reported per tenant: latency, throughput, P95, slowdown, SLO, violation
      ratio, achieved window. Reported for the system: STP, ANTT, aggregate
      throughput, throughput gain, pair power/energy/memory, clock config.

Metrics follow the edge multi-tenancy literature (Hao/Subedi/Ramaswamy/Kim,
arXiv:2107.12486 and ACM TOIT 2023): THROUGHPUT versus a single-tenancy
baseline is primary, memory is a first-class factor. On top of that we report
per-tenant slowdown (the interference lens) and P95 tail latency (the SLO lens
used by Clockwork / EdgeServing). Power (W, instantaneous) and energy (J, power
integrated over time) are distinct and both reported; on Jetson both are
pair-aggregate only because the board exposes one shared INA3221 rail.

System throughput metrics follow Eyerman and Eeckhout, "System-Level Performance
Metrics for Multiprogram Workloads", IEEE Micro, May 2008, DOI 10.1109/mm.2008.44:
  STP  (system throughput) = sum_i(thru_shared_i / thru_solo_i)
  ANTT (avg normalised turnaround) = mean_i(slowdown_i)
STP > 1 means co-location is profitable.

No new inference code: each tenant is the same tested single-exit run as
Experiment 1 (bench_jetson.py), first alone then concurrently.

Usage (Jetson). fam:exit[:sub]  (yolo sub 0/1/2 = P3/P4/P5):
    python multitenant_run.py --scenario llama_yolo        # heterogeneous grid
    python multitenant_run.py --scenario yolo_scale        # tenancy 2/3/4
    python multitenant_run.py --grid bert vision           # any 6x6 pair
    python multitenant_run.py --pair llama:8 yolo:5:2      # one manual cell
    python multitenant_run.py --pair bert:12 vision:12 --duration 60
    python multitenant_run.py --pair bert:0 yolo:0:0 --repeats 3
    python multitenant_run.py --grid bert vision --holdout 4
    python multitenant_run.py --list
    python multitenant_run.py --mode-label 15W --pair bert:0 yolo:0:0
Off-device check (reads datasource, no GPU):
    python multitenant_run.py --selftest
"""
import argparse
import hashlib
import itertools
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LOGS = REPO_ROOT / "logs"
OUT_DIR = REPO_ROOT / "result" / "multitenant"

# resident cost per instance, GB (weights + ~0.5 GB CUDA context)
RESIDENT_GB = {"yolo": 0.54, "vision": 1.1, "bert": 1.2, "llama": 3.0, "llama3b": 6.4}
HEADROOM_GB = 1.0

# Hard cap on the tenant count that --grow will ever produce.
# A mis-read of free memory cannot spawn an unbounded number of processes.
MAX_TENANTS = 12

# target concurrent window, seconds (phase 2 calibrates sample counts to this)
DEFAULT_DURATION = 30.0
MIN_SAMPLES, MAX_SAMPLES = 20, 20000

# overlap gate: cells below this fraction are suspect (see idea/exp2_scheme.md)
OVERLAP_GATE = 0.9

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


# ---- output path helpers ---------------------------------------------------

def _normalize_label(label: str) -> str:
    """Return a filesystem-safe lowercase token from a mode label.

    Lowercases, then replaces any character that is not alphanumeric, a dot,
    an underscore or a hyphen with an underscore. This ensures that a label
    like "15W" becomes "15w", "MAXN_SUPER" becomes "maxn_super", and labels
    containing spaces or slashes are safely sanitised before use as a path
    component.
    """
    return re.sub(r"[^a-z0-9._-]", "_", label.lower())


def _detect_mode_label() -> str:
    """Query the current nvpmodel mode name cheaply and return a normalised label.

    Querying the current mode does not require root on a Jetson; only setting a
    mode does. Returns "unknown" on any failure (nvpmodel absent, parse error,
    timeout, non-Jetson dev box) so callers always get a usable label.

    This helper is intentionally self-contained and does NOT import from
    multitenant_sweep to avoid a circular dependency (multitenant_sweep imports
    multitenant_run).
    """
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout + result.stderr
        for line in output.splitlines():
            m = re.search(r"NV Power Mode\s*:\s*(\S+)", line, re.IGNORECASE)
            if m:
                return _normalize_label(m.group(1))
    except Exception:
        pass
    return "unknown"


# ---- anchors ---------------------------------------------------------------
def _k_points(n, k, lo=0):
    """Return k evenly-spaced integer indices spanning [lo, n-1] inclusive.

    Indices are computed as round(lo + i*(n-1-lo)/(k-1)) for i in 0..k-1,
    deduplicated and sorted. With lo=0 the result is identical to the
    original two-argument form (backward compatible).

    Raises ValueError when lo >= n-1 (degenerate range) or k < 2 (division
    by zero in the spacing formula).
    """
    if k < 2:
        raise ValueError(f"k must be at least 2 (got {k})")
    if lo >= n - 1:
        raise ValueError(
            f"lo={lo} must be less than n-1={n - 1} (degenerate range)"
        )
    hi = n - 1
    return sorted({round(lo + i * (hi - lo) / (k - 1)) for i in range(k)})


def _anchor(fam, k=6, min_exit=0):
    """k evenly-spaced (exit, sub) anchors starting at min_exit.

    min_exit=0 (default) keeps every existing caller's result unchanged.
    min_exit=1 skips exit 0, which produced physically impossible slowdown
    below 1.0 in the 2026-09 measurement campaign.
    yolo maps leaf l -> (l//3, l%3).
    """
    if fam == "yolo":
        return [(l // 3, l % 3) for l in _k_points(YOLO_LEAVES, k, min_exit)]
    return [(e, None) for e in _k_points(FAM_N[fam], k, min_exit)]


def _holdout_anchors(fam, k, n_holdout, seed_tag):
    """Return n_holdout off-anchor (exit, sub) points, deterministically from seed_tag.

    Points are sampled from the complement of the anchor set so they can be
    predicted by interpolation and compared against measurement. The selection
    is stable: identical (fam, k, n_holdout, seed_tag) always yields the same
    set, with no dependence on wall-clock time or unseeded randomness.
    """
    import random as _random
    anchors = set(_anchor(fam, k))
    if fam == "yolo":
        universe = [(l // 3, l % 3) for l in range(YOLO_LEAVES)]
    else:
        universe = [(e, None) for e in range(FAM_N[fam])]
    complement = [p for p in universe if p not in anchors]
    if not complement:
        return []
    seed = int(hashlib.sha256(f"{fam}:{k}:{n_holdout}:{seed_tag}".encode()).hexdigest(), 16) % (2**32)
    rng = _random.Random(seed)
    return rng.sample(complement, min(n_holdout, len(complement)))


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


# ---- dataset pinning -------------------------------------------------------
# Each family runs bench_jetson with a single pinned dataset so a concurrent
# cell covers exactly one hw_results.json per (family, exit) and the window
# is the controlled 15-30 seconds it was designed to be.
#
# Values read from benchmark_config/:
#   bert:   benchmark_config/bert.py   line 56  TASKS[0] = "SST-2"
#   vision: benchmark_config/vision.py line 49  TRAINED_DATASETS[0] = "uoft-cs/cifar10"
#   yolo:   benchmark_config/yolo.py   line 69  HW_DATASETS[0] = "coco"
#   llama:  benchmark_config/llama.py  line 47  HW_DATASET = "cnn_dailymail"
#   llama3b: bench_jetson.py lines 913-915: no --dataset flag exposed for llama3b.
#            Left unpinned; a single dataset is used internally by the config.
#
# ponytail: dict lookup is O(1); no class needed.
DATASET_PIN = {
    "bert":   ("--task",    "SST-2"),
    "vision": ("--dataset", "uoft-cs/cifar10"),
    "yolo":   ("--dataset", "coco"),
    "llama":  ("--dataset", "cnn_dailymail"),
    # llama3b: bench_jetson exposes no --dataset flag for this subcommand.
}


# ---- running one tenant ----------------------------------------------------
def bench_cmd(fam, ex, sub, subdir, n_samples=None, task=None, dataset=None, duration=None):
    """Build the argv list for one bench_jetson.py tenant invocation.

    task:     override for bert's --task flag (default: the DATASET_PIN value).
    dataset:  override for vision/yolo/llama's --dataset flag.
    duration: when set, passes --duration to the child so the tenant stops on
              wall clock rather than sample count. --n-samples is also passed as
              a safety ceiling so the loop cannot run forever if the duration
              mechanism is absent or fails.

    When neither task/dataset override is given, the family's DATASET_PIN entry
    is appended automatically so a cell always runs exactly one dataset. llama3b
    has no dataset flag in bench_jetson and is left unpinned.
    """
    argv = [sys.executable, str(REPO_ROOT / "bench_jetson.py"), fam,
            "--exit", str(ex), "--no-quality"]
    if sub is not None:
        argv += ["--sub-exit", str(sub)]
    if n_samples:
        argv += ["--n-samples", str(int(n_samples))]
    if duration is not None:
        argv += ["--duration", str(float(duration))]
    # Apply the dataset pin: CLI overrides take precedence over the family default.
    pin = DATASET_PIN.get(fam)
    if pin is not None:
        flag, default_val = pin
        if flag == "--task":
            val = task if task is not None else default_val
        else:
            val = dataset if dataset is not None else default_val
        argv += [flag, val]
    return argv, {"BENCH_SUBDIR": subdir}


def run_one(fam, ex, sub, subdir, import_os, n_samples=None, task=None, dataset=None,
            _extra_env=None, duration=None):
    argv, envextra = bench_cmd(fam, ex, sub, subdir, n_samples, task=task, dataset=dataset,
                               duration=duration)
    env = dict(import_os.environ)
    env.update(envextra)
    if _extra_env:
        env.update(_extra_env)
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


def _violation_ratio(samples, slo_sec):
    """Fraction of per-sample latencies that exceed slo_sec.

    The SLO is defined as twice the solo latency (gpu-let / Choi et al. 2021
    convention). Computed here rather than carried around as a full list.
    """
    if slo_sec is None or slo_sec <= 0:
        return None
    lats = [s["end_to_end_sec"] for s in samples
            if isinstance(s.get("end_to_end_sec"), (int, float))]
    if not lats:
        return None
    return round(sum(1 for l in lats if l > slo_sec) / len(lats), 4)


def _min_gpu_sm_clock(samples):
    """Minimum GPU SM clock (MHz) observed across per-sample rows.

    The minimum matters more than the mean for thermal throttling detection:
    a single dip is the signature of the governor stepping down. Returns None
    when the field is absent (older runs predate Jetson SM clock capture).
    """
    clocks = [s["gpu_sm_clock_mhz"] for s in samples
              if isinstance(s.get("gpu_sm_clock_mhz"), (int, float))]
    return min(clocks) if clocks else None


def _read_hw(path, slo_sec=None):
    """Parse hw_results.json into a flat dict.

    slo_sec: when provided (solo latency * 2), also compute the SLO violation
    ratio from the per-sample list before discarding it.
    """
    if path is None or not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    a = d.get("aggregate", {})
    caps = d.get("device_caps", {})
    samples = d.get("samples") or []
    result = {
        "lat": a.get("per_sample_sec_mean"),
        "thru": a.get("throughput_samples_per_sec"),
        "p95": _p95(samples),
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
        "timed_end_unix": a.get("timed_end_unix"),
        # Clock configuration. avg_gpu_sm_clock_mhz comes from aggregate_hw (via
        # sample_jetson_hw) and is absent in runs predating Jetson SM clock
        # capture; min is derived from per-sample data because aggregate_hw only
        # emits avg_* and max_*, and the minimum is the throttling signal.
        # nvpmodel lives in device_caps (written by jetson_caps()), never in
        # aggregate, so it is read from there: this is what makes the "clocks
        # were pinned" claim checkable from the artefact alone.
        "nvpmodel": caps.get("nvpmodel"),
        "jetson_clocks": caps.get("jetson_clocks"),
        "avg_gpu_sm_clock_mhz": a.get("avg_gpu_sm_clock_mhz"),
        "min_gpu_sm_clock_mhz": _min_gpu_sm_clock(samples),
        # SLO violation ratio: fraction of shared latencies exceeding 2x solo.
        "violation_ratio": _violation_ratio(samples, slo_sec) if slo_sec is not None else None,
    }
    return result


# ---- PHASE 1: solo baseline ------------------------------------------------
def measure_solo(fam, ex, sub, tag, import_os, n_samples=None, mode_label=None,
                 task=None, dataset=None, duration=None):
    sd = f"mt_solo_{tag}_{fam}_{ex}" + (f"_P{sub}" if sub is not None else "")
    if n_samples is not None:
        sd = sd + f"_n{n_samples}"
    # When a mode label is set, nest all output under logs/multitenant.<mode_lc>/.
    if mode_label:
        subdir = f"multitenant.{mode_label}/{sd}"
        log_root = LOGS / f"multitenant.{mode_label}" / sd
    else:
        subdir = sd
        log_root = LOGS / sd
    proc, _ = run_one(fam, ex, sub, subdir, import_os, n_samples=n_samples,
                      task=task, dataset=dataset, duration=duration)
    proc.wait()
    # With dataset pinning, there is exactly one hw_results.json per (family, exit).
    # The max-mtime tiebreak in _find_hw is retained as a harmless fallback.
    return _read_hw(_find_hw(log_root, fam, ex, sub))


# ---- PHASE 2: duration calibration -----------------------------------------
def calibrate(solo_hw, duration):
    """How many samples keeps this tenant busy for `duration` seconds?

    Uses per-sample WALL time (total_sec / n) rather than forward-only latency.
    Forward timing misses dataloading, .cuda() transfer and per-sample telemetry
    reads. Across Experiment 1 runs, the measured overhead raises actual process
    time by 7-40% above the sum of forward passes, depending on the model family.

    Falls back to forward latency when total_sec or n are absent or invalid.
    The fallback is visible: a printed warning and a calib_fallback flag.
    Returns a (count, fallback_used) tuple so callers can record the flag.
    """
    if not solo_hw:
        return None, False

    total_sec = solo_hw.get("total_sec")
    n = solo_hw.get("n")
    lat_forward = solo_hw.get("lat")

    lat_wall = None
    if total_sec and n and n > 0:
        candidate = total_sec / n
        if candidate > 0:
            lat_wall = candidate

    if lat_wall is not None:
        count = int(duration / lat_wall)
        fallback = False
    elif lat_forward:
        print(f"[calibrate] WARNING: wall time unavailable (total_sec={total_sec!r}, "
              f"n={n!r}); falling back to forward latency. "
              "This produces a biased window -- rerun on hardware with full profiler output.")
        count = int(duration / lat_forward)
        fallback = True
    else:
        return None, False

    return max(MIN_SAMPLES, min(MAX_SAMPLES, count)), fallback


# ---- PHASE 3: concurrent ---------------------------------------------------
def measure_concurrent(tenants, tag, counts, import_os, mode_label=None,
                       task=None, dataset=None, duration=None):
    """Launch every tenant at once with its calibrated sample count."""
    # Create a fresh per-cell barrier directory so readiness files from a
    # previous cell can never be miscounted by this cell.
    barrier_dir = tempfile.mkdtemp(prefix="bench_barrier_")
    n_tenants = len(tenants)
    procs, starts = {}, {}
    for i, (fam, ex, sub) in enumerate(tenants):
        sd = f"mt_conc_{tag}_{i}_{fam}_{ex}"
        if mode_label:
            subdir = f"multitenant.{mode_label}/{sd}"
        else:
            subdir = sd
        # Thread the barrier env vars into this child's environment alongside
        # the existing BENCH_SUBDIR and MALLOC_ARENA_MAX that run_one sets.
        barrier_env = {
            "BENCH_BARRIER_DIR": barrier_dir,
            "BENCH_BARRIER_N":   str(n_tenants),
            "BENCH_BARRIER_ID":  str(i),
        }
        procs[i], starts[i] = run_one(fam, ex, sub, subdir, import_os, counts.get(i),
                                      task=task, dataset=dataset,
                                      _extra_env=barrier_env, duration=duration)
    ends = {}
    for i, p in procs.items():
        p.wait()
        ends[i] = time.perf_counter()
    # Remove the barrier directory now that all tenants have finished.
    shutil.rmtree(barrier_dir, ignore_errors=True)
    # Process-lifetime overlap (kept for backward compat; biased high because it
    # includes model-load time, not just the measurement window).
    span = max(ends.values()) - min(starts.values())
    overlap = max(0.0, min(ends.values()) - max(starts.values()))
    overlap_frac = round(overlap / span, 3) if span > 0 else 0.0
    # Build the log root paths for reading results.
    if mode_label:
        shared = {i: _read_hw(_find_hw(LOGS / f"multitenant.{mode_label}" / f"mt_conc_{tag}_{i}_{f}_{e}", f, e, s))
                  for i, (f, e, s) in enumerate(tenants)}
    else:
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
              timed_overlap_frac=None, repeat_idx=0, calib_fallbacks=None,
              is_holdout=False):
    """One CSV row: per-tenant metrics + system-level aggregates.

    System-level throughput metrics follow Eyerman and Eeckhout (2008):
      STP  = sum_i(thru_shared_i / thru_solo_i)   -- profitable when > 1
      ANTT = mean_i(slowdown_i)                    -- SLA/QoS metric
    Note: STP = sum of the reciprocals of the per-tenant slowdowns.
    DOI: 10.1109/mm.2008.44
    """
    if calib_fallbacks is None:
        calib_fallbacks = {}
    row = {"tag": tag, "n_tenants": len(tenants),
           # overlap_frac: process-lifetime overlap (kept for backward compat).
           # timed_overlap_frac: profiler-timed window overlap (the honest figure).
           "overlap_frac": overlap_frac,
           "timed_overlap_frac": timed_overlap_frac,
           "target_window_sec": duration,
           "repeat_idx": repeat_idx,
           "is_holdout": is_holdout}
    powers, busy, thru_sh, thru_so, vram, rams = [], [], [], [], [], []
    statics, peaks = [], []
    slowdowns, stp_terms = [], []
    for i, (fam, ex, sub) in enumerate(tenants):
        so, sh = solo.get(i), shared.get(i)
        row[f"t{i}"] = f"{fam}@{ex}" + (f"_P{sub}" if sub is not None else "")
        row[f"t{i}_n_samples"] = counts.get(i)
        row[f"t{i}_calib_fallback"] = calib_fallbacks.get(i, False)
        # latency + the interference lens
        lat_solo = so["lat"] if so else None
        lat_shared = sh["lat"] if sh else None
        row[f"t{i}_lat_solo"] = lat_solo
        row[f"t{i}_lat_shared"] = lat_shared
        slowdown = (round(lat_shared / lat_solo, 3)
                    if lat_shared and lat_solo else None)
        row[f"t{i}_slowdown"] = slowdown
        # throughput (primary metric)
        thru_solo_i = so["thru"] if so else None
        thru_shared_i = sh["thru"] if sh else None
        row[f"t{i}_thru_solo"] = thru_solo_i
        row[f"t{i}_thru_shared"] = thru_shared_i
        # tail latency (the SLO lens) -- SLO = 2x solo latency per gpu-let convention
        slo_sec_i = round(2.0 * lat_solo, 6) if lat_solo else None
        row[f"t{i}_p95_solo"] = round(so["p95"], 6) if so and so.get("p95") else None
        row[f"t{i}_p95_shared"] = round(sh["p95"], 6) if sh and sh.get("p95") else None
        row[f"t{i}_p95_ratio"] = (round(sh["p95"] / so["p95"], 3)
                                  if sh and so and sh.get("p95") and so.get("p95") else None)
        row[f"t{i}_slo_sec"] = slo_sec_i
        # violation_ratio is already computed in _read_hw when slo_sec is threaded in;
        # here we use the value stored in sh (see run_cells where slo is threaded).
        row[f"t{i}_slo_violation_ratio"] = sh.get("violation_ratio") if sh else None
        # achieved window per tenant (from profiler-timed timestamps when available,
        # total_sec otherwise). Records actual measurement coverage for verification.
        if sh and sh.get("timed_start_unix") is not None and sh.get("timed_end_unix") is not None:
            row[f"t{i}_window_shared_sec"] = round(sh["timed_end_unix"] - sh["timed_start_unix"], 3)
        elif sh and sh.get("total_sec"):
            row[f"t{i}_window_shared_sec"] = round(sh["total_sec"], 3)
        else:
            row[f"t{i}_window_shared_sec"] = None
        if so and so.get("timed_start_unix") is not None and so.get("timed_end_unix") is not None:
            row[f"t{i}_window_solo_sec"] = round(so["timed_end_unix"] - so["timed_start_unix"], 3)
        elif so and so.get("total_sec"):
            row[f"t{i}_window_solo_sec"] = round(so["total_sec"], 3)
        else:
            row[f"t{i}_window_solo_sec"] = None
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
        # clock configuration (nvpmodel mode and GPU SM clock; None for older runs)
        row[f"t{i}_nvpmodel"] = sh.get("nvpmodel") if sh else None
        row[f"t{i}_jetson_clocks"] = sh.get("jetson_clocks") if sh else None
        row[f"t{i}_avg_gpu_sm_clock_mhz"] = sh.get("avg_gpu_sm_clock_mhz") if sh else None
        row[f"t{i}_min_gpu_sm_clock_mhz"] = sh.get("min_gpu_sm_clock_mhz") if sh else None
        if sh:
            if sh.get("power_w"):
                powers.append(sh["power_w"])
            if sh.get("total_sec"):
                busy.append(sh["total_sec"])
            if thru_shared_i:
                thru_sh.append(thru_shared_i)
            if sh.get("vram_mb"):
                vram.append(sh["vram_mb"])
            if sh.get("ram_mb"):
                rams.append(sh["ram_mb"])
            if sh.get("gpu_mem_static_mb"):
                statics.append(sh["gpu_mem_static_mb"])
            if sh.get("gpu_mem_peak_mb"):
                peaks.append(sh["gpu_mem_peak_mb"])
        if thru_solo_i:
            thru_so.append(thru_solo_i)
        if slowdown is not None:
            slowdowns.append(slowdown)
        if thru_shared_i and thru_solo_i:
            stp_terms.append(thru_shared_i / thru_solo_i)
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
    # agg_throughput is only meaningful (same units) when all tenants share a family.
    row["agg_throughput_comparable"] = len(set(f for f, _, _ in tenants)) == 1
    # throughput_gain: normalised by SUM of solo throughputs (not max).
    # Using max would be arbitrary for heterogeneous pairs and understate the gain
    # for fast tenants. Sum gives the correct "fraction of solo capacity recovered".
    row["throughput_gain"] = (round(sum(thru_sh) / sum(thru_so), 3)
                              if thru_sh and thru_so else None)
    # STP and ANTT (Eyerman & Eeckhout 2008, DOI 10.1109/mm.2008.44).
    # STP = sum_i(thru_sh_i / thru_so_i) -- equals sum of reciprocals of slowdowns.
    # ANTT = mean_i(slowdown_i) -- the SLA / QoS perspective.
    row["stp"] = round(sum(stp_terms), 3) if stp_terms else None
    row["antt"] = round(sum(slowdowns) / len(slowdowns), 3) if slowdowns else None
    row["pair_vram_mb"] = round(sum(vram), 1) if vram else None
    row["pair_ram_mb"] = round(max(rams), 1) if rams else None
    # Two distinct system-level memory questions:
    #   static sum -> can these tenants coexist on the board at all (admission)
    #   peak sum   -> do they survive if their transient peaks land together
    row["pair_gpu_mem_static_mb"] = round(sum(statics), 1) if statics else None
    row["pair_gpu_mem_peak_mb"] = round(sum(peaks), 1) if peaks else None
    return row


# ---- driver ----------------------------------------------------------------
def run_cells(cells, tag, duration=DEFAULT_DURATION, repeats=1, holdout_n=0,
              keep_suspect=False, mode_label=None, task=None, dataset=None):
    """cells: list of tenant-lists [(fam,ex,sub),...]. Phase 1 solo (cached),
    phase 2 calibrate, phase 3 concurrent. One CSV row per cell per repeat.

    A probe run is used to obtain wall latency for each unique key, then the
    real solo run uses the calibrated count so both solo and concurrent cover
    the same target window. Total phase-1 cost is 2 * unique_keys runs (probe
    + real). This is stated here so the operator is not surprised.

    Each cell passes preflight independently. Cells that do not fit are skipped
    with a printed explanation; the scenario only aborts when NO cell fits.
    This means a bert_scale run where n=4 does not fit still measures n=2 and
    n=3 rather than aborting everything.

    Cells whose overlap falls below OVERLAP_GATE are written to
    concurrent_slowdown.suspect.csv rather than the main CSV, unless
    keep_suspect=True restores the old flat-file behaviour.

    mode_label: when set, all logs are nested under logs/multitenant.<mode_label>/
    and all CSVs are written to result/multitenant/<mode_label>/.
    When None, the old flat layout is used (result/multitenant/).
    """
    import os
    # Determine the effective output directory for CSVs.
    if mode_label:
        csv_dir = OUT_DIR / mode_label
    else:
        csv_dir = OUT_DIR
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Per-cell preflight: evaluate each cell individually.
    # Cells that exceed free memory are skipped; only abort when nothing fits.
    import psutil
    free_gb = psutil.virtual_memory().available / 1e9
    runnable = []
    for cell in cells:
        need = sum(RESIDENT_GB.get(f, 1.0) for f, _, _ in cell) + HEADROOM_GB
        if need <= free_gb:
            runnable.append(cell)
        else:
            fam_str = "+".join(f for f, _, _ in cell)
            n_str = str(len(cell))
            print(f"[preflight] skip {fam_str} n={n_str}: "
                  f"need ~{need:.1f} GB (+{HEADROOM_GB} headroom), "
                  f"free {free_gb:.1f} GB")
    if not runnable:
        print("[abort] no cell fits in available memory; refusing to launch (no OOM).")
        return None
    cells = runnable

    # Mark holdout cells before measurement.
    holdout_set = set()
    if holdout_n > 0:
        # Collect all unique families in cells to derive holdout points.
        for cell in cells:
            for fam, _, _ in cell:
                for pt in _holdout_anchors(fam, 6, holdout_n, tag):
                    holdout_set.add((fam,) + pt)

    print(f"[phase 1] solo baselines (target concurrent window {duration:.0f}s)")
    print(f"[phase 1] two passes per unique key: probe (config default) then "
          f"calibrated run. Unique keys x 2 = total phase-1 runs.")

    # Pass 1: probe runs to obtain wall latency (config default, no n_samples).
    probe_hw = {}
    for cell in cells:
        for key in cell:
            if key not in probe_hw:
                fam, ex, sub = key
                probe_hw[key] = measure_solo(fam, ex, sub, f"probe_{tag}", import_os=os,
                                             mode_label=mode_label,
                                             task=task, dataset=dataset)
                hw = probe_hw[key]
                lat_wall = None
                if hw and hw.get("total_sec") and hw.get("n") and hw["n"] > 0:
                    lat_wall = hw["total_sec"] / hw["n"]
                print(f"  probe  {fam}@{ex}: lat_forward={hw['lat'] if hw else None} "
                      f"lat_wall={round(lat_wall, 6) if lat_wall else None}")

    # Phase 2: calibrate counts using wall latency from probe.
    # NOTE: with wall-clock --duration now passed to every tenant, the calibrated
    # count here acts as a safety ceiling (bench_jetson honours whichever limit
    # fires first). It is no longer the primary mechanism that sets the window
    # length; --duration is. The count is retained because a missing or
    # unimplemented duration_sec in a backend would otherwise leave the loop
    # unbounded.
    calibrated_counts = {}
    calib_fallbacks_per_key = {}
    for key in probe_hw:
        n, fb = calibrate(probe_hw[key], duration)
        calibrated_counts[key] = n
        calib_fallbacks_per_key[key] = fb

    # Pass 2: real solo runs at calibrated counts (so window matches concurrent).
    solo = {}
    for key in probe_hw:
        fam, ex, sub = key
        n = calibrated_counts[key]
        # Thread the SLO threshold into _read_hw via a two-step: read solo first
        # without SLO (solo latency not known yet), then compute SLO from solo lat.
        # Violation ratio for solo runs is not meaningful (no concurrent stress), so
        # we do not thread it in here.
        solo[key] = measure_solo(fam, ex, sub, tag, import_os=os, n_samples=n,
                                 mode_label=mode_label, task=task, dataset=dataset,
                                 duration=duration)
        hw = solo[key]
        print(f"  solo   {fam}@{ex}: lat={hw['lat'] if hw else None} "
              f"thru={hw['thru'] if hw else None} "
              f"calib_fallback={calib_fallbacks_per_key[key]}")

    # ponytail: build_row/_append_csv moved to offline analyzer (multitenant_analyze.py).
    # Derivation of slowdown/STP/ANTT/fairness no longer happens on the device;
    # the raw per-tenant hw_results.json files are the only output.
    n_cells = 0
    for ci, cell in enumerate(cells):
        for rep in range(repeats):
            ctag = f"{tag}_{ci}_r{rep}"
            counts = {i: calibrated_counts[k] for i, k in enumerate(cell)}
            calib_fallbacks = {i: calib_fallbacks_per_key[k] for i, k in enumerate(cell)}
            print(f"[phase 2] {ctag} calibrated n_samples: "
                  + ", ".join(f"{k[0]}@{k[1]}={counts[i]}" for i, k in enumerate(cell)))
            # Launch concurrent run; each tenant writes its own hw_results.json
            # via BenchmarkProfiler (which now embeds intended_tenants in aggregate).
            measure_concurrent(cell, ctag, counts, os,
                               mode_label=mode_label,
                               task=task, dataset=dataset,
                               duration=duration)
            print(f"[phase 3] {ctag} done — raw hw_results.json written per tenant")
            n_cells += 1
    print(f"[done] {n_cells} cells measured; interference metrics recomputed offline "
          f"by multitenant_analyze.py from raw hw_results.json")
    return []


def _print_cv(rows, cell_count):
    """Print coefficient of variation of slowdown across repeats for each cell."""
    import math
    from collections import defaultdict
    cells_reps = defaultdict(list)
    for r in rows:
        # group by cell tag without the _r{N} suffix
        base = "_r".join(r["tag"].split("_r")[:-1]) if "_r" in r["tag"] else r["tag"]
        cells_reps[base].append(r)
    for base, reps in cells_reps.items():
        if len(reps) < 2:
            continue
        for i in range(reps[0]["n_tenants"]):
            sds = [rep.get(f"t{i}_slowdown") for rep in reps if rep.get(f"t{i}_slowdown") is not None]
            if len(sds) < 2:
                continue
            mean = sum(sds) / len(sds)
            if mean == 0:
                continue
            std = math.sqrt(sum((x - mean) ** 2 for x in sds) / (len(sds) - 1))
            cv = std / mean
            print(f"  [cv] {base} t{i}: slowdown mean={mean:.3f} std={std:.3f} cv={cv:.3f} "
                  f"(n={len(sds)} repeats)")


def _grow_counts(fam, start=2):
    """Compute the largest tenant count list [start..n_max] that fits in memory.

    n_max is the highest n whose preflight passes at current free memory, capped
    at MAX_TENANTS. Free memory is read fresh from psutil so the result reflects
    the board's actual state at call time (power mode, desktop, other tenants).

    Returns (counts, stop_reason) where counts is a list and stop_reason is a
    short human-readable string naming why growth stopped.
    """
    import psutil
    free_gb = psutil.virtual_memory().available / 1e9
    inst_gb = RESIDENT_GB.get(fam, 1.0)
    counts = []
    stop_reason = f"n={MAX_TENANTS} cap reached"
    for n in range(start, MAX_TENANTS + 1):
        need = inst_gb * n + HEADROOM_GB
        if need > free_gb:
            stop_reason = (
                f"n={n} would need {need:.1f} GB "
                f"(+{HEADROOM_GB} headroom), free {free_gb:.1f} GB"
            )
            break
        counts.append(n)
    if not counts:
        stop_reason = (
            f"n={start} already needs {inst_gb * start + HEADROOM_GB:.1f} GB "
            f"(+{HEADROOM_GB} headroom), free {free_gb:.1f} GB"
        )
    return counts, stop_reason


def run_pair(tenants, tag="run", duration=DEFAULT_DURATION, repeats=1, holdout_n=0,
             keep_suspect=False, mode_label=None, task=None, dataset=None):
    return run_cells([tenants], tag, duration, repeats=repeats, holdout_n=holdout_n,
                     keep_suspect=keep_suspect, mode_label=mode_label,
                     task=task, dataset=dataset)


def run_grid(fam_a, fam_b, tag="grid", k=6, duration=DEFAULT_DURATION, repeats=1,
             holdout_n=0, keep_suspect=False, mode_label=None, min_exit=0,
             task=None, dataset=None):
    anchors_a, anchors_b = _anchor(fam_a, k, min_exit), _anchor(fam_b, k, min_exit)
    cells = [[(fam_a, ea, sa), (fam_b, eb, sb)]
             for (ea, sa) in anchors_a for (eb, sb) in anchors_b]
    # Add holdout cells (off-anchor points for interpolation validation).
    if holdout_n > 0:
        ho_a = _holdout_anchors(fam_a, k, holdout_n, tag)
        ho_b = _holdout_anchors(fam_b, k, holdout_n, tag)
        for ea, sa in ho_a:
            for eb, sb in anchors_b:
                cells.append([(fam_a, ea, sa), (fam_b, eb, sb)])
        for ea, sa in anchors_a:
            for eb, sb in ho_b:
                cells.append([(fam_a, ea, sa), (fam_b, eb, sb)])
    print(f"[grid] {fam_a} x {fam_b} = {len(cells)} cells "
          f"({holdout_n} holdout anchors per family)")
    return run_cells(cells, tag, duration, repeats=repeats, holdout_n=holdout_n,
                     keep_suspect=keep_suspect, mode_label=mode_label,
                     task=task, dataset=dataset)


def run_scenario(name, tag=None, duration=DEFAULT_DURATION, k=6, repeats=1,
                 holdout_n=0, keep_suspect=False, mode_label=None, grow=False,
                 min_exit=0, task=None, dataset=None):
    tag = tag or name
    if name in SCALING:
        fam, default_counts = SCALING[name]
        if grow:
            counts, stop_reason = _grow_counts(fam)
            if not counts:
                print(f"[grow] {name}: no tenant count fits, {stop_reason}")
                return None
            inst_gb = RESIDENT_GB.get(fam, 1.0)
            import psutil
            free_gb = psutil.virtual_memory().available / 1e9
            print(
                f"[grow] {name}: {fam} at {inst_gb:.2f} GB/instance, "
                f"{free_gb:.1f} GB free, sweeping n={counts[0]}..{counts[-1]}; "
                f"{stop_reason}"
            )
        else:
            counts = default_counts
        cells = [[(fam, ex, sub)] * c
                 for (ex, sub) in _anchor(fam, k, min_exit)
                 for c in counts]
        print(f"[scenario {name}] tenancy scaling {counts} of {fam}, "
              f"{k} exit anchors = {len(cells)} cells")
        return run_cells(cells, tag, duration, repeats=repeats, holdout_n=holdout_n,
                         keep_suspect=keep_suspect, mode_label=mode_label,
                         task=task, dataset=dataset)
    if name not in SCENARIOS:
        print(f"[scenario] unknown '{name}'; see --list")
        return None
    if grow:
        print(f"[grow] --grow has no effect on heterogeneous scenario '{name}'; "
              f"running with standard cell list")
    fams = SCENARIOS[name]
    eff_k = k if len(fams) == 2 else min(k, 3)   # triple at k=6 would be 216 cells
    anchors = [_anchor(f, eff_k, min_exit) for f in fams]
    cells = [[(fams[j], e, s) for j, (e, s) in enumerate(combo)]
             for combo in itertools.product(*anchors)]
    print(f"[scenario {name}] {fams} = {len(cells)} cells (k={eff_k})")
    return run_cells(cells, tag, duration, repeats=repeats, holdout_n=holdout_n,
                     keep_suspect=keep_suspect, mode_label=mode_label,
                     task=task, dataset=dataset)


def _append_csv(row, suspect=False, csv_dir=None):
    import csv
    if csv_dir is None:
        csv_dir = OUT_DIR
    fname = "concurrent_slowdown.suspect.csv" if suspect else "concurrent_slowdown.csv"
    path = Path(csv_dir) / fname
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

    # Item 1: calibrate uses wall time.
    n, fallback = calibrate(so, 30.0)
    assert MIN_SAMPLES <= n <= MAX_SAMPLES, f"calibration out of range: {n}"
    # The datasource has total_sec and n, so wall time is available.
    # Wall latency = total_sec / n; for bert exit_0: 22.84s / 869 = 0.02629s/sample.
    # Forward latency = per_sample_sec_mean = 0.01441s.
    # Wall gives n = 30.0 / 0.02629 = ~1141; forward gives 30.0 / 0.01441 = ~2082.
    # Wall count must be smaller.
    n_forward = int(30.0 / so["lat"])
    assert n < n_forward, (
        f"calibrate should use wall time (smaller count {n}) not forward time ({n_forward})"
    )
    assert not fallback, "should not fall back when total_sec and n are present"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    row = build_row([("bert", 0, None)], "selftest", {0: so}, {0: sh}, 1.0, {0: n}, 30.0,
                   timed_overlap_frac=None, repeat_idx=0, calib_fallbacks={0: fallback})
    assert row["t0_slowdown"] and row["agg_throughput"], "core metrics missing"
    assert row["t0_p95_ratio"], "p95 ratio missing"
    assert row["pair_power_w"] and row["pair_energy_j"], "power/energy missing"
    assert "timed_overlap_frac" in row, "timed_overlap_frac missing from row"
    assert row["timed_overlap_frac"] is None, "timed_overlap_frac should be None (old data)"
    # Item 3: STP and ANTT present.
    assert "stp" in row, "stp missing"
    assert "antt" in row, "antt missing"
    # Item 3: throughput_gain uses sum of solo throughputs.
    assert row["agg_throughput_comparable"] is True, "homogeneous cell should be comparable"
    # Item 6: SLO fields present.
    assert "t0_slo_sec" in row, "slo_sec missing"
    # Item 7: overlap gate fields present.
    assert "overlap_gate_used" not in row, \
        "overlap_gate_used is only set in run_cells, not build_row"
    # Item 8: clock fields present (None for datasource runs).
    assert "t0_nvpmodel" in row, "nvpmodel field missing"
    assert "t0_avg_gpu_sm_clock_mhz" in row, "avg_gpu_sm_clock_mhz field missing"
    assert "t0_min_gpu_sm_clock_mhz" in row, "min_gpu_sm_clock_mhz field missing"
    # nvpmodel is read from device_caps, which datasource runs DO carry, so the
    # power mode is recoverable and the "clocks were pinned" claim is checkable.
    assert row["t0_nvpmodel"] == "15W", \
        f"nvpmodel should come from device_caps, got {row['t0_nvpmodel']!r}"
    assert "t0_jetson_clocks" in row, "jetson_clocks field missing"
    assert row["t0_avg_gpu_sm_clock_mhz"] is None, "avg_gpu_sm_clock_mhz should be None for datasource run"
    # Item 4: repeat_idx present.
    assert row["repeat_idx"] == 0, "repeat_idx should be 0"
    # Item 5: is_holdout present.
    assert row["is_holdout"] is False, "is_holdout should be False"
    # Item 2: window fields present.
    assert "t0_window_solo_sec" in row, "window_solo_sec missing"
    assert "t0_window_shared_sec" in row, "window_shared_sec missing"
    # Label normalisation.
    assert _normalize_label("MAXN_SUPER") == "maxn_super", "label normalisation failed"
    assert _normalize_label("15W") == "15w", "15W normalisation failed"
    assert "/" not in _normalize_label("a/b"), "slash not sanitised"
    assert " " not in _normalize_label("a b"), "space not sanitised"
    # Auto-detect returns a string (unknown on dev box where nvpmodel is absent).
    detected = _detect_mode_label()
    assert isinstance(detected, str) and detected, "detect returned empty"
    _append_csv(row, suspect=False)
    print(f"[selftest] calibrate(30s)={n} samples (wall-time, not forward) "
          f"fallback={fallback} | slowdown={row['t0_slowdown']} "
          f"gain={row['throughput_gain']} p95_ratio={row['t0_p95_ratio']} "
          f"power={row['pair_power_w']}W energy={row['pair_energy_j']}J "
          f"stp={row['stp']} antt={row['antt']} "
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
    ap.add_argument("--repeats", type=int, default=1,
                    help="number of times to repeat each cell (default 1); "
                         "each repeat writes a separate row with repeat_idx")
    ap.add_argument("--holdout", type=int, default=0, metavar="N",
                    help="add N off-anchor cells per family for interpolation "
                         "validation (default 0); deterministic from --tag")
    ap.add_argument("--keep-suspect", action="store_true",
                    help="write low-overlap cells to the main CSV instead of "
                         "the suspect sidecar file")
    ap.add_argument("--mode-label", default=None, metavar="LABEL",
                    help="power-mode label for output scoping: logs go under "
                         "logs/multitenant.<label>/ and CSVs under "
                         "result/multitenant/<label>/. "
                         "When omitted, the current mode is auto-detected; "
                         "on failure the label 'unknown' is used.")
    ap.add_argument("--grow", action="store_true",
                    help="for scaling scenarios: sweep tenant counts from 2 up to "
                         "the largest n whose preflight passes at current free memory, "
                         f"capped at MAX_TENANTS={MAX_TENANTS}. "
                         "Ignores the hardcoded count list. "
                         "Has no effect on heterogeneous pair/triple scenarios.")
    ap.add_argument("--min-exit", type=int, default=0, metavar="IDX",
                    help="lowest exit index to sample (default 0; use 1 to skip "
                         "the shallowest exit, which produced physically impossible "
                         "slowdown below 1.0 in the 2026-09 campaign)")
    ap.add_argument("--task", default=None, metavar="TASK",
                    help="override the bert --task flag for all cells "
                         "(default: SST-2 from DATASET_PIN). "
                         "Example: --task QNLI")
    ap.add_argument("--dataset", default=None, metavar="DATASET",
                    help="override the --dataset flag for vision/yolo/llama cells "
                         "(default: uoft-cs/cifar10 / coco / cnn_dailymail from DATASET_PIN). "
                         "Example: --dataset uoft-cs/cifar100")
    a = ap.parse_args()
    if a.k < 2:
        print(f"[error] --k must be at least 2 (got {a.k}); "
              f"_k_points divides by k-1 and would raise ZeroDivisionError")
        sys.exit(1)
    if a.min_exit < 0:
        print(f"[error] --min-exit must be non-negative (got {a.min_exit})")
        sys.exit(1)
    if a.selftest:
        _selftest()
        return

    # Resolve the mode label: explicit flag wins; auto-detect otherwise.
    if a.mode_label is not None:
        mode_label = _normalize_label(a.mode_label)
    else:
        mode_label = _detect_mode_label()
    print(f"[run] mode label: {mode_label!r}")

    if a.list:
        print("scenarios:", ", ".join(list(SCENARIOS) + list(SCALING)))
    elif a.scenario:
        run_scenario(a.scenario, a.tag, a.duration, k=a.k, repeats=a.repeats,
                     holdout_n=a.holdout, keep_suspect=a.keep_suspect,
                     mode_label=mode_label, grow=a.grow, min_exit=a.min_exit,
                     task=a.task, dataset=a.dataset)
    elif a.grid:
        run_grid(a.grid[0], a.grid[1], a.tag or "grid", k=a.k, duration=a.duration,
                 repeats=a.repeats, holdout_n=a.holdout, keep_suspect=a.keep_suspect,
                 mode_label=mode_label, min_exit=a.min_exit,
                 task=a.task, dataset=a.dataset)
    elif a.pair:
        run_pair([parse_tenant(s) for s in a.pair], a.tag or "run", a.duration,
                 repeats=a.repeats, holdout_n=a.holdout, keep_suspect=a.keep_suspect,
                 mode_label=mode_label, task=a.task, dataset=a.dataset)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
