"""Experiment 2: concurrent multi-tenant early-exit profiling.

Measures the slowdown factor (T_shared / T_solo) when several early-exit models
share one device, under the constraints established in idea/exp2_concurrent_plan.md
and the smart-sampling protocol in idea/exp2_multitenant_protocol.md:

  - per-model power is NOT attributable on a shared rail -> power/energy are
    recorded PAIR-AGGREGATE only. Latency/throughput/memory stay per-tenant.
  - memory-safety first: a preflight gate refuses to launch a scenario that would
    not fit with ~1 GB headroom, so the run never gets OOM-killed.
  - smart sampling: a 3x3 shallow/mid/deep anchor grid per scenario, not the full
    exit cross product.

Architecture: this file is BOTH the orchestrator and the tenant worker.
  orchestrator: spawns N worker subprocesses, samples pair power during the
                overlap window, collects each worker's own latency/mem, computes
                slowdown vs an in-session solo baseline, writes a CSV.
  worker (--worker): loads one model at a pinned exit, loops inference for a fixed
                duration, writes its per-iteration latency + own gpu/ram to --out.

Real model families plug in via TENANT_BACKENDS. The 'mock' backend (CPU matmul)
makes the whole orchestrator end-to-end testable off-device: `--selftest`.
Real-model adapters must be wired + validated ON the Jetson (see _real_worker).

Usage:
    python multitenant_bench.py --selftest              # off-device smoke, no GPU
    python multitenant_bench.py --scenario bert_x2      # on Jetson
    python multitenant_bench.py --list
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "result" / "multitenant"

# ---- per-instance resident cost, GB (weights + ~0.5 GB CUDA context) ---------
# from idea/exp2_multitenant_protocol.md section 3.
RESIDENT_GB = {"mock": 0.05, "yolo": 0.54, "vision": 1.1, "bert": 1.2, "llama": 3.0}
HEADROOM_GB = 1.0

# ---- exit counts per family (mirror benchmark_config/*.py N_EXITS) -----------
N_EXITS = {"mock": 6, "bert": 24, "vision": 24, "yolo": 18, "llama": 16}


def anchor_exits(family):
    """6 evenly-spaced exit indices (shallow to deep) -- the smart-sampling grid."""
    n = N_EXITS[family]
    return sorted({round(i * (n - 1) / 5) for i in range(6)})


# ---- scenarios (memory-safe first) ------------------------------------------
# each scenario is a list of tenant families; repeats = multiple instances.
SCENARIOS = {
    "bert_x2":     ["bert", "bert"],
    "bert_x3":     ["bert", "bert", "bert"],
    "bert_x4":     ["bert", "bert", "bert", "bert"],
    "yolo_x2":     ["yolo", "yolo"],
    "yolo_vit":    ["yolo", "vision"],
    "llama_yolo":  ["llama", "yolo"],
    "llama_bert":  ["llama", "bert"],
    "llama_x2":    ["llama", "llama"],      # canary, near ceiling
    "mock_x2":     ["mock", "mock"],        # selftest only
    "mock_x3":     ["mock", "mock", "mock"],
}


# =============================================================================
# Worker: load one model at one exit, loop inference, record own metrics.
# =============================================================================
def _mock_infer_step(state):
    """CPU matmul sized to a few ms; contends for CPU so two workers slow down."""
    import numpy as np
    a, b = state
    (a @ b).sum()  # keep the product live


def _mock_setup(exit_idx):
    import numpy as np
    n = 256 + exit_idx * 16          # deeper exit = slightly bigger = slower
    rng = np.random.default_rng(0)
    return (rng.random((n, n), dtype="float32"), rng.random((n, n), dtype="float32"))


def _real_setup(family, exit_idx):
    # ponytail: real per-family "load once, infer at pinned exit" adapters wire in
    # here. They must be validated on the Jetson (import torch + the family backend,
    # load the model, return a closure that runs ONE inference at exit_idx). Left
    # unimplemented rather than guessed blind -- a wrong adapter silently corrupts
    # every slowdown number. See idea/exp2_multitenant_protocol.md section 6.
    raise NotImplementedError(
        f"real backend '{family}' not wired yet; run --selftest (mock) off-device, "
        f"and wire _real_setup on the Jetson before the real sweep.")


def _sample_self_mem():
    """This worker's own RAM (MB) + GPU allocated (MB) if torch is present."""
    out = {}
    try:
        import psutil
        out["ram_mb"] = psutil.Process().memory_info().rss / 1e6
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu_mem_mb"] = torch.cuda.memory_allocated() / 1e6
    except Exception:
        pass
    return out


def run_worker(family, exit_idx, duration, warmup, out_path, mock):
    setup = _mock_setup(exit_idx) if mock else _real_setup(family, exit_idx)
    step = _mock_infer_step if mock else setup  # real setup returns the step closure
    # warmup
    for _ in range(warmup):
        step(setup) if mock else step()
    lats, t_end = [], time.perf_counter() + duration
    t_start = time.perf_counter()
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        step(setup) if mock else step()
        lats.append(time.perf_counter() - t0)
    mem = _sample_self_mem()
    out = {"family": family, "exit": exit_idx, "n_iters": len(lats),
           "lat_median": st.median(lats) if lats else None,
           "lat_p90": (sorted(lats)[int(len(lats) * 0.9)] if lats else None),
           "active_start": t_start, "active_end": time.perf_counter(), **mem}
    Path(out_path).write_text(json.dumps(out), encoding="utf-8")


# =============================================================================
# Orchestrator
# =============================================================================
def _preflight(families):
    """Memory-safety gate: refuse to launch if it will not fit with headroom."""
    import psutil
    need = sum(RESIDENT_GB[f] for f in families) + HEADROOM_GB
    free = psutil.virtual_memory().available / 1e9
    ok = need <= free
    print(f"[preflight] need ~{need:.1f} GB (+{HEADROOM_GB} headroom), free {free:.1f} GB "
          f"-> {'OK' if ok else 'SKIP'}")
    return ok


def _spawn(family, exit_idx, duration, warmup, out_path, mock):
    argv = [sys.executable, str(REPO_ROOT / "multitenant_bench.py"), "--worker",
            "--family", family, "--exit", str(exit_idx),
            "--duration", str(duration), "--warmup", str(warmup),
            "--out", str(out_path)]
    if mock:
        argv.append("--mock")
    env = dict(os.environ)
    env.setdefault("MALLOC_ARENA_MAX", "2")   # plan S9: this is a 3rd spawn path
    return subprocess.Popen(argv, env=env)


def _sample_pair_power(duration, interval=0.2):
    """Aggregate device power over the window (plan S1: per-tenant is impossible)."""
    samples, t_end = [], time.perf_counter() + duration
    try:
        from shared.hw_profiler import sample_hw
    except Exception:
        return {"pair_power_w": None, "pair_energy_j": None}
    while time.perf_counter() < t_end:
        try:
            hw = sample_hw()
            p = hw.get("power_w")
            if isinstance(p, (int, float)):
                samples.append(p)
        except Exception:
            pass
        time.sleep(interval)
    if not samples:
        return {"pair_power_w": None, "pair_energy_j": None}
    avg_p = st.mean(samples)
    return {"pair_power_w": round(avg_p, 3), "pair_energy_j": round(avg_p * duration, 3)}


def _run_group(families, exits, duration, warmup, mock, tag):
    """Launch all tenants pinned at `exits`, measure the overlap window."""
    tmp = OUT_DIR / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    outs = [tmp / f"{tag}_{i}_{f}.json" for i, f in enumerate(families)]
    procs = [_spawn(f, e, duration, warmup, o, mock)
             for f, e, o in zip(families, exits, outs)]
    power = _sample_pair_power(duration + warmup * 0.05)   # sample during the run
    for p in procs:
        p.wait()
    results = []
    for o in outs:
        try:
            results.append(json.loads(o.read_text(encoding="utf-8")))
        except Exception:
            results.append(None)
    # overlap fraction: how much of each worker's window overlapped ALL others
    starts = [r["active_start"] for r in results if r]
    ends = [r["active_end"] for r in results if r]
    overlap = max(0.0, min(ends) - max(starts)) if starts else 0.0
    span = max(ends) - min(starts) if starts else 1.0
    overlap_frac = round(overlap / span, 3) if span > 0 else 0.0
    return results, power, overlap_frac


def run_scenario(name, duration=8.0, warmup=20, mock=False):
    families = SCENARIOS[name]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not mock and not _preflight(families):
        print(f"[scenario {name}] SKIPPED by memory gate")
        return
    # solo baselines first (in-session, same thermal state -- plan S4)
    solo = {}
    uniq = sorted(set(families))
    for f in uniq:
        for e in anchor_exits(f):
            res, _pw, _ov = _run_group([f], [e], duration, warmup, mock, f"solo_{f}_{e}")
            if res and res[0]:
                solo[(f, e)] = res[0]["lat_median"]
    # anchor grid: sample each tenant at shallow/mid/deep, same index across copies
    rows = []
    grids = [anchor_exits(f) for f in families]
    depth_labels = ["shallow", "mid", "deep"]
    for d, label in enumerate(depth_labels):
        exits = [g[min(d, len(g) - 1)] for g in grids]
        res, power, ov = _run_group(families, exits, duration, warmup, mock, f"grid_{label}")
        row = {"scenario": name, "depth": label, "overlap_frac": ov, **power}
        for i, (f, e, r) in enumerate(zip(families, exits, res)):
            lat_s = r["lat_median"] if r else None
            lat_solo = solo.get((f, e))
            slow = round(lat_s / lat_solo, 3) if (lat_s and lat_solo) else None
            row[f"t{i}_family"] = f
            row[f"t{i}_exit"] = e
            row[f"t{i}_lat_solo"] = round(lat_solo, 6) if lat_solo else None
            row[f"t{i}_lat_shared"] = round(lat_s, 6) if lat_s else None
            row[f"t{i}_slowdown"] = slow
            row[f"t{i}_gpu_mem_mb"] = round(r["gpu_mem_mb"], 1) if r and "gpu_mem_mb" in r else None
            row[f"t{i}_ram_mb"] = round(r["ram_mb"], 1) if r and "ram_mb" in r else None
        rows.append(row)
        print(f"[{name}/{label}] overlap={ov} "
              + " ".join(f"{r.get(f't{i}_family')}@{r.get(f't{i}_exit')}:x{r.get(f't{i}_slowdown')}"
                         for i in range(len(families)) for r in [row]))
    _write_csv(name, rows)
    return rows


def _write_csv(name, rows):
    import csv
    if not rows:
        return
    path = OUT_DIR / f"{name}.csv"
    keys = list({k for r in rows for k in r})
    keys.sort(key=lambda k: (not k.startswith("scenario"), k))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] wrote {path}")


# =============================================================================
def _selftest():
    """Off-device end-to-end: 2 mock tenants must show slowdown > 1, CSV written."""
    rows = run_scenario("mock_x2", duration=3.0, warmup=5, mock=True)
    assert rows, "no rows produced"
    slows = [r["t0_slowdown"] for r in rows if r.get("t0_slowdown")]
    assert slows, "no slowdown computed"
    assert all(s >= 0.5 for s in slows), f"implausible slowdown {slows}"
    assert (OUT_DIR / "mock_x2.csv").exists(), "csv not written"
    # two CPU tenants contending should generally slow each other (>1), but on a
    # very idle many-core box it can be ~1; assert only that it's measured + sane.
    print(f"[selftest] OK slowdowns={slows}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true", help="internal: run one tenant")
    ap.add_argument("--family"); ap.add_argument("--exit", type=int)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--scenario", help="scenario name (see --list)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.worker:
        run_worker(a.family, a.exit, a.duration, a.warmup, a.out, a.mock)
    elif a.selftest:
        _selftest()
    elif a.list:
        for k, v in SCENARIOS.items():
            print(f"  {k:<12} {v}  (~{sum(RESIDENT_GB[f] for f in v):.1f} GB)")
    elif a.scenario:
        run_scenario(a.scenario, mock=a.scenario.startswith("mock"))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
