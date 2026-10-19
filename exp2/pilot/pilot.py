"""Pilot gates (steps 0-9 of the paper's "Experimental stages" table).

    python -m exp2.pilot.pilot [--dry-run] --out DIR [--steps 0,1,...] [--config exp2/configs/pilot.json]
    exit 0 all gates passed, 1 a gate failed, 2 usage error

Each gate_* function is pure: measurements in, {"value", "threshold", "pass", "step"} out. Missing or empty
measurements never pass (value None). Steps 0 and 1 are measured here (sysfs root + root helper). Steps 2-9 read
`<out>/measurements/step<N>.json`, written by the sessions in RUNBOOK.md. `--dry-run` uses a fake sysfs tree and
built-in synthetic measurements; it never starts a model.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
from pathlib import Path

from exp2 import contracts as C

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pilot.json"
GATE_IDS = ("inventory", "power_mode_id", "coresident_min_mem_mb", "warm_over_cold_load", "gpu_lock_hold",
            "emc_lock_hold", "switch_latency_s", "ckpt_overhead_frac", "yolo_solo_cv", "energy_vs_f_h2")


def _gate(step, value, threshold, ok):
    return {"value": value, "threshold": threshold, "pass": bool(ok), "step": step}


def _missing(step, threshold):
    return _gate(step, None, threshold, False)


def _nearest_rank(xs, p):
    s = sorted(xs)
    return s[max(1, math.ceil(p * len(s))) - 1]


# ---- gates ---------------------------------------------------------------------------------------------------------
def gate_inventory(info, required, threshold=1.0):
    """Step 0: fraction of required inventory checks that are True (info: {check: bool})."""
    if not info or not required:
        return _missing(0, threshold)
    v = sum(1 for k in required if info.get(k) is True) / len(required)
    return _gate(0, v, threshold, v >= threshold)


def gate_power_mode(mode_id, threshold=2):
    """Step 1: the board reports power mode id 2 (MAXN_SUPER)."""
    if mode_id is None:
        return _missing(1, threshold)
    return _gate(1, mode_id, threshold, mode_id == threshold)


def gate_coresident(m, threshold=500.0):
    """Step 2: all four servers ready and MemAvailable stays above the threshold, idle and under load."""
    if not m or m.get("min_mem_available_mb_idle") is None or m.get("min_mem_available_mb_load") is None:
        return _missing(2, threshold)
    v = float(min(m["min_mem_available_mb_idle"], m["min_mem_available_mb_load"]))
    return _gate(2, v, threshold, bool(m.get("all_ready")) and v >= threshold)


def gate_warm_over_cold(m, threshold=0.5):
    """Step 3: worst family ratio warm_load_s / cold_load_s (m: {family: {cold_load_s, warm_load_s}})."""
    if not m:
        return _missing(3, threshold)
    v = max(f["warm_load_s"] / f["cold_load_s"] for f in m.values())
    return _gate(3, v, threshold, v <= threshold)


def gate_gpu_lock(observed_mhz, target_mhz, threshold=0.99):
    """Step 4: fraction of samples where the GPU clock equals the locked level."""
    if not observed_mhz:
        return _missing(4, threshold)
    v = sum(1 for x in observed_mhz if abs(x - target_mhz) < 0.5) / len(observed_mhz)
    return _gate(4, v, threshold, v >= threshold)


def gate_emc_lock(sysfs_mhz, tegrastats_mhz, target_mhz, threshold=0.99, rel_tol=0.01):
    """Step 5: fraction of paired samples where sysfs AND tegrastats show the locked EMC rate."""
    n = min(len(sysfs_mhz or []), len(tegrastats_mhz or []))
    if n == 0:
        return _missing(5, threshold)
    ok = sum(1 for a, b in zip(sysfs_mhz, tegrastats_mhz)
             if abs(a - target_mhz) <= rel_tol * target_mhz and abs(b - target_mhz) <= rel_tol * target_mhz)
    return _gate(5, ok / n, threshold, ok / n >= threshold)


def gate_switch_latency(latencies_s, threshold=0.5, p=0.95):
    """Step 6: nearest-rank p-quantile of the frequency-switch delay in seconds."""
    if not latencies_s:
        return _missing(6, threshold)
    v = _nearest_rank(latencies_s, p)
    return _gate(6, v, threshold, v <= threshold)


def gate_ckpt_overhead(m, threshold=0.1):
    """Step 7: worst family median(with checkpoints) / median(without) - 1 (m: {family: {with_s, without_s}})."""
    if not m:
        return _missing(7, threshold)
    v = max(statistics.median(f["with_s"]) / statistics.median(f["without_s"]) - 1.0 for f in m.values())
    return _gate(7, v, threshold, v <= threshold)


def gate_yolo_cv(latencies_s, threshold=0.1):
    """Step 8: coefficient of variation (population std / mean) of solo detector latency, strictly below."""
    if not latencies_s or len(latencies_s) < 2:
        return _missing(8, threshold)
    v = statistics.pstdev(latencies_s) / statistics.fmean(latencies_s)
    return _gate(8, v, threshold, v < threshold)


def energy_per_request(cell):
    """Window energy (idle gaps included) divided by completed requests."""
    return cell["window_energy_j"] / cell["n_done"]


def gate_energy_vs_f(m, f_max, threshold=1.0):
    """Step 9 (H2): 1.0 if, for at least one (family, load), energy per request incl. idle is NOT minimal at f_max.

    m: {family: {load: {f_mhz: {window_energy_j, n_done}}}}; the window covers idle gaps and expired requests.
    """
    interior = []
    for fam in (m or {}).values():
        for cells in fam.values():
            if len(cells) < 2 or any(c["n_done"] <= 0 for c in cells.values()):
                continue
            best = min(cells, key=lambda f: energy_per_request(cells[f]))
            interior.append(float(best) != float(f_max))
    if not interior:
        return _missing(9, threshold)
    v = 1.0 if any(interior) else 0.0
    return _gate(9, v, threshold, v >= threshold)


# ---- measurement ---------------------------------------------------------------------------------------------------
def collect_inventory(root):
    root = Path(root)
    return {
        "tegra_release": (root / "etc/nv_tegra_release").is_file(),
        "gpu_devfreq": (root / "sys/class/devfreq/17000000.gpu/cur_freq").is_file(),
        "emc_rate": (root / "sys/kernel/debug/bpmp/debug/clk/emc/rate").is_file(),
        "ina3221": any(p.read_text().strip() == "ina3221" for p in root.glob("sys/class/hwmon/hwmon*/name")),
        "tegrastats": shutil.which("tegrastats") is not None,
        "mps_binary": (root / "usr/bin/nvidia-cuda-mps-control").exists(),
    }


def _synthetic():
    """Built-in passing measurements for --dry-run (shape = what the sessions write)."""
    cells = {"612": {"window_energy_j": 90.0, "n_done": 10}, "1020": {"window_energy_j": 110.0, "n_done": 10}}
    fams = {f: {"cold_load_s": 10.0, "warm_load_s": 2.0} for f in C.FAMILIES}
    return {
        2: {"all_ready": True, "min_mem_available_mb_idle": 1900.0, "min_mem_available_mb_load": 1850.0},
        3: fams,
        4: {"observed_mhz": [1020] * 100},
        5: {"sysfs_mhz": [3199.0] * 100, "tegrastats_mhz": [3199.0] * 100},
        6: {"latencies_s": [0.08, 0.09, 0.1]},
        7: {f: {"with_s": [1.04, 1.04], "without_s": [1.0, 1.0]} for f in C.FAMILIES},
        8: {"latencies_s": [1.0, 1.05, 0.95, 1.0]},
        9: {"bert": {"0.6": cells}},
    }


def _load_measurement(out, step):
    p = Path(out) / "measurements" / f"step{step}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def run_pilot(out, cfg, steps, dry, board=None, root="/"):
    """Evaluate the requested steps and return the pilot_summary dict (also validated)."""
    th, gates = cfg["thresholds"], {}
    synth = _synthetic() if dry else {}
    for s in steps:
        gid = GATE_IDS[s]
        if s == 0:
            info = collect_inventory(root)
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "inventory.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            req = cfg["inventory_required_dry" if dry else "inventory_required"]
            g = gate_inventory(info, req, th[gid])
        elif s == 1:
            mode = None
            if board is not None:
                board.ensure_power_mode(C.POWER_MODE_ID)
                mode = board.query_power_mode()[1]
            g = gate_power_mode(mode, th[gid])
        else:
            m = synth[s] if dry else _load_measurement(out, s)
            g = _eval_measured(s, m, cfg, th[gid])
        gates[gid] = g
    summary = {"schema": C.SCHEMAS["pilot"], "gates": gates, "dry": bool(dry)}
    return C.validate_pilot(summary)


def _eval_measured(s, m, cfg, t):
    if m is None:
        return _missing(s, t)
    if s == 2:
        return gate_coresident(m, t)
    if s == 3:
        return gate_warm_over_cold(m, t)
    if s == 4:
        return gate_gpu_lock(m.get("observed_mhz"), cfg["gpu_target_mhz"], t)
    if s == 5:
        return gate_emc_lock(m.get("sysfs_mhz"), m.get("tegrastats_mhz"), cfg["emc_target_mhz"], t)
    if s == 6:
        return gate_switch_latency(m.get("latencies_s"), t, cfg["switch_p"])
    if s == 7:
        return gate_ckpt_overhead(m, t)
    if s == 8:
        return gate_yolo_cv(m.get("latencies_s"), t)
    return gate_energy_vs_f(m, cfg["f_max_mhz"], t)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exp2.pilot.pilot")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default=",".join(str(i) for i in range(10)))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        return 2
    try:
        steps = sorted({int(x) for x in a.steps.split(",") if x != ""})
        if not steps or min(steps) < 0 or max(steps) > 9:
            raise ValueError("steps must be in 0..9")
        cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        print(f"pilot: {e}", file=sys.stderr)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    helper = None
    try:
        root = "/"
        if a.dry_run:
            from exp2.board.dryroot import make_dry_root
            root = str(make_dry_root(out / "dry_root"))
        if 1 in steps:
            from exp2.board.clocks import Board
            from exp2.board.sysfs import RootHelper
            # real mode: the password comes from env JETSON_SUDO_PW or a getpass prompt, never from a file
            helper = RootHelper.start(mode="dry", fake_root=root) if a.dry_run else RootHelper.start(mode="sudo")
        summary = run_pilot(out, cfg, steps, a.dry_run, Board(helper) if helper else None, root)
    except Exception as e:  # noqa: BLE001 - report any board failure as a usage-level error
        print(f"pilot: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    finally:
        if helper is not None:
            helper.close()
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    failed = [g for g, v in summary["gates"].items() if not v["pass"]]
    print("pilot: %d gates, failed: %s" % (len(summary["gates"]), ", ".join(failed) or "none"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
