"""Recompute multi-tenant interference metrics from raw hw_results.json files.

Every number here comes from the raw per-tenant hw_results.json trees.  The
existing concurrent_slowdown.csv is used only to look up the intended tenant
count per cell; nothing else is taken from it.

A cell is COMPLETE when every intended tenant has a parseable hw_results.json
and every tenant has a valid solo baseline.  INCOMPLETE cells are listed
separately and excluded from the main metrics table.

Usage
-----
    python multitenant_analyze.py <logroot> [--mode-label LABEL] \\
        [--out result/multitenant/analysis.csv]

Directory conventions (auto-detected under <logroot>)
------------------------------------------------------
Concurrent tenant folder:
    mt_conc_<basetag>_<cellidx>_r0_<tid>_<fam>_<exit>/
    <fam>/<dataset>/<workspace>/exit_<N>/hw_results.json

Solo baseline folder (calibrated; the _n<count> suffix distinguishes it from
mt_solo_probe_* folders which are ignored):
    mt_solo_<basetag>_<fam>_<exit>_n<count>/
    <fam>/<dataset>/<workspace>/exit_<N>/hw_results.json
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

# shared.csv_export._load is the proven hw_results.json reader used in the
# non-multi-tenant pipeline.  It returns the aggregate dict with std_<key>
# fields injected from per-sample distributions.
#
# NOTE: the coordinator referred to this function as "load_hw" but the real
# name in shared/csv_export.py is the private function _load; there is no
# public load_hw alias anywhere in the repo.
#
# We load the module directly via importlib so that shared/__init__.py (which
# imports torch at load time) is never executed.  This keeps offline/test
# environments working without torch installed.
import importlib.util as _ilu

_csv_export_spec = _ilu.spec_from_file_location(
    "shared.csv_export",
    Path(__file__).parent / "shared" / "csv_export.py",
)
_csv_export_mod = _ilu.module_from_spec(_csv_export_spec)
_csv_export_spec.loader.exec_module(_csv_export_mod)
_load_hw = _csv_export_mod._load  # loads aggregate dict + std_<key> from samples

_fairness_spec = _ilu.spec_from_file_location(
    "shared.fairness",
    Path(__file__).parent / "shared" / "fairness.py",
)
_fairness_mod = _ilu.module_from_spec(_fairness_spec)
_fairness_spec.loader.exec_module(_fairness_mod)
_jain_fairness = _fairness_mod.jain_fairness
_latency_dispersion = _fairness_mod.latency_dispersion

import json as _json


def _load_samples(hw_json: Path) -> list:
    """Return the raw per-sample list from hw_results.json, or [] if absent."""
    try:
        data = _json.loads(hw_json.read_text(encoding="utf-8"))
        return data.get("samples", [])
    except Exception:
        return []


def compute_timeseries(samples: list) -> tuple[list, bool]:
    """Bucket samples by elapsed_sec into per-second rows.

    Returns (buckets, has_elapsed) where has_elapsed is False when none of the
    samples carry the elapsed_sec field (recorded before this change).

    Each bucket dict has:
      t          -- integer second index (0, 1, 2, ...)
      mean_power_w
      energy_j   -- mean_power_w * bucket_duration_sec  (energy = power * time)
      n_samples  -- count of samples in this bucket (throughput for that second)
      mean_lat_sec
      duration_sec -- actual bucket duration (last bucket is usually partial)
    """
    # ponytail: check the first sample; if elapsed_sec absent in any, treat run as old.
    has_elapsed = any(s.get("elapsed_sec") is not None for s in samples)
    if not has_elapsed:
        return [], False

    timed = [s for s in samples if s.get("elapsed_sec") is not None]
    if not timed:
        return [], False

    max_elapsed = max(s["elapsed_sec"] for s in timed)

    buckets = []
    # Build bucket index once: t -> list of samples
    from collections import defaultdict as _dd
    bucket_map: dict = _dd(list)
    for s in timed:
        t = int(s["elapsed_sec"])
        bucket_map[t].append(s)

    all_t = sorted(bucket_map)
    for t in all_t:
        slist = bucket_map[t]
        # Actual bucket duration: full second except for the final bucket.
        # Final bucket ends at max_elapsed; its duration is max_elapsed - t.
        is_last = (t == all_t[-1])
        if is_last:
            duration = max_elapsed - t
            if duration <= 0:
                duration = max_elapsed - (t - 1) if t > 0 else max_elapsed
        else:
            duration = 1.0

        powers = [s["power_w"] for s in slist
                  if isinstance(s.get("power_w"), (int, float))]
        mean_power = sum(powers) / len(powers) if powers else None

        # energy_j = mean power (W) * duration (s), i.e. energy = power * time
        energy = round(mean_power * duration, 6) if mean_power is not None else None

        lats = []
        for s in slist:
            for k in ("end_to_end_sec", "forward_sec"):
                v = s.get(k)
                if isinstance(v, (int, float)) and v > 0:
                    lats.append(v)
                    break
        mean_lat = sum(lats) / len(lats) if lats else None

        clocks = [s["gpu_sm_clock_mhz"] for s in slist
                  if isinstance(s.get("gpu_sm_clock_mhz"), (int, float))]
        mean_clock = round(sum(clocks) / len(clocks), 2) if clocks else None
        min_clock = round(min(clocks), 2) if clocks else None

        temps = [s["gpu_temperature_c"] for s in slist
                 if isinstance(s.get("gpu_temperature_c"), (int, float))]
        mean_temp = round(sum(temps) / len(temps), 2) if temps else None

        buckets.append({
            "t": t,
            "mean_power_w": round(mean_power, 4) if mean_power is not None else None,
            "energy_j": energy,
            "n_samples": len(slist),
            "mean_lat_sec": round(mean_lat, 6) if mean_lat is not None else None,
            "duration_sec": round(duration, 6),
            "mean_gpu_clock_mhz": mean_clock,
            "min_gpu_clock_mhz": min_clock,
            "mean_gpu_temp_c": mean_temp,
        })

    return buckets, True


# A power reading below this floor under GPU load is physically implausible on
# Jetson Orin Nano and is treated as missing rather than averaged in.
# ponytail: empirical floor; raise if a lighter power mode is added.
_MIN_POWER_W = 0.5


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _collect_hw_results(folder: Path) -> list[Path]:
    """Return all hw_results.json paths nested under folder."""
    return list(folder.rglob("hw_results.json"))


def _tenant_metrics(hw_files: list[Path]) -> Optional[dict]:
    """Average metrics across all hw_results.json files for one tenant.

    Uses shared.csv_export._load (the same reader used by the non-multi-tenant
    pipeline) to load each hw_results.json.  Returns None if no file is
    parseable.  Power readings below _MIN_POWER_W are excluded from the power
    average; the exclusion count is noted in the returned dict.
    """
    records = []
    for p in hw_files:
        try:
            a = _load_hw(p)
        except Exception:
            continue
        # Require the two throughput-and-latency fields.
        if a.get("per_sample_sec_mean") is None or a.get("throughput_samples_per_sec") is None:
            continue
        records.append(a)

    if not records:
        return None

    def _mean(key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    power_vals = [r["avg_power_w"] for r in records
                  if r.get("avg_power_w") is not None and r["avg_power_w"] >= _MIN_POWER_W]
    power_excluded = len(records) - len(power_vals)

    return {
        "lat": _mean("per_sample_sec_mean"),
        "thru": _mean("throughput_samples_per_sec"),
        "power": sum(power_vals) / len(power_vals) if power_vals else None,
        "power_excluded_datasets": power_excluded,
        "energy": _mean("avg_energy_j"),
        "gpu_mem_static_mb": _mean("gpu_mem_static_mb"),
        "gpu_mem_dynamic_mb": _mean("gpu_mem_dynamic_mb"),
        "peak_vram_mb": _mean("peak_vram_allocated_mb"),
        "n_datasets": len(records),
    }


# ---------------------------------------------------------------------------
# Folder discovery
# ---------------------------------------------------------------------------

def _parse_conc_folder(name: str) -> Optional[tuple]:
    """Parse mt_conc_<basetag>_<cellidx>_r0_<tid>_<fam>_<exit> folder name.

    Returns (basetag, cellidx, tid, fam, exit_n) or None.
    The tricky part is that basetag itself may contain underscores.
    The known fixed suffix pattern is: ..._<cellidx>_r0_<tid>_<fam>_<exit>
    where cellidx, tid, exit_n are integers and fam is a word.
    """
    if not name.startswith("mt_conc_"):
        return None
    rest = name[len("mt_conc_"):]
    # Split from the right: exit_n (int), fam (word), tid (int), "r0", cellidx (int), then basetag
    parts = rest.rsplit("_", 5)
    if len(parts) != 6:
        return None
    basetag_part, cellidx_s, r0, tid_s, fam, exit_s = parts
    if r0 != "r0":
        return None
    try:
        cellidx = int(cellidx_s)
        tid = int(tid_s)
        exit_n = int(exit_s)
    except ValueError:
        return None
    return basetag_part, cellidx, tid, fam, exit_n


def _parse_solo_folder(name: str) -> Optional[tuple]:
    """Parse mt_solo_<basetag>_<fam>_<exit>_n<count> folder name.

    Returns (basetag, fam, exit_n) or None.
    Ignores mt_solo_probe_* folders.
    """
    if not name.startswith("mt_solo_"):
        return None
    if name.startswith("mt_solo_probe_"):
        return None  # explicitly excluded
    rest = name[len("mt_solo_"):]
    # Suffix is _n<count> where count is an integer.
    # Split off _n<count> first.
    idx = rest.rfind("_n")
    if idx < 0:
        return None
    core = rest[:idx]
    count_s = rest[idx + 2:]
    if not count_s.isdigit():
        return None
    # core = <basetag>_<fam>_<exit>
    # Split from right: exit (int), fam (word), basetag.
    parts = core.rsplit("_", 2)
    if len(parts) != 3:
        return None
    basetag_part, fam, exit_s = parts
    try:
        exit_n = int(exit_s)
    except ValueError:
        return None
    return basetag_part, fam, exit_n


def discover_cells(logroot: Path) -> dict:
    """Return {(basetag, cellidx): {tid: Path}} for all concurrent cells."""
    cells: dict = {}
    for entry in sorted(logroot.iterdir()):
        if not entry.is_dir():
            continue
        parsed = _parse_conc_folder(entry.name)
        if parsed is None:
            continue
        basetag, cellidx, tid, fam, exit_n = parsed
        key = (basetag, cellidx)
        cells.setdefault(key, {})[tid] = entry
    return cells


def discover_solos(logroot: Path) -> dict:
    """Return {(basetag, fam, exit_n): Path} for all solo baseline folders.

    mt_solo_probe_* folders are ignored.
    """
    solos: dict = {}
    for entry in sorted(logroot.iterdir()):
        if not entry.is_dir():
            continue
        parsed = _parse_solo_folder(entry.name)
        if parsed is None:
            continue
        basetag, fam, exit_n = parsed
        key = (basetag, fam, exit_n)
        # Prefer the most recent entry (sorted last) if there are duplicates.
        solos[key] = entry
    return solos


# ---------------------------------------------------------------------------
# CSV lookup for intended tenant count
# ---------------------------------------------------------------------------

def load_intended_counts(logroot: Path) -> tuple[dict, str]:
    """Search for concurrent_slowdown.csv near logroot.

    Looks in <logroot> itself, then ../result/multitenant/<modename>/ where
    modename is derived from the logroot directory name after the last dot.

    Returns ({csv_tag: int}, source_description).
    """
    candidates = []
    # Check <logroot>/concurrent_slowdown.csv first (covers test fixtures and ad-hoc layouts).
    candidates.append(logroot / "concurrent_slowdown.csv")
    # Peer result directory: e.g. logroot = .../logs/multitenant.maxn_super
    # -> .../result/multitenant/maxn_super/concurrent_slowdown.csv
    mode_label = logroot.name.split(".")[-1] if "." in logroot.name else logroot.name
    peer_result = logroot.parent.parent / "result" / "multitenant" / mode_label / "concurrent_slowdown.csv"
    candidates.append(peer_result)
    candidates.append(peer_result.parent.parent / "concurrent_slowdown.csv")
    # Also check mt_all result tree when logroot is inside mt_all
    for part in logroot.parts:
        if part.startswith("mt_"):
            root = Path(*logroot.parts[:logroot.parts.index(part) + 1])
            candidates.append(root / "result" / "multitenant" / mode_label / "concurrent_slowdown.csv")
            candidates.append(root / "result" / "multitenant" / "concurrent_slowdown.csv")
            break

    # Deduplicate while preserving order.
    seen: set = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    for csv_path in unique_candidates:
        if not csv_path.exists():
            continue
        counts: dict = {}
        try:
            with csv_path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    tag = row.get("tag", "").strip()
                    n_s = row.get("n_tenants", "").strip()
                    if tag and n_s:
                        try:
                            counts[tag] = int(float(n_s))
                        except ValueError:
                            pass
        except Exception:
            continue
        if counts:
            return counts, str(csv_path)

    return {}, "not found"


# ---------------------------------------------------------------------------
# Cell-level analysis
# ---------------------------------------------------------------------------

def _infer_cell_attrs(cell_tid_map: dict) -> tuple[str, str, int]:
    """Extract (basetag, fam, exit_n) from the set of folder paths in a cell."""
    # All folders in the cell share the same basetag/fam/exit; pick the first.
    folder = next(iter(cell_tid_map.values()))
    parsed = _parse_conc_folder(folder.name)
    if parsed is None:
        return "", "", 0
    basetag, _, _, fam, exit_n = parsed
    return basetag, fam, exit_n


def analyze_cell(
    basetag: str,
    cellidx: int,
    tid_map: dict,
    solos: dict,
    intended: int,
    intended_source: str,
) -> dict:
    """Analyze one concurrent cell and return a result dict.

    Result has 'status' = 'complete' or 'incomplete', plus all metrics for
    complete cells or a 'reason' string for incomplete ones.
    """
    cell_tag = f"{basetag}_{cellidx}_r0"
    _, fam, exit_n = _infer_cell_attrs(tid_map)

    # Compute per-tenant metrics.
    tenant_results: dict = {}
    tenant_samples: dict = {}  # tid -> flat list of latency floats (all hw_files combined)
    failed_tids: list = []
    for tid, folder in sorted(tid_map.items()):
        hw_files = _collect_hw_results(folder)
        m = _tenant_metrics(hw_files)
        if m is None:
            failed_tids.append(tid)
        else:
            tenant_results[tid] = m
            # Collect all per-sample latencies for within-run dispersion.
            lats: list = []
            for p in hw_files:
                for s in _load_samples(p):
                    for k in ("end_to_end_sec", "forward_sec"):
                        v = s.get(k)
                        if isinstance(v, (int, float)) and v > 0:
                            lats.append(v)
                            break
            tenant_samples[tid] = lats

    logged = len(tenant_results)

    # Check completeness: logged count must match intended.
    reasons: list = []
    if failed_tids:
        reasons.append(f"tids {failed_tids} had no parseable hw_results.json")
    if logged < intended:
        missing_tids = sorted(set(range(intended)) - set(tid_map.keys()))
        # Note: missing_tids is approximate when tids are not contiguous.
        reasons.append(
            f"logged={logged} < intended={intended}"
            + (f"; absent tid folders approx {missing_tids}" if missing_tids else "")
        )

    # Check solo baseline.
    solo_key = (basetag, fam, exit_n)
    solo_folder = solos.get(solo_key)
    solo_metrics = None
    if solo_folder is None:
        reasons.append(f"no solo baseline for ({basetag}, {fam}, exit {exit_n})")
    else:
        hw_files = _collect_hw_results(solo_folder)
        solo_metrics = _tenant_metrics(hw_files)
        if solo_metrics is None:
            reasons.append(f"solo baseline folder present but no parseable hw_results.json")

    if reasons or logged != intended:
        return {
            "status": "incomplete",
            "tag": cell_tag,
            "fam": fam,
            "exit": exit_n,
            "intended": intended,
            "intended_source": intended_source,
            "logged": logged,
            "reason": "; ".join(reasons) if reasons else f"logged={logged} != intended={intended}",
        }

    # All checks passed: compute derived metrics.
    lat_solo = solo_metrics["lat"]
    thru_solo = solo_metrics["thru"]

    per_tenant: list = []
    power_warnings: list = []
    for tid in sorted(tenant_results):
        m = tenant_results[tid]
        if m["power"] is None:
            power_warnings.append(f"tid {tid}: all power readings below floor ({_MIN_POWER_W} W), excluded")
        if m["power"] is not None and m["power"] < _MIN_POWER_W:
            # Redundant guard: _tenant_metrics already filters these out, but
            # belt-and-suspenders for any edge case.
            m = dict(m, power=None)
            power_warnings.append(f"tid {tid}: power below floor, excluded")
        slowdown = m["lat"] / lat_solo if lat_solo and lat_solo > 0 else None
        thru_ratio = m["thru"] / thru_solo if thru_solo and thru_solo > 0 else None
        disp = _latency_dispersion(tenant_samples.get(tid, []))
        lat_cv = disp["cv"] if disp else None
        per_tenant.append({
            "tid": tid,
            "lat_shared": m["lat"],
            "thru_shared": m["thru"],
            "slowdown": slowdown,
            "thru_ratio": thru_ratio,
            "power": m["power"],
            "energy": m["energy"],
            "gpu_mem_static_mb": m["gpu_mem_static_mb"],
            "gpu_mem_dynamic_mb": m["gpu_mem_dynamic_mb"],
            "peak_vram_mb": m["peak_vram_mb"],
            "power_excluded_datasets": m["power_excluded_datasets"],
            "lat_cv": lat_cv,
        })

    slowdowns = [t["slowdown"] for t in per_tenant if t["slowdown"] is not None]
    thru_ratios = [t["thru_ratio"] for t in per_tenant if t["thru_ratio"] is not None]

    antt = sum(slowdowns) / len(slowdowns) if slowdowns else None
    stp = sum(thru_ratios) if thru_ratios else None
    agg_thru = sum(t["thru_shared"] for t in per_tenant if t["thru_shared"] is not None)

    # agg_thru is only comparable across cells when all tenants share one family.
    all_same_fam = True  # by construction: all folders in a cell share the same fam/exit

    fairness = _jain_fairness(slowdowns)

    return {
        "status": "complete",
        "tag": cell_tag,
        "fam": fam,
        "exit": exit_n,
        "n_tenants": intended,
        "lat_solo": lat_solo,
        "thru_solo": thru_solo,
        "per_tenant": per_tenant,
        "stp": stp,
        "antt": antt,
        "agg_thru": agg_thru,
        "agg_thru_comparable": all_same_fam,
        "power_warnings": power_warnings,
        "fairness": fairness,
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def _emit_timeseries(logroot: Path, mode_label: str, ts_out: Optional[Path]) -> None:
    """Print and optionally write per-second time-series for all hw_results.json found.

    Terminology: energy is measured in joules; power is measured in watts and equals
    joules per second; energy equals power multiplied by time.  There is no standard
    named quantity for power divided by time.

    Runs whose samples lack elapsed_sec (recorded before benchmark_profiler gained the
    field) are reported as skipped rather than silently dropped.
    """
    all_hw = list(logroot.rglob("hw_results.json"))
    if not all_hw:
        print("[timeseries] no hw_results.json files found under", logroot)
        return

    ts_rows: list = []
    print("=" * 80)
    print("TIME-SERIES (per-second buckets)")
    if mode_label:
        print(f"Mode: {mode_label}")
    print(
        f"  {'run':<60} {'t':>4}  {'power_w':>8}  {'energy_j':>9}  "
        f"{'n_samples':>9}  {'lat_sec':>8}  {'dur_sec':>8}  "
        f"{'clk_mean':>9}  {'clk_min':>8}  {'temp_c':>7}"
    )
    print("-" * 145)

    skipped = 0
    for hw_path in sorted(all_hw):
        run_label = str(hw_path.relative_to(logroot))
        samples = _load_samples(hw_path)
        if not samples:
            print(f"  {run_label:<60} [no samples]")
            skipped += 1
            continue
        buckets, has_elapsed = compute_timeseries(samples)
        if not has_elapsed:
            print(f"  {run_label:<60} [SKIPPED: samples lack elapsed_sec, recorded before this feature]")
            skipped += 1
            continue
        for b in buckets:
            pw_s = f"{b['mean_power_w']:.4f}" if b["mean_power_w"] is not None else "  N/A"
            ej_s = f"{b['energy_j']:.6f}" if b["energy_j"] is not None else "     N/A"
            lat_s = f"{b['mean_lat_sec']:.6f}" if b["mean_lat_sec"] is not None else "     N/A"
            clkm_s = f"{b['mean_gpu_clock_mhz']:.2f}" if b["mean_gpu_clock_mhz"] is not None else "     N/A"
            clki_s = f"{b['min_gpu_clock_mhz']:.2f}" if b["min_gpu_clock_mhz"] is not None else "    N/A"
            temp_s = f"{b['mean_gpu_temp_c']:.2f}" if b["mean_gpu_temp_c"] is not None else "   N/A"
            print(
                f"  {run_label:<60} {b['t']:>4}  {pw_s:>8}  {ej_s:>9}  "
                f"{b['n_samples']:>9}  {lat_s:>8}  {b['duration_sec']:>8.6f}  "
                f"{clkm_s:>9}  {clki_s:>8}  {temp_s:>7}"
            )
            if ts_out is not None:
                ts_rows.append({
                    "run": run_label,
                    "t": b["t"],
                    "mean_power_w": b["mean_power_w"],
                    "energy_j": b["energy_j"],
                    "n_samples": b["n_samples"],
                    "mean_lat_sec": b["mean_lat_sec"],
                    "duration_sec": b["duration_sec"],
                    "mean_gpu_clock_mhz": b["mean_gpu_clock_mhz"],
                    "min_gpu_clock_mhz": b["min_gpu_clock_mhz"],
                    "mean_gpu_temp_c": b["mean_gpu_temp_c"],
                })

    print()
    if skipped:
        print(f"  {skipped} run(s) skipped (no elapsed_sec in samples).")
    print()

    if ts_out and ts_rows:
        ts_out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["run", "t", "mean_power_w", "energy_j", "n_samples", "mean_lat_sec", "duration_sec",
                      "mean_gpu_clock_mhz", "min_gpu_clock_mhz", "mean_gpu_temp_c"]
        with ts_out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ts_rows)
        print(f"[timeseries] CSV written to {ts_out} ({len(ts_rows)} rows)")


def run(logroot: Path, mode_label: str, out_path: Optional[Path],
        timeseries: bool = False, ts_out: Optional[Path] = None) -> int:
    """Run the full analysis and return an exit code (0 = success).

    timeseries: when True, emit per-second power/energy/throughput buckets for
                each run whose samples carry elapsed_sec.
    ts_out:     optional CSV path; when given, write one row per (run, second).
    """
    if not logroot.is_dir():
        print(f"ERROR: logroot does not exist or is not a directory: {logroot}", file=sys.stderr)
        return 1

    cells = discover_cells(logroot)
    solos = discover_solos(logroot)
    intended_counts, csv_source = load_intended_counts(logroot)

    if not cells:
        print("No concurrent cell folders found under", logroot)
        return 0

    results: list = []
    for (basetag, cellidx), tid_map in sorted(cells.items()):
        csv_tag = f"{basetag}_{cellidx}_r0"
        if csv_tag in intended_counts:
            intended = intended_counts[csv_tag]
            intended_source = "csv"
        else:
            # Fall back: max observed tid + 1.
            intended = max(tid_map.keys()) + 1
            intended_source = "inferred (max tid + 1, no CSV row)"
        result = analyze_cell(basetag, cellidx, tid_map, solos, intended, intended_source)
        results.append(result)

    complete = [r for r in results if r["status"] == "complete"]
    incomplete = [r for r in results if r["status"] == "incomplete"]

    # Print mode label header.
    if mode_label:
        print(f"Mode: {mode_label}")
    print(f"Logroot: {logroot}")
    print(f"Intended-count source: {csv_source}")
    print()

    # --- COMPLETE cells table ---
    if complete:
        print("=" * 80)
        print("COMPLETE CELLS")
        print("=" * 80)
        hdr = (
            f"{'tag':<42} {'fam':<6} {'exit':>4} {'n':>2}  "
            f"{'lat_solo':>9} {'lat_sh[mean]':>12}  "
            f"{'thru_solo':>10} {'thru_sh[sum]':>12}  "
            f"{'ANTT':>6} {'STP':>6}  "
            f"{'agg_thru':>9} {'comp':>4}  "
            f"{'power[mean]':>11} {'energy[mean]':>12}  "
            f"{'static_mb':>9} {'dyn_mb':>7}"
        )
        print(hdr)
        print("-" * 160)
        for r in complete:
            pt = r["per_tenant"]
            mean_lat_sh = sum(t["lat_shared"] for t in pt if t["lat_shared"] is not None) / len(pt)
            sum_thru_sh = r["agg_thru"]
            mean_power = [t["power"] for t in pt if t["power"] is not None]
            mean_power_v = sum(mean_power) / len(mean_power) if mean_power else float("nan")
            mean_energy = [t["energy"] for t in pt if t["energy"] is not None]
            mean_energy_v = sum(mean_energy) / len(mean_energy) if mean_energy else float("nan")
            mean_static = [t["gpu_mem_static_mb"] for t in pt if t["gpu_mem_static_mb"] is not None]
            mean_static_v = sum(mean_static) / len(mean_static) if mean_static else float("nan")
            mean_dyn = [t["gpu_mem_dynamic_mb"] for t in pt if t["gpu_mem_dynamic_mb"] is not None]
            mean_dyn_v = sum(mean_dyn) / len(mean_dyn) if mean_dyn else float("nan")
            comp_flag = "Y" if r["agg_thru_comparable"] else "N"
            antt_s = f"{r['antt']:.4f}" if r["antt"] is not None else "  N/A"
            stp_s = f"{r['stp']:.4f}" if r["stp"] is not None else "  N/A"
            print(
                f"{r['tag']:<42} {r['fam']:<6} {r['exit']:>4} {r['n_tenants']:>2}  "
                f"{r['lat_solo']:>9.5f} {mean_lat_sh:>12.5f}  "
                f"{r['thru_solo']:>10.4f} {sum_thru_sh:>12.4f}  "
                f"{antt_s:>6} {stp_s:>6}  "
                f"{r['agg_thru']:>9.4f} {comp_flag:>4}  "
                f"{mean_power_v:>11.3f} {mean_energy_v:>12.5f}  "
                f"{mean_static_v:>9.2f} {mean_dyn_v:>7.2f}"
            )
            # Print per-tenant slowdown breakdown.
            for t in pt:
                sd_s = f"{t['slowdown']:.4f}" if t["slowdown"] is not None else "  N/A"
                th_s = f"{t['thru_shared']:.4f}" if t["thru_shared"] is not None else "  N/A"
                pw_s = f"{t['power']:.3f}" if t["power"] is not None else "  N/A"
                cv_s = f"{t['lat_cv']:.4f}" if t.get("lat_cv") is not None else "  N/A"
                print(
                    f"  tid {t['tid']}: slowdown={sd_s}  thru_shared={th_s}  power={pw_s} W  lat_cv={cv_s}"
                    + (f"  [WARN: {t['power_excluded_datasets']} dataset(s) had power below floor]"
                       if t["power_excluded_datasets"] > 0 else "")
                )
            # Print cell-level fairness summary.
            fa = r.get("fairness")
            if fa is not None:
                ji_s = f"{fa['jain_index']:.4f}" if fa["jain_index"] is not None else "N/A"
                ur_s = f"{fa['unfairness_ratio']:.4f}" if fa["unfairness_ratio"] is not None else "N/A"
                excl_note = f"  [{fa['n_excluded']} excl]" if fa["n_excluded"] else ""
                trivial_note = " (trivial: n=1)" if fa.get("trivial") else ""
                print(
                    f"  fairness: jain={ji_s}  unfairness_ratio={ur_s}"
                    f"  min_sd={fa['min_slowdown']:.4f}  max_sd={fa['max_slowdown']:.4f}"
                    f"{excl_note}{trivial_note}"
                )
            if r["power_warnings"]:
                for w in r["power_warnings"]:
                    print(f"  [POWER WARNING: {w}]")
        print()

    # --- INCOMPLETE cells section ---
    if incomplete:
        print("=" * 80)
        print("INCOMPLETE CELLS (excluded from metrics)")
        print("=" * 80)
        hdr2 = f"{'tag':<42} {'fam':<6} {'exit':>4}  {'intended':>8} {'logged':>6}  reason"
        print(hdr2)
        print("-" * 110)
        for r in incomplete:
            src_note = "" if r["intended_source"] == "csv" else f" [{r['intended_source']}]"
            print(
                f"{r['tag']:<42} {r['fam']:<6} {r['exit']:>4}  "
                f"{r['intended']:>8}{src_note} {r['logged']:>6}  {r['reason']}"
            )
        print()

    # --- Summary ---
    exits_seen = sorted({r["exit"] for r in results})
    tenant_counts = sorted({r.get("n_tenants", r.get("intended", "?")) for r in results})
    print(
        f"Summary: {len(complete)} complete, {len(incomplete)} incomplete  "
        f"|  exits: {exits_seen}  |  tenant counts seen: {tenant_counts}"
    )
    print()

    # --- CSV output ---
    # --- Time-series output (opt-in, --timeseries) ---
    if timeseries:
        _emit_timeseries(logroot, mode_label, ts_out)

    if out_path and complete:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "tag", "fam", "exit", "n_tenants",
            "lat_solo", "thru_solo",
            "tid", "lat_shared", "thru_shared", "slowdown", "lat_cv",
            "stp", "antt", "agg_thru", "agg_thru_comparable",
            "jain_index", "unfairness_ratio",
            "power_w", "energy_j", "gpu_mem_static_mb", "gpu_mem_dynamic_mb", "peak_vram_mb",
        ]
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in complete:
                fa = r.get("fairness") or {}
                for t in r["per_tenant"]:
                    writer.writerow({
                        "tag": r["tag"],
                        "fam": r["fam"],
                        "exit": r["exit"],
                        "n_tenants": r["n_tenants"],
                        "lat_solo": _round(r["lat_solo"], 6),
                        "thru_solo": _round(r["thru_solo"], 4),
                        "tid": t["tid"],
                        "lat_shared": _round(t["lat_shared"], 6),
                        "thru_shared": _round(t["thru_shared"], 4),
                        "slowdown": _round(t["slowdown"], 4),
                        "lat_cv": _round(t.get("lat_cv"), 6),
                        "stp": _round(r["stp"], 4),
                        "antt": _round(r["antt"], 4),
                        "agg_thru": _round(r["agg_thru"], 4),
                        "agg_thru_comparable": r["agg_thru_comparable"],
                        "jain_index": _round(fa.get("jain_index"), 6),
                        "unfairness_ratio": _round(fa.get("unfairness_ratio"), 6),
                        "power_w": _round(t["power"], 3),
                        "energy_j": _round(t["energy"], 5),
                        "gpu_mem_static_mb": _round(t["gpu_mem_static_mb"], 2),
                        "gpu_mem_dynamic_mb": _round(t["gpu_mem_dynamic_mb"], 2),
                        "peak_vram_mb": _round(t["peak_vram_mb"], 2),
                    })
        print(f"CSV written to {out_path} ({len(complete)} complete cells)")

    return 0


def _round(v, decimals):
    if v is None:
        return ""
    return round(v, decimals)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Recompute multi-tenant metrics from raw hw_results.json files."
    )
    parser.add_argument("logroot", help="Directory containing mt_conc_* and mt_solo_* folders.")
    parser.add_argument("--mode-label", default="", help="Human-readable label for the power mode.")
    parser.add_argument("--out", default=None, help="Path for CSV output (optional).")
    parser.add_argument(
        "--timeseries", action="store_true",
        help="Emit per-second power/energy/throughput buckets for each run.",
    )
    parser.add_argument(
        "--ts-out", default=None,
        help="Path for time-series CSV output (one row per run+second). Requires --timeseries.",
    )
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else None
    ts_out = Path(args.ts_out) if args.ts_out else None
    code = run(Path(args.logroot), args.mode_label, out_path,
               timeseries=args.timeseries, ts_out=ts_out)
    sys.exit(code)


if __name__ == "__main__":
    main()
