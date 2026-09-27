"""Power-mode sweep wrapper for Experiment 2 multi-tenant runs.

Runs the same multi-tenant scenario at every requested Jetson stock power mode
in sequence, so co-location behaviour can be compared across MAXN, 25W and 15W
the way Experiment 1 compared single-tenant numbers across modes.

Mode IDs are discovered from /etc/nvpmodel.conf rather than hardcoded, because
Orin Nano variants and JetPack versions use different IDs and names. Stock modes
are selected with "sudo nvpmodel -m <id>", which is separate from set_config.py
(that file rewrites a local custom conf; this wrapper uses the stock system conf).

Usage examples (Jetson):
    python multitenant_sweep.py --list-modes
    python multitenant_sweep.py --scenario llama_yolo --modes MAXN,25W,15W
    python multitenant_sweep.py --scenario bert_yolo  --modes MAXN,15W --duration 60
    python multitenant_sweep.py --scenario yolo_scale --modes 25W --k 4 --tag sweep1

Off-device check (no GPU, no hardware):
    python multitenant_sweep.py --selftest
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import multitenant_run as mr

# Default settle time after a mode switch: clocks and the governor need a moment.
DEFAULT_SETTLE_SEC = 5.0

# Path to the stock Jetson nvpmodel config.
STOCK_NVPMODEL_CONF = Path("/etc/nvpmodel.conf")


# ---- mode discovery ----------------------------------------------------------

def parse_nvpmodel_conf(text: str) -> dict:
    """Parse a nvpmodel.conf text and return {id: name}.

    Recognises lines of the form:
        < POWER_MODEL ID=N NAME=SOMENAME >
    Returns an empty dict if no entries are found; callers treat that as an
    error rather than silently continuing with an empty mapping.
    """
    mapping = {}
    for m in re.finditer(r"<\s*POWER_MODEL\s+ID=(\d+)\s+NAME=(\S+)\s*>", text):
        mapping[int(m.group(1))] = m.group(2)
    return mapping


def load_mode_table(conf_path: Path = STOCK_NVPMODEL_CONF) -> dict:
    """Read conf_path and return the id-to-name mapping.

    Raises RuntimeError with a clear message when the file is absent, unreadable,
    or contains no POWER_MODEL entries, so callers see the problem rather than
    an empty dict that looks like success.
    """
    if not conf_path.exists():
        raise RuntimeError(
            f"nvpmodel config not found: {conf_path}\n"
            "This file is only present on a Jetson device. "
            "To run offline, use --selftest instead of --scenario."
        )
    try:
        text = conf_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RuntimeError(f"Cannot read {conf_path}: {exc}") from exc
    table = parse_nvpmodel_conf(text)
    if not table:
        raise RuntimeError(
            f"No POWER_MODEL entries found in {conf_path}. "
            "The file exists but its format is unrecognised. "
            "Expected lines of the form: < POWER_MODEL ID=N NAME=SOMENAME >"
        )
    return table


def resolve_mode_names(requested: list, table: dict) -> list:
    """Resolve a list of requested name strings to (id, canonical_name) pairs.

    Matching is case-insensitive and prefix-tolerant: "MAXN" matches
    "MAXN_SUPER". Fails before returning when any requested name matches zero
    or more than one entry in the table.

    Returns a list of (id, name) tuples in the same order as requested.
    Raises ValueError with the discovered table printed when any name fails.
    """
    results = []
    errors = []
    for req in requested:
        req_up = req.upper()
        # Exact match wins: if the requested string equals a table entry name
        # exactly (case-insensitive), use it and do not look for prefix matches.
        # Prefix fallback applies only when no exact entry exists, so "25W" picks
        # ID=2 NAME=25W (not also ID=3 NAME=25W_2CORE) when both are present, but
        # "MAXN" picks MAXN_SUPER when the table contains only MAXN_SUPER.
        exact = [(mid, mname) for mid, mname in table.items()
                 if mname.upper() == req_up]
        if exact:
            candidates = exact
        else:
            candidates = [
                (mid, mname) for mid, mname in table.items()
                if mname.upper().startswith(req_up + "_")
            ]
        if len(candidates) == 1:
            results.append(candidates[0])
        elif len(candidates) == 0:
            errors.append(
                f"  '{req}' matches nothing in the table (checked {len(table)} entries)"
            )
        else:
            names = ", ".join(f"ID={mid} NAME={mname}" for mid, mname in candidates)
            errors.append(
                f"  '{req}' is ambiguous, matches {len(candidates)} entries: {names}"
            )
    if errors:
        table_lines = "\n".join(f"  ID={mid}  NAME={mname}" for mid, mname in sorted(table.items()))
        raise ValueError(
            "Mode name resolution failed:\n"
            + "\n".join(errors)
            + "\n\nDiscovered power modes:\n"
            + table_lines
        )
    return results


# ---- sudo handling ----------------------------------------------------------

def check_sudo_noninteractive() -> bool:
    """Return True if non-interactive sudo works right now.

    Uses 'sudo -n true': -n means non-interactive (returns error instead of
    prompting for a password).
    """
    try:
        result = subprocess.run(
            ["sudo", "-n", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


SUDO_HELP = (
    "Non-interactive sudo is not available right now.\n"
    "To fix this, run one of:\n"
    "  sudo -v                       (cache credentials for the current session)\n"
    "  or configure NOPASSWD for nvpmodel in /etc/sudoers, e.g.:\n"
    "    %sudo ALL=(ALL) NOPASSWD: /usr/sbin/nvpmodel\n"
    "Then retry the sweep."
)


# ---- mode switching and read-back -------------------------------------------

def switch_mode(mode_id: int) -> None:
    """Switch to the stock nvpmodel mode by ID using sudo nvpmodel -m."""
    result = subprocess.run(
        ["sudo", "nvpmodel", "-m", str(mode_id)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"nvpmodel -m {mode_id} failed (exit {result.returncode}): "
            + (result.stderr.strip() or result.stdout.strip())
        )


def read_current_mode_name() -> str:
    """Read the current nvpmodel mode name from the device.

    Uses 'nvpmodel -q' which prints something like:
        NV Power Mode: 25W
        2
    and extracts the mode name from the 'NV Power Mode:' line.
    Falls back to 'nvpmodel -q --verbose' output if the short form is absent.
    Raises RuntimeError when the mode cannot be determined.
    """
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
            return m.group(1)
    raise RuntimeError(
        f"Cannot determine current nvpmodel mode from 'nvpmodel -q' output:\n{output}"
    )


def verify_mode(requested_name: str, tolerance_sec: float = 0.0) -> None:
    """Read back the current mode and raise RuntimeError if it does not match.

    tolerance_sec is unused here but kept as a parameter so callers can express
    the settle delay separately from the verification call.
    """
    actual = read_current_mode_name()
    if actual.upper() != requested_name.upper():
        raise RuntimeError(
            f"Mode verification failed: requested '{requested_name}', "
            f"device reports '{actual}'. "
            "The switch may not have taken effect. Aborting sweep to avoid "
            "mislabelled data."
        )


# ---- scenario cell count ----------------------------------------------------

def count_cells(scenario: str, k: int) -> int:
    """Return the number of cells a scenario produces at the given k.

    Mirrors the logic in multitenant_run.run_scenario without running anything.
    Returns 0 for unknown scenarios.
    """
    if scenario in mr.SCALING:
        fam, counts = mr.SCALING[scenario]
        return _anchor_len(fam, k) * len(counts)
    if scenario in mr.SCENARIOS:
        fams = mr.SCENARIOS[scenario]
        eff_k = k if len(fams) == 2 else min(k, 3)
        total = 1
        for f in fams:
            total *= _anchor_len(f, eff_k)
        return total
    return 0


def _anchor_len(fam: str, k: int) -> int:
    return len(mr._anchor(fam, k))


# ---- sweep ------------------------------------------------------------------

def run_sweep(scenario: str, modes: list, tag: str,
              duration: float, k: int, repeats: int,
              holdout_n: int, keep_suspect: bool,
              settle_sec: float,
              conf_path: Path = STOCK_NVPMODEL_CONF) -> dict:
    """Run one scenario at each of the requested power modes in sequence.

    modes: list of name strings (e.g. ["MAXN", "25W", "15W"]).
    Returns a dict mapping mode_name -> list of rows (or None on failure).

    Guarantees: the original mode is restored in the finally block, whether the
    sweep completes normally, raises an exception, or is interrupted.
    """
    table = load_mode_table(conf_path)
    resolved = resolve_mode_names(modes, table)  # [(id, canonical_name), ...]

    # Capture original mode before touching anything.
    original_mode_name = read_current_mode_name()
    print(f"[sweep] current mode before sweep: {original_mode_name}")

    # Pre-flight sudo check: fail fast before any measurement.
    if not check_sudo_noninteractive():
        print(SUDO_HELP)
        sys.exit(1)

    cells_per_mode = count_cells(scenario, k)
    total_cells = cells_per_mode * len(resolved)
    print(
        f"[sweep] scenario={scenario}  modes={[n for _, n in resolved]}  "
        f"k={k}  cells/mode={cells_per_mode}  total={total_cells}  "
        f"repeats={repeats}  duration={duration:.0f}s"
    )
    print(
        f"[sweep] estimated minimum wall time: "
        f"{total_cells * repeats * duration / 60:.0f} min "
        f"(assuming perfect overlap, no probe overhead)"
    )

    completed = []
    results = {}

    try:
        for mode_id, mode_name in resolved:
            # Re-check sudo before each switch because credentials expire.
            if not check_sudo_noninteractive():
                print(
                    f"\n[sweep] sudo credentials have expired before switching to "
                    f"{mode_name}. Modes completed so far: {completed}\n"
                    + SUDO_HELP
                )
                break

            print(f"\n[sweep] switching to {mode_name} (ID={mode_id})")
            switch_mode(mode_id)
            if settle_sec > 0:
                print(f"[sweep] settling for {settle_sec:.1f}s")
                time.sleep(settle_sec)

            verify_mode(mode_name)
            print(f"[sweep] mode verified: {mode_name}")

            # Build a per-mode tag that carries the mode name into every CSV row.
            mode_tag = f"{tag}_{mode_name}"
            print(f"[sweep] running scenario '{scenario}' with tag '{mode_tag}'")

            rows = mr.run_scenario(
                scenario,
                tag=mode_tag,
                duration=duration,
                k=k,
                repeats=repeats,
                holdout_n=holdout_n,
                keep_suspect=keep_suspect,
            )
            results[mode_name] = rows
            completed.append(mode_name)
            print(f"[sweep] completed mode {mode_name}")

    finally:
        # Always restore the original mode. Never leave the board at a low cap.
        print(f"\n[sweep] restoring original mode: {original_mode_name}")
        try:
            # Resolve original mode name back to an ID.
            original_id = None
            for mid, mname in table.items():
                if mname.upper() == original_mode_name.upper():
                    original_id = mid
                    break
            if original_id is not None:
                switch_mode(original_id)
                print(f"[sweep] restored to {original_mode_name} (ID={original_id})")
            else:
                print(
                    f"[sweep] WARNING: original mode '{original_mode_name}' not found "
                    f"in table, cannot restore automatically. "
                    f"Run: sudo nvpmodel -m <id>  to restore manually."
                )
        except Exception as exc:
            print(f"[sweep] WARNING: restore failed: {exc}")

    if completed:
        print(f"\n[sweep] done. completed modes: {completed}")
        missed = [n for _, n in resolved if n not in completed]
        if missed:
            print(f"[sweep] modes not reached: {missed}")

    return results


# ---- selftest ---------------------------------------------------------------

_SAMPLE_CONF = """\
# Sample nvpmodel.conf for selftest (representative Orin Nano format)
< POWER_MODEL ID=0 NAME=MAXN >
CPU_ONLINE CORE_0 1
GPU MAX_FREQ -1

< POWER_MODEL ID=2 NAME=25W >
CPU_ONLINE CORE_0 1
GPU MAX_FREQ 624000

< POWER_MODEL ID=3 NAME=25W_2CORE >
CPU_ONLINE CORE_0 1
CPU_ONLINE CORE_3 0
GPU MAX_FREQ 624000

< POWER_MODEL ID=4 NAME=15W >
CPU_ONLINE CORE_0 1
GPU MAX_FREQ 408000

< PM_CONFIG DEFAULT=0 >
"""


def _selftest():
    errors = []

    # 1. Parsing: correct id-to-name mapping.
    table = parse_nvpmodel_conf(_SAMPLE_CONF)
    expected = {0: "MAXN", 2: "25W", 3: "25W_2CORE", 4: "15W"}
    if table != expected:
        errors.append(f"parse: expected {expected}, got {table}")
    else:
        print("[selftest] parse_nvpmodel_conf: OK")

    # 2. Empty / malformed input produces empty dict, not a crash.
    empty = parse_nvpmodel_conf("# no entries here\n")
    if empty != {}:
        errors.append(f"parse empty: expected {{}}, got {empty}")
    else:
        print("[selftest] parse_nvpmodel_conf empty: OK")

    # 3. Name matching: exact.
    resolved = resolve_mode_names(["MAXN"], table)
    if resolved != [(0, "MAXN")]:
        errors.append(f"resolve exact: expected [(0, 'MAXN')], got {resolved}")
    else:
        print("[selftest] resolve_mode_names exact: OK")

    # 4. Name matching: case-insensitive.
    resolved_ci = resolve_mode_names(["maxn"], table)
    if resolved_ci != [(0, "MAXN")]:
        errors.append(f"resolve case-insensitive: expected [(0, 'MAXN')], got {resolved_ci}")
    else:
        print("[selftest] resolve_mode_names case-insensitive: OK")

    # 5. Prefix matching: "25W" should match "25W" only, not "25W_2CORE"
    #    (exact match wins over prefix when an exact entry exists).
    resolved_25 = resolve_mode_names(["25W"], table)
    if resolved_25 != [(2, "25W")]:
        errors.append(f"resolve prefix 25W: expected [(2, '25W')], got {resolved_25}")
    else:
        print("[selftest] resolve_mode_names 25W exact (not prefix): OK")

    # 6. Ambiguous name: no exact match, two prefix candidates should reject "MAXN".
    ambig_table = {0: "MAXN_SUPER", 1: "MAXN_PLUS"}
    try:
        resolve_mode_names(["MAXN"], ambig_table)
        errors.append("resolve ambiguous: expected ValueError, got no error")
    except ValueError as exc:
        if "ambiguous" in str(exc).lower():
            print("[selftest] resolve_mode_names ambiguous: OK")
        else:
            errors.append(f"resolve ambiguous: unexpected error text: {exc}")

    # 7. Unknown name raises ValueError.
    try:
        resolve_mode_names(["99W"], table)
        errors.append("resolve unknown: expected ValueError, got no error")
    except ValueError as exc:
        if "matches nothing" in str(exc):
            print("[selftest] resolve_mode_names unknown: OK")
        else:
            errors.append(f"resolve unknown: unexpected error text: {exc}")

    # 8. Cell count helper works for known scenarios.
    nc = count_cells("bert_scale", 6)
    if nc != 18:  # bert_scale: 6 exit anchors x 3 tenant counts = 18
        errors.append(f"count_cells bert_scale: expected 18, got {nc}")
    else:
        print(f"[selftest] count_cells bert_scale k=6: {nc} OK")

    if errors:
        for e in errors:
            print(f"[selftest] FAIL: {e}")
        sys.exit(1)
    else:
        print("[selftest] all checks passed")


# ---- CLI --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--scenario", metavar="NAME",
        help="scenario name (same as multitenant_run --scenario)"
    )
    ap.add_argument(
        "--modes", metavar="NAMES",
        help="comma-separated power mode names to sweep, e.g. MAXN,25W,15W"
    )
    ap.add_argument(
        "--tag", default=None,
        help="base tag; each mode appends _MODENAME automatically"
    )
    ap.add_argument(
        "--duration", type=float, default=mr.DEFAULT_DURATION,
        help=f"target concurrent window in seconds (default {mr.DEFAULT_DURATION:.0f})"
    )
    ap.add_argument(
        "--k", type=int, default=6,
        help="exit anchors sampled per model (default 6, minimum 2)"
    )
    ap.add_argument(
        "--repeats", type=int, default=1,
        help="number of repeats per cell (default 1)"
    )
    ap.add_argument(
        "--holdout", type=int, default=0, metavar="N",
        help="off-anchor holdout points per family (default 0)"
    )
    ap.add_argument(
        "--keep-suspect", action="store_true",
        help="write low-overlap cells to main CSV instead of sidecar"
    )
    ap.add_argument(
        "--settle", type=float, default=DEFAULT_SETTLE_SEC,
        help=f"seconds to wait after a mode switch before measuring "
             f"(default {DEFAULT_SETTLE_SEC:.0f})"
    )
    ap.add_argument(
        "--list-modes", action="store_true",
        help="print discovered power modes from /etc/nvpmodel.conf and exit"
    )
    ap.add_argument(
        "--selftest", action="store_true",
        help="run offline parsing and matching checks against an embedded "
             "sample config, then exit"
    )
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return

    if a.list_modes:
        table = load_mode_table()
        print("Discovered power modes:")
        for mid, mname in sorted(table.items()):
            print(f"  ID={mid}  NAME={mname}")
        return

    if not a.scenario:
        ap.print_help()
        sys.exit(1)

    if not a.modes:
        print("[error] --modes is required (e.g. --modes MAXN,25W,15W)")
        ap.print_help()
        sys.exit(1)

    if a.k < 2:
        print(f"[error] --k must be at least 2 (got {a.k})")
        sys.exit(1)

    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    tag = a.tag or f"sweep_{a.scenario}"

    run_sweep(
        scenario=a.scenario,
        modes=modes,
        tag=tag,
        duration=a.duration,
        k=a.k,
        repeats=a.repeats,
        holdout_n=a.holdout,
        keep_suspect=a.keep_suspect,
        settle_sec=a.settle,
    )


if __name__ == "__main__":
    main()
