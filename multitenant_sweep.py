"""Power-mode sweep wrapper for Experiment 2 multi-tenant runs.

Runs the same multi-tenant scenario at every requested Jetson stock power mode
in sequence, so co-location behaviour can be compared across MAXN, 25W and 15W
the way Experiment 1 compared single-tenant numbers across modes.

Mode IDs are discovered from /etc/nvpmodel.conf rather than hardcoded, because
Orin Nano variants and JetPack versions use different IDs and names. Stock modes
are selected with "sudo nvpmodel -m <id>" (or a direct call when the process is
root or the board grants unprivileged access), which is separate from set_config.py
(that file rewrites a local custom conf; this wrapper uses the stock system conf).

When mode switching is unavailable the sweep degrades: it runs once at the
current mode rather than refusing to produce any data. Use --require-all-modes
to restore the strict fail-before-measuring behaviour.

Usage examples (Jetson):
    python multitenant_sweep.py --list-modes
    python multitenant_sweep.py --scenario llama_yolo --modes MAXN,25W,15W
    python multitenant_sweep.py --scenario bert_yolo  --modes MAXN,15W --duration 60
    python multitenant_sweep.py --scenario yolo_scale --modes 25W --k 4 --tag sweep1
    python multitenant_sweep.py --campaign
    python multitenant_sweep.py --campaign bert_scale,yolo_vit --modes MAXN,15W

Background daemon (survives SSH disconnect):
    python multitenant_sweep.py --campaign -d   # start detached
    python multitenant_sweep.py -ss             # snapshot progress
    python multitenant_sweep.py -s              # stop

Off-device check (no GPU, no hardware):
    python multitenant_sweep.py --selftest
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import multitenant_run as mr

REPO_ROOT = Path(__file__).resolve().parent

# Default settle time after a mode switch: clocks and the governor need a moment.
DEFAULT_SETTLE_SEC = 5.0

# Path to the stock Jetson nvpmodel config.
STOCK_NVPMODEL_CONF = Path("/etc/nvpmodel.conf")

# Exit code for a degraded run: measurement completed but not all modes covered.
EXIT_DEGRADED = 2

# Campaign order: cheapest and most diagnostic first so a late failure does not
# cost the early results. Scaling studies validate the harness; heterogeneous
# pairs come after.
CAMPAIGN_ORDER = [
    "bert_scale",
    "yolo_scale",
    "vit_scale",
    "yolo_vit",
    "bert_yolo",
    "llama_vit",
    "llama_yolo",
    "triple",
]


# ---- background daemon control ----------------------------------------------
# Separate pid and log paths from bench_jetson.py so a benchmark and a sweep
# can run concurrently without each one thinking the other is itself.

SWEEP_PID = REPO_ROOT / "logs" / "sweep_daemon.pid"
SWEEP_LOG = REPO_ROOT / "logs" / "sweep_daemon.log"

# CSV written by multitenant_run -- read-only here, for snapshot reporting.
_SWEEP_CSV = REPO_ROOT / "result" / "multitenant" / "concurrent_slowdown.csv"
_SWEEP_SUSPECT_CSV = REPO_ROOT / "result" / "multitenant" / "concurrent_slowdown.suspect.csv"


def _sweep_read_pid():
    """Return the integer pid from SWEEP_PID, or None when absent or unreadable."""
    try:
        return int(SWEEP_PID.read_text().strip())
    except Exception:
        return None


def _sweep_running() -> bool:
    """Return True when a sweep daemon pid exists and that process is alive."""
    import psutil
    pid = _sweep_read_pid()
    return pid is not None and psutil.pid_exists(pid)


def _sweep_start():
    """Relaunch this script (minus the -d flag) as a detached background process.

    Mirrors _daemon_start in bench_jetson.py exactly: records the pid, redirects
    stdout and stderr to SWEEP_LOG, and returns immediately. The detached child
    re-runs the full sweep independently.
    """
    if _sweep_running():
        pid = _sweep_read_pid()
        print(f"[sweep-daemon] already running pid={pid}")
        print(f"[sweep-daemon]   snapshot: python multitenant_sweep.py -ss")
        print(f"[sweep-daemon]   stop:     python multitenant_sweep.py -s")
        return
    SWEEP_LOG.parent.mkdir(parents=True, exist_ok=True)
    argv = [a for a in sys.argv if a not in ("-d", "--daemon")]
    cmd = [sys.executable] + argv      # argv[0] is already the script path
    logf = open(SWEEP_LOG, "a", buffering=1, encoding="utf-8")
    import datetime
    logf.write(f"\n==== sweep daemon start {datetime.datetime.now()} ====\n")
    # start_new_session detaches on POSIX (survives SSH disconnect).
    # creationflags=DETACHED_PROCESS is the Windows equivalent but not needed
    # for correctness on the Jetson; keep the branch for test portability.
    spawn = {"start_new_session": True} if os.name == "posix" else {"creationflags": 0x00000008}
    # glibc opens ~8 malloc arenas per core by default; on a 6-core 8 GB Tegra
    # with multiple tenants each spawning further processes this is enough to
    # trip the OOM reaper. Cap to 2 for the child. setdefault so an explicit
    # environment value from the caller still wins.
    env = dict(os.environ)
    env.setdefault("MALLOC_ARENA_MAX", "2")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            env=env, **spawn)
    SWEEP_PID.write_text(str(proc.pid))
    print(f"[sweep-daemon] started pid={proc.pid}")
    print(f"[sweep-daemon]   log:      {SWEEP_LOG}")
    print(f"[sweep-daemon]   snapshot: python multitenant_sweep.py -ss")
    print(f"[sweep-daemon]   stop:     python multitenant_sweep.py -s")


def _sweep_stop():
    """Terminate the sweep daemon and all child processes it has spawned.

    The sweep launches tenants which each launch benchmark processes, so killing
    only the top-level pid would leave orphaned processes holding GPU memory.
    psutil.children(recursive=True) collects the whole tree.
    """
    import psutil
    pid = _sweep_read_pid()
    if pid is None or not psutil.pid_exists(pid):
        print("[sweep-daemon] no running background sweep")
        SWEEP_PID.unlink(missing_ok=True)
        return
    proc = psutil.Process(pid)
    kids = proc.children(recursive=True)
    for p in [proc, *kids]:
        _sweep_terminate_quietly(p)
    _, alive = psutil.wait_procs([proc, *kids], timeout=10)
    for p in alive:
        _sweep_terminate_quietly(p, kill=True)
    SWEEP_PID.unlink(missing_ok=True)
    print(f"[sweep-daemon] stopped pid={pid} (+{len(kids)} children)")


def _sweep_terminate_quietly(proc, kill: bool = False):
    """Terminate or kill a process, ignoring errors (process may have exited)."""
    try:
        proc.kill() if kill else proc.terminate()
    except Exception:
        pass


def _sweep_snapshot():
    """Print a progress report useful for monitoring a running campaign.

    Reports:
      - whether the daemon is alive and its pid
      - row count in the main CSV (read defensively: may be mid-write)
      - row count in the suspect sidecar (if it exists)
      - distinct tags seen so far (indicates which scenario/mode is in progress)
      - last 30 lines of the sweep log
    """
    pid = _sweep_read_pid()
    status = f"RUNNING pid={pid}" if _sweep_running() else "not running"
    print(f"[sweep-daemon] {status}")

    main_rows, main_note = _count_csv_rows(_SWEEP_CSV)
    suspect_rows, suspect_note = _count_csv_rows(_SWEEP_SUSPECT_CSV)
    print(f"[sweep-daemon] main CSV rows:    {main_rows}{main_note}  ({_SWEEP_CSV})")
    if _SWEEP_SUSPECT_CSV.exists():
        print(f"[sweep-daemon] suspect CSV rows: {suspect_rows}{suspect_note}  ({_SWEEP_SUSPECT_CSV})")

    tags = _read_csv_tags(_SWEEP_CSV)
    if tags:
        print(f"[sweep-daemon] distinct tags seen ({len(tags)}): {sorted(tags)}")
    else:
        print("[sweep-daemon] no tags yet (CSV empty or not started)")

    _sweep_print_log_tail(30)


def _count_csv_rows(path: Path):
    """Return (count, note) for a CSV file read defensively.

    Skips the header row. The final line may be truncated because a live process
    is appending, so a line that does not parse is discarded rather than raising.
    Returns (0, note) when the file is absent or empty.
    """
    if not path.exists():
        return 0, " (file not found)"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0, " (unreadable)"
    lines = text.splitlines()
    if len(lines) < 2:
        return 0, " (header only or empty)"
    # Header is line 0; count non-blank lines after it.
    # A partially written last line is counted only when it has at least one
    # comma, which means it got past the first field.
    count = 0
    partial = False
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if "," in stripped:
            count += 1
        else:
            partial = True
    note = " (last line truncated, not counted)" if partial else ""
    return count, note


def _read_csv_tags(path: Path) -> set:
    """Return the set of distinct values in the 'tag' column of a CSV file.

    Reads defensively: skips malformed lines, handles a truncated final line.
    Returns an empty set when the file is absent or has no parseable rows.
    """
    if not path.exists():
        return set()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    lines = text.splitlines()
    if len(lines) < 2:
        return set()
    # Find the 'tag' column index from the header.
    header = lines[0].split(",")
    try:
        tag_idx = header.index("tag")
    except ValueError:
        return set()
    tags = set()
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) > tag_idx:
            val = parts[tag_idx].strip()
            if val:
                tags.add(val)
    return tags


def _sweep_print_log_tail(n: int):
    """Print the last n lines of the sweep log."""
    if not SWEEP_LOG.exists():
        print("[sweep-daemon] no log yet")
        return
    lines = SWEEP_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"[sweep-daemon] --- last {min(n, len(lines))} log lines ---")
    for ln in lines[-n:]:
        print(ln)


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


# ---- capability probe -------------------------------------------------------
# Checked in order; the first path that works is used for all switches in this
# run. The method is printed once so the operator knows what happened.

def check_is_root() -> bool:
    """Return True when the effective user ID is zero (process is already root).

    os.geteuid is not present on Windows; guard the call so dev-box tests work.
    """
    if not hasattr(os, "geteuid"):
        return False
    return os.geteuid() == 0  # type: ignore[attr-defined]


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


def check_nvpmodel_direct() -> bool:
    """Return True if nvpmodel can be invoked without sudo.

    Some Jetson configurations grant unprivileged access. A no-op query
    ('nvpmodel -q') is sufficient to probe availability; the actual switch
    call omits sudo when this path is selected.
    """
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ponytail: named tuple would add nothing here; a plain string is fine.
def probe_switch_capability() -> tuple:
    """Check switching capability in priority order.

    Returns (method, description) where method is one of:
      "root"            -- process is already root, no sudo needed
      "sudo"            -- non-interactive sudo works
      "nvpmodel_direct" -- nvpmodel runs without sudo
      None              -- no switching path available

    Always prints which path was selected.
    """
    if check_is_root():
        msg = "[sweep] switch capability: process is root, sudo not needed"
        print(msg)
        return ("root", msg)
    if check_sudo_noninteractive():
        msg = "[sweep] switch capability: non-interactive sudo available"
        print(msg)
        return ("sudo", msg)
    if check_nvpmodel_direct():
        msg = "[sweep] switch capability: nvpmodel usable directly without sudo"
        print(msg)
        return ("nvpmodel_direct", msg)
    print("[sweep] switch capability: none of root/sudo/nvpmodel_direct available")
    return (None, "none available")


SUDO_HELP = (
    "Non-interactive sudo is not available right now.\n"
    "To fix this, run one of:\n"
    "  sudo -v                       (cache credentials for the current session)\n"
    "  or configure NOPASSWD for nvpmodel in /etc/sudoers, e.g.:\n"
    "    %sudo ALL=(ALL) NOPASSWD: /usr/sbin/nvpmodel\n"
    "Then retry the sweep."
)


# ---- mode switching and read-back -------------------------------------------

def switch_mode(mode_id: int, method: str = "sudo") -> None:
    """Switch to the stock nvpmodel mode by ID.

    method selects the privilege path discovered by probe_switch_capability:
      "root" or "sudo"            -- prepend sudo (root does not need it but it
                                     also does not hurt; skip it for cleanliness)
      "nvpmodel_direct"           -- call nvpmodel directly without sudo
    """
    if method in ("root", "nvpmodel_direct"):
        cmd = ["nvpmodel", "-m", str(mode_id)]
    else:
        cmd = ["sudo", "nvpmodel", "-m", str(mode_id)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
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
              require_all_modes: bool = False,
              conf_path: Path = STOCK_NVPMODEL_CONF) -> dict:
    """Run one scenario at each of the requested power modes in sequence.

    modes: list of name strings (e.g. ["MAXN", "25W", "15W"]).
    Returns a dict mapping mode_name -> list of rows (or None on failure).

    When switching is unavailable and require_all_modes is False (the default),
    the sweep degrades: it runs once at the current mode and exits with
    EXIT_DEGRADED (2) after writing results. When require_all_modes is True it
    refuses to start instead, matching the old strict behaviour.

    Guarantees: the original mode is restored in the finally block, whether the
    sweep completes normally, raises an exception, or is interrupted. Results
    already written are preserved on abort.
    """
    table = load_mode_table(conf_path)
    resolved = resolve_mode_names(modes, table)  # [(id, canonical_name), ...]

    # Capture original mode before touching anything.
    original_mode_name = read_current_mode_name()
    print(f"[sweep] current mode before sweep: {original_mode_name}")

    # Probe capability once before any measurement.
    switch_method, _cap_desc = probe_switch_capability()
    can_switch = switch_method is not None

    if not can_switch:
        if require_all_modes:
            print(
                f"\n[sweep] ABORTED: --require-all-modes is set and mode switching "
                f"is not available. No measurement was taken.\n{SUDO_HELP}"
            )
            sys.exit(1)
        # Degraded path: measure once at the current mode.
        _run_degraded(
            scenario=scenario,
            requested_modes=[n for _, n in resolved],
            current_mode=original_mode_name,
            tag=tag,
            duration=duration,
            k=k,
            repeats=repeats,
            holdout_n=holdout_n,
            keep_suspect=keep_suspect,
        )
        # Exit with a distinct code so a script can detect a degraded run.
        sys.exit(EXIT_DEGRADED)

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
            # Re-probe before each switch because sudo credentials expire.
            cur_method, _ = probe_switch_capability()
            if cur_method is None:
                print(
                    f"\n[sweep] switching capability lost before switching to "
                    f"{mode_name}. Modes completed so far: {completed}\n"
                    + SUDO_HELP
                )
                break

            print(f"\n[sweep] switching to {mode_name} (ID={mode_id})")
            switch_mode(mode_id, method=cur_method)
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
            original_id = None
            for mid, mname in table.items():
                if mname.upper() == original_mode_name.upper():
                    original_id = mid
                    break
            if original_id is not None:
                # Use the last known good method; fall back to sudo.
                restore_method = switch_method or "sudo"
                switch_mode(original_id, method=restore_method)
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


def _run_degraded(scenario: str, requested_modes: list, current_mode: str,
                  tag: str, duration: float, k: int, repeats: int,
                  holdout_n: int, keep_suspect: bool) -> None:
    """Run one scenario once at the current mode when switching is unavailable.

    Rows are tagged with the REAL mode discovered from the device, never with
    any requested mode name. Prints a prominent notice before and after.
    """
    skipped = [m for m in requested_modes if m.upper() != current_mode.upper()]
    print(
        f"\n[sweep] DEGRADED RUN: mode switching is not available.\n"
        f"  Board is at: {current_mode}\n"
        f"  Requested modes: {requested_modes}\n"
        f"  Measuring once at current mode only.\n"
        f"  Skipped modes: {skipped}\n"
        f"  Rows will be tagged with the actual device mode: {current_mode}\n"
        f"  Exit status will be {EXIT_DEGRADED} to signal a degraded run.\n"
    )
    mode_tag = f"{tag}_{current_mode}"
    print(f"[sweep] running scenario '{scenario}' with tag '{mode_tag}' (degraded)")
    mr.run_scenario(
        scenario,
        tag=mode_tag,
        duration=duration,
        k=k,
        repeats=repeats,
        holdout_n=holdout_n,
        keep_suspect=keep_suspect,
    )
    print(
        f"\n[sweep] DEGRADED RUN COMPLETE.\n"
        f"  Mode measured: {current_mode}\n"
        f"  Modes NOT measured: {skipped}\n"
        f"  Reason: no switching path was available (not root, no sudo, "
        f"no unprivileged nvpmodel).\n"
        f"  Exit status: {EXIT_DEGRADED}\n"
    )


# ---- campaign ---------------------------------------------------------------

def run_campaign(scenarios: list, modes: list, tag: str,
                 duration: float, k: int, repeats: int,
                 holdout_n: int, keep_suspect: bool,
                 settle_sec: float,
                 require_all_modes: bool = False,
                 conf_path: Path = STOCK_NVPMODEL_CONF) -> dict:
    """Run multiple scenarios in sequence under the same mode logic.

    Prints the full plan before starting. Continues to the next scenario when
    one fails rather than aborting the campaign. Returns a summary dict:
      {"completed": [...], "failed": {name: reason}, "degraded": bool}

    The degraded flag is set when switching was unavailable and the run fell
    back to single-mode measurement. In that case the exit status will be
    EXIT_DEGRADED after all scenarios finish.
    """
    table = load_mode_table(conf_path)
    resolved = resolve_mode_names(modes, table)
    original_mode_name = read_current_mode_name()
    print(f"[campaign] current mode before campaign: {original_mode_name}")

    switch_method, _cap_desc = probe_switch_capability()
    can_switch = switch_method is not None

    if not can_switch and require_all_modes:
        print(
            f"\n[campaign] ABORTED: --require-all-modes is set and mode switching "
            f"is not available. No measurement was taken.\n{SUDO_HELP}"
        )
        sys.exit(1)

    # --- print the full plan before starting ---
    mode_names = [n for _, n in resolved] if can_switch else [original_mode_name]
    total_cells_all = sum(count_cells(s, k) for s in scenarios)
    total_runs = total_cells_all * len(mode_names) * repeats
    print(
        f"\n[campaign] PLAN: {len(scenarios)} scenario(s) x "
        f"{len(mode_names)} mode(s) x {repeats} repeat(s)"
    )
    print(f"[campaign] modes: {mode_names}")
    print(f"[campaign] k={k}  duration={duration:.0f}s  holdout={holdout_n}")
    print()
    for s in scenarios:
        nc = count_cells(s, k)
        print(f"  {s:<14}  {nc:>4} cells/mode  x {len(mode_names)} modes = "
              f"{nc * len(mode_names) * repeats} total runs")
    est_min = total_runs * duration / 60
    print(
        f"\n[campaign] total cells across all scenarios and modes: "
        f"{total_cells_all * len(mode_names) * repeats}"
    )
    print(
        f"[campaign] estimated minimum wall time: {est_min:.0f} min "
        f"(assuming perfect overlap, no probe overhead)\n"
    )

    if not can_switch:
        print(
            f"[campaign] DEGRADED: switching unavailable, all scenarios will run "
            f"once at current mode ({original_mode_name}) only.\n"
        )

    completed = []
    failed = {}
    degraded = not can_switch

    try:
        for scenario in scenarios:
            print(f"\n[campaign] ===== scenario: {scenario} =====")
            try:
                _run_one_campaign_scenario(
                    scenario=scenario,
                    resolved=resolved,
                    original_mode_name=original_mode_name,
                    can_switch=can_switch,
                    switch_method=switch_method,
                    tag=tag,
                    duration=duration,
                    k=k,
                    repeats=repeats,
                    holdout_n=holdout_n,
                    keep_suspect=keep_suspect,
                    settle_sec=settle_sec,
                )
                completed.append(scenario)
                print(f"[campaign] scenario {scenario}: OK")
            except Exception as exc:
                failed[scenario] = str(exc)
                print(f"[campaign] scenario {scenario}: FAILED: {exc}")
                print("[campaign] continuing to next scenario.")
    finally:
        if can_switch:
            print(f"\n[campaign] restoring original mode: {original_mode_name}")
            try:
                original_id = next(
                    (mid for mid, mname in table.items()
                     if mname.upper() == original_mode_name.upper()), None
                )
                if original_id is not None:
                    switch_mode(original_id, method=switch_method)
                    print(f"[campaign] restored to {original_mode_name} (ID={original_id})")
                else:
                    print(
                        f"[campaign] WARNING: original mode '{original_mode_name}' not in "
                        f"table. Run: sudo nvpmodel -m <id>  to restore manually."
                    )
            except Exception as exc:
                print(f"[campaign] WARNING: restore failed: {exc}")

    # --- summary ---
    print(f"\n[campaign] SUMMARY")
    print(f"  completed ({len(completed)}): {completed}")
    if failed:
        print(f"  failed    ({len(failed)}):")
        for name, reason in failed.items():
            print(f"    {name}: {reason}")
    else:
        print(f"  failed    (0): none")
    if degraded:
        print(
            f"  DEGRADED: only mode '{original_mode_name}' was measured. "
            f"Requested: {[n for _, n in resolved]}"
        )

    return {"completed": completed, "failed": failed, "degraded": degraded}


def _run_one_campaign_scenario(scenario, resolved, original_mode_name,
                                can_switch, switch_method,
                                tag, duration, k, repeats,
                                holdout_n, keep_suspect, settle_sec):
    """Run a single scenario across all modes within the campaign loop.

    Raises on scenario failure so the campaign loop can catch and continue.
    """
    if not can_switch:
        mode_tag = f"{tag}_{scenario}_{original_mode_name}"
        print(
            f"[campaign] running '{scenario}' at current mode "
            f"{original_mode_name} (degraded, tag={mode_tag})"
        )
        mr.run_scenario(
            scenario,
            tag=mode_tag,
            duration=duration,
            k=k,
            repeats=repeats,
            holdout_n=holdout_n,
            keep_suspect=keep_suspect,
        )
        return

    for mode_id, mode_name in resolved:
        # Re-probe before each switch.
        cur_method, _ = probe_switch_capability()
        if cur_method is None:
            raise RuntimeError(
                f"Switching capability lost before switching to {mode_name} "
                f"during scenario {scenario}."
            )
        print(f"[campaign] switching to {mode_name} (ID={mode_id})")
        switch_mode(mode_id, method=cur_method)
        if settle_sec > 0:
            time.sleep(settle_sec)
        verify_mode(mode_name)

        mode_tag = f"{tag}_{scenario}_{mode_name}"
        print(f"[campaign] running '{scenario}' at {mode_name} (tag={mode_tag})")
        mr.run_scenario(
            scenario,
            tag=mode_tag,
            duration=duration,
            k=k,
            repeats=repeats,
            holdout_n=holdout_n,
            keep_suspect=keep_suspect,
        )
        print(f"[campaign] completed '{scenario}' at {mode_name}")


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

    # 9. probe_switch_capability returns a known method string or None.
    # On the dev box (Windows, no nvpmodel) it should return None without raising.
    method, desc = probe_switch_capability()
    if method not in (None, "root", "sudo", "nvpmodel_direct"):
        errors.append(f"probe_switch_capability: unexpected method {method!r}")
    else:
        print(f"[selftest] probe_switch_capability: method={method!r} OK")

    # 10. check_is_root does not raise on Windows (no os.geteuid).
    try:
        result = check_is_root()
        assert isinstance(result, bool)
        print(f"[selftest] check_is_root: {result} OK (no AttributeError)")
    except Exception as exc:
        errors.append(f"check_is_root raised: {exc}")

    # 11. Campaign order covers every known scenario exactly once.
    all_known = set(mr.SCENARIOS) | set(mr.SCALING)
    missing = set(CAMPAIGN_ORDER) - all_known
    if missing:
        errors.append(f"CAMPAIGN_ORDER references unknown scenarios: {missing}")
    else:
        print(f"[selftest] CAMPAIGN_ORDER references: OK ({len(CAMPAIGN_ORDER)} scenarios)")

    # 12. Daemon argv reconstruction: -d/-d/--daemon must be stripped from the
    # relaunch command; every other argument must be preserved exactly.
    # This is pure string logic -- no process spawning needed.
    test_cases = [
        # (input_sys_argv, expected_argv_without_script)
        (
            ["multitenant_sweep.py", "--campaign", "-d"],
            ["multitenant_sweep.py", "--campaign"],
        ),
        (
            ["multitenant_sweep.py", "--scenario", "bert_yolo",
             "--modes", "MAXN,15W", "--k", "4", "--duration", "120",
             "--tag", "run1", "--daemon"],
            ["multitenant_sweep.py", "--scenario", "bert_yolo",
             "--modes", "MAXN,15W", "--k", "4", "--duration", "120",
             "--tag", "run1"],
        ),
        (
            ["multitenant_sweep.py", "-d", "--campaign", "bert_scale,yolo_vit",
             "--modes", "MAXN", "--tag", "exp2"],
            ["multitenant_sweep.py", "--campaign", "bert_scale,yolo_vit",
             "--modes", "MAXN", "--tag", "exp2"],
        ),
    ]
    for original, expected in test_cases:
        filtered = [a for a in original if a not in ("-d", "--daemon")]
        if filtered != expected:
            errors.append(
                f"daemon argv reconstruction: expected {expected}, got {filtered}"
            )
        else:
            print(f"[selftest] daemon argv reconstruction: OK ({original!r})")

    # 13. Sweep pid/log paths are distinct from bench_jetson's daemon paths.
    bench_pid = REPO_ROOT / "logs" / "bench_daemon.pid"
    bench_log = REPO_ROOT / "logs" / "bench_daemon.log"
    if SWEEP_PID == bench_pid:
        errors.append(f"SWEEP_PID collides with bench_jetson's DAEMON_PID: {SWEEP_PID}")
    else:
        print(f"[selftest] SWEEP_PID distinct from bench_jetson pid: OK")
    if SWEEP_LOG == bench_log:
        errors.append(f"SWEEP_LOG collides with bench_jetson's DAEMON_LOG: {SWEEP_LOG}")
    else:
        print(f"[selftest] SWEEP_LOG distinct from bench_jetson log: OK")

    # 14. _count_csv_rows handles a truncated last line without raising.
    # The truncated line has no comma, which is the signal used to detect a
    # partially written row (the writer was interrupted before the second field).
    import tempfile, os as _os
    _csv_content = "tag,n_tenants,other\nmytag,2,val1\nmytag,2,val2\nmytag_partial"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False, encoding="utf-8") as _f:
        _f.write(_csv_content)
        _tmp = _f.name
    try:
        count, note = _count_csv_rows(Path(_tmp))
        if count != 2:
            errors.append(f"_count_csv_rows truncated: expected 2, got {count}")
        elif "truncated" not in note:
            errors.append(f"_count_csv_rows truncated: expected note about truncation, got {note!r}")
        else:
            print(f"[selftest] _count_csv_rows truncated line: OK (count={count}, note={note!r})")
    finally:
        _os.unlink(_tmp)

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
        default="MAXN,25W,15W",
        help="comma-separated power mode names to sweep (default: MAXN,25W,15W)"
    )
    ap.add_argument(
        "--campaign", metavar="SCENARIOS", nargs="?", const="",
        help="run all scenarios in CAMPAIGN_ORDER (or a comma-separated subset). "
             "Example: --campaign bert_scale,yolo_vit"
    )
    ap.add_argument(
        "--tag", default=None,
        help="base tag; each mode and scenario append automatically"
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
        "--require-all-modes", action="store_true",
        help="refuse to start (exit 1) when mode switching is unavailable, "
             "instead of running a degraded single-mode measurement"
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
    # Background daemon control: mutually exclusive, dispatched before any
    # measurement work begins so the launching process stays light.
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "-d", "--daemon", action="store_true",
        help="run this sweep in the background (detached, survives SSH disconnect)"
    )
    g.add_argument(
        "-s", "--stop", action="store_true",
        help="stop the background sweep"
    )
    g.add_argument(
        "-ss", "--snapshot", action="store_true",
        help="print a progress snapshot of the background sweep"
    )
    a = ap.parse_args()

    # Daemon dispatch: handle before --selftest so "-ss" always works even when
    # no scenario flag was given.
    if a.snapshot:
        _sweep_snapshot()
        return
    if a.stop:
        _sweep_stop()
        return
    if a.daemon:
        _sweep_start()
        return

    if a.selftest:
        _selftest()
        return

    if a.list_modes:
        table = load_mode_table()
        print("Discovered power modes:")
        for mid, mname in sorted(table.items()):
            print(f"  ID={mid}  NAME={mname}")
        return

    if a.k < 2:
        print(f"[error] --k must be at least 2 (got {a.k})")
        sys.exit(1)

    modes = [m.strip() for m in a.modes.split(",") if m.strip()]

    if a.campaign is not None:
        # --campaign with no value runs CAMPAIGN_ORDER; with a value runs the subset.
        if a.campaign:
            scenarios = [s.strip() for s in a.campaign.split(",") if s.strip()]
            unknown = [s for s in scenarios
                       if s not in mr.SCENARIOS and s not in mr.SCALING]
            if unknown:
                print(f"[error] unknown scenario(s) in --campaign: {unknown}")
                print(f"  known: {list(mr.SCENARIOS) + list(mr.SCALING)}")
                sys.exit(1)
        else:
            scenarios = list(CAMPAIGN_ORDER)

        tag = a.tag or "campaign"
        summary = run_campaign(
            scenarios=scenarios,
            modes=modes,
            tag=tag,
            duration=a.duration,
            k=a.k,
            repeats=a.repeats,
            holdout_n=a.holdout,
            keep_suspect=a.keep_suspect,
            settle_sec=a.settle,
            require_all_modes=a.require_all_modes,
        )
        if summary["degraded"]:
            sys.exit(EXIT_DEGRADED)
        if summary["failed"]:
            sys.exit(3)
        return

    if not a.scenario:
        ap.print_help()
        sys.exit(1)

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
        require_all_modes=a.require_all_modes,
    )


if __name__ == "__main__":
    main()
