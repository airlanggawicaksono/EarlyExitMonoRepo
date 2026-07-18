"""Jetson (Orin Nano / Xavier / NX) ARM64 benchmark CLI.

Pulls trained self-distill checkpoints from HF and runs HW + quality sweeps
using the same code paths as benchmark_colab. HW metrics are auto-routed to
jetson-stats (jtop) via shared/jetson_profiler.py when running on Tegra.

Usage:
    # full grid: every backend, every task, every mode, every exit
    python bench_jetson.py all

    # one backend, narrow scope
    python bench_jetson.py bert --task SST-2 --mode segd
    python bench_jetson.py bert --task SST-2 --mode segd --exit 12
    python bench_jetson.py vision --dataset uoft-cs/cifar10
    python bench_jetson.py yolo --mode joint --sub-exit 0
    python bench_jetson.py llama --mode pairwise

    # HW only (skip quality), 5-sample smoke
    python bench_jetson.py bert --no-quality --dry-run

    # disable torch.compile (default ON)
    python bench_jetson.py bert --no-compile

Output: identical schema to benchmark_colab. Writes under
    logs/benchmark/{backend}/{task_or_dataset}/{mode}/exit_<k>/{hw,quality}_results.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from shared import load_env, load_hf_dataset, resolve_hf_dataset  # noqa: E402

load_env()


def _patch_compile(cfg_mod, enable: bool):
    if hasattr(cfg_mod, "USE_TORCH_COMPILE"):
        cfg_mod.USE_TORCH_COMPILE = enable


def _exit_sort_key(method: str):
    nums = [int(n) for n in __import__("re").findall(r"\d+", str(method))]
    return nums or [10 ** 9]


def _collect_run_dirs(out_dir: Path):
    hw = {p.parent for p in out_dir.rglob("hw_results.json")}
    q = {p.parent for p in out_dir.rglob("quality_results.json")}
    return list(hw | q)


def _group_runs(out_dir: Path, run_dirs):
    """Group run dirs by their parent (task/mode) -> {group_key: {exit_name: dir}}."""
    from collections import defaultdict
    groups = defaultdict(dict)
    for rd in run_dirs:
        rel_parent = rd.parent.relative_to(out_dir).as_posix()
        group_key = rel_parent.replace("/", "_") or "root"
        groups[group_key][rd.name] = rd
    return groups


def _export_one(cfg):
    """Per-task CSVs + cross-task averages + curated plots for one backend.
    Also writes the per-TASK grouped (pivoted) view: rows=exit, cols=series."""
    from shared import (
        write_benchmark_csvs, write_average_csvs, plot_model_panel,
        write_grouped_csvs, plot_grouped_csvs, group_by_metric,
    )

    out_dir = Path(cfg.OUT_DIR)
    csv_root = REPO_ROOT / "results" / cfg.NAME
    csv_root.mkdir(parents=True, exist_ok=True)

    run_dirs = _collect_run_dirs(out_dir)
    if not run_dirs:
        print(f"[{cfg.NAME}] no results found; skip")
        return
    groups = _group_runs(out_dir, run_dirs)
    for group_key, runs in sorted(groups.items()):
        ordered = sorted(runs.keys(), key=_exit_sort_key)
        write_benchmark_csvs(results_files=runs, out_dir=csv_root / group_key,
                             baseline_key=None, method_order=ordered)
        print(f"  [{cfg.NAME}] {group_key}: {len(runs)} runs")
    write_average_csvs(csv_root)
    # per-task grouped view (modes/baseline as columns, metric-correct) + plots
    try:
        write_grouped_csvs(out_dir, REPO_ROOT / "results_grouped", cfg.NAME)
        plot_grouped_csvs(REPO_ROOT / "results_grouped", cfg.NAME)
        # bucket this backend's tasks by metric family (accuracy/glue/detection/…)
        group_by_metric(REPO_ROOT / "results_grouped", cfg.NAME)
    except Exception as e:
        print(f"[{cfg.NAME}] grouped export failed: {e}")
    try:
        plot_model_panel(csv_root)
    except Exception as e:
        print(f"[{cfg.NAME}] plot failed: {e}")


def cmd_export(args):
    """Export CSVs (per-task + averages) + curated per-metric plots for all 4."""
    from benchmark_config import bert, vision, yolo, llama
    for cfg in (bert, vision, yolo, llama):
        _export_one(cfg)
    print(f"[export] CSVs + plots under {REPO_ROOT / 'results'}")


def _force_single_thread_inductor() -> None:
    """Compile in-process so inductor never pickles its worker state.

    On Tegra the parallel async-compile pool raises
    `TypeError: cannot pickle '_thread.RLock'` when forking workers. Setting one
    compile thread keeps torch.compile in-process (avoids that specific crash)."""
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    try:
        import torch._inductor.config as _ind

        _ind.compile_threads = 1
    except Exception as e:
        print(f"[bench_jetson] inductor single-thread setup skipped: {e}")


def _compile_smoke_ok() -> bool:
    """Actually try to torch.compile a trivial kernel. Returns True only if it
    runs end-to-end (so a fixed triton -> compile auto-enables; a broken one ->
    eager fallback). Cheap, catches the Tegra triton/inductor mismatch live."""
    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"

        @torch.compile
        def _f(x):
            return x * 2 + 1

        _f(torch.randn(8, device=dev))
        return True
    except Exception as e:
        print(f"[bench_jetson] compile smoke-test failed ({type(e).__name__}: {e})")
        return False


def _resolve_jetson_compile(args) -> None:
    """On Jetson, keep torch.compile ONLY if it actually works. Always single-thread
    inductor (avoids the RLock-pickle crash); --force-jetson-compile skips the probe."""
    _force_single_thread_inductor()
    if getattr(args, "force_jetson_compile", False):
        print("[bench_jetson] --force-jetson-compile: compile ON (probe skipped)")
        return
    if _compile_smoke_ok():
        print("[bench_jetson] compile smoke-test OK -> compile ON (single-thread inductor)")
        return
    print("[bench_jetson] compile unavailable on this Jetson -> eager "
          "(fix: pin triton to torch 2.8's match, e.g. triton==3.4.0)")
    args.compile = False


def _check_jetson() -> bool:
    try:
        from shared.jetson_profiler import is_jetson

        return is_jetson()
    except Exception:
        return False


DRY_ROOT = REPO_ROOT / "logs.dry_run" / "benchmark"


def _validate_leaf(leaf: Path):
    """Return list of issues for one run dir; empty list = logged correctly."""
    issues = []
    hw, q = leaf / "hw_results.json", leaf / "quality_results.json"
    if not hw.exists():
        issues.append("hw MISSING")
    else:
        try:
            d = json.load(open(hw))
            if not d.get("aggregate"):
                issues.append("hw empty aggregate")
            if not d.get("samples"):
                issues.append("hw 0 samples")
        except Exception as e:
            issues.append(f"hw BAD ({e})")
    if not q.exists():
        issues.append("quality MISSING")
    else:
        try:
            d = json.load(open(q))
            mm = d.get("main_metric")
            if mm is None or not isinstance(d.get(mm), (int, float)):
                issues.append("quality no main_metric")
            # n_samples is optional (yolo logs mAP, no n_samples) — only flag if
            # present AND zero/empty.
            if "n_samples" in d and not d["n_samples"]:
                issues.append("quality 0 samples")
        except Exception as e:
            issues.append(f"quality BAD ({e})")
    return issues


def _hw_summary(leaf: Path) -> str:
    """Compact HW line from a run's hw_results.json aggregate (for the dry report)."""
    try:
        agg = json.load(open(leaf / "hw_results.json")).get("aggregate", {})
    except Exception:
        return ""
    lat = agg.get("end_to_end_sec_mean")
    thr = agg.get("throughput_samples_per_sec")
    pw = agg.get("avg_power_w")
    vram = agg.get("peak_vram_allocated_mb")
    # Jetson = unified memory (no discrete VRAM); this is the real board-mem metric.
    uni = agg.get("max_unified_mem_used_mb") or agg.get("avg_unified_mem_used_mb")
    parts = []
    if isinstance(lat, (int, float)):
        parts.append(f"lat={lat * 1000:.1f}ms")
    if isinstance(thr, (int, float)):
        parts.append(f"thr={thr:.1f}/s")
    if isinstance(pw, (int, float)):
        parts.append(f"pw={pw:.1f}W")
    if isinstance(vram, (int, float)):
        parts.append(f"cuda={vram:.0f}MB")  # torch CUDA pool (slice of unified mem)
    if isinstance(uni, (int, float)) and uni > 0:
        parts.append(f"unified={uni:.0f}MB")  # board-wide unified RAM (the Jetson metric)
    return "  ".join(parts)


REAL_ROOT = REPO_ROOT / "logs" / os.environ.get("BENCH_SUBDIR", "benchmark")


def _verify_dry(only: str = None, root: Path = None) -> bool:
    """Walk a log tree and print PASS/FAIL per run so a bad backend/task/exit is
    obvious. Default root = logs.dry_run (pre-sweep smoke); pass root=REAL_ROOT
    (`verify --real`) to audit the actual sweep. Returns True if all good."""
    root = root or DRY_ROOT
    is_dry = root == DRY_ROOT
    print("=" * 60)
    print(f"[verify] checking logs under {root}")
    print("=" * 60)
    if not root.exists():
        print("[verify] no logs found" + (" — run `--dry-run` first" if is_dry else ""))
        return False
    n_ok = n_bad = 0
    sample_leaf = None
    backends = sorted(p for p in root.iterdir() if p.is_dir())
    seen = {b.name for b in backends}
    for b in backends:
        if only and b.name != only:
            continue
        leaves = {p.parent for p in b.rglob("hw_results.json")} | \
                 {p.parent for p in b.rglob("quality_results.json")}
        if not leaves:
            print(f"  FAIL [{b.name}] NO OUTPUT (errored before logging?)")
            n_bad += 1
            continue
        for leaf in sorted(leaves):
            rel = leaf.relative_to(root)
            issues = _validate_leaf(leaf)
            if issues:
                print(f"  FAIL {rel}: {'; '.join(issues)}")
                n_bad += 1
            else:
                hw = _hw_summary(leaf)
                print(f"  ok   {rel}" + (f"   [{hw}]" if hw else ""))
                n_ok += 1
                if sample_leaf is None:
                    sample_leaf = leaf
    for want in (["bert", "vision", "yolo", "llama"] if not only else [only]):
        if want not in seen:
            print(f"  FAIL [{want}] NO OUTPUT DIR (backend never ran / errored at import)")
            n_bad += 1
    # dump ONE good item's full logged keys so the schema/values are eyeballable
    if sample_leaf is not None:
        print("-" * 60)
        print(f"[verify] sample item keys -> {sample_leaf.relative_to(root)}")
        for name in ("hw_results.json", "quality_results.json"):
            p = sample_leaf / name
            if not p.exists():
                continue
            try:
                d = json.load(open(p))
            except Exception as e:
                print(f"  {name}: unreadable ({e})")
                continue
            body = d.get("aggregate", d) if name == "hw_results.json" else d
            print(f"  {name}:")
            for k, v in body.items():
                if isinstance(v, (list, dict)):
                    v = f"<{type(v).__name__} len={len(v)}>"
                print(f"    {k} = {v}")
    if not is_dry and FAIL_LOG.exists():
        print("-" * 60)
        print(f"[verify] recorded failures ({FAIL_LOG}):")
        for ln in FAIL_LOG.read_text(encoding="utf-8").splitlines()[-10:]:
            print(f"  {ln}")
    print("-" * 60)
    print(f"[verify] {n_ok} ok, {n_bad} bad")
    if n_bad:
        print("[verify] ^^ FAIL rows re-run automatically on the next sweep "
              "(error-tagged / missing results are not skipped)")
    else:
        print("[verify] all runs logged correctly")
    return n_bad == 0


def _dir_size_mb(p: Path) -> float:
    try:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
    except Exception:
        return 0.0


def _purge_artifacts(label: str = "") -> None:
    """Delete re-downloadable benchmark INPUTS (HF model snapshots + HF dataset
    cache + local yolo ckpts) while KEEPING every log/result. The bench only
    needs the JSON under logs/benchmark/; everything purged here re-pulls on
    demand. Called after a backend's sweep finishes (disk-constrained boxes).

    Kept on purpose: logs/, results/, results_grouped/, AnyTime*/datasets
    (coco/GLUE — heavy to re-pull or cheap and reused)."""
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    targets = [
        Path(os.environ.get("HUGGINGFACE_HUB_CACHE", home / "hub")),
        Path(os.environ.get("HF_DATASETS_CACHE", home / "datasets")),
        REPO_ROOT / "AnyTimeYolo" / "ckpts",
    ]
    freed = 0.0
    tag = f":{label}" if label else ""
    for t in targets:
        if t.exists():
            mb = _dir_size_mb(t)
            shutil.rmtree(t, ignore_errors=True)
            freed += mb
            print(f"[purge{tag}] rm {t} (~{mb:.0f} MB)")
    print(f"[purge{tag}] freed ~{freed:.0f} MB; logs/results kept")


def cmd_bert(args):
    from benchmark_config import bert

    _patch_compile(bert, args.compile)
    bert.run_all(
        only_task=args.task,
        only_mode=args.mode,
        only_weight_source=args.weight_source,
        only_exit=args.exit,
        skip_quality=args.no_quality,
        skip_hw=args.no_hw,
        dry_run=args.dry_run,
    )


def cmd_vision(args):
    from benchmark_config import vision

    _patch_compile(vision, args.compile)
    vision.run_all(
        only_dataset=args.dataset,
        only_mode=args.mode,
        only_weight_source=args.weight_source,
        only_exit=args.exit,
        skip_quality=args.no_quality,
        skip_hw=args.no_hw,
        dry_run=args.dry_run,
    )


def cmd_yolo(args):
    from benchmark_config import yolo

    _patch_compile(yolo, args.compile)
    yolo.run_all(
        only_dataset=args.dataset,
        only_mode=args.mode,
        only_weight_source=args.weight_source,
        only_exit=args.exit,
        only_sub_exit=args.sub_exit,
        skip_quality=args.no_quality,
        skip_hw=args.no_hw,
        dry_run=args.dry_run,
    )


def cmd_llama(args):
    from benchmark_config import llama

    _patch_compile(llama, args.compile)
    llama.run_all(
        only_mode=args.mode,
        only_dataset=args.dataset,
        only_weight_source=args.weight_source,
        only_exit=args.exit,
        skip_quality=args.no_quality,
        skip_hw=args.no_hw,
        dry_run=args.dry_run,
    )


DAEMON_PID = REPO_ROOT / "logs" / "bench_daemon.pid"
DAEMON_LOG = REPO_ROOT / "logs" / "bench_daemon.log"


def _read_daemon_pid():
    try:
        return int(DAEMON_PID.read_text().strip())
    except Exception:
        return None


def _daemon_running() -> bool:
    import psutil

    pid = _read_daemon_pid()
    return pid is not None and psutil.pid_exists(pid)


def _daemon_start(extra_note: str = ""):
    """Relaunch `all` (minus the -d flag) as a detached background process."""
    if _daemon_running():
        if extra_note:
            print(f"[daemon] WARNING: {extra_note}")
        print(f"[daemon] already running pid={_read_daemon_pid()} "
              f"(-ss snapshot, -s stop)")
        return
    DAEMON_LOG.parent.mkdir(parents=True, exist_ok=True)
    argv = [a for a in sys.argv if a not in ("-d", "--daemon")]
    cmd = [sys.executable] + argv                      # argv[0] is this script
    logf = open(DAEMON_LOG, "a", buffering=1, encoding="utf-8")
    logf.write(f"\n==== daemon start {__import__('datetime').datetime.now()} ====\n")
    spawn = {"start_new_session": True} if os.name == "posix" else {"creationflags": 0x00000008}
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, **spawn)
    DAEMON_PID.write_text(str(proc.pid))
    print(f"[daemon] started pid={proc.pid}")
    print(f"[daemon]   log:      {DAEMON_LOG}")
    print(f"[daemon]   snapshot: python bench_jetson.py all -ss")
    print(f"[daemon]   stop:     python bench_jetson.py all -s")


def _daemon_stop():
    import psutil

    pid = _read_daemon_pid()
    if pid is None or not psutil.pid_exists(pid):
        print("[daemon] no running background bench")
        DAEMON_PID.unlink(missing_ok=True)
        return
    proc = psutil.Process(pid)
    kids = proc.children(recursive=True)
    for p in [proc, *kids]:
        _terminate_quietly(p)
    _, alive = psutil.wait_procs([proc, *kids], timeout=10)
    for p in alive:
        _terminate_quietly(p, kill=True)
    DAEMON_PID.unlink(missing_ok=True)
    print(f"[daemon] stopped pid={pid} (+{len(kids)} children)")


def _terminate_quietly(proc, kill: bool = False):
    try:
        proc.kill() if kill else proc.terminate()
    except Exception:
        pass


def _daemon_snapshot():
    from benchmark_config import bert, vision, yolo, llama

    pid = _read_daemon_pid()
    status = f"RUNNING pid={pid}" if _daemon_running() else "not running"
    print(f"[daemon] {status}")
    print("[daemon] progress (completed result files per backend):")
    for cfg in (bert, vision, yolo, llama):
        out_dir = Path(cfg.OUT_DIR)
        hw = len(list(out_dir.rglob("hw_results.json"))) if out_dir.exists() else 0
        q = len(list(out_dir.rglob("quality_results.json"))) if out_dir.exists() else 0
        print(f"[daemon]   {cfg.NAME:<7} hw={hw:<4} quality={q}")
    _print_log_tail(30)


def _print_log_tail(n: int):
    if not DAEMON_LOG.exists():
        print("[daemon] no log yet")
        return
    lines = DAEMON_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"[daemon] --- last {min(n, len(lines))} log lines ---")
    for ln in lines[-n:]:
        print(ln)


# ---- nvpmodel power-mode sweep ------------------------------------------------
# Primary interface = jetson-stats (jtop) python API: `jetson.nvpmodel = "15W"`
# goes through the root jtop.service, so NO sudo is needed and the result is
# VERIFIED by reading the mode back. CLI (`sudo nvpmodel -m`) is the fallback.
# Mode names are device-specific (Orin Nano: "15W"/"7W", Super adds
# "MAXN SUPER") -> matched case-insensitively, substring allowed.
def _match_mode(requested: str, models) -> Optional[str]:
    req = requested.strip().upper()
    exact = [m for m in models if m.upper() == req]
    if exact:
        return exact[0]
    sub = [m for m in models if req in m.upper()]
    return sub[0] if len(sub) == 1 else None


def _set_power_mode(mode_name: str) -> Optional[str]:
    """Switch nvpmodel and VERIFY it took effect.

    Returns the VERIFIED mode name (e.g. "15W", "25W", "MAXN SUPER"), or None
    if the switch could not be confirmed. Callers must label output dirs from
    this return value, never from the requested name — on Super firmware
    "maxn" can resolve to "MAXN SUPER", and requests like "7w" may not exist
    at all."""
    import time

    try:
        from jtop import jtop  # type: ignore

        with jtop() as jetson:
            if not jetson.ok():
                raise RuntimeError("jtop service not responding")
            models = list(jetson.nvpmodel.models)
            target = _match_mode(mode_name, models)
            if target is None:
                print(f"[nvpmodel] '{mode_name}' does not match available modes {models} — skipped")
                return None
            if str(jetson.nvpmodel) == target:
                print(f"[nvpmodel] already in {target}")
                return target
            jetson.nvpmodel = target          # set via root jtop service (no sudo)
            for _ in range(30):               # verify: poll until the switch lands
                if not jetson.ok():
                    break
                if str(jetson.nvpmodel) == target:
                    time.sleep(5)             # let DVFS/clocks settle before timing
                    print(f"[nvpmodel] mode -> {target} (verified via jtop)")
                    return target
                time.sleep(1)
            print(f"[nvpmodel] set '{target}' sent but never confirmed (still {jetson.nvpmodel})")
            return None
    except ImportError:
        pass  # no jetson-stats -> CLI fallback below
    except Exception as e:
        print(f"[nvpmodel] jtop path failed ({e}); trying sudo nvpmodel CLI")

    # Fallback: parse conf for the id, sudo CLI, verify with -q.
    import re as _re
    table = {}
    try:
        for line in Path("/etc/nvpmodel.conf").read_text(errors="ignore").splitlines():
            m = _re.search(r"<\s*POWER_MODEL\s+ID=(\d+)\s+NAME=(\S+?)\s*>", line)
            if m:
                table[m.group(2).upper()] = int(m.group(1))
    except Exception as e:
        print(f"[nvpmodel] cannot read /etc/nvpmodel.conf: {e}")
        return None
    name = _match_mode(mode_name, list(table))
    if name is None:
        print(f"[nvpmodel] '{mode_name}' not in {list(table)} — skipped")
        return None
    r = subprocess.run(["sudo", "-n", "nvpmodel", "-m", str(table[name])],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[nvpmodel] CLI set failed: {r.stderr.strip() or r.stdout.strip()}\n"
              f"[nvpmodel] hint: install jetson-stats (preferred) or add visudo NOPASSWD for nvpmodel")
        return None
    import time
    time.sleep(5)
    cur = _current_power_mode()
    ok = cur is not None and _match_mode(mode_name, [cur]) is not None
    print(f"[nvpmodel] mode -> {cur} ({'verified' if ok else 'UNVERIFIED'})")
    return cur if ok else None


def _current_power_mode() -> Optional[str]:
    try:
        from jtop import jtop  # type: ignore
        with jtop() as jetson:
            if jetson.ok():
                return str(jetson.nvpmodel)
    except Exception:
        pass
    try:
        r = subprocess.run(["sudo", "-n", "nvpmodel", "-q"], capture_output=True, text=True)
        lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            if "NV Power Mode" in ln:
                after_colon = ln.split(":")[-1].strip()
                return after_colon or (lines[i + 1] if i + 1 < len(lines) else None)
    except Exception:
        pass
    return None


def _run_backends(args, backends):
    bar = "=" * 60
    for name, fn in backends:
        print(bar)
        print(f"  {name}")
        print(bar)
        try:
            if args.hot_reload:
                _hot_reload_backend(name, args)
            else:
                fn(args)
        except Exception as e:
            print(f"[{name}] failed: {e}")
            traceback.print_exc()
        finally:
            if getattr(args, "delete_artifacts", False):
                _purge_artifacts(name)


def cmd_all(args):
    if getattr(args, "snapshot", False):
        _daemon_snapshot()
        return
    if getattr(args, "stop", False):
        _daemon_stop()
        return
    if getattr(args, "daemon", False):
        note = ""
        if getattr(args, "power_modes", None) and _daemon_running():
            note = (f"--power-modes {args.power_modes} IGNORED: the running daemon "
                    f"keeps ITS OWN flags from when it was started. Stop it first "
                    f"(python bench_jetson.py all -s), then rerun with -d, or run "
                    f"in the foreground without -d.")
        _daemon_start(extra_note=note)
        return

    backends = []
    if not args.skip_bert:
        backends.append(("bert", cmd_bert))
    if not args.skip_vision:
        backends.append(("vision", cmd_vision))
    if not args.skip_yolo:
        backends.append(("yolo", cmd_yolo))
    if not args.skip_llama:
        backends.append(("llama", cmd_llama))

    power_modes = [m.strip() for m in (args.power_modes or "").split(",") if m.strip()]
    if not power_modes:
        _run_backends(args, backends)
    else:
        # One full sweep per power mode. Everything stays under logs/: MAXN (the
        # default profile) writes to logs/benchmark/, other modes write to
        # logs/benchmark.{mode}/ via BENCH_SUBDIR, which hot-reload children
        # inherit through the environment.
        original = _current_power_mode()
        try:
            for pm in power_modes:
                actual = _set_power_mode(pm)
                if not actual:
                    _log_failure(f"[nvpmodel] could not switch to {pm}; mode skipped")
                    continue
                # Label the output dir from the VERIFIED mode, not the request:
                # "maxn" may resolve to "MAXN SUPER", or land elsewhere on other
                # firmware. MAXN* keeps the plain benchmark/ dir; everything
                # else gets benchmark.{mode} (spaces stripped, lowercased).
                if "MAXN" in actual.upper():
                    sub = "benchmark"
                    os.environ.pop("BENCH_SUBDIR", None)
                else:
                    sub = "benchmark." + actual.lower().replace(" ", "")
                    os.environ["BENCH_SUBDIR"] = sub
                print(f"[nvpmodel] verified mode '{actual}' -> logging under logs/{sub}/")
                _run_backends(args, backends)
        finally:
            os.environ.pop("BENCH_SUBDIR", None)
            if original:
                print(f"[nvpmodel] restoring original mode: {original}")
                _set_power_mode(original)

    if getattr(args, "dry_run", False):
        _verify_dry()


def cmd_verify(args):
    root = REAL_ROOT if getattr(args, "real", False) else None
    _verify_dry(only=getattr(args, "backend", None), root=root)


def cmd_download(args):
    """Pre-fetch all benchmark datasets to local disk/cache BEFORE benching, so
    the hot-reload loop never stalls on a download. Idempotent (skips cached)."""
    print("[download] BERT GLUE TSVs ...")
    from AnyTimeBert import prepare_all as bert_prepare
    bert_prepare()

    # Vision: SKIP pre-fetch. The ViT loader streams the test/val split at bench
    # time (load_dataset(streaming=True)), so caching is pointless — and pre-warming
    # imagenet-1k here would download the WHOLE ~150GB and blow the disk. imagenet
    # streams; cifar streams. Nothing to download for vision.
    print("[download] Vision: streamed at bench (no pre-fetch needed) — skip")

    print("[download] YOLO COCO val2017 + labels ...")
    from benchmark_config import yolo as _yolo
    for ds in _yolo.HW_DATASETS:
        _yolo._ensure_dataset(ds)

    print("[download] LLaMA eval sets (best-effort HF cache) ...")
    from benchmark_config import llama as _llama
    _warm_hf_datasets(dict.fromkeys([_llama.HW_DATASET, *_llama.QUALITY_DATASETS]))
    print("[download] done.")


def _warm_hf_datasets(datasets) -> None:
    """Best-effort: touch HF datasets so they cache locally. Config-specific
    loads happen at bench time; this just pre-pulls the common ones."""
    try:
        import datasets as _datasets  # noqa: F401
    except Exception as e:
        print(f"[download]   datasets lib unavailable: {e}")
        return
    for dataset in datasets:
        spec = resolve_hf_dataset(dataset)
        try:
            load_hf_dataset(spec, split=spec.default_split)
            print(f"[download]   cached {spec.label} ({spec.default_split})")
        except Exception as e:
            print(f"[download]   {spec.label} lazy (loads at bench time): {e}")


def _common(parser: argparse.ArgumentParser):
    parser.add_argument("--mode", choices=["pairwise", "segd"], default=None,
                        help="None = sweep all")
    parser.add_argument("--exit", type=int, default=None, help="None = sweep all")
    parser.add_argument("--weight-source", dest="weight_source",
                        choices=["pretrained", "trained"], default="pretrained",
                        help="pretrained base (default; backbone pretrained, random head) vs "
                             "trained self-distill ckpts pulled from HF (use once training done)")
    parser.add_argument("-da", "--delete-artifacts", dest="delete_artifacts",
                        action="store_true",
                        help="after each backend finishes, delete its re-downloadable "
                             "inputs (HF model snapshots + HF dataset cache + yolo ckpts); "
                             "keeps all logs/results. For disk-constrained boxes.")
    parser.add_argument("--no-quality", action="store_true", help="HW only")
    parser.add_argument("--no-hw", action="store_true", help="Quality only")
    parser.add_argument("--dry-run", action="store_true", help="5-sample smoke")
    parser.add_argument("--no-compile", dest="compile", action="store_false",
                        help="disable torch.compile (default: ON)")
    parser.add_argument("--force-jetson-compile", dest="force_jetson_compile",
                        action="store_true",
                        help="force torch.compile on Jetson (default: auto-disabled, "
                             "triton wheel is broken there)")
    parser.add_argument("--no-hot-reload", dest="hot_reload", action="store_false",
                        help="run all exits in ONE process (default: fresh process per exit, RAM-safe)")
    parser.set_defaults(compile=True, hot_reload=True)


def _passthrough_flags(args) -> list:
    """Rebuild scope/flag argv for a hot-reload child process (one exit)."""
    flags = ["--weight-source", args.weight_source]
    if getattr(args, "task", None):
        flags += ["--task", args.task]
    if getattr(args, "dataset", None):
        flags += ["--dataset", args.dataset]
    if getattr(args, "mode", None):
        flags += ["--mode", args.mode]
    if getattr(args, "sub_exit", None) is not None:
        flags += ["--sub-exit", str(args.sub_exit)]
    if args.no_quality:
        flags.append("--no-quality")
    if args.no_hw:
        flags.append("--no-hw")
    if args.dry_run:
        flags.append("--dry-run")
    if not args.compile:
        flags.append("--no-compile")
    return flags


FAIL_LOG = REPO_ROOT / "logs" / "benchmark" / "_failures.log"


def _log_failure(msg: str) -> None:
    """Loud print + durable line in logs/benchmark/_failures.log — a child killed
    by the OOM reaper dies with NO traceback, so without this the sweep 'just
    stops' recording and nothing says why."""
    print(msg)
    try:
        import datetime
        FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(FAIL_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}\n")
    except Exception:
        pass


def _print_board_mem(tag: str) -> None:
    try:
        import psutil
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        print(f"[mem:{tag}] board RAM {vm.percent:.0f}% used, "
              f"{vm.available / 1e9:.2f} GB free | swap {sw.percent:.0f}% used")
    except Exception:
        pass


def _hot_reload_backend(backend_name: str, args):
    """Spawn ONE fresh process for the whole backend so its RAM is fully freed
    before the next model family loads (8GB Jetson can't hold all 4 at once).

    The child runs every exit IN-PROCESS (--no-hot-reload): the per-backend
    sweep_hw loads the model once and loops force_exit, so there is no reason to
    reload per exit (RAM peak is identical — one model resident either way)."""
    base = [sys.executable, str(Path(__file__).resolve()), backend_name, "--no-hot-reload"]
    argv = base + _passthrough_flags(args)
    if args.exit is not None:
        argv += ["--exit", str(args.exit)]
    print(f"[hot-reload] {backend_name} -> fresh process (all exits in-process)")
    _print_board_mem(f"{backend_name}:start")
    rc = subprocess.run(argv, check=False).returncode
    if rc != 0:
        # posix signal death = negative rc (-9 = SIGKILL); 137 = 128+9 via shell.
        oom = rc in (-9, 137)
        hint = (" — SIGKILL, almost certainly the kernel OOM reaper "
                "(confirm: `dmesg | grep -i 'killed process'`); finished exits are "
                "kept, rerun `all` to resume the rest" if oom else
                " — see traceback above; finished exits kept, rerun to resume")
        _log_failure(f"[hot-reload] {backend_name} exited rc={rc}{hint}")
    _print_board_mem(f"{backend_name}:end")


def main():
    p = argparse.ArgumentParser(description="Jetson ARM benchmark CLI.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_bert = sub.add_parser("bert", help="ElasticBERT-large × GLUE")
    p_bert.add_argument("--task", default=None, help="None = sweep all GLUE tasks")
    _common(p_bert)
    p_bert.set_defaults(func=cmd_bert)

    p_vision = sub.add_parser("vision", help="ViT-large × CIFAR-10/100")
    p_vision.add_argument("--dataset", default=None, help="e.g. uoft-cs/cifar10")
    _common(p_vision)
    p_vision.set_defaults(func=cmd_vision)

    p_yolo = sub.add_parser("yolo", help="gelan-m-ee × COCO")
    p_yolo.add_argument("--dataset", default=None, help="e.g. coco")
    p_yolo.add_argument("--sub-exit", dest="sub_exit", type=int, default=None,
                        choices=[0, 1, 2], help="0=P3, 1=P4, 2=P5")
    _common(p_yolo)
    p_yolo.set_defaults(func=cmd_yolo)

    p_llama = sub.add_parser("llama", help="LLaMA-3.2-1B × C4-derived eval")
    p_llama.add_argument("--dataset", default=None, help="e.g. cnn_dailymail | gsm8k")
    _common(p_llama)
    p_llama.set_defaults(func=cmd_llama)

    p_dl = sub.add_parser("download", help="Pre-fetch all benchmark datasets, then exit")
    p_dl.set_defaults(func=cmd_download, hot_reload=False, weight_source="pretrained", compile=True)

    p_ex = sub.add_parser("export", help="Export CSVs + curated plots for all backends, then exit")
    p_ex.set_defaults(func=cmd_export, hot_reload=False, weight_source="pretrained", compile=True)

    p_vf = sub.add_parser("verify", help="Check logs — print which runs logged OK vs bad")
    p_vf.add_argument("backend", nargs="?", default=None, help="restrict to one backend")
    p_vf.add_argument("--real", action="store_true",
                      help="audit the real sweep (logs/benchmark) instead of logs.dry_run")
    p_vf.set_defaults(func=cmd_verify, hot_reload=False, weight_source="pretrained", compile=True)

    p_all = sub.add_parser("all", help="Run every backend sequentially")
    p_all.add_argument("--skip-bert", action="store_true")
    p_all.add_argument("--skip-vision", action="store_true")
    p_all.add_argument("--skip-yolo", action="store_true")
    p_all.add_argument("--skip-llama", action="store_true")
    p_all.add_argument("--task", default=None)       # unused for vision/yolo/llama
    p_all.add_argument("--dataset", default=None)    # unused for bert
    p_all.add_argument("--sub-exit", dest="sub_exit", type=int, default=None)
    p_all.add_argument("--power-modes", dest="power_modes", default=None,
                       help="comma list of nvpmodel modes to sweep, e.g. 'maxn,15w,7w'. "
                            "Each mode runs the full grid; maxn logs to logs/benchmark/, "
                            "others to logs/benchmark.{mode}/. Set + verified via jtop "
                            "(sudo nvpmodel CLI fallback). Original mode restored at the end.")
    # Background daemon control (mutually exclusive).
    g_daemon = p_all.add_mutually_exclusive_group()
    g_daemon.add_argument("-d", "--daemon", action="store_true",
                          help="run the whole sweep in the background (detached)")
    g_daemon.add_argument("-s", "--stop", action="store_true",
                          help="stop the background sweep")
    g_daemon.add_argument("-ss", "--snapshot", action="store_true",
                          help="print a progress snapshot of the background sweep")
    _common(p_all)
    p_all.set_defaults(func=cmd_all)

    args = p.parse_args()
    on_jetson = _check_jetson()
    # On Jetson, torch.compile only works if the installed triton matches torch's
    # inductor (e.g. torch 2.8 <-> triton 3.4.0; triton 3.7 lacks the
    # KernelMetadata.cluster_dims field inductor reads -> crash). Probe it live:
    # compile stays ON when it works, falls back to eager only when it truly
    # can't. So once triton is pinned correctly, compiled numbers come for free.
    #
    # BUT: `all` + hot-reload runs every backend in a fresh child which re-probes
    # compile itself. Probing here would pull torch + a CUDA context + inductor
    # (~1.5-2 GB unified RAM on Tegra) into THIS parent, which then sits on that
    # memory for the entire multi-hour sweep and starves the children — prime
    # OOM-kill trigger on 8 GB boards. Skip the probe in the parent.
    _probe_in_children = args.cmd == "all" and getattr(args, "hot_reload", False)
    if on_jetson and getattr(args, "compile", False) and not _probe_in_children:
        _resolve_jetson_compile(args)
    print(f"[bench_jetson] jetson={on_jetson} compile={getattr(args, 'compile', False)} "
          f"ws={getattr(args, 'weight_source', '-')} "
          f"hot_reload={getattr(args, 'hot_reload', False)} cmd={args.cmd}")
    # A single-backend run loads its model once and sweeps all exits in-process
    # (sweep_hw is "load once, loop force_exit"). No per-exit subprocess needed —
    # the process exits afterward, freeing RAM. Hot-reload only matters for `all`,
    # where cmd_all isolates each of the 4 model families in its own process.
    args.func(args)
    # `all` purges between backends inside cmd_all; for a single-backend run do it
    # here once the sweep is done.
    if args.cmd in ("bert", "vision", "yolo", "llama") and getattr(args, "delete_artifacts", False):
        _purge_artifacts(args.cmd)
    print("[bench_jetson] done.")


if __name__ == "__main__":
    main()
