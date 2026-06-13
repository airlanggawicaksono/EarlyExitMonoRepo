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
import os
import subprocess
import sys
import traceback
from pathlib import Path

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


def _check_jetson() -> bool:
    try:
        from shared.jetson_profiler import is_jetson

        return is_jetson()
    except Exception:
        return False


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


def _daemon_start():
    """Relaunch `all` (minus the -d flag) as a detached background process."""
    if _daemon_running():
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


def cmd_all(args):
    if getattr(args, "snapshot", False):
        _daemon_snapshot()
        return
    if getattr(args, "stop", False):
        _daemon_stop()
        return
    if getattr(args, "daemon", False):
        _daemon_start()
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


def cmd_download(args):
    """Pre-fetch all benchmark datasets to local disk/cache BEFORE benching, so
    the hot-reload loop never stalls on a download. Idempotent (skips cached)."""
    print("[download] BERT GLUE TSVs ...")
    from AnyTimeBert import prepare_all as bert_prepare
    bert_prepare()

    print("[download] Vision CIFAR-10/100 (HF cache for ViT loader) ...")
    # The pretrained/trained ViT loader pulls these via HF load_dataset(split="test").
    # (The legacy AnyTimeVisionenc.prepare_all is MSDNet-only — not used by ViT.)
    from benchmark_config import vision as _vis
    _warm_hf_datasets(_vis.DATASETS)

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
    subprocess.run(argv, check=False)


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

    p_all = sub.add_parser("all", help="Run every backend sequentially")
    p_all.add_argument("--skip-bert", action="store_true")
    p_all.add_argument("--skip-vision", action="store_true")
    p_all.add_argument("--skip-yolo", action="store_true")
    p_all.add_argument("--skip-llama", action="store_true")
    p_all.add_argument("--task", default=None)       # unused for vision/yolo/llama
    p_all.add_argument("--dataset", default=None)    # unused for bert
    p_all.add_argument("--sub-exit", dest="sub_exit", type=int, default=None)
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
    # torch.compile/inductor is broken on this Jetson's wheel stack: the bundled
    # triton lacks `KernelMetadata.cluster_dims` that torch 2.8's inductor
    # codegen expects (version skew). It cannot compile a single kernel. Run
    # EAGER on Jetson — eager latency is production-representative anyway; the
    # compiled numbers come from the x86 (Colab) path. Override with
    # --force-jetson-compile if you've fixed the triton/torch match yourself.
    if on_jetson and getattr(args, "compile", False) and not getattr(args, "force_jetson_compile", False):
        print("[bench_jetson] torch.compile disabled on Jetson (triton/inductor "
              "incompatible with torch 2.8 wheel: missing KernelMetadata.cluster_dims). "
              "Running eager. Use --force-jetson-compile to override.")
        args.compile = False
    elif on_jetson and getattr(args, "compile", False):
        _force_single_thread_inductor()
        print("[bench_jetson] --force-jetson-compile: compile ON, inductor single-thread")
    print(f"[bench_jetson] jetson={on_jetson} compile={getattr(args, 'compile', False)} "
          f"ws={getattr(args, 'weight_source', '-')} "
          f"hot_reload={getattr(args, 'hot_reload', False)} cmd={args.cmd}")
    # A single-backend run loads its model once and sweeps all exits in-process
    # (sweep_hw is "load once, loop force_exit"). No per-exit subprocess needed —
    # the process exits afterward, freeing RAM. Hot-reload only matters for `all`,
    # where cmd_all isolates each of the 4 model families in its own process.
    args.func(args)
    print("[bench_jetson] done.")


if __name__ == "__main__":
    main()
