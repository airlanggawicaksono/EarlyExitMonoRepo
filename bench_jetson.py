"""Jetson (Orin Nano / Xavier / NX) ARM64 benchmark CLI.

Pulls trained self-distill checkpoints from HF and runs HW + quality sweeps
using the same code paths as benchmark_colab. HW metrics are auto-routed to
jetson-stats (jtop) via shared/jetson_profiler.py when running on Tegra.

Usage:
    # full grid: every backend, every task, every mode, every exit
    python bench_jetson.py all

    # one backend, narrow scope
    python bench_jetson.py bert --task SST-2 --mode cascade
    python bench_jetson.py bert --task SST-2 --mode cascade --exit 12
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

from shared import load_env  # noqa: E402

load_env()

# Per-backend exit counts (for hot-reload enumeration). Mirrors benchmark_config.
_N_EXITS = {"bert": 24, "vision": 24, "llama": 16, "yolo": 6}


def _patch_compile(cfg_mod, enable: bool):
    if hasattr(cfg_mod, "USE_TORCH_COMPILE"):
        cfg_mod.USE_TORCH_COMPILE = enable


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


def cmd_all(args):
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
                _hot_reload_exits(name, args)
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

    print("[download] Vision CIFAR-10/100 ...")
    from AnyTimeVisionenc import prepare_all as vision_prepare
    vision_prepare()

    print("[download] YOLO COCO val2017 + labels ...")
    from benchmark_config import yolo as _yolo
    for ds in _yolo.HW_DATASETS:
        _yolo._ensure_dataset(ds)

    print("[download] LLaMA eval sets (best-effort HF cache) ...")
    _warm_hf_datasets(["cnn_dailymail", "gsm8k"])
    print("[download] done.")


def _warm_hf_datasets(names) -> None:
    """Best-effort: touch HF datasets so they cache locally. Config-specific
    loads happen at bench time; this just pre-pulls the common ones."""
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"[download]   datasets lib unavailable: {e}")
        return
    for name in names:
        try:
            load_dataset(name, split="test")
            print(f"[download]   cached {name}")
        except Exception as e:
            print(f"[download]   {name} lazy (loads at bench time): {e}")


def _common(parser: argparse.ArgumentParser):
    parser.add_argument("--mode", choices=["joint", "pairwise", "cascade"], default=None,
                        help="None = sweep all")
    parser.add_argument("--exit", type=int, default=None, help="None = sweep all")
    parser.add_argument("--weight-source", dest="weight_source",
                        choices=["pretrained", "trained"], default="pretrained",
                        help="default model (pretrained, copy head to exits) vs trained ckpts")
    parser.add_argument("--no-quality", action="store_true", help="HW only")
    parser.add_argument("--no-hw", action="store_true", help="Quality only")
    parser.add_argument("--dry-run", action="store_true", help="5-sample smoke")
    parser.add_argument("--no-compile", dest="compile", action="store_false",
                        help="disable torch.compile (default: ON)")
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


def _hot_reload_exits(backend_name: str, args):
    """Spawn one FRESH process per exit so RAM is fully freed between exits
    (8GB Jetson). Each child runs a single exit in-process (--no-hot-reload)."""
    n = _N_EXITS[backend_name]
    exits = [args.exit] if args.exit is not None else list(range(n))
    base = [sys.executable, str(Path(__file__).resolve()), backend_name, "--no-hot-reload"]
    for k in exits:
        argv = base + ["--exit", str(k)] + _passthrough_flags(args)
        print(f"[hot-reload] {backend_name} exit={k} -> fresh process")
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

    p_all = sub.add_parser("all", help="Run every backend sequentially")
    p_all.add_argument("--skip-bert", action="store_true")
    p_all.add_argument("--skip-vision", action="store_true")
    p_all.add_argument("--skip-yolo", action="store_true")
    p_all.add_argument("--skip-llama", action="store_true")
    p_all.add_argument("--task", default=None)       # unused for vision/yolo/llama
    p_all.add_argument("--dataset", default=None)    # unused for bert
    p_all.add_argument("--sub-exit", dest="sub_exit", type=int, default=None)
    _common(p_all)
    p_all.set_defaults(func=cmd_all)

    args = p.parse_args()
    on_jetson = _check_jetson()
    print(f"[bench_jetson] jetson={on_jetson} compile={args.compile} "
          f"ws={args.weight_source} hot_reload={args.hot_reload} cmd={args.cmd}")
    # Single-backend + hot-reload: spawn one fresh process per exit (RAM-safe).
    # `all` handles its own hot-reload inside cmd_all (per backend).
    if args.cmd != "all" and args.hot_reload:
        _hot_reload_exits(args.cmd, args)
    else:
        args.func(args)
    print("[bench_jetson] done.")


if __name__ == "__main__":
    main()
