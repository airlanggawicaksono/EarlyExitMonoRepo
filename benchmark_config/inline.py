"""Inline post-train benchmark hook for the training grid.

Lets the training pipeline benchmark each model the moment it finishes training,
instead of a separate pass in the notebook. Wire it into the runner:

    from benchmark_config.inline import bench_trained_item
    run_grid(items, ..., post_item=bench_trained_item)

`run_grid` calls `post_item(item, run_dir)` after an item trains + its checkpoints
are pushed to HF, so the bench pulls the just-pushed repo and interleaves with
training in the same round-robin order.

After the grid, build the CSVs + plots once:

    from benchmark_config.inline import export_all
    export_all()
"""

import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Trained-weight HW + quality sweep for the just-trained item.
_BENCH_OPTS = dict(only_weight_source="trained", skip_quality=False)


def set_hf_cache(cache_dir):
    """Point the HF datasets/hub cache at a persistent dir (e.g. Drive) so eval
    sets download ONCE and survive Colab session resets. Call before any
    load_dataset (prefetch + the bench loaders read HF_DATASETS_CACHE at call
    time, so setting it here covers both)."""
    import os

    cache_dir = str(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
    print(f"[prefetch] HF cache -> {cache_dir}")


def _prefetch_one(name, n_samples):
    """Cache one eval set. Try the bench slice; fall back to the dataset's
    default split (hellaswag has no test split). Helper so prefetch_datasets
    stays a flat loop."""
    from shared import load_hf_dataset

    try:
        load_hf_dataset(name, split=f"test[:{n_samples}]")
        print(f"[prefetch] cached {name}")
    except Exception:
        load_hf_dataset(name)  # default split (e.g. hellaswag validation)
        print(f"[prefetch] cached {name} (default split)")


def prefetch_datasets(datasets=None, n_samples=128, cache_dir=None):
    """Warm the HF datasets cache before the grid so the inline bench doesn't
    stall mid-round-robin downloading each eval set on first use. Call ONCE at
    the top of the notebook, before run_grid. Network blip up front fails loud
    here instead of silently killing a quality leaf later.

    cache_dir: persistent path (e.g. Drive) -> download once, reused next
    session. The bench loaders read the same HF_DATASETS_CACHE, so they hit the
    Drive copy too."""
    from benchmark_config import llama

    if cache_dir is not None:
        set_hf_cache(cache_dir)
    names = datasets if datasets is not None else list(llama.QUALITY_DATASETS)
    for name in names:
        try:
            _prefetch_one(name, n_samples)
        except Exception as exc:
            print(f"[prefetch] {name} failed: {exc}")


def _bench_bert(cfg, mode):
    from benchmark_config import bert
    bert.run_all(only_task=cfg.task, only_mode=mode, **_BENCH_OPTS)


def _bench_vision(cfg, mode):
    from benchmark_config import vision
    vision.run_all(only_dataset=cfg.dataset, only_mode=mode, **_BENCH_OPTS)


def _bench_yolo(cfg, mode):
    from benchmark_config import yolo
    yolo.run_all(only_dataset="coco", only_mode=mode, **_BENCH_OPTS)


def _bench_llama(cfg, mode):
    from benchmark_config import llama
    llama.run_all(only_mode=mode, **_BENCH_OPTS)


# label prefix -> bench fn. Dict dispatch avoids a long if/elif chain.
_BENCH = {
    "bert": _bench_bert,
    "vision": _bench_vision,
    "yolo": _bench_yolo,
    "llama": _bench_llama,
}


def bench_trained_item(item, run_dir=None):
    """run_grid post_item hook: benchmark one just-trained item (trained weights).

    Scoped to the item's own task/dataset + mode, so it benches only what just
    trained. Exceptions propagate to run_grid, which isolates them per item."""
    backend = item.label.split("-")[0]
    fn = _BENCH.get(backend)
    if fn is None:
        print(f"[inline-bench] no mapping for {item.label}; skip")
        return
    print(f"[inline-bench] {item.label} (trained) ...")
    fn(item.cfg, getattr(item.cfg, "mode", None))


# ---- CSV + plot export (run once after the grid) ----------------------------
def _exit_key(method):
    nums = [int(n) for n in re.findall(r"\d+", method)]
    return nums or [10 ** 9]


def _export_one(cfg, results_root):
    """Per-task CSVs + cross-task averages + curated plots for one backend."""
    from shared import write_benchmark_csvs, write_average_csvs, plot_model_panel

    out_dir = Path(cfg.OUT_DIR)
    csv_root = results_root / cfg.NAME
    csv_root.mkdir(parents=True, exist_ok=True)
    run_dirs = ({p.parent for p in out_dir.rglob("hw_results.json")}
                | {p.parent for p in out_dir.rglob("quality_results.json")})
    if not run_dirs:
        print(f"[inline-bench] {cfg.NAME}: no results; skip export")
        return
    groups = defaultdict(dict)
    for rd in run_dirs:
        key = rd.parent.relative_to(out_dir).as_posix().replace("/", "_") or "root"
        groups[key][rd.name] = rd
    for key, runs in sorted(groups.items()):
        write_benchmark_csvs(results_files=runs, out_dir=csv_root / key,
                             baseline_key=None, method_order=sorted(runs, key=_exit_key))
    write_average_csvs(csv_root)
    try:
        plot_model_panel(csv_root)
    except Exception as exc:
        print(f"[inline-bench] {cfg.NAME}: plot failed: {exc}")


def export_all(results_root=None):
    """Build CSVs + plots for every backend from logs/benchmark/. Call once after
    the grid finishes (bench JSONs are written inline per item)."""
    from benchmark_config import bert, vision, yolo, llama

    root = Path(results_root) if results_root else REPO_ROOT / "results"
    for cfg in (bert, vision, yolo, llama):
        _export_one(cfg, root)
    print(f"[inline-bench] CSVs + plots under {root}")
