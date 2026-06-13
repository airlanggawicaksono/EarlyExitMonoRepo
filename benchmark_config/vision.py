"""Multi-exit ViT per-exit benchmark config + sweep runner.

ALL benchmark knobs here. AnyTimeVisionenc/src/benchmark_trained_vit.py is the
trained-model entrypoint. The legacy MSDNet path (benchmark.py) stays for
weight_source="pretrained" random-init HW timing.

Sweep: datasets × modes × exits.

Repo naming on HF (pushed by GeneralizableSelfDistilationEarlyExitTraining.sync):
    {HF_USER}/selfdistill-vision-{dataset_slug}-{mode}
    dataset_slug = dataset.split("/")[-1].lower()  (e.g. "cifar10")

Output layout:
    logs/benchmark/vision/{dataset_slug}/{mode}/exit_{k}/{hw,quality}_results.json
"""

import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared import load_env, has_valid_result

load_env()

NAME = "vision"
MODEL_FAMILY = "vit-large-patch16-224"

HF_USER = os.environ.get("HF_USER", "wicaksonolxn")
HF_TOKEN = os.environ.get("HF_TOKEN")


def _slug(dataset: str) -> str:
    return dataset.split("/")[-1].lower()


def hf_trained_repo(dataset: str, mode: str) -> str:
    return f"{HF_USER}/selfdistill-vision-{_slug(dataset)}-{mode}"


# ---- Sweep ------------------------------------------------------------------
# Use full HF dataset ids (matches train_colab.ipynb VISION_DATASETS).
# TRAINED_DATASETS: self-distill ckpts exist on HF (training did these).
# PRETRAINED_ONLY_DATASETS: no trained ckpt — meaningful ONLY for the pretrained
#   baseline, where ViT-large's native 1000-way head matches exactly (real quality).
TRAINED_DATASETS = ["uoft-cs/cifar10", "uoft-cs/cifar100"]
PRETRAINED_ONLY_DATASETS = ["imagenet-1k"]
DATASETS = TRAINED_DATASETS + PRETRAINED_ONLY_DATASETS
MODES = ["pairwise", "segd"]
WEIGHT_SOURCES = ["trained"]
N_EXITS = 24                    # ViT-large-patch16-224 = 24 blocks
NUM_LABELS_FOR_DATASET = {
    "uoft-cs/cifar10": 10,
    "uoft-cs/cifar100": 100,
    "imagenet-1k": 1000,        # head matches ViT pretrained classifier -> REAL quality
}

# ---- Bench hparams ----------------------------------------------------------
BENCH_BATCH = 1
WARMUP_STEPS = 3
USE_TORCH_COMPILE = True

OUT_DIR = REPO_ROOT / "logs" / "benchmark" / NAME

# =============================================================================


def run_all(
    only_dataset: Optional[str] = None,
    only_mode: Optional[str] = None,
    only_weight_source: Optional[str] = None,
    only_exit: Optional[int] = None,
    skip_quality: bool = True,
    skip_hw: bool = False,
    dry_run: bool = False,
):
    max_samples = 5 if dry_run else None
    out_root_base = REPO_ROOT / "logs.dry_run" / "benchmark" / NAME if dry_run else OUT_DIR
    datasets = [only_dataset] if only_dataset else (DATASETS[:1] if dry_run else DATASETS)
    modes = [only_mode] if only_mode else MODES
    weight_sources = [only_weight_source] if only_weight_source else WEIGHT_SOURCES
    exits = [only_exit] if only_exit is not None else list(range(N_EXITS))
    if dry_run:
        print(f"[vision] DRY RUN: 5 samples -> {out_root_base} | datasets={datasets} modes={modes}")

    for ws in weight_sources:
        for ds in datasets:
            if ws == "trained" and ds in PRETRAINED_ONLY_DATASETS:
                print(f"[vision] skip {ds} for trained (no ckpt; pretrained/baseline-only dataset)")
                continue
            num_labels = NUM_LABELS_FOR_DATASET[ds]
            out_root_ds = out_root_base / _slug(ds)
            if ws == "pretrained":
                _bench_pretrained(ds, num_labels, exits, out_root_ds / "pretrained",
                                  skip_hw, skip_quality, max_samples)
                continue
            for mode in modes:
                _bench_trained(ds, mode, num_labels, exits, out_root_ds / mode, ws,
                               skip_hw, skip_quality, max_samples)


def _bench_pretrained(ds, num_labels, exits, out_root, skip_hw, skip_quality, max_samples):
    """Pretrained ViT (default model) HW + quality at each exit."""
    from AnyTimeVisionenc import sweep_hw_pretrained, evaluate_quality_pretrained

    if not skip_hw:
        try:
            sweep_hw_pretrained(
                dataset=ds, exits=exits, n_exits=N_EXITS, num_labels=num_labels,
                out_root=out_root, bench_batch=BENCH_BATCH, warmup_steps=WARMUP_STEPS,
                use_torch_compile=USE_TORCH_COMPILE, max_samples=max_samples,
            )
        except Exception as e:
            print(f"[vision] pretrained hw sweep failed {ds}: {e}")

    if skip_quality:
        return
    for k in exits:
        run_dir = out_root / f"exit_{k}"
        if has_valid_result(run_dir / "quality_results.json"):
            print(f"[skip] quality exists: {run_dir / 'quality_results.json'}")
            continue
        try:
            evaluate_quality_pretrained(
                dataset=ds, force_exit=k, n_exits=N_EXITS, num_labels=num_labels,
                out_dir=run_dir, bench_batch=BENCH_BATCH, max_samples=max_samples,
            )
        except Exception as e:
            print(f"[vision] pretrained quality failed {ds} exit={k}: {e}")


def _bench_trained(ds, mode, num_labels, exits, out_root, ws, skip_hw, skip_quality, max_samples):
    """Trained ViT (HF ckpt) HW + quality at each exit for one mode."""
    from AnyTimeVisionenc import sweep_hw_trained, evaluate_quality_trained

    repo_id = hf_trained_repo(ds, mode)
    if not skip_hw:
        try:
            sweep_hw_trained(
                repo_id=repo_id, dataset=ds, mode=mode, exits=exits, n_exits=N_EXITS,
                num_labels=num_labels, out_root=out_root, weight_source=ws,
                bench_batch=BENCH_BATCH, warmup_steps=WARMUP_STEPS,
                use_torch_compile=USE_TORCH_COMPILE, max_samples=max_samples,
            )
        except Exception as e:
            print(f"[vision] hw sweep failed {ds}/{mode}: {e}")

    if skip_quality:
        return
    for k in exits:
        run_dir = out_root / f"exit_{k}"
        if has_valid_result(run_dir / "quality_results.json"):
            print(f"[skip] quality exists: {run_dir / 'quality_results.json'}")
            continue
        try:
            evaluate_quality_trained(
                repo_id=repo_id, dataset=ds, mode=mode, force_exit=k, n_exits=N_EXITS,
                num_labels=num_labels, out_dir=run_dir, weight_source=ws,
                bench_batch=BENCH_BATCH, max_samples=max_samples,
            )
        except Exception as e:
            print(f"[vision] quality failed {ds}/{mode} exit={k}: {e}")
