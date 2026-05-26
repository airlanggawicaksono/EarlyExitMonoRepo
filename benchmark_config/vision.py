"""MSDNet per-exit benchmark config + sweep runner.

ALL benchmark knobs live here. AnyTimeVisionenc/src/benchmark.py is pure functions.

Sweep: datasets x weight_sources x exits (0..n_exits_for(ds)-1).
weight_source = trained only (no public HF pretrained MSDNet).

MSDNet anytime config recommends nBlocks=7. Training must use same nBlocks
as benchmark (ckpts pinned to arch). Default here matches MSDNet README.
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
MODEL_FAMILY = "msdnet"

# ---- HuggingFace ------------------------------------------------------------
HF_USER = os.environ.get("HF_USER", "wicaksonolxn")
HF_TOKEN = os.environ.get("HF_TOKEN")


def hf_trained_repo(dataset: str) -> str:
    return f"{HF_USER}/msdnet-{dataset.lower()}-ee"


def resolve_model_id(dataset: str, weight_source: str) -> Optional[str]:
    """Returns HF id, or None for random-init (HW-only profiling)."""
    if weight_source == "trained":
        return hf_trained_repo(dataset)
    if weight_source == "pretrained":
        return None  # no upstream MSDNet on HF; random init OK for HW timing
    raise ValueError(f"weight_source invalid: {weight_source}")


# ---- Model arch facts (MSDNet anytime config) -------------------------------
# Per-dataset MAX nBlocks for finer latency graph. Limited by spatial dim
# degeneration: nChannels/stride/grFactor must keep H,W >= 1 at every block.
ARCH = "msdnet"
BASE = 4
STEP_MODE = "even"

NBLOCKS_PER_DATASET = {
    "cifar10":      10,   # 32x32 input, 3-scale grFactor
    "cifar100":     10,
    "svhn":         10,   # 32x32 -> remap to cifar10 arch for HW
    "tinyimagenet": 5,    # 64x64 ImageNet-style arch; stride 4 limits nBlocks
    "imagenet":     7,    # 224x224, 4-scale grFactor, stride 4
    "stl10":        10,   # 96x96 -> resize 32x32, cifar10 arch (10 classes)
    "mnist":        10,   # 28x28 -> resize 32x32 + 3ch, cifar10 arch (10 classes)
    "fashionmnist": 10,   # 28x28 -> resize 32x32 + 3ch, cifar10 arch (10 classes)
}

# MSDNet ref only knows cifar10/cifar100/ImageNet. Map others for HW timing.
DATA_KEY_FOR_MSDNET = {
    "cifar10":      "cifar10",
    "cifar100":     "cifar100",
    "svhn":         "cifar10",       # same 32x32, HW unchanged
    "tinyimagenet": "ImageNet",      # 64x64 with ImageNet-style arch
    "imagenet":     "ImageNet",      # case-sensitive in ref
    "stl10":        "cifar10",       # resize to 32x32, same 10 classes
    "mnist":        "cifar10",       # resize to 32x32 + RGB, 10 classes
    "fashionmnist": "cifar10",       # resize to 32x32 + RGB, 10 classes
}

ARCH_CIFAR = dict(
    arch=ARCH,
    nChannels=16, growthRate=6,
    grFactor=[1, 2, 4], bnFactor=[1, 2, 4],
    base=BASE, step=2, stepmode=STEP_MODE,
    bottleneck=True, prune="max", reduction=0.5,
)
ARCH_IMAGENET = dict(
    arch=ARCH,
    nChannels=32, growthRate=16,
    grFactor=[1, 2, 4, 4], bnFactor=[1, 2, 4, 4],
    base=BASE, step=4, stepmode=STEP_MODE,
    bottleneck=True, prune="max", reduction=0.5,
)


def arch_kwargs_for(dataset: str) -> dict:
    ds = dataset.lower()
    if ds in ("imagenet", "tinyimagenet"):
        base = ARCH_IMAGENET.copy()
    else:
        base = ARCH_CIFAR.copy()
    base["nBlocks"] = NBLOCKS_PER_DATASET[ds]
    return base


def n_exits_for(dataset: str) -> int:
    return NBLOCKS_PER_DATASET[dataset.lower()]


def msdnet_data_key(dataset: str) -> str:
    """Map our dataset name to MSDNet ref's accepted 'data' arg."""
    return DATA_KEY_FOR_MSDNET[dataset.lower()]


# ---- Sweep ------------------------------------------------------------------
# HW sweep always uses cifar10 (latency measurement, content-independent).
HW_DATASET = "cifar10"

# Quality sweep: top-1/top-5 accuracy per exit per dataset.
QUALITY_DATASETS = [
    "cifar10",      # 10-class natural images (32x32)
    "cifar100",     # 100-class natural images (32x32)
    "svhn",         # digit recognition (32x32)
    "imagenet",     # 1000-class large-scale (224x224) — requires local data
    "stl10",        # 10-class natural images (96x96 → resized 32x32)
    "mnist",        # 10-class handwritten digits (28x28 → resized 32x32 + RGB)
    "fashionmnist", # 10-class fashion items (28x28 → resized 32x32 + RGB)
]

WEIGHT_SOURCES = ["pretrained"]  # random-init for HW-only sweep (no upstream MSDNet on HF)

# ---- Bench hparams ----------------------------------------------------------
BENCH_BATCH = 1
WARMUP_STEPS = 3
USE_TORCH_COMPILE = True

DATA_DIR = REPO_ROOT / "AnyTimeVisionenc" / "data"
OUT_DIR = REPO_ROOT / "logs" / "benchmark" / NAME

# =============================================================================


def run_all(
    only_weight_source: Optional[str] = None,
    only_exit: Optional[int] = None,
    only_dataset: Optional[str] = None,
    skip_quality: bool = True,   # HW-only default
    skip_hw: bool = False,
):
    from AnyTimeVisionenc import evaluate_quality, sweep_hw_all_exits

    weight_sources = [only_weight_source] if only_weight_source else WEIGHT_SOURCES
    datasets = [only_dataset] if only_dataset else QUALITY_DATASETS

    for ws in weight_sources:
        for ds in datasets:
            arch_key = msdnet_data_key(ds)
            arch_kw = arch_kwargs_for(ds)
            n_exits_ds = n_exits_for(ds)
            exits_ds = [only_exit] if only_exit is not None else list(range(n_exits_ds))
            model_id = resolve_model_id(ds, ws)
            data_dir = DATA_DIR / ds

            # HW pass: load model + per-submodule compile once per (ds, ws)
            if not skip_hw:
                try:
                    sweep_hw_all_exits(
                        model_id=model_id,
                        dataset=ds,
                        exits=exits_ds,
                        data_dir=data_dir,
                        out_root=OUT_DIR / ds,
                        arch_kwargs=arch_kw,
                        arch_key=arch_key,
                        weight_source=ws,
                        bench_batch=BENCH_BATCH,
                        warmup_steps=WARMUP_STEPS,
                        use_torch_compile=USE_TORCH_COMPILE,
                    )
                except Exception as e:
                    print(f"[vision] hw sweep failed {ds}/{ws}: {e}")

            # Quality pass: separate (no compile, fresh load per k)
            if not skip_quality:
                for k in exits_ds:
                    q_path = OUT_DIR / ds / f"exit_{k}" / "quality_results.json"
                    if has_valid_result(q_path):
                        print(f"[skip] quality exists: {q_path}")
                        continue
                    try:
                        evaluate_quality(
                            model_id=model_id,
                            dataset=ds,
                            arch_key=arch_key,
                            force_exit=k,
                            data_dir=data_dir,
                            out_dir=OUT_DIR / ds / f"exit_{k}",
                            arch_kwargs=arch_kw,
                            weight_source=ws,
                            bench_batch=BENCH_BATCH,
                        )
                    except Exception as e:
                        print(f"[vision] quality failed {ds} exit={k}: {e}")
