"""MSDNet vision benchmark config + sweep runner."""

import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared import load_env

load_env()

NAME = "vision"
MODEL_FAMILY = "msdnet"

HF_USER = os.environ.get("HF_USER", "wicaksonolxn")
HF_TEMPLATE = f"{HF_USER}/msdnet-{{dataset}}-ee"

DATASETS = ["cifar10", "cifar100", "svhn", "tinyimagenet", "ImageNet"]
EVAL_MODES = ["anytime", "dynamic"]
N_BLOCKS = 5
BENCH_BATCH = 1
WARMUP_STEPS = 3

DATA_DIR = REPO_ROOT / "AnyTimeVisionenc" / "data"
OUT_DIR = REPO_ROOT / "logs" / "benchmark" / NAME

# =============================================================================


def run_all(
    only_dataset: Optional[str] = None,
    only_mode: Optional[str] = None,
    skip_quality: bool = False,
    skip_hw: bool = False,
):
    from AnyTimeVisionenc.benchmark import profile_hw, evaluate_quality

    datasets = [only_dataset] if only_dataset else DATASETS
    modes = [only_mode] if only_mode else EVAL_MODES

    for ds in datasets:
        model_id = HF_TEMPLATE.format(dataset=ds)
        for mode in modes:
            run_dir = OUT_DIR / ds / mode
            kw = dict(
                model_id=model_id,
                dataset=ds,
                eval_mode=mode,
                data_dir=DATA_DIR / ds,
                out_dir=run_dir,
                n_blocks=N_BLOCKS,
                bench_batch=BENCH_BATCH,
                warmup_steps=WARMUP_STEPS,
            )
            if not skip_hw:
                profile_hw(**kw)
            if not skip_quality:
                evaluate_quality(
                    model_id=model_id,
                    dataset=ds,
                    eval_mode=mode,
                    data_dir=DATA_DIR / ds,
                    out_dir=run_dir,
                    n_blocks=N_BLOCKS,
                    bench_batch=BENCH_BATCH,
                )
