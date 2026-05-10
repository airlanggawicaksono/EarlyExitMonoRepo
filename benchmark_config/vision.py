"""MSDNet vision benchmark config + sweep runner."""

from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

NAME = "vision"
MODEL_FAMILY = "msdnet"

HF_USER     = "your-username"
HF_TEMPLATE = f"{HF_USER}/msdnet-{{dataset}}-ee"

DATASETS  = ["cifar10", "cifar100", "svhn", "tinyimagenet", "imagenet"]
EVAL_MODES = ["anytime", "dynamic"]
N_BLOCKS  = 5         # MSDNet block count (5 or 7)
BENCH_BATCH  = 1
WARMUP_STEPS = 3

DATA_DIR = REPO_ROOT / "AnyTimeVisionenc" / "data"
OUT_DIR  = REPO_ROOT / "logs" / "benchmark" / NAME

# =============================================================================

def run_all(only_dataset: Optional[str] = None, only_mode: Optional[str] = None):
    from AnyTimeVisionenc.benchmark import benchmark as _bench

    datasets = [only_dataset] if only_dataset else DATASETS
    modes    = [only_mode]    if only_mode    else EVAL_MODES

    for ds in datasets:
        model_id = HF_TEMPLATE.format(dataset=ds)
        for mode in modes:
            run_dir = OUT_DIR / ds / mode
            _bench(
                model_id=model_id,
                dataset=ds,
                eval_mode=mode,
                data_dir=DATA_DIR / ds,
                out_dir=run_dir,
                n_blocks=N_BLOCKS,
                bench_batch=BENCH_BATCH,
                warmup_steps=WARMUP_STEPS,
            )
