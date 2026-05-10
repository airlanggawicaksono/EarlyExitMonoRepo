"""ElasticBERT benchmark config + sweep runner."""

import os, sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared import load_env
load_env()

NAME = "bert"
MODEL_FAMILY = "elasticbert-base"

HF_USER     = os.environ.get("HF_USER", "wicaksonolxn")
HF_TEMPLATE = f"{HF_USER}/elasticbert-base-{{task}}-ee"

TASKS = ["SST-2", "MRPC", "QNLI", "RTE", "CoLA"]
SWEEPS = {
    "entropy":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "patience": [0,   1,   2,   3,   4,   6,   8],
}

MAX_SEQ_LENGTH    = 128
BENCH_BATCH       = 1
WARMUP_STEPS      = 3
USE_TORCH_COMPILE = True

DATA_DIR = REPO_ROOT / "AnyTimeBert" / "glue_data"
OUT_DIR  = REPO_ROOT / "logs" / "benchmark" / NAME

# =============================================================================

def run_all(
    only_task: Optional[str] = None,
    only_strategy: Optional[str] = None,
    skip_quality: bool = False,
    skip_hw: bool = False,
):
    """Iterate full sweep. Calls profile_hw + evaluate_quality per run."""
    from AnyTimeBert.benchmark import profile_hw, evaluate_quality

    tasks = [only_task] if only_task else TASKS
    for task in tasks:
        model_id = HF_TEMPLATE.format(task=task.lower())
        for strategy, values in SWEEPS.items():
            if only_strategy and strategy != only_strategy:
                continue
            for v in values:
                run_dir = OUT_DIR / task / f"{strategy}_{v}"
                if not skip_hw:
                    profile_hw(
                        model_id=model_id, task=task, strategy=strategy, threshold=v,
                        data_dir=DATA_DIR / task, out_dir=run_dir,
                        max_seq_length=MAX_SEQ_LENGTH, warmup_steps=WARMUP_STEPS,
                        use_torch_compile=USE_TORCH_COMPILE,
                    )
                if not skip_quality:
                    evaluate_quality(
                        model_id=model_id, task=task, strategy=strategy, threshold=v,
                        data_dir=DATA_DIR / task, out_dir=run_dir,
                        max_seq_length=MAX_SEQ_LENGTH,
                    )
