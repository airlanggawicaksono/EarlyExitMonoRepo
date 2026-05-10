"""LLaMa-3 early-exit benchmark config + sweep runner."""

import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared import load_env

load_env()

NAME = "llama"
MODEL_FAMILY = "llama-3-8b"

HF_USER = os.environ.get("HF_USER", "wicaksonolxn")
HF_BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
HF_EXIT_HEADS = f"{HF_USER}/llama3-8b-ee-heads"

EXIT_LAYERS = [8, 16, 24]
CONFIDENCE_THRESHOLDS = [0.5, 0.7, 0.9]
MAX_NEW_TOKENS = 128
N_SAMPLES = 100

OUT_DIR = REPO_ROOT / "logs" / "benchmark" / NAME

# =============================================================================


def run_all(
    only_threshold: Optional[float] = None,
    skip_quality: bool = False,
    skip_hw: bool = False,
):
    from AnyTimeLLaMa.benchmark import profile_hw, evaluate_quality

    thresholds = (
        [only_threshold] if only_threshold is not None else CONFIDENCE_THRESHOLDS
    )

    for thr in thresholds:
        for exit_layer in EXIT_LAYERS + [None]:  # None = full dynamic
            run_name = (
                f"thr_{thr}_exit_{exit_layer if exit_layer is not None else 'dyn'}"
            )
            run_dir = OUT_DIR / run_name
            kw = dict(
                base_model_id=HF_BASE_MODEL,
                exit_heads_id=HF_EXIT_HEADS,
                exit_layers=EXIT_LAYERS,
                out_dir=run_dir,
                confidence_threshold=thr,
                force_exit_layer=exit_layer,
                n_samples=N_SAMPLES,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            if not skip_hw:
                profile_hw(**kw)
            if not skip_quality:
                evaluate_quality(
                    base_model_id=HF_BASE_MODEL,
                    exit_heads_id=HF_EXIT_HEADS,
                    exit_layers=EXIT_LAYERS,
                    out_dir=run_dir,
                    n_samples=N_SAMPLES,
                )
