"""LLaMa-3-8B early-exit benchmark config + sweep runner."""

from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

NAME = "llama"
MODEL_FAMILY = "llama-3-8b"

HF_USER         = "your-username"
HF_BASE_MODEL   = "meta-llama/Meta-Llama-3-8B"
HF_EXIT_HEADS   = f"{HF_USER}/llama3-8b-ee-heads"

EXIT_LAYERS           = [8, 16, 24]
CONFIDENCE_THRESHOLDS = [0.5, 0.7, 0.9]
MAX_NEW_TOKENS        = 128
N_SAMPLES             = 100

OUT_DIR = REPO_ROOT / "logs" / "benchmark" / NAME

# =============================================================================

def run_all(only_threshold: Optional[float] = None):
    from AnyTimeLLaMa.benchmark import benchmark as _bench

    for thr in (CONFIDENCE_THRESHOLDS if only_threshold is None else [only_threshold]):
        for exit_layer in EXIT_LAYERS + [None]:   # None = dynamic / full sweep
            run_name = f"thr_{thr}_exit_{exit_layer if exit_layer is not None else 'dyn'}"
            run_dir  = OUT_DIR / run_name
            _bench(
                base_model_id=HF_BASE_MODEL,
                exit_heads_id=HF_EXIT_HEADS,
                exit_layers=EXIT_LAYERS,
                confidence_threshold=thr,
                force_exit_layer=exit_layer,
                n_samples=N_SAMPLES,
                max_new_tokens=MAX_NEW_TOKENS,
                out_dir=run_dir,
            )
