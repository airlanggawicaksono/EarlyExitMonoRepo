"""LLaMa-3.2-1B per-layer benchmark config + sweep runner.

ALL benchmark knobs live here. AnyTimeLLaMa/src/benchmark.py is pure functions.

Sweep: weight_sources x exits (per-layer 0..N_EXITS-1).
- weight_source=trained:    use trained head if exit in EXIT_LAYERS, else base.lm_head
- weight_source=pretrained: base.lm_head at every layer (no trained heads loaded)
"""

import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared import load_env

load_env()

NAME = "llama"
MODEL_FAMILY = "llama-3.2-1b"

# ---- HuggingFace ------------------------------------------------------------
HF_USER = os.environ.get("HF_USER", "wicaksonolxn")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_BASE_MODEL = "meta-llama/Llama-3.2-1B"
HF_EXIT_HEADS = f"{HF_USER}/llama-3.2-1b-ee-heads"

# ---- Model arch facts -------------------------------------------------------
N_LAYERS = 16  # Llama-3.2-1B = 16 transformer layers
EXIT_LAYERS = [4, 8, 12]  # which layers have trained heads
N_EXITS = N_LAYERS  # benchmark per-layer (0..15)
WEIGHT_SOURCES = ["pretrained"]  # HW-only sweep; add "trained" once heads pushed

# ---- Bench hparams ----------------------------------------------------------
N_SAMPLES = 100
MAX_NEW_TOKENS = 128
WARMUP_STEPS = 3
USE_TORCH_COMPILE = True

OUT_DIR = REPO_ROOT / "logs" / "benchmark" / NAME

# =============================================================================


def _resolve_exit_heads_id(weight_source: str) -> Optional[str]:
    if weight_source == "trained":
        return HF_EXIT_HEADS
    if weight_source == "pretrained":
        return None
    raise ValueError(f"weight_source must be in {WEIGHT_SOURCES}, got {weight_source}")


def run_all(
    only_weight_source: Optional[str] = None,
    only_exit: Optional[int] = None,
    skip_quality: bool = True,   # HW-only default
    skip_hw: bool = False,
):
    from AnyTimeLLaMa import profile_hw, evaluate_quality

    weight_sources = [only_weight_source] if only_weight_source else WEIGHT_SOURCES
    exits = [only_exit] if only_exit is not None else list(range(N_EXITS))

    for ws in weight_sources:
        heads_id = _resolve_exit_heads_id(ws)
        for k in exits:
            run_dir = OUT_DIR / ws / f"exit_{k}"
            if not skip_hw:
                profile_hw(
                    base_model_id=HF_BASE_MODEL,
                    exit_heads_id=heads_id,
                    exit_layers=EXIT_LAYERS,
                    force_exit=k,
                    out_dir=run_dir,
                    weight_source=ws,
                    n_samples=N_SAMPLES,
                    max_new_tokens=MAX_NEW_TOKENS,
                    warmup_steps=WARMUP_STEPS,
                    use_torch_compile=USE_TORCH_COMPILE,
                )
            if not skip_quality:
                evaluate_quality(
                    base_model_id=HF_BASE_MODEL,
                    exit_heads_id=heads_id,
                    exit_layers=EXIT_LAYERS,
                    force_exit=k,
                    out_dir=run_dir,
                    weight_source=ws,
                    n_samples=N_SAMPLES,
                )
