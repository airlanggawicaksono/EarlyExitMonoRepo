"""LLaMa-3.2-1B per-layer benchmark config + sweep runner.

ALL benchmark knobs live here. AnyTimeLLaMa/src/benchmark.py is pure functions.

Sweep: weight_sources x datasets x exits (per-layer 0..N_EXITS-1).
- weight_source=trained:    use trained head if exit in EXIT_LAYERS, else base.lm_head
- weight_source=pretrained: base.lm_head at every layer (no trained heads loaded)

HW pass: always cnn_dailymail (prompt length distribution matters for latency, not content).
Quality pass: all QUALITY_DATASETS — perplexity for generation tasks, MCQ accuracy for others.
"""

import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared import load_env, has_valid_result

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

# ---- Benchmark datasets -----------------------------------------------------
# HW sweep always uses cnn_dailymail (latency measurement, not content-dependent).
HW_DATASET = "cnn_dailymail"

# Quality sweep: perplexity for generation tasks, MCQ accuracy for mcq tasks.
QUALITY_DATASETS = [
    "cnn_dailymail",   # generation — perplexity on news text
    "gsm8k",           # generation — perplexity on math solutions
    "arc_challenge",   # mcq       — science reasoning accuracy
    "hellaswag",       # mcq       — commonsense completion accuracy
    "mmlu",            # mcq       — broad knowledge accuracy (57 subjects)
]

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
    only_dataset: Optional[str] = None,
    skip_quality: bool = True,   # HW-only default
    skip_hw: bool = False,
):
    from AnyTimeLLaMa import sweep_exit

    weight_sources = [only_weight_source] if only_weight_source else WEIGHT_SOURCES
    exits = [only_exit] if only_exit is not None else list(range(N_EXITS))
    quality_datasets = [only_dataset] if only_dataset else QUALITY_DATASETS

    for ws in weight_sources:
        heads_id = _resolve_exit_heads_id(ws)
        for k in exits:
            # hw-only mode: no quality datasets, run hw on HW_DATASET
            hw_dir = OUT_DIR / HW_DATASET / f"exit_{k}" if (not skip_hw and skip_quality) else None
            if hw_dir is not None and has_valid_result(hw_dir / "hw_results.json"):
                print(f"[skip] hw exists: {hw_dir / 'hw_results.json'}")
                hw_dir = None
            q_dirs_full = (
                {ds: OUT_DIR / ds / f"exit_{k}" for ds in quality_datasets}
                if not skip_quality else {}
            )
            q_dirs = {}
            for ds, qd in q_dirs_full.items():
                if has_valid_result(qd / "quality_results.json"):
                    print(f"[skip] quality exists: {qd / 'quality_results.json'}")
                else:
                    q_dirs[ds] = qd
            if hw_dir is None and not q_dirs:
                print(f"[skip] all done for exit_{k}")
                continue
            sweep_exit(
                base_model_id=HF_BASE_MODEL,
                exit_heads_id=heads_id,
                exit_layers=EXIT_LAYERS,
                force_exit=k,
                hw_out_dir=hw_dir,
                hw_dataset=HW_DATASET,
                quality_out_dirs=q_dirs,
                weight_source=ws,
                n_samples=N_SAMPLES,
                max_new_tokens=MAX_NEW_TOKENS,
                warmup_steps=WARMUP_STEPS,
                use_torch_compile=USE_TORCH_COMPILE,
                hw_quality_datasets=(not skip_hw),
            )
