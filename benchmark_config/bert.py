"""ElasticBERT per-exit benchmark config + sweep runner.

ALL benchmark knobs live here. AnyTimeBert/src/benchmark.py is pure functions.

Sweep: tasks x weight_sources x all exits (0..N_EXITS-1).
"""

import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared import load_env, has_valid_result

load_env()

NAME = "bert"
MODEL_FAMILY = "elasticbert-base"

# ---- HuggingFace ------------------------------------------------------------
HF_USER = os.environ.get("HF_USER", "wicaksonolxn")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_PRETRAINED_MODEL = "OpenMOSS-Team/elasticbert-base"


def hf_trained_repo(task: str) -> str:
    return f"{HF_USER}/elasticbert-base-{task.lower()}-ee"


def resolve_model_id(task: str, weight_source: str) -> str:
    if weight_source == "trained":
        return hf_trained_repo(task)
    if weight_source == "pretrained":
        return HF_PRETRAINED_MODEL
    raise ValueError(f"weight_source must be in {WEIGHT_SOURCES}, got {weight_source}")


# ---- Sweep ------------------------------------------------------------------
TASKS = ["SST-2", "MRPC", "QNLI", "RTE", "CoLA", "MNLI", "QQP"]
WEIGHT_SOURCES = ["pretrained"]  # HW-only sweep; add "trained" once ckpts pushed
N_EXITS = 12  # ElasticBERT-base = 12 layers -> 12 exits

# ---- Bench hparams ----------------------------------------------------------
MAX_SEQ_LENGTH = 128
BENCH_BATCH = 1
WARMUP_STEPS = 3
USE_TORCH_COMPILE = True

DATA_DIR = REPO_ROOT / "AnyTimeBert" / "glue_data"
OUT_DIR = REPO_ROOT / "logs" / "benchmark" / NAME

# =============================================================================


def run_all(
    only_task: Optional[str] = None,
    only_weight_source: Optional[str] = None,
    only_exit: Optional[int] = None,
    skip_quality: bool = True,   # HW-only default
    skip_hw: bool = False,
    dry_run: bool = False,
):
    from AnyTimeBert import evaluate_quality, prepare_task, sweep_hw

    max_samples = 5 if dry_run else None
    out_root_base = REPO_ROOT / "logs.dry_run" / "benchmark" / NAME if dry_run else OUT_DIR
    tasks = [only_task] if only_task else (TASKS[:1] if dry_run else TASKS)
    weight_sources = [only_weight_source] if only_weight_source else WEIGHT_SOURCES
    exits = [only_exit] if only_exit is not None else list(range(N_EXITS))
    if dry_run:
        print(f"[bert] DRY RUN: 5 samples per (task, exit) -> {out_root_base} | tasks={tasks}")

    for task in tasks:
        prepare_task(task, out_root=DATA_DIR)
        for ws in weight_sources:
            model_id = resolve_model_id(task, ws)
            out_root = out_root_base / task

            # HW pass: one model load + per-layer compile shared across all k.
            if not skip_hw:
                sweep_hw(
                    model_id=model_id,
                    task=task,
                    exits=exits,
                    data_dir=DATA_DIR / task,
                    out_root=out_root,
                    weight_source=ws,
                    max_seq_length=MAX_SEQ_LENGTH,
                    warmup_steps=WARMUP_STEPS,
                    use_torch_compile=USE_TORCH_COMPILE,
                    max_samples=max_samples,
                )

            # Quality pass: separate (no compile, model loaded fresh per call).
            if not skip_quality:
                for k in exits:
                    run_dir = out_root / f"exit_{k}"
                    q_path = run_dir / "quality_results.json"
                    if has_valid_result(q_path):
                        print(f"[skip] quality exists: {q_path}")
                        continue
                    evaluate_quality(
                        model_id=model_id,
                        task=task,
                        force_exit=k,
                        data_dir=DATA_DIR / task,
                        out_dir=run_dir,
                        weight_source=ws,
                        max_seq_length=MAX_SEQ_LENGTH,
                        max_samples=max_samples,
                    )
