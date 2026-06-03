"""ElasticBERT per-exit benchmark config + sweep runner.

ALL benchmark knobs live here. AnyTimeBert/src/benchmark.py is pure functions.

Sweep: tasks x modes x exits (0..N_EXITS-1).

Repo naming on HF (pushed by GeneralizableSelfDistilationEarlyExitTraining.sync):
    {HF_USER}/selfdistill-bert-{task.lower()}-{mode}

Output layout (per-mode dir added):
    logs/benchmark/bert/{task}/{mode}/exit_{k}/{hw_results.json, quality_results.json}
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
MODEL_FAMILY = "elasticbert-large"

# ---- HuggingFace ------------------------------------------------------------
HF_USER = os.environ.get("HF_USER", "wicaksonolxn")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_PRETRAINED_MODEL = "OpenMOSS-Team/elasticbert-large"


def hf_trained_repo(task: str, mode: str) -> str:
    """Matches GeneralizableSelfDistilationEarlyExitTraining.runner push naming:
    repo_id = f"{HF_USER}/{repo_prefix}-{item.label}"
    where repo_prefix = "selfdistill" and item.label = f"bert-{task.lower()}-{mode}"."""
    return f"{HF_USER}/selfdistill-bert-{task.lower()}-{mode}"


def resolve_model_id(task: str, weight_source: str, mode: Optional[str] = None) -> str:
    if weight_source == "trained":
        if mode is None:
            raise ValueError("mode required for weight_source='trained'")
        return hf_trained_repo(task, mode)
    if weight_source == "pretrained":
        return HF_PRETRAINED_MODEL
    raise ValueError(f"weight_source must be in {WEIGHT_SOURCES}, got {weight_source}")


# ---- Sweep ------------------------------------------------------------------
TASKS = ["SST-2", "MRPC", "QNLI", "RTE", "CoLA", "MNLI", "QQP"]
MODES = ["joint", "pairwise", "cascade"]
WEIGHT_SOURCES = ["trained"]  # default: benchmark trained ckpts pushed by trainer
N_EXITS = 24  # ElasticBERT-large = 24 layers -> 24 exits

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
    only_mode: Optional[str] = None,
    only_weight_source: Optional[str] = None,
    only_exit: Optional[int] = None,
    skip_quality: bool = True,   # HW-only default
    skip_hw: bool = False,
    dry_run: bool = False,
):
    from AnyTimeBert import (
        evaluate_quality, evaluate_quality_trained,
        prepare_task, sweep_hw, sweep_hw_trained,
    )

    max_samples = 5 if dry_run else None
    out_root_base = REPO_ROOT / "logs.dry_run" / "benchmark" / NAME if dry_run else OUT_DIR
    tasks = [only_task] if only_task else (TASKS[:1] if dry_run else TASKS)
    modes = [only_mode] if only_mode else MODES
    weight_sources = [only_weight_source] if only_weight_source else WEIGHT_SOURCES
    exits = [only_exit] if only_exit is not None else list(range(N_EXITS))
    if dry_run:
        print(f"[bert] DRY RUN: 5 samples per (task, mode, exit) -> {out_root_base} | tasks={tasks} modes={modes}")

    for task in tasks:
        prepare_task(task, out_root=DATA_DIR)
        for ws in weight_sources:
            for mode in modes:
                model_id = resolve_model_id(task, ws, mode=mode if ws == "trained" else None)
                out_root = out_root_base / task / mode

                # HW pass
                if not skip_hw:
                    if ws == "trained":
                        sweep_hw_trained(
                            repo_id=model_id,
                            task=task,
                            mode=mode,
                            exits=exits,
                            n_exits=N_EXITS,
                            data_dir=DATA_DIR / task,
                            out_root=out_root,
                            weight_source=ws,
                            max_seq_length=MAX_SEQ_LENGTH,
                            warmup_steps=WARMUP_STEPS,
                            use_torch_compile=USE_TORCH_COMPILE,
                            max_samples=max_samples,
                            bench_batch=BENCH_BATCH,
                        )
                    else:
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
                            bench_batch=BENCH_BATCH,
                        )

                # Quality pass
                if not skip_quality:
                    for k in exits:
                        run_dir = out_root / f"exit_{k}"
                        q_path = run_dir / "quality_results.json"
                        if has_valid_result(q_path):
                            print(f"[skip] quality exists: {q_path}")
                            continue
                        if ws == "trained":
                            evaluate_quality_trained(
                                repo_id=model_id,
                                task=task,
                                mode=mode,
                                force_exit=k,
                                n_exits=N_EXITS,
                                data_dir=DATA_DIR / task,
                                out_dir=run_dir,
                                weight_source=ws,
                                max_seq_length=MAX_SEQ_LENGTH,
                                max_samples=max_samples,
                            )
                        else:
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
