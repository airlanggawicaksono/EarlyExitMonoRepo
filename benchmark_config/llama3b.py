"""LLaMa-3.2-3B pretrained multi-exit per-exit benchmark config + sweep.

Same early-exit as benchmark_config.llama but NO training: the base model +
its lm_head are broadcast to every transformer block (weight_source="pretrained"),
so each exit is just a truncated forward through k blocks. Nothing is fine-tuned.

HW-only by default (latency/energy/power/memory per exit) — quality of an
untrained early exit is meaningless, so skip_quality defaults True.
"""

import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared import load_env, has_valid_result

load_env()

NAME = "llama3b"
MODEL_FAMILY = "llama-3.2-3b"

HF_BASE_MODEL = "meta-llama/Llama-3.2-3B"
N_EXITS = 28        # per-layer (Llama-3.2-3B = 28 transformer blocks)

HW_DATASET = "cnn_dailymail"

# ---- Bench hparams (match llama.py so the two are comparable) ----------------
SEQ_LEN = 256
N_SAMPLES = 100
WARMUP_STEPS = 3
# ponytail: None = no wall-clock bound; _patch_duration in bench_jetson.py sets
# this when --duration is supplied (wall-clock stop after warmup).
DURATION_SEC = None  # type: Optional[float]
# 3B BF16 is ~6.4 GB of weights; on an 8 GB unified-memory board the inductor
# compile of 28 blocks pushes it into the OOM reaper. Eager only here.
USE_TORCH_COMPILE = False
DRY_SAMPLES = 10

# BENCH_SUBDIR overrides the subdir under logs/ for power-mode sweeps
# (e.g. "benchmark.15w"); default = plain "benchmark".
OUT_DIR = REPO_ROOT / "logs" / os.environ.get("BENCH_SUBDIR", "benchmark") / NAME

# =============================================================================


def run_all(
    only_exit: Optional[int] = None,
    skip_quality: bool = True,   # untrained exits -> quality meaningless; HW only
    skip_hw: bool = False,
    dry_run: bool = False,
    **_ignored,                  # accept the trained-config kwargs (only_mode, ...)
):
    """Pretrained 3B sweep: load base ONCE, truncate per exit (sweep_all_exits).
    A per-exit reload would stack 28 full 3B loads and OOM; one resident model
    via layer-view truncation keeps memory flat."""
    from AnyTimeLLaMa import sweep_all_exits

    n_samples = DRY_SAMPLES if dry_run else N_SAMPLES
    duration_sec = None if dry_run else DURATION_SEC
    out_root_base = (REPO_ROOT / "logs.dry_run" / "benchmark" / NAME) if dry_run else OUT_DIR
    exits = [only_exit] if only_exit is not None else list(range(N_EXITS))
    if dry_run and only_exit is None:
        exits = exits[:1]
    if dry_run:
        print(f"[llama3b] DRY RUN: {DRY_SAMPLES} samples -> {out_root_base}")

    def hw_factory(k):
        if skip_hw:
            return None
        d = out_root_base / HW_DATASET / "pretrained" / f"exit_{k}"
        if has_valid_result(d / "hw_results.json"):
            print(f"[skip] hw exists: {d / 'hw_results.json'}")
            return None
        return d

    try:
        sweep_all_exits(
            base_model_id=HF_BASE_MODEL,
            exit_heads_id=None,
            exit_layers=[],
            exits=list(exits),
            hw_out_dir_factory=hw_factory,
            hw_dataset=HW_DATASET,
            quality_out_dir_factories={},   # skip_quality: HW only
            weight_source="pretrained",
            n_samples=n_samples,
            warmup_steps=WARMUP_STEPS,
            use_torch_compile=USE_TORCH_COMPILE,
            hw_quality_datasets=False,
            duration_sec=duration_sec,
        )
    except Exception as exc:
        print(f"[llama3b] pretrained sweep failed: {exc}")
