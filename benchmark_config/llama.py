"""LLaMa-3.2-1B multi-exit (self-distill) per-exit benchmark config + sweep.

Pulls trained MultiExitLM from HF repos pushed by the self-distill trainer:
    {HF_USER}/selfdistill-llama-c4-{mode}

HW pass: cnn_dailymail prompts (latency only depends on tokenisation).
Quality pass: per-dataset perplexity (generation tasks) or MCQ accuracy
(only via legacy path — trained MultiExitLM here only emits perplexity).
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


def hf_trained_repo(dataset_tag: str, mode: str) -> str:
    """Matches train_colab item.label = f'llama-c4-{mode}'."""
    return f"{HF_USER}/selfdistill-llama-{dataset_tag}-{mode}"


# ---- Arch / sweep -----------------------------------------------------------
DATASET_TAG = "c4"  # what train_colab uses in the item.label slug
N_EXITS = 16        # per-layer (Llama-3.2-1B = 16 transformer blocks)
MODES = ["pairwise", "segd"]
WEIGHT_SOURCES = ["trained"]

# Quality datasets — metric picked per dataset by evaluate_quality_trained:
#   MCQ sets        -> acc_norm (log-likelihood over choices, length-normalized)
#   summarization   -> ROUGE-L F1 (greedy generation vs reference highlights)
#   math            -> exact-match accuracy (final number, GSM8K convention)
HW_DATASET = "cnn_dailymail"
QUALITY_DATASETS = [
    "cnn_dailymail",   # generation — ROUGE-L F1
    "gsm8k",           # generation — exact-match accuracy
    "arc_challenge",   # mcq        — acc_norm (science reasoning)
    "hellaswag",       # mcq        — acc_norm (commonsense completion)
    "mmlu",            # mcq        — acc_norm (broad knowledge, 57 subjects)
]

# ---- Bench hparams ----------------------------------------------------------
SEQ_LEN = 256
N_SAMPLES = 100
WARMUP_STEPS = 3
USE_TORCH_COMPILE = True

OUT_DIR = REPO_ROOT / "logs" / "benchmark" / NAME

# =============================================================================


DRY_SAMPLES = 10  # dry-run sample count (smoke test)


def run_all(
    only_mode: Optional[str] = None,
    only_weight_source: Optional[str] = None,
    only_exit: Optional[int] = None,
    only_dataset: Optional[str] = None,
    skip_quality: bool = True,
    skip_hw: bool = False,
    dry_run: bool = False,
):
    from AnyTimeLLaMa import sweep_hw_trained, evaluate_quality_trained

    n_samples = DRY_SAMPLES if dry_run else N_SAMPLES
    out_root_base = REPO_ROOT / "logs.dry_run" / "benchmark" / NAME if dry_run else OUT_DIR
    modes = [only_mode] if only_mode else MODES
    weight_sources = [only_weight_source] if only_weight_source else WEIGHT_SOURCES
    exits = [only_exit] if only_exit is not None else list(range(N_EXITS))
    quality_datasets = (
        [only_dataset] if only_dataset
        else (QUALITY_DATASETS[:1] if dry_run else QUALITY_DATASETS)
    )
    if dry_run:
        print(f"[llama] DRY RUN: {DRY_SAMPLES} samples -> {out_root_base} | modes={modes}")

    for ws in weight_sources:
        if ws == "pretrained":
            _bench_llama_pretrained(exits, n_samples, quality_datasets,
                                    out_root_base, skip_hw, skip_quality)
            continue
        if ws != "trained":
            print(f"[llama] weight_source={ws} not implemented; skipping")
            continue
        for mode in modes:
            repo_id = hf_trained_repo(DATASET_TAG, mode)

            # HW pass on HW_DATASET only
            if not skip_hw:
                try:
                    sweep_hw_trained(
                        repo_id=repo_id,
                        dataset=HW_DATASET,
                        mode=mode,
                        exits=exits,
                        n_exits=N_EXITS,
                        out_root=out_root_base / HW_DATASET / mode,
                        base_model_id=HF_BASE_MODEL,
                        weight_source=ws,
                        seq_len=SEQ_LEN,
                        n_samples=n_samples,
                        warmup_steps=WARMUP_STEPS,
                        use_torch_compile=USE_TORCH_COMPILE,
                    )
                except Exception as exc:
                    print(f"[llama] hw sweep failed {mode}/{ws}: {exc}")

            # Quality pass per dataset
            if not skip_quality:
                for ds in quality_datasets:
                    for k in exits:
                        run_dir = out_root_base / ds / mode / f"exit_{k}"
                        q_path = run_dir / "quality_results.json"
                        if has_valid_result(q_path):
                            print(f"[skip] quality exists: {q_path}")
                            continue
                        try:
                            evaluate_quality_trained(
                                repo_id=repo_id,
                                dataset=ds,
                                mode=mode,
                                force_exit=k,
                                n_exits=N_EXITS,
                                out_dir=run_dir,
                                base_model_id=HF_BASE_MODEL,
                                weight_source=ws,
                                seq_len=SEQ_LEN,
                                n_samples=n_samples,
                            )
                        except Exception as exc:
                            print(f"[llama] quality failed {ds}/{mode} exit={k}: {exc}")


def _llama_pretrained_qdirs(force_exit, quality_datasets, out_root_base):
    """Per-dataset pretrained quality dirs for one exit, skipping finished ones."""
    q = {}
    for ds in quality_datasets:
        run_dir = out_root_base / ds / "pretrained" / f"exit_{force_exit}"
        if has_valid_result(run_dir / "quality_results.json"):
            print(f"[skip] quality exists: {run_dir / 'quality_results.json'}")
            continue
        q[ds] = run_dir
    return q


def _bench_llama_pretrained(exits, n_samples, quality_datasets, out_root_base, skip_hw, skip_quality):
    """Pretrained LLaMA (base + base.lm_head broadcast to every exit).

    Loads the base model ONCE and truncates per exit (sweep_all_exits). A per-exit
    reload (the old sweep_exit loop) OOMs an 8GB Jetson: the harness runs all exits
    in one process (fresh process per *backend*, not per exit), so 16 full 1B
    reloads stack up -> NvMap ENOMEM (error 12). sweep_all_exits reuses one
    resident model via a layer-view truncation, so memory stays flat."""
    from AnyTimeLLaMa import sweep_all_exits

    def hw_factory(k):
        if skip_hw:
            return None
        d = out_root_base / HW_DATASET / "pretrained" / f"exit_{k}"
        if has_valid_result(d / "hw_results.json"):
            print(f"[skip] hw exists: {d / 'hw_results.json'}")
            return None
        return d

    def _q_factory(ds, k):
        d = out_root_base / ds / "pretrained" / f"exit_{k}"
        if has_valid_result(d / "quality_results.json"):
            print(f"[skip] quality exists: {d / 'quality_results.json'}")
            return None
        return d

    q_factories = {} if skip_quality else {
        ds: (lambda k, _ds=ds: _q_factory(_ds, k)) for ds in quality_datasets
    }
    try:
        sweep_all_exits(
            base_model_id=HF_BASE_MODEL,
            exit_heads_id=None,
            exit_layers=[],
            exits=list(exits),
            hw_out_dir_factory=hw_factory,
            hw_dataset=HW_DATASET,
            quality_out_dir_factories=q_factories,
            weight_source="pretrained",
            n_samples=n_samples,
            warmup_steps=WARMUP_STEPS,
            use_torch_compile=USE_TORCH_COMPILE,
        )
    except Exception as exc:
        print(f"[llama] pretrained sweep failed: {exc}")
