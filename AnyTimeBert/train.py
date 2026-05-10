"""ElasticBERT training function. Called by Colab notebook or local CLI.

Public API:
    train(task, **overrides) -> Path
        Returns checkpoint dir. Auto-pushes to HF if HF_AUTO_PUSH=True.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import config as C
from shared import auto_push


def train(
    task: str,
    *,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    push_to_hub: Optional[bool] = None,
    hf_repo: Optional[str] = None,
    skip_if_exists: bool = False,
) -> Path:
    """Fine-tune ElasticBERT on one GLUE task. Returns local checkpoint dir.

    skip_if_exists=True : check HF first; skip training if checkpoint already pushed.
    """
    out_dir = C.CKPT_DIR / "elasticbert-base/glue" / task

    if skip_if_exists:
        repo = hf_repo or C.hf_repo_for(task)
        try:
            from huggingface_hub import list_repo_files
            files = list_repo_files(repo, token=C.HF_TOKEN)
            if any(f.endswith(".bin") or f.endswith(".safetensors") for f in files):
                print(f"[train] HF checkpoint already exists: {repo}, skipping training")
                return out_dir
        except Exception:
            pass   # repo missing or no access — train

    cmd = [
        sys.executable, "./run_glue.py",
        "--model_name_or_path", C.HF_MODEL_NAME,
        "--task_name", task,
        "--do_train", "--do_eval", "--do_lower_case",
        "--data_dir",   str(C.GLUE_DIR / task),
        "--log_dir",    str(C.LOG_DIR  / "elasticbert-base/glue" / task),
        "--output_dir", str(out_dir),
        "--num_hidden_layers", str(C.NUM_HIDDEN_LAYERS),
        "--num_output_layers", str(C.NUM_OUTPUT_LAYERS),
        "--max_seq_length",    str(C.MAX_SEQ_LENGTH),
        "--per_gpu_train_batch_size", str(batch_size or C.TRAIN_BATCH),
        "--per_gpu_eval_batch_size",  str(C.EVAL_BATCH),
        "--gradient_accumulation_steps", str(C.GRAD_ACCUM),
        "--learning_rate", str(lr or C.LR),
        "--weight_decay",  str(C.WEIGHT_DECAY),
        "--logging_steps",    str(C.LOGGING_STEPS),
        "--early_stop_steps", str(C.EARLY_STOP),
        "--num_train_epochs", str(epochs or C.NUM_EPOCHS),
        "--warmup_rate",      str(C.WARMUP_RATE),
        "--evaluate_during_training",
        "--overwrite_output_dir",
    ]
    if C.USE_FP16:
        cmd.append("--fp16")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = C.GPU_ID
    env["PYTHONPATH"] = f"{C.REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, cwd=C.REF_STATIC, env=env, check=True)

    do_push = C.HF_AUTO_PUSH if push_to_hub is None else push_to_hub
    if do_push:
        repo = hf_repo or C.hf_repo_for(task)
        try:
            auto_push(local_path=out_dir, repo_id=repo,
                      commit_msg=f"AnyTimeBert: {task} fine-tune",
                      private=C.HF_PRIVATE,
                      token=C.HF_TOKEN)
        except Exception as e:
            print(f"[train] HF push failed for {task}: {e}")

    return out_dir


def train_all() -> None:
    """Convenience: train every task in C.TASKS."""
    for task in C.TASKS:
        train(task)
