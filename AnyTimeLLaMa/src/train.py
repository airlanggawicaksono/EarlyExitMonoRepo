"""LLaMa early-exit training function. Wraps finetune_ee.py.

Public API:
    train(base_model=None, **overrides) -> Path
        Returns checkpoint dir. Auto-pushes to HF if HF_AUTO_PUSH=True.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

_HERE  = Path(__file__).resolve().parent     # AnyTimeLLaMa/src/
_MODEL = _HERE.parent
_REPO  = _MODEL.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_MODEL))

import config as C  # type: ignore
from shared import BgHwPoller


def _model_short(model_name: str) -> str:
    return model_name.split("/")[-1].lower().replace(".", "")


def _write_config(
    out_path: Path,
    base_model: str,
    exit_layers: List[int],
    exit_weights: List[float],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    seq_len: int,
    max_train_samples: int,
    max_val_samples: int,
    hf_repo: Optional[str],
) -> None:
    """Write key=value config consumed by finetune_ee.py."""
    use_local_c4 = (
        C.USE_LOCAL_C4_CACHE
        and (C.C4_CACHE / "c4_train.jsonl").exists()
        and (C.C4_CACHE / "c4_validation.jsonl").exists()
    )

    if use_local_c4:
        dataset_block = (
            f"train_file = {C.C4_CACHE / 'c4_train.jsonl'}\n"
            f"validation_file = {C.C4_CACHE / 'c4_validation.jsonl'}\n"
            f"text_column = text"
        )
    else:
        dataset_block = (
            "dataset_name = allenai/c4\ndataset_config_name = en\ntext_column = text"
        )

    exit_layers_str = "[" + ", ".join(str(x) for x in exit_layers) + "]"
    exit_weights_str = "[" + ", ".join(str(x) for x in exit_weights) + "]"

    body = f"""model_name_or_path = {base_model}
{dataset_block}
output_dir = {output_dir}

max_train_samples = {max_train_samples}
max_val_samples = {max_val_samples}

max_seq_length = {seq_len}
per_device_train_batch_size = {batch_size}
per_device_eval_batch_size = {batch_size}
gradient_accumulation_steps = {grad_accum}
learning_rate = {lr}
num_train_epochs = {epochs}

logging_steps = {C.LOGGING_STEPS}
save_steps = {C.SAVE_STEPS}
eval_steps = {C.EVAL_STEPS}
seed = 42

torch_dtype = {C.TORCH_DTYPE}
padding_side = right
report_to = ["none"]
overwrite_output_dir = true

exit_layer_indices = {exit_layers_str}
exit_loss_weights = {exit_weights_str}
init_exit_from_base = true
exit_confidence_threshold = {C.CONFIDENCE_THRESHOLD}
"""
    if hf_repo:
        body += f"hub_exit_heads_repo = {hf_repo}\n"

    out_path.write_text(body, encoding="utf-8")


def train(
    base_model: Optional[str] = None,
    *,
    exit_layers: Optional[List[int]] = None,
    exit_weights: Optional[List[float]] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    grad_accum: Optional[int] = None,
    lr: Optional[float] = None,
    seq_len: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    push_to_hub: Optional[bool] = None,
    hf_repo: Optional[str] = None,
    skip_if_exists: bool = False,
) -> Path:
    """Train EE heads. Returns local checkpoint dir."""
    bm = base_model or C.BASE_MODEL
    short = _model_short(bm)
    out_dir = C.CKPT_DIR / f"{short}-ee"
    repo = hf_repo or C.hf_repo_for(short)

    if skip_if_exists:
        try:
            from huggingface_hub import list_repo_files

            files = list_repo_files(repo, token=C.HF_TOKEN)
            if any(f.endswith(".safetensors") or f.endswith(".pt") for f in files):
                print(f"[train] HF checkpoint exists: {repo}, skip")
                return out_dir
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "ee_config"
    _write_config(
        out_path=config_path,
        base_model=bm,
        exit_layers=exit_layers or C.EXIT_LAYERS,
        exit_weights=exit_weights or C.EXIT_WEIGHTS,
        output_dir=out_dir,
        epochs=epochs or C.NUM_EPOCHS,
        batch_size=batch_size or C.TRAIN_BATCH,
        grad_accum=grad_accum or C.GRAD_ACCUM,
        lr=lr or C.LR,
        seq_len=seq_len or C.SEQ_LEN,
        max_train_samples=max_train_samples or C.MAX_TRAIN_SAMPLES,
        max_val_samples=max_val_samples or C.MAX_VAL_SAMPLES,
        hf_repo=repo
        if (push_to_hub if push_to_hub is not None else C.HF_AUTO_PUSH)
        else None,
    )

    cmd = [sys.executable, "finetune_ee.py", "--config", str(config_path)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = C.GPU_ID
    env["PYTHONPATH"] = f"{C.REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"

    # Background HW poller in case finetune_ee internal callback misses something
    hw_log = out_dir / "train_metrics_hw.json"
    with BgHwPoller(hw_log, interval_sec=2.0):
        subprocess.run(cmd, cwd=_HERE, env=env, check=True)

    # finetune_ee.py auto-pushes to HF when hub_exit_heads_repo set in config.
    return out_dir
