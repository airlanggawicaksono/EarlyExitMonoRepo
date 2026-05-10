"""MSDNet training function. Wraps reference/main.py via subprocess.

Public API:
    train(dataset, **overrides) -> Path
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import config as C   # type: ignore
from shared import auto_push, BgHwPoller


def train(
    dataset: str,
    *,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    n_blocks: Optional[int] = None,
    push_to_hub: Optional[bool] = None,
    hf_repo: Optional[str] = None,
    skip_if_exists: bool = False,
) -> Path:
    """Train MSDNet on one dataset. Returns local checkpoint dir."""
    out_dir = C.CKPT_DIR / f"msdnet-{dataset}"

    if skip_if_exists:
        repo = hf_repo or C.hf_repo_for(dataset)
        try:
            from huggingface_hub import list_repo_files
            files = list_repo_files(repo, token=C.HF_TOKEN)
            if any(f.endswith(".pth.tar") or f.endswith(".pt") for f in files):
                print(f"[train] HF checkpoint exists: {repo}, skip")
                return out_dir
        except Exception:
            pass

    # ImageNet config differs from CIFAR
    is_imagenet = dataset.lower() == "imagenet"
    n_chan   = 32 if is_imagenet else C.N_CHANNELS
    growth   = 16 if is_imagenet else C.GROWTH_RATE
    grf      = "1-2-4-4" if is_imagenet else C.GR_FACTOR
    bnf      = "1-2-4-4" if is_imagenet else C.BN_FACTOR
    step     = 4 if is_imagenet else C.STEP

    cmd = [
        sys.executable, "main.py",
        "--data-root", str(C.DATA_DIR / dataset),
        "--data",      dataset,
        "--save",      str(out_dir),
        "--arch",      C.ARCH,
        "--batch-size", str(batch_size or C.TRAIN_BATCH),
        "--epochs",    str(epochs or C.EPOCHS),
        "--nBlocks",   str(n_blocks or C.N_BLOCKS),
        "--stepmode",  C.STEP_MODE,
        "--step",      str(step),
        "--base",      str(C.BASE),
        "--nChannels", str(n_chan),
        "--growthRate", str(growth),
        "--grFactor",  grf,
        "--bnFactor",  bnf,
        "--lr",        str(lr or C.LR),
        "--lr-type",   C.LR_TYPE,
        "--momentum",  str(C.MOMENTUM),
        "--weight-decay", str(C.WEIGHT_DECAY),
        "-j",          str(C.WORKERS),
        "--use-valid",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = C.GPU_ID
    env["PYTHONPATH"] = f"{C.REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"

    # HW poller runs as background thread, samples every 2s while subprocess trains
    hw_log = out_dir / "train_metrics_hw.json"
    with BgHwPoller(hw_log, interval_sec=2.0):
        subprocess.run(cmd, cwd=C.REF, env=env, check=True)

    do_push = C.HF_AUTO_PUSH if push_to_hub is None else push_to_hub
    if do_push:
        repo = hf_repo or C.hf_repo_for(dataset)
        try:
            auto_push(local_path=out_dir, repo_id=repo,
                      commit_msg=f"AnyTimeVisionenc: MSDNet {dataset}",
                      private=C.HF_PRIVATE,
                      token=C.HF_TOKEN)
        except Exception as e:
            print(f"[train] HF push failed for {dataset}: {e}")

    return out_dir


def train_all() -> None:
    """Train every dataset in C.DATASETS."""
    for ds in C.DATASETS:
        train(ds)
