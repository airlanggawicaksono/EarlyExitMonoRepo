"""YOLOv9 + early-exit training. Wraps model/yolov9/train.py.

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

import config as C  # type: ignore
from shared import auto_push, BgHwPoller


def train(
    dataset: str = "coco",
    *,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    img_size: Optional[int] = None,
    weights: Optional[str] = None,
    push_to_hub: Optional[bool] = None,
    hf_repo: Optional[str] = None,
    skip_if_exists: bool = False,
) -> Path:
    """Train YOLOv9 with EE heads on dataset. Returns local checkpoint dir."""
    out_name = f"yolov9-{dataset}-ee"
    out_dir = C.CKPT_DIR / out_name
    repo = hf_repo or C.hf_repo_for(dataset)

    if skip_if_exists:
        try:
            from huggingface_hub import list_repo_files

            files = list_repo_files(repo, token=C.HF_TOKEN)
            if any(f.endswith(".pt") for f in files):
                print(f"[train] HF checkpoint exists: {repo}, skip")
                return out_dir
        except Exception:
            pass

    data_yaml = C.DATA_DIR / dataset / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} missing. Run prepare_data.py first.")

    cmd = [
        sys.executable,
        "train.py",
        "--data",
        str(data_yaml),
        "--weights",
        weights or C.WEIGHTS_BASE,
        "--img",
        str(img_size or C.IMG_SIZE),
        "--batch",
        str(batch_size or C.TRAIN_BATCH),
        "--epochs",
        str(epochs or C.EPOCHS),
        "--device",
        C.GPU_ID,
        "--project",
        str(C.CKPT_DIR),
        "--name",
        out_name,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = C.GPU_ID
    env["PYTHONPATH"] = f"{C.REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"

    hw_log = out_dir / "train_metrics_hw.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with BgHwPoller(hw_log, interval_sec=2.0):
        subprocess.run(cmd, cwd=C.YOLO_REF, env=env, check=True)

    do_push = C.HF_AUTO_PUSH if push_to_hub is None else push_to_hub
    if do_push:
        try:
            auto_push(
                local_path=out_dir / "weights",
                repo_id=repo,
                commit_msg=f"AnyTimeYolo: {dataset} train",
                private=C.HF_PRIVATE,
                token=C.HF_TOKEN,
            )
        except Exception as e:
            print(f"[train] HF push failed: {e}")

    return out_dir


def train_all():
    for ds in C.DATASETS:
        train(ds)
