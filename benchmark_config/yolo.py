"""YOLO early-exit benchmark config + sweep runner."""

import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared import load_env

load_env()

NAME = "yolo"
MODEL_FAMILY = "yolov9-ee"

HF_USER = os.environ.get("HF_USER", "wicaksonolxn")
HF_REPO = f"{HF_USER}/yolov9-ee"

DATASETS = ["coco"]
CONF_THRESHOLDS = [0.25, 0.5, 0.75]
IOU_THRESHOLDS = [0.45, 0.5, 0.65]
IMG_SIZES = [640]
BENCH_BATCH = 1
WARMUP_STEPS = 3

DATA_DIR = REPO_ROOT / "AnyTimeYolo" / "datasets"
OUT_DIR = REPO_ROOT / "logs" / "benchmark" / NAME

# =============================================================================


def run_all(
    only_dataset: Optional[str] = None,
    skip_quality: bool = False,
    skip_hw: bool = False,
):
    from AnyTimeYolo import profile_hw, evaluate_quality

    datasets = [only_dataset] if only_dataset else DATASETS

    for ds in datasets:
        for conf in CONF_THRESHOLDS:
            for iou in IOU_THRESHOLDS:
                for sz in IMG_SIZES:
                    run_dir = OUT_DIR / ds / f"conf{conf}_iou{iou}_sz{sz}"
                    kw = dict(
                        model_id=HF_REPO,
                        dataset=ds,
                        out_dir=run_dir,
                        data_dir=DATA_DIR / ds,
                        conf_threshold=conf,
                        iou_threshold=iou,
                        img_size=sz,
                        bench_batch=BENCH_BATCH,
                        warmup_steps=WARMUP_STEPS,
                    )
                    if not skip_hw:
                        profile_hw(**kw)
                    if not skip_quality:
                        evaluate_quality(
                            model_id=HF_REPO,
                            dataset=ds,
                            out_dir=run_dir,
                            data_dir=DATA_DIR / ds,
                            img_size=sz,
                            bench_batch=32,
                        )
