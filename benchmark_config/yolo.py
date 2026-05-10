"""YOLO early-exit benchmark config + sweep runner."""

from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

NAME = "yolo"
MODEL_FAMILY = "yolov9-ee"

HF_USER  = "your-username"
HF_REPO  = f"{HF_USER}/yolov9-ee"

DATASETS         = ["coco", "voc"]
CONF_THRESHOLDS  = [0.25, 0.5, 0.75]
IOU_THRESHOLDS   = [0.45, 0.5, 0.65]
IMG_SIZES        = [640]
BENCH_BATCH      = 1
WARMUP_STEPS     = 3

DATA_DIR = REPO_ROOT / "AnyTimeYolo" / "datasets"
OUT_DIR  = REPO_ROOT / "logs" / "benchmark" / NAME

# =============================================================================

def run_all(only_dataset: Optional[str] = None):
    from AnyTimeYolo.benchmark import benchmark as _bench

    datasets = [only_dataset] if only_dataset else DATASETS

    for ds in datasets:
        for conf in CONF_THRESHOLDS:
            for iou in IOU_THRESHOLDS:
                for sz in IMG_SIZES:
                    run_dir = OUT_DIR / ds / f"conf{conf}_iou{iou}_sz{sz}"
                    _bench(
                        model_id=HF_REPO,
                        dataset=ds,
                        data_dir=DATA_DIR / ds,
                        conf_threshold=conf,
                        iou_threshold=iou,
                        img_size=sz,
                        out_dir=run_dir,
                        bench_batch=BENCH_BATCH,
                        warmup_steps=WARMUP_STEPS,
                    )
