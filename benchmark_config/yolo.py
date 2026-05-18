"""YOLOv9 gelan-s-ee per-sub-exit benchmark config + sweep runner.

ALL benchmark knobs live here. AnyTimeYolo/src/benchmark.py is pure functions.

HW sweep:      HW_DATASETS x weight_sources x exits (5) x sub_exits (3 scales).
Quality sweep: QUALITY_DATASETS x weight_sources x exits x sub_exits.

Cross-dataset generalization: labels must use COCO class IDs (0-79).
  - coco: native COCO 80-class labels.
  - voc:  20-class subset; download Ultralytics YOLO-format VOC and remap
          labels to COCO class IDs using VOC_COCO_CLASS_IDS below.
          Eval filters predictions to the 20 VOC-equivalent COCO classes only.
"""

import os
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared import load_env, auto_pull

load_env()

NAME = "yolo"
MODEL_FAMILY = "gelan-s-ee"

# ---- HuggingFace + paths ----------------------------------------------------
HF_USER = os.environ.get("HF_USER", "wicaksonolxn")
HF_TOKEN = os.environ.get("HF_TOKEN")

MODEL_ROOT = REPO_ROOT / "AnyTimeYolo"
EE_YAML = MODEL_ROOT / "src" / "early_exit" / "configs" / "gelan-s-ee.yaml"
CKPT_DIR = MODEL_ROOT / "ckpts"
DATA_DIR = MODEL_ROOT / "datasets"
OUT_DIR = REPO_ROOT / "logs" / "benchmark" / NAME

PRETRAINED_URL = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-s.pt"
PRETRAINED_FILE = "gelan-s.pt"


def hf_trained_repo(dataset: str) -> str:
    return f"{HF_USER}/gelan-s-{dataset.lower()}-ee"


def resolve_weights_path(dataset: str, weight_source: str) -> Path:
    if weight_source == "trained":
        repo_dir = auto_pull(hf_trained_repo(dataset), token=HF_TOKEN)
        for name in ("best.pt", "last.pt"):
            f = repo_dir / name
            if f.exists():
                return f
        pts = list(repo_dir.glob("*.pt"))
        if pts:
            return pts[0]
        raise FileNotFoundError(f"No .pt in {repo_dir}")
    if weight_source == "pretrained":
        dest = CKPT_DIR / PRETRAINED_FILE
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"[yolo] downloading {PRETRAINED_URL} -> {dest}")
            urllib.request.urlretrieve(PRETRAINED_URL, dest)
        return dest
    raise ValueError(f"weight_source invalid: {weight_source}")


# ---- Sweep datasets ----------------------------------------------------------
# HW timing: latency on diverse image distributions.
# Each entry needs DATA_DIR/<name>/val/ with images in YOLO format.
HW_DATASETS: List[str] = [
    "coco",   # 80-class, 5k val images — primary HW benchmark
    "voc",    # 20-class, ~5k test images — different scene distribution
]

# Quality / generalization: mAP per (exit, sub_exit).
# Labels MUST use COCO class IDs (0-79), not dataset-native IDs.
# For "voc": remap VOC labels to COCO class IDs before eval.
QUALITY_DATASETS: List[str] = [
    "coco",   # primary benchmark — all 80 COCO classes
    "voc",    # cross-dataset generalization — 20 COCO-equivalent classes
]

# COCO class IDs present in each quality dataset.
# None = evaluate all 80 classes. List = filter predictions + labels to these only.
# VOC 20 classes mapped to their COCO equivalent IDs (0-indexed):
#   person=0, bicycle=1, car=2, motorcycle=3, airplane=4, bus=5, train=6,
#   boat=8, bottle=39, chair=56, couch=57, potted plant=58, dining table=60,
#   tv=62, bird=14, cat=15, dog=16, horse=17, sheep=18, cow=19
DATASET_COCO_CLASS_IDS = {
    "coco": None,
    "voc": [0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62],
}

WEIGHT_SOURCES = ["pretrained"]  # add "trained" once per-dataset ckpts pushed
N_EXITS = 5
N_SUB_EXITS = 3
SUB_EXIT_NAMES = ["P3", "P4", "P5"]

# ---- Bench hparams ----------------------------------------------------------
IMG_SIZE = 640
BENCH_BATCH = 1
WARMUP_STEPS = 3
USE_TORCH_COMPILE = True
N_SAMPLES = 200

# =============================================================================


def run_all(
    only_dataset: Optional[str] = None,
    only_weight_source: Optional[str] = None,
    only_exit: Optional[int] = None,
    only_sub_exit: Optional[int] = None,
    skip_quality: bool = True,   # HW-only default
    skip_hw: bool = False,
):
    from AnyTimeYolo import profile_hw, evaluate_quality

    weight_sources = [only_weight_source] if only_weight_source else WEIGHT_SOURCES
    exits = [only_exit] if only_exit is not None else list(range(N_EXITS))
    sub_exits = [only_sub_exit] if only_sub_exit is not None else list(range(N_SUB_EXITS))

    # ---- HW pass ------------------------------------------------------------
    if not skip_hw:
        hw_datasets = [only_dataset] if only_dataset else HW_DATASETS
        for ds in hw_datasets:
            for ws in weight_sources:
                try:
                    weights = resolve_weights_path(ds, ws)
                except Exception as exc:
                    print(f"[yolo] weights not found for hw {ds}/{ws}: {exc}")
                    continue
                for ei in exits:
                    for s in sub_exits:
                        run_dir = OUT_DIR / ds / f"exit_{ei}_{SUB_EXIT_NAMES[s]}"
                        try:
                            profile_hw(
                                ee_yaml=EE_YAML,
                                weights_path=weights,
                                dataset=ds,
                                force_exit=ei,
                                data_dir=DATA_DIR / ds,
                                out_dir=run_dir,
                                sub_exit=s,
                                weight_source=ws,
                                img_size=IMG_SIZE,
                                bench_batch=BENCH_BATCH,
                                warmup_steps=WARMUP_STEPS,
                                use_torch_compile=USE_TORCH_COMPILE,
                                n_samples=N_SAMPLES,
                            )
                        except Exception as exc:
                            print(f"[yolo] hw failed {ds} exit={ei} sub={s}: {exc}")

    # ---- Quality pass -------------------------------------------------------
    if not skip_quality:
        quality_datasets = [only_dataset] if only_dataset else QUALITY_DATASETS
        for ds in quality_datasets:
            for ws in weight_sources:
                try:
                    weights = resolve_weights_path(ds, ws)
                except Exception as exc:
                    print(f"[yolo] weights not found for quality {ds}/{ws}: {exc}")
                    continue
                valid_cls = DATASET_COCO_CLASS_IDS.get(ds)
                for ei in exits:
                    for s in sub_exits:
                        run_dir = OUT_DIR / ds / f"exit_{ei}_{SUB_EXIT_NAMES[s]}"
                        try:
                            evaluate_quality(
                                ee_yaml=EE_YAML,
                                weights_path=weights,
                                dataset=ds,
                                force_exit=ei,
                                data_dir=DATA_DIR / ds,
                                out_dir=run_dir,
                                sub_exit=s,
                                weight_source=ws,
                                img_size=IMG_SIZE,
                                bench_batch=BENCH_BATCH,
                                valid_classes=valid_cls,
                            )
                        except Exception as exc:
                            print(f"[yolo] quality failed {ds} exit={ei} sub={s}: {exc}")
