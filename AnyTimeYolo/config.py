"""AnyTimeYolo TRAINING config. Loads .env on import.

Benchmark settings live in benchmark_config/yolo.py (at repo root). This file
is for training only.
"""

from pathlib import Path
import os
import sys

ROOT = Path(__file__).parent.resolve()
REPO_ROOT = ROOT.parent

sys.path.insert(0, str(REPO_ROOT))
from shared import load_env

load_env()

# ----- paths ------------------------------------------------------------------
DATA_DIR = ROOT / "datasets"
CKPT_DIR = ROOT / "ckpts"
LOG_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"
YOLO_REF = ROOT / "model" / "yolov9"

# ----- model arch (training) --------------------------------------------------
ARCH = "gelan-m-ee"
EE_YAML = ROOT / "src" / "early_exit" / "configs" / "gelan-m-ee.yaml"
PRETRAINED_WEIGHTS = "gelan-m.pt"
PRETRAINED_URL = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-m.pt"
IMG_SIZE = 640

# ----- HuggingFace push -------------------------------------------------------
HF_USER = os.environ.get("HF_USER", "wicaksonolxn")
HF_TOKEN = os.environ.get("HF_TOKEN")
RF_API_KEY = os.environ.get("RF_API_KEY")
HF_AUTO_PUSH = True
HF_PRIVATE = True


def hf_repo_for(dataset: str) -> str:
    return f"{HF_USER}/gelan-m-{dataset.lower()}-ee"


# ----- Roboflow projects (workspace, project, version) ------------------------
# COCO: Universe microsoft/coco v17 IS the yolov9-c-640 (-gelan-) export — YOLO
# txt labels + data.yaml, 123k imgs. Exactly matches gelan-m's format.
ROBOFLOW_PROJECTS = {
    "coco": ("microsoft", "coco", 17),           # universe.roboflow.com/microsoft/coco/dataset/17
    "voc":  ("public", "pascal-voc-2012", 13),   # placeholder coords — verify before enabling voc
}

# ----- training hparams -------------------------------------------------------
DATASETS = ["coco", "voc"]
EPOCHS = 100
TRAIN_BATCH = 16
EVAL_BATCH = 16
LR0 = 0.01
LRF = 0.01

# ----- runtime ----------------------------------------------------------------
GPU_ID = "0"
