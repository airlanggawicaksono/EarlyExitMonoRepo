"""Central config for AnyTimeYolo. Loads .env on import."""

from pathlib import Path
import os, sys

ROOT      = Path(__file__).parent.resolve()
REPO_ROOT = ROOT.parent

sys.path.insert(0, str(REPO_ROOT))
from shared import load_env
load_env()

# ----- paths ------------------------------------------------------------------
DATA_DIR    = ROOT / "datasets"
CKPT_DIR    = ROOT / "ckpts"
LOG_DIR     = ROOT / "logs"
RESULTS_DIR = ROOT / "results"
YOLO_REF    = ROOT / "model" / "yolov9"

# ----- model ------------------------------------------------------------------
WEIGHTS_BASE = "yolov9-c.pt"
IMG_SIZE     = 640

# ----- HuggingFace I/O --------------------------------------------------------
HF_USER       = os.environ.get("HF_USER", "wicaksonolxn")
HF_TOKEN      = os.environ.get("HF_TOKEN")
RF_API_KEY    = os.environ.get("RF_API_KEY")
HF_AUTO_PUSH  = True
HF_PRIVATE    = True
def hf_repo_for(dataset: str) -> str:
    return f"{HF_USER}/yolov9-{dataset.lower()}-ee"

# ----- Roboflow projects ------------------------------------------------------
ROBOFLOW_WORKSPACE = "your-workspace"   # edit
ROBOFLOW_PROJECTS  = {
    "coco": ("microsoft", "coco", 1),
    # "voc":  ("workspace", "project", version),
}

# ----- training ---------------------------------------------------------------
DATASETS      = ["coco"]
EPOCHS        = 100
TRAIN_BATCH   = 16
EVAL_BATCH    = 16
LR0           = 0.01
LRF           = 0.01

# ----- benchmark --------------------------------------------------------------
CONF_THRESHOLDS = [0.25, 0.5, 0.75]
IOU_THRESHOLDS  = [0.45, 0.5, 0.65]
BENCH_BATCH     = 1
WARMUP_STEPS    = 3
USE_TORCH_COMPILE = True

# ----- runtime ----------------------------------------------------------------
GPU_ID = "0"
