"""AnyTimeVisionenc TRAINING config. Loads .env on import.

Benchmark settings live in benchmark_config/vision.py (at repo root). This file
is for training only.

NOTE: MSDNet ckpts pinned to nBlocks. If you bump N_BLOCKS here for training,
update benchmark_config/vision.py to match.
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
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "ckpts"
LOG_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"
REF = ROOT / "reference"

# ----- model arch (training; must match benchmark) ----------------------------
ARCH = "msdnet"
N_BLOCKS = 7  # anytime config (was 5 = batch); retrain ckpts on change
N_CHANNELS = 16
GROWTH_RATE = 6
BASE = 4
STEP = 2
STEP_MODE = "even"
GR_FACTOR = "1-2-4"
BN_FACTOR = "1-2-4"

# ----- HuggingFace push -------------------------------------------------------
HF_USER = os.environ.get("HF_USER", "wicaksonolxn")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_AUTO_PUSH = True
HF_PRIVATE = True


def hf_repo_for(dataset: str) -> str:
    return f"{HF_USER}/msdnet-{dataset.lower()}-ee"


# ----- training hparams -------------------------------------------------------
DATASETS = ["cifar10", "cifar100", "svhn", "tinyimagenet", "ImageNet"]
EPOCHS = 300
TRAIN_BATCH = 64
EVAL_BATCH = 64
LR = 0.1
LR_TYPE = "multistep"
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
WORKERS = 4

# ----- runtime ----------------------------------------------------------------
GPU_ID = "0"
