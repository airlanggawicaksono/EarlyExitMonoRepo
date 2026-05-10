"""Central config for AnyTimeVisionenc. Loads .env on import."""

from pathlib import Path
import os, sys

ROOT      = Path(__file__).parent.resolve()
REPO_ROOT = ROOT.parent

sys.path.insert(0, str(REPO_ROOT))
from shared import load_env
load_env()

# ----- paths ------------------------------------------------------------------
DATA_DIR    = ROOT / "data"
CKPT_DIR    = ROOT / "ckpts"
LOG_DIR     = ROOT / "logs"
RESULTS_DIR = ROOT / "results"
REF         = ROOT / "reference"

# ----- model (MSDNet) ---------------------------------------------------------
ARCH         = "msdnet"
N_BLOCKS     = 5            # number of multi-exit blocks (5 for batch, 7 for anytime)
N_CHANNELS   = 16           # CIFAR; use 32 for ImageNet
GROWTH_RATE  = 6            # CIFAR; use 16 for ImageNet
BASE         = 4
STEP         = 2
STEP_MODE    = "even"       # "even" or "lin_grow"
GR_FACTOR    = "1-2-4"
BN_FACTOR    = "1-2-4"

# ----- HuggingFace I/O --------------------------------------------------------
HF_USER       = os.environ.get("HF_USER", "wicaksonolxn")
HF_TOKEN      = os.environ.get("HF_TOKEN")
HF_AUTO_PUSH  = True
HF_PRIVATE    = True
def hf_repo_for(dataset: str) -> str:
    return f"{HF_USER}/msdnet-{dataset.lower()}-ee"

# ----- training ---------------------------------------------------------------
DATASETS      = ["cifar10", "cifar100", "svhn", "tinyimagenet", "ImageNet"]
EPOCHS        = 300
TRAIN_BATCH   = 64
EVAL_BATCH    = 64
LR            = 0.1
LR_TYPE       = "multistep"
MOMENTUM      = 0.9
WEIGHT_DECAY  = 1e-4
WORKERS       = 4

# ----- benchmark --------------------------------------------------------------
EVAL_MODES        = ["anytime", "dynamic"]
BENCH_BATCH       = 1
WARMUP_STEPS      = 3
USE_TORCH_COMPILE = True

# ----- runtime ----------------------------------------------------------------
GPU_ID = "0"
