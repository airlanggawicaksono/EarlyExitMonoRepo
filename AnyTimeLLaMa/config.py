"""AnyTimeLLaMa TRAINING config. Loads .env on import.

Benchmark settings live in benchmark_config/llama.py (at repo root). This file
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
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "ckpts"
LOG_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"
C4_CACHE = DATA_DIR / "c4_cache"

# ----- model arch (training) --------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.2-1B"  # Jetson Nano 4GB friendly (~2GB BF16)
EXIT_LAYERS = [4, 8, 12]  # trained exit positions (1B = 16 layers)
EXIT_WEIGHTS = [1.0, 1.0, 1.0]
SEQ_LEN = 1024

# ----- HuggingFace push -------------------------------------------------------
HF_USER = os.environ.get("HF_USER", "wicaksonolxn")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_AUTO_PUSH = True
HF_PRIVATE = True


def hf_repo_for(model_short: str) -> str:
    """Where training pushes the exit heads."""
    return f"{HF_USER}/{model_short}-ee-heads"


HF_EXIT_HEADS = f"{HF_USER}/llama-3.2-1b-ee-heads"

# ----- training hparams -------------------------------------------------------
TRAIN_BATCH = 4
EVAL_BATCH = 4
GRAD_ACCUM = 8
LR = 1e-4
NUM_EPOCHS = 1
LOGGING_STEPS = 10
SAVE_STEPS = 500
EVAL_STEPS = 500
TORCH_DTYPE = "bfloat16"
MAX_TRAIN_SAMPLES = 200_000
MAX_VAL_SAMPLES = 2_000

USE_LOCAL_C4_CACHE = True

# ----- runtime ----------------------------------------------------------------
GPU_ID = "0"
