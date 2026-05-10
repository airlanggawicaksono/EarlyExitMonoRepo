"""Central config for AnyTimeLLaMa. Loads .env on import."""

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
C4_CACHE    = DATA_DIR / "c4_cache"

# ----- model ------------------------------------------------------------------
BASE_MODEL    = "meta-llama/Llama-3.2-1B"   # local-friendly default; override per env
EXIT_LAYERS   = [4, 8, 12]                  # for 1B (16 layers); use [8,16,24] for 8B
EXIT_WEIGHTS  = [1.0, 1.0, 1.0]
SEQ_LEN       = 1024

# ----- HuggingFace I/O --------------------------------------------------------
HF_USER       = os.environ.get("HF_USER", "wicaksonolxn")
HF_TOKEN      = os.environ.get("HF_TOKEN")
HF_AUTO_PUSH  = True
HF_PRIVATE    = True
def hf_repo_for(model_short: str) -> str:
    return f"{HF_USER}/{model_short}-ee-heads"

# ----- training ---------------------------------------------------------------
TRAIN_BATCH   = 4
EVAL_BATCH    = 4
GRAD_ACCUM    = 8
LR            = 1e-4
NUM_EPOCHS    = 1
LOGGING_STEPS = 10
SAVE_STEPS    = 500
EVAL_STEPS    = 500
TORCH_DTYPE   = "bfloat16"
MAX_TRAIN_SAMPLES = 200_000
MAX_VAL_SAMPLES   = 2_000

# ----- C4 cache (optional pre-download to skip repeated remote pulls) ---------
USE_LOCAL_C4_CACHE = True

# ----- benchmark --------------------------------------------------------------
BENCH_DATASET            = "cnn_dailymail"
BENCH_N_SAMPLES          = 100
BENCH_MAX_NEW_TOKENS     = 128
CONFIDENCE_THRESHOLD     = 0.9
USE_TORCH_COMPILE        = True
WARMUP_STEPS             = 3

# ----- runtime ----------------------------------------------------------------
GPU_ID = "0"
