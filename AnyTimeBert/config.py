"""Central config for AnyTimeBert. Loads .env on import."""

from pathlib import Path
import os, sys

ROOT      = Path(__file__).parent.resolve()
REPO_ROOT = ROOT.parent

sys.path.insert(0, str(REPO_ROOT))
from shared import load_env
load_env()

# ----- paths ------------------------------------------------------------------
GLUE_DIR    = ROOT / "glue_data"
CKPT_DIR    = ROOT / "ckpts"
LOG_DIR     = ROOT / "logs"
RESULTS_DIR = ROOT / "results"
REF_STATIC  = ROOT / "reference" / "finetune-static"
REF_DYNAMIC = ROOT / "reference" / "finetune-dynamic"

# ----- model ------------------------------------------------------------------
HF_MODEL_NAME     = "OpenMOSS-Team/elasticbert-base"
NUM_HIDDEN_LAYERS = 12
NUM_OUTPUT_LAYERS = 12
MAX_SEQ_LENGTH    = 128

# ----- HuggingFace I/O --------------------------------------------------------
HF_USER       = os.environ.get("HF_USER", "wicaksonolxn")
HF_TOKEN      = os.environ.get("HF_TOKEN")  # from .env
HF_AUTO_PUSH  = True
HF_PRIVATE    = True
def hf_repo_for(task: str) -> str:
    return f"{HF_USER}/elasticbert-base-{task.lower()}-ee"

# ----- training ---------------------------------------------------------------
TRAIN_BATCH    = 16
EVAL_BATCH     = 32
GRAD_ACCUM     = 2
LR             = 2e-5
WEIGHT_DECAY   = 0.1
WARMUP_RATE    = 0.06
NUM_EPOCHS     = 5
LOGGING_STEPS  = 50
EARLY_STOP     = 10
USE_FP16       = True

# ----- benchmark --------------------------------------------------------------
TASKS         = ["SST-2", "MRPC", "QNLI", "RTE", "CoLA"]
BENCH_BATCH   = 1
WARMUP_STEPS  = 3
USE_TORCH_COMPILE = True

# ----- runtime ----------------------------------------------------------------
GPU_ID = "0"
