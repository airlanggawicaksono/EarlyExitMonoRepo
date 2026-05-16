"""YOLOv9 gelan-s-ee per-sub-exit benchmark config + sweep runner.

ALL benchmark knobs live here. AnyTimeYolo/src/benchmark.py is pure functions.

Sweep: datasets x weight_sources x exits (5) x sub_exits (3 scales).
"""

import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

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


# ---- Sweep ------------------------------------------------------------------
DATASETS = ["coco"]
WEIGHT_SOURCES = ["pretrained"]  # HW-only sweep; add "trained" once ckpts pushed
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

    datasets = [only_dataset] if only_dataset else DATASETS
    weight_sources = [only_weight_source] if only_weight_source else WEIGHT_SOURCES
    exits = [only_exit] if only_exit is not None else list(range(N_EXITS))
    sub_exits = [only_sub_exit] if only_sub_exit is not None else list(range(N_SUB_EXITS))

    for ds in datasets:
        for ws in weight_sources:
            weights = resolve_weights_path(ds, ws)
            for e in exits:
                for s in sub_exits:
                    run_dir = OUT_DIR / ds / ws / f"exit_{e}_{SUB_EXIT_NAMES[s]}"
                    if not skip_hw:
                        profile_hw(
                            ee_yaml=EE_YAML,
                            weights_path=weights,
                            dataset=ds,
                            force_exit=e,
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
                    if not skip_quality:
                        evaluate_quality(
                            ee_yaml=EE_YAML,
                            weights_path=weights,
                            dataset=ds,
                            force_exit=e,
                            data_dir=DATA_DIR / ds,
                            out_dir=run_dir,
                            sub_exit=s,
                            weight_source=ws,
                            img_size=IMG_SIZE,
                            bench_batch=BENCH_BATCH,
                        )
