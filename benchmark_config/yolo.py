"""YOLOv9 gelan-m-ee per-sub-exit benchmark config + sweep runner.

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

from shared import load_env, auto_pull, has_valid_result

load_env()

NAME = "yolo"
MODEL_FAMILY = "gelan-m-ee"

# ---- HuggingFace + paths ----------------------------------------------------
HF_USER = os.environ.get("HF_USER", "wicaksonolxn")
HF_TOKEN = os.environ.get("HF_TOKEN")

MODEL_ROOT = REPO_ROOT / "AnyTimeYolo"
EE_YAML = MODEL_ROOT / "src" / "early_exit" / "configs" / "gelan-m-ee.yaml"
CKPT_DIR = MODEL_ROOT / "ckpts"
DATA_DIR = MODEL_ROOT / "datasets"
OUT_DIR = REPO_ROOT / "logs" / "benchmark" / NAME

PRETRAINED_URL = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-m.pt"
PRETRAINED_FILE = "gelan-m.pt"


def hf_trained_repo(dataset: str, mode: Optional[str] = None) -> str:
    """Mode-aware (selfdistill push naming) when mode is set; legacy when None."""
    if mode is None:
        return f"{HF_USER}/gelan-m-{dataset.lower()}-ee"
    return f"{HF_USER}/selfdistill-yolo-{dataset.lower()}-{mode}"


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
    # "voc",  # disabled — no working auto-download source. Install manually to enable.
]

# Quality / generalization: mAP per (exit, sub_exit).
# Labels MUST use COCO class IDs (0-79), not dataset-native IDs.
# For "voc": remap VOC labels to COCO class IDs before eval.
QUALITY_DATASETS: List[str] = [
    "coco",   # primary benchmark — all 80 COCO classes
    # "voc",  # disabled — no working auto-download source. Install manually to enable.
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

WEIGHT_SOURCES = ["trained"]
MODES = ["pairwise", "segd"]
N_EXITS = 6
N_SUB_EXITS = 3
SUB_EXIT_NAMES = ["P3", "P4", "P5"]

# ---- Bench hparams ----------------------------------------------------------
IMG_SIZE = 640
BENCH_BATCH = 1
WARMUP_STEPS = 3
USE_TORCH_COMPILE = True
N_SAMPLES = 200

# =============================================================================


# Roboflow public/coco exports purged from their CDN (all formats return 404 from GCS).
# Use direct official URLs instead.
_DIRECT_DOWNLOADS = {
    "coco": [
        ("http://images.cocodataset.org/zips/val2017.zip",
         "val2017 images (~778 MB, 5k imgs)"),
        ("https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip",
         "YOLO labels (~48 MB)"),
    ],
}


def _has_val_dir(root: Path) -> bool:
    """Check if YOLO val-style directory with images exists anywhere under root."""
    if not root.exists():
        return False
    for sub in ("val", "valid/images", "valid", "images/val2017", "val2017"):
        p = root / sub
        if p.exists() and any(p.iterdir()):
            return True
    for cand in list(root.rglob("val2017")) + list(root.rglob("valid/images")):
        if cand.is_dir() and any(cand.iterdir()):
            return True
    return False


def _download_and_extract(url: str, dest: Path, label: str) -> None:
    import urllib.request
    import zipfile

    zip_path = dest / Path(url).name
    print(f"[yolo]   downloading {label}: {url}")
    urllib.request.urlretrieve(url, str(zip_path))
    sz_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"[yolo]   got {sz_mb:.1f} MB. Extracting...")
    with zipfile.ZipFile(str(zip_path), "r") as z:
        z.extractall(str(dest))
    zip_path.unlink()


def _ensure_dataset(ds: str) -> None:
    """Download dataset via direct official URLs. Hard-fails on error."""
    dest = DATA_DIR / ds
    if _has_val_dir(dest):
        print(f"[yolo] dataset '{ds}' already present at {dest}")
        return
    if ds not in _DIRECT_DOWNLOADS:
        print(f"[yolo] no auto-download configured for '{ds}' — install manually")
        return
    print(f"[yolo] dataset '{ds}' missing — downloading via direct URLs...")
    dest.mkdir(parents=True, exist_ok=True)
    for url, label in _DIRECT_DOWNLOADS[ds]:
        _download_and_extract(url, dest, label)

    # coco2017labels.zip nests under coco/ — flatten if needed
    nested = dest / "coco"
    if nested.exists() and nested.is_dir():
        print(f"[yolo]   flattening nested coco/ dir")
        import shutil
        for item in nested.iterdir():
            target = dest / item.name
            if target.exists():
                continue
            shutil.move(str(item), str(target))
        nested.rmdir()

    # val2017.zip extracts to dest/val2017/ — move into dest/images/val2017/
    # coco2017labels.zip creates an EMPTY images/val2017/ placeholder; move files into it
    val_raw = dest / "val2017"
    val_target = dest / "images" / "val2017"
    if val_raw.exists() and val_raw.is_dir():
        import shutil
        val_target.parent.mkdir(parents=True, exist_ok=True)
        if val_target.exists():
            for item in val_raw.iterdir():
                shutil.move(str(item), str(val_target / item.name))
            val_raw.rmdir()
        else:
            shutil.move(str(val_raw), str(val_target))

    print(f"[yolo]   final contents of {dest}:")
    for p in sorted(dest.iterdir()):
        print(f"     {p.name}{'/' if p.is_dir() else ''}")
    if not _has_val_dir(dest):
        raise RuntimeError(f"download done but no val dir found under {dest}")


def run_all(
    only_dataset: Optional[str] = None,
    only_mode: Optional[str] = None,
    only_weight_source: Optional[str] = None,
    only_exit: Optional[int] = None,
    only_sub_exit: Optional[int] = None,
    skip_quality: bool = True,   # HW-only default
    skip_hw: bool = False,
    dry_run: bool = False,
):
    from AnyTimeYolo import (
        evaluate_quality, evaluate_quality_trained,
        sweep_hw_all_exits, sweep_hw_trained,
    )

    n_samples = 5 if dry_run else N_SAMPLES
    max_samples = 5 if dry_run else None
    out_root_base = REPO_ROOT / "logs.dry_run" / "benchmark" / NAME if dry_run else OUT_DIR

    all_datasets = set()
    if not skip_hw:
        hw_ds_list = [only_dataset] if only_dataset else (HW_DATASETS[:1] if dry_run else HW_DATASETS)
        all_datasets.update(hw_ds_list)
    if not skip_quality:
        q_ds_list = [only_dataset] if only_dataset else (QUALITY_DATASETS[:1] if dry_run else QUALITY_DATASETS)
        all_datasets.update(q_ds_list)
    for ds in all_datasets:
        _ensure_dataset(ds)

    weight_sources = [only_weight_source] if only_weight_source else WEIGHT_SOURCES
    modes = [only_mode] if only_mode else MODES
    exits = [only_exit] if only_exit is not None else list(range(N_EXITS))
    sub_exits = [only_sub_exit] if only_sub_exit is not None else list(range(N_SUB_EXITS))
    if dry_run:
        print(f"[yolo] DRY RUN: 5 samples -> {out_root_base} | datasets={sorted(all_datasets)} modes={modes}")

    # Pretrained base weights — used only as backbone seed when building MultiExitYolo;
    # adapters + heads come from the HF repo. Best-effort fetch; if missing, train side
    # already pushed full weights so this is optional.
    try:
        pretrained_path = resolve_weights_path("coco", "pretrained")
    except Exception:
        pretrained_path = None

    # ---- HW pass -----------------------------------------------------------
    if not skip_hw:
        hw_datasets = [only_dataset] if only_dataset else (HW_DATASETS[:1] if dry_run else HW_DATASETS)
        for ds in hw_datasets:
            for ws in weight_sources:
                # pretrained = identical across modes -> single "pretrained" pseudo-mode
                for mode in (modes if ws == "trained" else ["pretrained"]):
                    if ws == "trained":
                        repo_id = hf_trained_repo(ds, mode)
                        try:
                            sweep_hw_trained(
                                repo_id=repo_id,
                                dataset=ds,
                                mode=mode,
                                exits=exits,
                                sub_exits=sub_exits,
                                n_exits=N_EXITS,
                                ee_yaml=EE_YAML,
                                data_dir=DATA_DIR / ds,
                                out_root=out_root_base / ds / mode,
                                pretrained_weights=pretrained_path,
                                weight_source=ws,
                                img_size=IMG_SIZE,
                                bench_batch=BENCH_BATCH,
                                warmup_steps=WARMUP_STEPS,
                                use_torch_compile=USE_TORCH_COMPILE,
                                n_samples=n_samples,
                            )
                        except Exception as exc:
                            print(f"[yolo] hw sweep failed {ds}/{mode}/{ws}: {exc}")
                    else:
                        try:
                            weights = resolve_weights_path(ds, ws)
                        except Exception as exc:
                            print(f"[yolo] weights not found for hw {ds}/{ws}: {exc}")
                            continue
                        try:
                            sweep_hw_all_exits(
                                ee_yaml=EE_YAML,
                                weights_path=weights,
                                dataset=ds,
                                exits=exits,
                                sub_exits=sub_exits,
                                data_dir=DATA_DIR / ds,
                                out_root=out_root_base / ds / mode,
                                weight_source=ws,
                                img_size=IMG_SIZE,
                                bench_batch=BENCH_BATCH,
                                warmup_steps=WARMUP_STEPS,
                                use_torch_compile=USE_TORCH_COMPILE,
                                n_samples=n_samples,
                            )
                        except Exception as exc:
                            print(f"[yolo] hw sweep failed {ds}/{ws}: {exc}")

    # ---- Quality pass -------------------------------------------------------
    if not skip_quality:
        quality_datasets = [only_dataset] if only_dataset else (QUALITY_DATASETS[:1] if dry_run else QUALITY_DATASETS)
        for ds in quality_datasets:
            for ws in weight_sources:
                # pretrained = identical across modes -> single "pretrained" pseudo-mode
                for mode in (modes if ws == "trained" else ["pretrained"]):
                    valid_cls = DATASET_COCO_CLASS_IDS.get(ds)
                    for ei in exits:
                        for s in sub_exits:
                            run_dir = out_root_base / ds / mode / f"exit_{ei}_{SUB_EXIT_NAMES[s]}"
                            q_path = run_dir / "quality_results.json"
                            if has_valid_result(q_path):
                                print(f"[skip] quality exists: {q_path}")
                                continue
                            if ws == "trained":
                                repo_id = hf_trained_repo(ds, mode)
                                try:
                                    evaluate_quality_trained(
                                        repo_id=repo_id,
                                        dataset=ds,
                                        mode=mode,
                                        force_exit=ei,
                                        sub_exit=s,
                                        n_exits=N_EXITS,
                                        ee_yaml=EE_YAML,
                                        data_dir=DATA_DIR / ds,
                                        out_dir=run_dir,
                                        pretrained_weights=pretrained_path,
                                        weight_source=ws,
                                        img_size=IMG_SIZE,
                                        bench_batch=BENCH_BATCH,
                                        valid_classes=valid_cls,
                                        max_samples=max_samples,
                                    )
                                except Exception as exc:
                                    print(f"[yolo] quality failed {ds}/{mode} exit={ei} sub={s}: {exc}")
                            else:
                                try:
                                    weights = resolve_weights_path(ds, ws)
                                except Exception as exc:
                                    print(f"[yolo] weights not found for quality {ds}/{ws}: {exc}")
                                    continue
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
                                        max_samples=max_samples,
                                    )
                                except Exception as exc:
                                    print(f"[yolo] quality failed {ds} exit={ei} sub={s}: {exc}")
