"""YOLOv9 benchmark. TWO passes:

profile_hw(...)        -> hw_results.json       (latency + memory + energy)
evaluate_quality(...)  -> quality_results.json  (mAP@0.5, mAP@0.5:0.95)
benchmark(...)         -> runs both
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from tqdm import tqdm

_HERE  = Path(__file__).resolve().parent     # AnyTimeYolo/src/
_MODEL = _HERE.parent
_REPO  = _MODEL.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_MODEL))

import config as C  # type: ignore
from shared import BenchmarkProfiler, auto_pull


def _resolve_weights(model_id: str) -> Path:
    """If HF repo, pull it; else local."""
    if "/" in model_id and not Path(model_id).exists():
        repo_dir = auto_pull(model_id, token=C.HF_TOKEN)
        # YOLOv9 best.pt expected
        for name in ("best.pt", "last.pt"):
            f = repo_dir / name
            if f.exists():
                return f
        # else first .pt
        pts = list(repo_dir.glob("*.pt"))
        if pts:
            return pts[0]
        raise FileNotFoundError(f"No .pt in {repo_dir}")
    return Path(model_id)


# =============================================================================
# HW pass — pure latency. NO mAP computation.
# =============================================================================
def profile_hw(
    model_id: str,
    dataset: str,
    out_dir: Union[str, Path],
    *,
    data_dir: Optional[Union[str, Path]] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    img_size: int = 640,
    bench_batch: int = 1,
    warmup_steps: int = 3,
    use_torch_compile: bool = True,
    n_samples: int = 200,
) -> Path:
    """Run forward passes only, sample HW per image. No NMS metric calc."""
    out_dir = Path(out_dir)
    out_path = out_dir / "hw_results.json"

    sys.path.insert(0, str(C.YOLO_REF))
    from models.common import DetectMultiBackend  # type: ignore
    from utils.dataloaders import LoadImagesAndLabels  # type: ignore

    weights = _resolve_weights(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(str(weights), device=device, fp16=False)
    if use_torch_compile and hasattr(torch, "compile"):
        try:
            model.model = torch.compile(model.model, mode="reduce-overhead")
        except Exception as e:
            print(f"[yolo.benchmark] compile failed: {e}")

    base = Path(data_dir) if data_dir else (C.DATA_DIR / dataset)
    val_dataset = LoadImagesAndLabels(
        str(base / "val"),
        img_size=img_size,
        batch_size=bench_batch,
        augment=False,
        hyp=None,
        rect=True,
    )
    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bench_batch, shuffle=False
    )

    n = 0
    with BenchmarkProfiler(
        out_path=out_path,
        task=dataset,
        strategy="conf",
        threshold=conf_threshold,
        warmup_steps=warmup_steps,
    ) as prof:
        for batch in tqdm(loader, desc=f"HW {dataset}"):
            imgs = batch[0].to(device).float() / 255.0
            with prof.timer() as t:
                with torch.no_grad():
                    _ = model(imgs)
            prof.log_sample(
                prediction=None,
                label=None,
                ttft_sec=t.elapsed_s,
                end_to_end_sec=t.elapsed_s,
            )
            n += 1
            if n >= n_samples:
                break
    return out_path


# =============================================================================
# Quality pass — mAP via YOLOv9's val.py (subprocess, separate from HW).
# =============================================================================
def evaluate_quality(
    model_id: str,
    dataset: str,
    out_dir: Union[str, Path],
    *,
    data_dir: Optional[Union[str, Path]] = None,
    conf_threshold: float = 0.001,  # standard val conf for mAP
    iou_threshold: float = 0.6,
    img_size: int = 640,
    bench_batch: int = 32,
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "quality_results.json"
    weights = _resolve_weights(model_id)
    base = Path(data_dir) if data_dir else (C.DATA_DIR / dataset)
    data_yaml = base / "data.yaml"

    val_out = out_dir / "yolo_val"
    cmd = [
        sys.executable,
        "val.py",
        "--data",
        str(data_yaml),
        "--weights",
        str(weights),
        "--img",
        str(img_size),
        "--batch",
        str(bench_batch),
        "--conf",
        str(conf_threshold),
        "--iou",
        str(iou_threshold),
        "--device",
        C.GPU_ID,
        "--project",
        str(out_dir),
        "--name",
        "yolo_val",
        "--save-json",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = C.GPU_ID
    subprocess.run(cmd, cwd=C.YOLO_REF, env=env, check=True)

    # Parse mAP from val output. YOLOv9 prints to stdout but also saves results.txt
    # Try to scrape the saved results.
    metrics = {}
    results_csv = val_out / "results.csv"
    if results_csv.exists():
        import csv

        with open(results_csv) as f:
            reader = csv.DictReader(f)
            row = next(reader, {})
            metrics = {k.strip(): v for k, v in row.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "weights": str(weights),
                "conf": conf_threshold,
                "iou": iou_threshold,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[evaluate_quality] {metrics}")
    return out_path


def benchmark(
    model_id: str,
    dataset: str,
    out_dir: Union[str, Path],
    *,
    data_dir: Optional[Union[str, Path]] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    img_size: int = 640,
    bench_batch: int = 1,
    warmup_steps: int = 3,
    use_torch_compile: bool = True,
) -> Tuple[Path, Path]:
    hw = profile_hw(
        model_id,
        dataset,
        out_dir,
        data_dir=data_dir,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        img_size=img_size,
        bench_batch=bench_batch,
        warmup_steps=warmup_steps,
        use_torch_compile=use_torch_compile,
    )
    q = evaluate_quality(
        model_id,
        dataset,
        out_dir,
        data_dir=data_dir,
        img_size=img_size,
        bench_batch=32,
    )
    return hw, q
