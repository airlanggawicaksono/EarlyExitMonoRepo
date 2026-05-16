"""YOLOv9 gelan-s-ee per-exit benchmark. TWO passes:

profile_hw(...)        -> hw_results.json       (latency + memory + energy)
evaluate_quality(...)  -> quality_results.json  (mAP@0.5 — currently placeholder)
benchmark(...)         -> runs both

Per-exit isolation: forward computes modules 0..EXIT_MAX_DEPTH[k], then applies
exit head k (module EXIT_HEAD_OFFSET+k). Skips intermediate FPN modules not
needed by exit k.

weight_source = trained (your gelan-s-ee HF) or pretrained (upstream gelan-s.pt).
Pretrained gives backbone weights only; EE heads random -> HW valid, quality not.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from tqdm import tqdm

_HERE  = Path(__file__).resolve().parent     # AnyTimeYolo/src/
_MODEL = _HERE.parent
_REPO  = _MODEL.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_MODEL / "model" / "yolov9"))

from shared import BenchmarkProfiler, load_env, model_metrics  # noqa: E402

load_env()

HF_TOKEN = os.environ.get("HF_TOKEN")

# ---- Architectural facts of gelan-s-ee.yaml ---------------------------------
# (these are properties of the EE yaml file, not user-tunable knobs)
EXIT_MAX_DEPTH = {0: 8, 1: 9, 2: 15, 3: 18, 4: 21}
EXIT_HEAD_OFFSET = 22
SUB_EXIT_NAMES = ["P3", "P4", "P5"]


def _download_pretrained(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[yolo.benchmark] downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def _load_ee_model(ee_yaml: Path, weights_path: Path, weight_source: str, compile_model: bool = False):
    """Load EarlyExitModel from yaml + weights."""
    from early_exit.model import EarlyExitModel  # type: ignore

    model = EarlyExitModel(str(ee_yaml), ch=3)
    ckpt = torch.load(str(weights_path), map_location="cpu")
    state = ckpt.get("model", ckpt)
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[yolo.benchmark] load (ws={weight_source}) "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    model.cuda().eval()
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"[yolo.benchmark] compile failed: {e}")
    return model


def _forward_to_exit(model, x, force_exit: int, sub_exit: Optional[int] = None):
    """Run modules 0..EXIT_MAX_DEPTH[k] + exit head k.

    If sub_exit is None, run full DDetect (all 3 scales: cv2[0..2] + cv3[0..2]).
    If sub_exit in {0,1,2}, only run cv2[sub_exit] + cv3[sub_exit] for scale s.
    """
    real = model._orig_mod if hasattr(model, "_orig_mod") else model
    max_d = EXIT_MAX_DEPTH[force_exit]
    head_i = EXIT_HEAD_OFFSET + force_exit

    y = []
    out = x
    for m in real.model:
        if m.i > max_d and m.i != head_i:
            y.append(None)
            continue
        if m.i == head_i and sub_exit is not None:
            head_input = real._resolve_input(m, out, y)  # list of 3 feature maps
            xs = head_input[sub_exit]
            out = torch.cat((m.cv2[sub_exit](xs), m.cv3[sub_exit](xs)), 1)
            y.append(None)
            continue
        x_in = real._resolve_input(m, out, y)
        out = m(x_in)
        y.append(out if m.i in real.save else None)
    return out


def _load_loader(dataset: str, data_dir: Union[str, Path], img_size: int, batch: int):
    from utils.dataloaders import LoadImagesAndLabels  # type: ignore
    base = Path(data_dir)
    val_dataset = LoadImagesAndLabels(
        str(base / "val"),
        img_size=img_size,
        batch_size=batch,
        augment=False,
        hyp=None,
        rect=True,
    )
    return torch.utils.data.DataLoader(val_dataset, batch_size=batch, shuffle=False)


# =============================================================================
# HW pass
# =============================================================================
def profile_hw(
    ee_yaml: Union[str, Path],
    weights_path: Union[str, Path],
    dataset: str,
    force_exit: int,
    data_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    sub_exit: Optional[int] = None,
    weight_source: str = "trained",
    img_size: int = 640,
    bench_batch: int = 1,
    warmup_steps: int = 3,
    use_torch_compile: bool = True,
    n_samples: int = 200,
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "hw_results.json"

    model = _load_ee_model(Path(ee_yaml), Path(weights_path), weight_source, compile_model=use_torch_compile)
    loader = _load_loader(dataset, data_dir, img_size, bench_batch)
    device = next(model.parameters()).device

    dummy = torch.zeros((1, 3, img_size, img_size), device=device)
    try:
        mm = model_metrics(model, dummy)
    except Exception as e:
        print(f"[yolo.benchmark] model_metrics skipped: {e}")
        mm = {}

    sub_tag = f"_{SUB_EXIT_NAMES[sub_exit]}" if sub_exit is not None else "_all"
    n = 0
    with BenchmarkProfiler(
        out_path=out_path,
        task=dataset,
        strategy=weight_source,
        threshold=f"E{force_exit}{sub_tag}",
        warmup_steps=warmup_steps,
        meta={
            "force_exit": force_exit,
            "sub_exit": sub_exit,
            "sub_exit_name": SUB_EXIT_NAMES[sub_exit] if sub_exit is not None else "all",
            "weight_source": weight_source,
            **mm,
        },
    ) as prof:
        desc = f"HW {dataset} E{force_exit}{sub_tag} ({weight_source})"
        for batch in tqdm(loader, desc=desc):
            imgs = batch[0].to(device).float() / 255.0
            with prof.timer() as t:
                with torch.no_grad():
                    _ = _forward_to_exit(model, imgs, force_exit, sub_exit)
            prof.log_sample(
                prediction=None,
                label=None,
                ttft_sec=t.elapsed_s,
                end_to_end_sec=t.elapsed_s,
                exit_layer=force_exit,
                sub_exit=sub_exit,
            )
            n += 1
            if n >= n_samples:
                break
    return out_path


# =============================================================================
# Quality pass — TODO: proper mAP via val.py for each exit head.
# Placeholder: record predictions, full eval pipeline pending.
# =============================================================================
def evaluate_quality(
    ee_yaml: Union[str, Path],
    weights_path: Union[str, Path],
    dataset: str,
    force_exit: int,
    data_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    sub_exit: Optional[int] = None,
    weight_source: str = "trained",
    img_size: int = 640,
    bench_batch: int = 1,
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "quality_results.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "weight_source": weight_source,
                "force_exit": force_exit,
                "sub_exit": sub_exit,
                "sub_exit_name": SUB_EXIT_NAMES[sub_exit] if sub_exit is not None else "all",
                "note": "TODO: per-(exit,scale) mAP requires custom val pipeline; HW pass only currently",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


# =============================================================================
def benchmark(
    ee_yaml: Union[str, Path],
    weights_path: Union[str, Path],
    dataset: str,
    force_exit: int,
    data_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    sub_exit: Optional[int] = None,
    weight_source: str = "trained",
    img_size: int = 640,
    bench_batch: int = 1,
    warmup_steps: int = 3,
    use_torch_compile: bool = True,
) -> Tuple[Path, Path]:
    hw = profile_hw(
        ee_yaml, weights_path, dataset, force_exit, data_dir, out_dir,
        sub_exit=sub_exit,
        weight_source=weight_source,
        img_size=img_size,
        bench_batch=bench_batch,
        warmup_steps=warmup_steps,
        use_torch_compile=use_torch_compile,
    )
    q = evaluate_quality(
        ee_yaml, weights_path, dataset, force_exit, data_dir, out_dir,
        sub_exit=sub_exit,
        weight_source=weight_source,
        img_size=img_size,
        bench_batch=bench_batch,
    )
    return hw, q
