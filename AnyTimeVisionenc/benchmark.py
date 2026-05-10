"""MSDNet benchmark. Two passes:

    profile_hw(...)        -> hw_results.json       (latency + memory + energy)
    evaluate_quality(...)  -> quality_results.json  (top-1 + top-5)
    benchmark(...)         -> runs both
"""

import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "reference"))

import config as C   # type: ignore
from shared import BenchmarkProfiler, auto_pull


def _maybe_pull_ckpt(model_id: str) -> Path:
    """If model_id is HF repo, pull it; else treat as local path."""
    if "/" in model_id and not Path(model_id).exists():
        return auto_pull(model_id, token=C.HF_TOKEN)
    return Path(model_id)


def _load_msdnet(model_id: str, dataset: str, compile_model: bool = False):
    from models.msdnet import MSDNet     # type: ignore
    import argparse

    is_imagenet = dataset.lower() == "imagenet"
    args = argparse.Namespace(
        arch=C.ARCH,
        nBlocks=C.N_BLOCKS,
        nChannels=32 if is_imagenet else C.N_CHANNELS,
        growthRate=16 if is_imagenet else C.GROWTH_RATE,
        grFactor=[int(x) for x in (("1-2-4-4" if is_imagenet else C.GR_FACTOR).split("-"))],
        bnFactor=[int(x) for x in (("1-2-4-4" if is_imagenet else C.BN_FACTOR).split("-"))],
        base=C.BASE,
        step=4 if is_imagenet else C.STEP,
        stepmode=C.STEP_MODE,
        bottleneck=True,
        prune="max",
        reduction=0.5,
        data=dataset,
    )
    model = MSDNet(args)

    ckpt_path = _maybe_pull_ckpt(model_id)
    ckpt_file = next((p for p in ckpt_path.glob("*.pth.tar")), None) \
                or next((p for p in ckpt_path.glob("*.pt")), None)
    if ckpt_file is None:
        raise FileNotFoundError(f"No checkpoint in {ckpt_path}")
    state = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))

    model.cuda().eval()
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[vision.benchmark] torch.compile enabled")
        except Exception as e:
            print(f"[vision.benchmark] torch.compile failed: {e}")
    return model


def _load_loader(dataset: str, data_dir: Union[str, Path], batch: int = 1):
    from dataloader import get_dataloaders   # type: ignore
    import argparse
    args = argparse.Namespace(
        data=dataset, data_root=str(data_dir),
        batch_size=batch, workers=2, use_valid=False,
    )
    _, _, test_loader = get_dataloaders(args)
    return test_loader


# =============================================================================
# HW pass — pure latency + memory + energy. NO quality.
# =============================================================================
def profile_hw(
    model_id: str, dataset: str, eval_mode: str,
    data_dir: Union[str, Path], out_dir: Union[str, Path],
    *, n_blocks: Optional[int] = None, bench_batch: int = 1,
    warmup_steps: int = 3, use_torch_compile: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "hw_results.json"

    model  = _load_msdnet(model_id, dataset, compile_model=use_torch_compile)
    loader = _load_loader(dataset, data_dir, batch=bench_batch)

    with BenchmarkProfiler(
        out_path=out_path, task=dataset, strategy=eval_mode,
        threshold=None, warmup_steps=warmup_steps,
    ) as prof:
        for inputs, _ in tqdm(loader, desc=f"HW {dataset} {eval_mode}"):
            inputs = inputs.cuda(non_blocking=True)
            with prof.timer() as t:
                with torch.no_grad():
                    _ = model(inputs)
            prof.log_sample(prediction=None, label=None,
                            ttft_sec=t.elapsed_s, end_to_end_sec=t.elapsed_s)
    return out_path


# =============================================================================
# Quality pass — top-1 / top-5 per exit. NO HW sampling.
# =============================================================================
def evaluate_quality(
    model_id: str, dataset: str, eval_mode: str,
    data_dir: Union[str, Path], out_dir: Union[str, Path],
    *, n_blocks: Optional[int] = None, bench_batch: int = 1,
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "quality_results.json"

    model  = _load_msdnet(model_id, dataset, compile_model=False)
    loader = _load_loader(dataset, data_dir, batch=bench_batch)

    n_exits = n_blocks or C.N_BLOCKS
    correct1 = [0] * n_exits
    correct5 = [0] * n_exits
    total    = 0

    for inputs, targets in tqdm(loader, desc=f"Q  {dataset} {eval_mode}"):
        inputs  = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        with torch.no_grad():
            outputs = model(inputs)   # list of logits per exit
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        for i, logits in enumerate(outputs):
            top1 = (logits.argmax(-1) == targets).sum().item()
            top5 = sum((targets[j] in logits[j].topk(5).indices) for j in range(targets.size(0)))
            correct1[i] += top1
            correct5[i] += top5
        total += targets.size(0)

    per_exit = []
    for i in range(n_exits):
        per_exit.append({
            "exit": i,
            "top1_acc": correct1[i] / total if total else 0.0,
            "top5_acc": correct5[i] / total if total else 0.0,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "dataset": dataset, "eval_mode": eval_mode, "n_samples": total,
        "per_exit": per_exit,
    }, indent=2), encoding="utf-8")
    print(f"[evaluate_quality] {per_exit}")
    return out_path


# =============================================================================
def benchmark(
    model_id: str, dataset: str, eval_mode: str,
    data_dir: Union[str, Path], out_dir: Union[str, Path],
    *, n_blocks: Optional[int] = None, bench_batch: int = 1,
    warmup_steps: int = 3, use_torch_compile: bool = True,
) -> Tuple[Path, Path]:
    hw = profile_hw(model_id, dataset, eval_mode, data_dir, out_dir,
                    n_blocks=n_blocks, bench_batch=bench_batch,
                    warmup_steps=warmup_steps, use_torch_compile=use_torch_compile)
    q  = evaluate_quality(model_id, dataset, eval_mode, data_dir, out_dir,
                          n_blocks=n_blocks, bench_batch=bench_batch)
    return hw, q
