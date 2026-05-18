"""MSDNet per-exit benchmark. Two passes:

profile_hw(...)        -> hw_results.json       (latency + memory + energy)
evaluate_quality(...)  -> quality_results.json  (top-1 + top-5)
benchmark(...)         -> runs both

Per-exit isolation: forward truncated to block K (run blocks 0..K, apply
classifier K, stop). Fair latency per exit.

weight_source=trained only. No public HF pretrained MSDNet.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE  = Path(__file__).resolve().parent     # AnyTimeVisionenc/src/
_MODEL = _HERE.parent
_REPO  = _MODEL.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_MODEL / "reference"))

from shared import BenchmarkProfiler, auto_pull, load_env, model_metrics  # noqa: E402

load_env()

HF_TOKEN = os.environ.get("HF_TOKEN")


def _maybe_pull_ckpt(model_id: str) -> Path:
    if "/" in model_id and not Path(model_id).exists():
        return auto_pull(model_id, token=HF_TOKEN)
    return Path(model_id)


def _load_msdnet(model_id: Optional[str], arch_data: str, arch_kwargs: dict, compile_model: bool = False):
    """arch_data: MSDNet data key ("cifar10", "cifar100", "ImageNet") — sets nClasses.
    model_id=None -> random-init (HW-only)."""
    from models.msdnet import MSDNet  # type: ignore
    import argparse

    args = argparse.Namespace(data=arch_data, **arch_kwargs)
    args.nScales = len(args.grFactor)
    model = MSDNet(args)

    if model_id is None:
        print("[vision.benchmark] random-init (no weights loaded) -- HW-only mode")
    else:
        ckpt_path = _maybe_pull_ckpt(model_id)
        ckpt_file = next((p for p in ckpt_path.glob("*.pth.tar")), None) or next(
            (p for p in ckpt_path.glob("*.pt")), None
        )
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


def _forward_to_exit(model, x, force_exit: int):
    """Run blocks 0..force_exit, apply classifier[force_exit]. Stop."""
    real = model._orig_mod if hasattr(model, "_orig_mod") else model
    for i in range(force_exit + 1):
        x = real.blocks[i](x)
    return real.classifier[force_exit](x)


_IMAGENET_NORM = tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_CIFAR_NORM    = tv_transforms.Normalize(mean=[0.4914, 0.4824, 0.4467], std=[0.2471, 0.2435, 0.2616])


def _make_loader_stl10(data_dir, batch):
    t = tv_transforms.Compose([
        tv_transforms.Resize(32),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2242, 0.2215, 0.2239]),
    ])
    ds = tv_datasets.STL10(str(data_dir), split="test", download=True, transform=t)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)


def _make_loader_mnist(data_dir, batch):
    t = tv_transforms.Compose([
        tv_transforms.Resize(32),
        tv_transforms.Grayscale(num_output_channels=3),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081]),
    ])
    ds = tv_datasets.MNIST(str(data_dir), train=False, download=True, transform=t)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)


def _make_loader_fashionmnist(data_dir, batch):
    t = tv_transforms.Compose([
        tv_transforms.Resize(32),
        tv_transforms.Grayscale(num_output_channels=3),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.2860, 0.2860, 0.2860], std=[0.3530, 0.3530, 0.3530]),
    ])
    ds = tv_datasets.FashionMNIST(str(data_dir), train=False, download=True, transform=t)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)


_CUSTOM_LOADERS = {
    "stl10":        _make_loader_stl10,
    "mnist":        _make_loader_mnist,
    "fashionmnist": _make_loader_fashionmnist,
}

_LEGACY_DATASETS = {"cifar10", "cifar100", "svhn", "tinyimagenet", "imagenet"}


def _load_loader(dataset: str, data_dir: Union[str, Path], batch: int = 1):
    if dataset in _CUSTOM_LOADERS:
        return _CUSTOM_LOADERS[dataset](data_dir, batch)

    from dataloader import get_dataloaders  # type: ignore
    import argparse

    args = argparse.Namespace(
        data=dataset,
        data_root=str(data_dir),
        batch_size=batch,
        workers=2,
        use_valid=False,
        splits=["test"],
        save=".",
    )
    _, _, test_loader = get_dataloaders(args)
    return test_loader


# =============================================================================
# HW pass
# =============================================================================
def profile_hw(
    model_id: Optional[str],
    dataset: str,
    force_exit: int,
    data_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    arch_kwargs: dict,
    arch_key: Optional[str] = None,
    weight_source: str = "trained",
    bench_batch: int = 1,
    warmup_steps: int = 3,
    use_torch_compile: bool = True,
) -> Path:
    """dataset: real dataset name (e.g. "svhn") — used for dataloader + JSON task label.
    arch_key: MSDNet data key (e.g. "cifar10") — controls nClasses in MSDNet.
              Defaults to dataset when not supplied (correct for cifar10/cifar100/ImageNet).
    """
    out_dir = Path(out_dir)
    out_path = out_dir / "hw_results.json"

    _arch_key = arch_key or dataset
    model = _load_msdnet(model_id, _arch_key, arch_kwargs, compile_model=use_torch_compile)
    loader = _load_loader(dataset, data_dir, batch=bench_batch)

    # Dummy input shape per dataset (HW arch derived from arch_key, not real dataset name)
    dummy_shape = (1, 3, 224, 224) if _arch_key.lower() == "imagenet" else (1, 3, 32, 32)
    dummy = torch.zeros(dummy_shape, device="cuda")
    try:
        mm = model_metrics(model, dummy)
    except Exception as e:
        print(f"[vision.benchmark] model_metrics skipped: {e}")
        mm = {}

    with BenchmarkProfiler(
        out_path=out_path,
        task=dataset,
        strategy=weight_source,
        threshold=force_exit,
        warmup_steps=warmup_steps,
        meta={
            "force_exit": force_exit,
            "weight_source": weight_source,
            "model_id": model_id,
            "arch_key": _arch_key,
            **mm,
        },
    ) as prof:
        for inputs, _ in tqdm(loader, desc=f"HW {dataset} exit={force_exit} ({weight_source})"):
            inputs = inputs.cuda(non_blocking=True)
            with prof.timer() as t:
                with torch.no_grad():
                    _ = _forward_to_exit(model, inputs, force_exit)
            prof.log_sample(
                prediction=None,
                label=None,
                ttft_sec=t.elapsed_s,
                end_to_end_sec=t.elapsed_s,
                exit_layer=force_exit,
            )
    return out_path


# =============================================================================
# Quality pass
# =============================================================================
def evaluate_quality(
    model_id: Optional[str],
    dataset: str,
    force_exit: int,
    data_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    arch_kwargs: dict,
    arch_key: Optional[str] = None,
    weight_source: str = "trained",
    bench_batch: int = 1,
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "quality_results.json"

    _arch_key = arch_key or dataset
    model = _load_msdnet(model_id, _arch_key, arch_kwargs, compile_model=False)
    loader = _load_loader(dataset, data_dir, batch=bench_batch)

    correct1 = 0
    correct5 = 0
    total = 0
    for inputs, targets in tqdm(loader, desc=f"Q  {dataset} exit={force_exit} ({weight_source})"):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        with torch.no_grad():
            logits = _forward_to_exit(model, inputs, force_exit)
        correct1 += (logits.argmax(-1) == targets).sum().item()
        correct5 += sum(
            (targets[j] in logits[j].topk(5).indices) for j in range(targets.size(0))
        )
        total += targets.size(0)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "arch_key": _arch_key,
                "weight_source": weight_source,
                "force_exit": force_exit,
                "model_id": model_id,
                "n_samples": total,
                "top1_acc": round(correct1 / total, 6) if total else 0.0,
                "top5_acc": round(correct5 / total, 6) if total else 0.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[evaluate_quality] exit={force_exit} top1={correct1/total:.4f}")
    return out_path


# =============================================================================
def benchmark(
    model_id: Optional[str],
    dataset: str,
    force_exit: int,
    data_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    arch_kwargs: dict,
    arch_key: Optional[str] = None,
    weight_source: str = "trained",
    bench_batch: int = 1,
    warmup_steps: int = 3,
    use_torch_compile: bool = True,
) -> Tuple[Path, Path]:
    hw = profile_hw(
        model_id, dataset, force_exit, data_dir, out_dir,
        arch_kwargs=arch_kwargs,
        arch_key=arch_key,
        weight_source=weight_source,
        bench_batch=bench_batch,
        warmup_steps=warmup_steps,
        use_torch_compile=use_torch_compile,
    )
    q = evaluate_quality(
        model_id, dataset, force_exit, data_dir, out_dir,
        arch_kwargs=arch_kwargs,
        arch_key=arch_key,
        weight_source=weight_source,
        bench_batch=bench_batch,
    )
    return hw, q
