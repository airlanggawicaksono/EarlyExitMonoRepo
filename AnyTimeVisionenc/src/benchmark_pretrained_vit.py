"""Pretrained ViT (multi-exit) per-exit benchmark — DEFAULT model, no training.

Loads `google/vit-large-patch16-224` (the upstream pretrained ViT), builds the
same MultiExitViT used by the trained path, then broadcasts the pretrained
classifier head into every exit slot. If the pretrained head's output dim does
not match the task label count (ViT-large = 1000 ImageNet classes vs CIFAR
10/100), the exit heads stay random-init — HW timing is still valid, quality is
not meaningful in that case.

Reuses the trained module's internals (truncation ctx, forward, loader, HW pass)
so emit format is identical: hw_results.json + quality_results.json.
"""

import json
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from .benchmark_trained_vit import (
    _exit_at_trained,
    _get_block_list_holder,
    _load_loader_trained,
    _run_hw_pass_trained,
    _trained_forward_exit,
)

PRETRAINED_MODEL_ID = "google/vit-large-patch16-224"

# Map task class index -> ImageNet-1k label keyword (substring-matched against the
# model's id2label). Lets us SLICE the matching rows out of the pretrained
# 1000-way classifier so exit heads carry real trained class directions instead
# of random weights. CIFAR-10 order: airplane, automobile, bird, cat, deer, dog,
# frog, horse, ship, truck. (deer has no ImageNet class -> "gazelle" proxy.)
_CIFAR10_IMAGENET_KW = {
    0: "airliner",
    1: "sports car",
    2: "robin",
    3: "tabby",
    4: "gazelle",
    5: "golden retriever",
    6: "bullfrog",
    7: "sorrel",
    8: "container ship",
    9: "trailer truck",
}

# dataset id (HF + bare alias) -> keyword map. cifar100 intentionally absent
# (100 classes, many absent from ImageNet) -> falls back to random-init.
_SLICE_MAPS = {
    "uoft-cs/cifar10": _CIFAR10_IMAGENET_KW,
    "cifar10": _CIFAR10_IMAGENET_KW,
}


def _imagenet_index(id2label: dict, keyword: str):
    """First ImageNet index whose label contains keyword (case-insensitive)."""
    kw = keyword.lower()
    for idx, label in id2label.items():
        if kw in str(label).lower():
            return int(idx)
    return None


def _slice_head(src: nn.Linear, dataset: str, num_labels: int, id2label: dict):
    """Slice the task's class rows out of the pretrained classifier. Returns
    (weight[num_labels,h], bias[num_labels]) or None if no map / a class can't
    be matched."""
    kwmap = _SLICE_MAPS.get(dataset)
    if kwmap is None or len(kwmap) != num_labels:
        return None
    idxs = []
    for c in range(num_labels):
        i = _imagenet_index(id2label, kwmap[c])
        if i is None:
            print(f"[vision.pretrained] no ImageNet match for class {c} (kw={kwmap[c]}); random fallback")
            return None
        idxs.append(i)
    return src.weight.data[idxs].clone(), src.bias.data[idxs].clone()


def _broadcast_or_random_head(model, num_labels: int, model_id: str, dataset: str) -> None:
    """Fill every exit head from the pretrained classifier:
    1. exact size match  -> copy whole head;
    2. size mismatch + slice-map -> slice the task's class rows (real weights);
    3. otherwise          -> leave random-init (HW still valid)."""
    from transformers import ViTForImageClassification  # type: ignore

    clf = ViTForImageClassification.from_pretrained(model_id)
    src = clf.classifier
    if not isinstance(src, nn.Linear):
        print(f"[vision.pretrained] no Linear classifier on {model_id}; exit heads random-init")
        return

    if src.out_features == num_labels:
        sd = src.state_dict()
        for h in model.heads:
            h.load_state_dict(sd)
        print(f"[vision.pretrained] broadcast full pretrained classifier -> {len(model.heads)} exits")
        return

    sliced = _slice_head(src, dataset, num_labels, clf.config.id2label)
    if sliced is None:
        print(f"[vision.pretrained] head {src.out_features} != labels {num_labels}, no slice-map "
              f"for {dataset}; exit heads random-init (HW still valid)")
        return
    w, b = sliced
    for h in model.heads:
        h.weight.data.copy_(w)
        h.bias.data.copy_(b)
    print(f"[vision.pretrained] sliced pretrained head {src.out_features}->{num_labels} "
          f"for {dataset} -> {len(model.heads)} exits (real class directions)")


def _load_pretrained_vit(
    dataset: str,
    n_exits: int,
    num_labels: int,
    *,
    model_id: str = PRETRAINED_MODEL_ID,
    compile_model: bool = False,
):
    """Build MultiExitViT on the pretrained backbone + broadcast head. No LoRA."""
    from GeneralizableSelfDistilationEarlyExitTraining.backends.vision.config import Cfg as _VCfg
    from GeneralizableSelfDistilationEarlyExitTraining.backends.vision.model import build_model

    vit_cfg = _VCfg(dataset=dataset, mode="joint", n_exits=n_exits, model_id=model_id)
    model = build_model(vit_cfg, num_labels)
    _broadcast_or_random_head(model, num_labels, model_id, dataset)

    model.cuda().eval()
    if compile_model and hasattr(torch, "compile"):
        try:
            holder, attr = _get_block_list_holder(model)
            blocks = getattr(holder, attr)
            for i in range(len(blocks)):
                blocks[i] = torch.compile(blocks[i])
            print(f"[vision.pretrained] torch.compile enabled per-layer ({len(blocks)} layers)")
        except Exception as e:
            print(f"[vision.pretrained] torch.compile failed: {e}")
    return model


def sweep_hw_pretrained(
    dataset: str,
    exits,
    n_exits: int,
    num_labels: int,
    out_root: Union[str, Path],
    *,
    weight_source: str = "pretrained",
    model_id: str = PRETRAINED_MODEL_ID,
    bench_batch: int = 1,
    warmup_steps: int = 3,
    use_torch_compile: bool = False,
    max_samples: Optional[int] = None,
):
    """HW sweep on the pretrained ViT. mode='joint' (no adapters)."""
    from shared import has_valid_result

    out_root = Path(out_root)
    model = _load_pretrained_vit(
        dataset, n_exits, num_labels, model_id=model_id, compile_model=use_torch_compile,
    )
    loader = _load_loader_trained(dataset, batch=bench_batch, model_id=model_id)
    paths = []
    for k in exits:
        out_path = out_root / f"exit_{k}" / "hw_results.json"
        if has_valid_result(out_path):
            print(f"[skip] hw exists: {out_path}")
            paths.append(out_path)
            continue
        _run_hw_pass_trained(
            model, loader, k, out_path,
            mode="joint", dataset=dataset, weight_source=weight_source, model_id=model_id,
            warmup_steps=warmup_steps, max_samples=max_samples,
        )
        paths.append(out_path)
    return paths


def evaluate_quality_pretrained(
    dataset: str,
    force_exit: int,
    n_exits: int,
    num_labels: int,
    out_dir: Union[str, Path],
    *,
    weight_source: str = "pretrained",
    model_id: str = PRETRAINED_MODEL_ID,
    bench_batch: int = 1,
    max_samples: Optional[int] = None,
) -> Path:
    """Quality pass on the pretrained ViT at one exit. mode='joint'."""
    import numpy as np
    from sklearn.metrics import f1_score as sk_f1

    from shared import compute_ece

    out_dir = Path(out_dir)
    out_path = out_dir / "quality_results.json"

    model = _load_pretrained_vit(dataset, n_exits, num_labels, model_id=model_id, compile_model=False)
    loader = _load_loader_trained(dataset, batch=bench_batch, model_id=model_id)
    exit_layer_1idx = model.exit_layers[force_exit]

    preds, labels, confidences, corrects = [], [], [], []
    top5_hits = 0
    n = 0
    for inputs, lbl in tqdm(loader, desc=f"Q  {dataset}/pretrained exit={force_exit}"):
        inputs = inputs.cuda(non_blocking=True)
        lbl = lbl.cuda(non_blocking=True)
        with torch.no_grad(), _exit_at_trained(model, exit_layer_1idx):
            logits = _trained_forward_exit(model, inputs, force_exit)
        pred = logits.argmax(-1)
        preds.extend(pred.cpu().tolist())
        labels.extend(lbl.cpu().tolist())
        _accumulate_conf(logits, lbl, pred, confidences, corrects)
        top5_hits += _top5_hits(logits, lbl)
        n += inputs.shape[0]
        if max_samples is not None and n >= max_samples:
            break

    preds_np = np.array(preds)
    labels_np = np.array(labels)
    acc = float((preds_np == labels_np).mean()) if len(preds_np) else 0.0
    top5_acc = top5_hits / len(preds_np) if len(preds_np) else 0.0
    f1 = float(sk_f1(labels_np, preds_np, average="weighted", zero_division=0)) if len(preds_np) else 0.0
    ece = compute_ece(np.array(confidences), np.array(corrects)) if confidences else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "main_metric": "top1",
            "task": dataset,
            "mode": "joint",
            "weight_source": weight_source,
            "force_exit": force_exit,
            "model_id": model_id,
            "n_samples": len(preds),
            "top1": round(acc, 6),
            "top5": round(top5_acc, 6),
            "acc": round(acc, 6),
            "f1": round(f1, 6),
            "ece": round(ece, 6),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[evaluate_quality_pretrained] top1={acc:.4f} top5={top5_acc:.4f} f1={f1:.4f} ece={ece:.4f}")
    return out_path


def _accumulate_conf(logits, lbl, pred, confidences, corrects) -> None:
    """Append softmax confidence + correctness for ECE. Skips degenerate 1-logit."""
    if logits.shape[-1] <= 1:
        return
    probs = torch.softmax(logits.float(), dim=-1)
    confidences.extend(probs.max(-1).values.cpu().tolist())
    corrects.extend((pred == lbl).cpu().tolist())


def _top5_hits(logits, lbl) -> int:
    """Count top-5 correct predictions in this batch."""
    if logits.shape[-1] <= 1:
        return 0
    probs = torch.softmax(logits.float(), dim=-1)
    top5 = probs.topk(min(5, logits.shape[-1]), dim=-1).indices
    return int((top5 == lbl.unsqueeze(-1)).any(-1).sum().item())
