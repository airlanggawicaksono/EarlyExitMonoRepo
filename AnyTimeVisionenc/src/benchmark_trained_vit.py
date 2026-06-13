"""Trained ViT (multi-exit) per-exit benchmark. Same emit format as MSDNet path:
- hw_results.json (latency + memory + energy + per-PID + FLOPs/params)
- quality_results.json (top1/top5/f1/ece)

Loads from HF Hub repos pushed by
GeneralizableSelfDistilationEarlyExitTraining.sync.push_ckpts_to_hf
naming: {HF_USER}/selfdistill-vision-{dataset_slug}-{mode}

Per-exit fair HW timing: truncate ViT encoder layers to exit_layers[k].
"""

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

from shared import BenchmarkProfiler, load_env, model_metrics  # noqa: E402

load_env()


# ---- stage label resolution -------------------------------------------------
def _stage_label_for_exit(mode: str, exit_k: int, deepest: int) -> Optional[str]:
    if mode == "joint":
        return "joint"
    if mode == "pairwise":
        return "teacher" if exit_k == deepest else f"pair_e{exit_k}"
    if mode == "segd":
        return "segd_teacher" if exit_k == deepest else f"segd_e{exit_k}"
    return None


# ---- model load -------------------------------------------------------------
def _load_trained_vit(
    repo_id: str,
    mode: str,
    dataset: str,
    n_exits: int,
    num_labels: int,
    *,
    hf_token: Optional[str] = None,
    compile_model: bool = False,
):
    from huggingface_hub import snapshot_download

    from GeneralizableSelfDistilationEarlyExitTraining.backends.vision.config import (
        Cfg as _VCfg,
    )
    from GeneralizableSelfDistilationEarlyExitTraining.backends.vision.model import (
        build_model,
    )
    from GeneralizableSelfDistilationEarlyExitTraining.backends.vision import adapters as _va

    token = hf_token or os.environ.get("HF_TOKEN")
    local = Path(snapshot_download(repo_id=repo_id, token=token, repo_type="model"))

    vit_cfg = _VCfg(dataset=dataset, mode=mode, n_exits=n_exits)
    model = build_model(vit_cfg, num_labels)

    if mode == "joint":
        full_path = local / "joint" / "full_model.pt"
        if not full_path.exists():
            raise FileNotFoundError(f"missing joint/full_model.pt in {repo_id}")
        sd = torch.load(full_path, map_location="cpu")
        model.load_state_dict(sd)
    else:
        _va.attach(model, vit_cfg)
        deepest = n_exits - 1
        for k in range(n_exits):
            stage_label = _stage_label_for_exit(mode, k, deepest)
            stage_dir = local / stage_label
            if not stage_dir.exists():
                print(f"[vision.benchmark.trained] missing stage {stage_label}; exit {k} skipped")
                continue
            try:
                _va.load_adapter(model, k, stage_dir / "adapter")
            except Exception as e:
                print(f"[vision.benchmark.trained] adapter load failed for exit {k}: {e}")
                continue
            head_pt = stage_dir / f"head_{k}.pt"
            if head_pt.exists():
                model.heads[k].load_state_dict(torch.load(head_pt, map_location="cpu"))
            else:
                print(f"[vision.benchmark.trained] missing head_{k}.pt in {stage_label}")

    model.cuda().eval()
    if compile_model and hasattr(torch, "compile"):
        try:
            holder, attr = _get_block_list_holder(model)
            blocks = getattr(holder, attr)
            for i in range(len(blocks)):
                blocks[i] = torch.compile(blocks[i])
            print(f"[vision.benchmark.trained] torch.compile enabled per-layer ({len(blocks)} layers)")
        except Exception as e:
            print(f"[vision.benchmark.trained] torch.compile failed: {e}")
    return model


def _get_block_list_holder(model):
    """Return (holder, attr) where getattr(holder, attr) is the ModuleList of ViT
    transformer blocks. Robust across:
      - peft wrapping            (backbone.base_model.model)
      - transformers <5          (ViTModel.encoder.layer)
      - transformers >=5         (ViTModel.layers — flattened, no .encoder)"""
    bb = model.backbone
    if hasattr(bb, "base_model") and hasattr(bb.base_model, "model"):
        bb = bb.base_model.model
    # transformers >=5: blocks directly on the model
    if hasattr(bb, "layers") and isinstance(bb.layers, nn.ModuleList):
        return bb, "layers"
    # transformers <5: wrapped in .encoder.{layer,layers}
    enc = getattr(bb, "encoder", None)
    if enc is not None:
        for attr in ("layer", "layers"):
            ml = getattr(enc, attr, None)
            if isinstance(ml, nn.ModuleList):
                return enc, attr
    # generic fallback: any submodule holding a ModuleList of >1 blocks
    for m in bb.modules():
        for attr in ("layers", "layer"):
            ml = getattr(m, attr, None)
            if isinstance(ml, nn.ModuleList) and len(ml) > 1:
                return m, attr
    raise AttributeError("ViT transformer-block ModuleList not found on backbone")


@contextlib.contextmanager
def _exit_at_trained(model, exit_layer_1idx: int):
    """exit_layer_1idx is 1-indexed: keep transformer blocks [0:exit_layer_1idx]."""
    holder, attr = _get_block_list_holder(model)
    full_layers = getattr(holder, attr)
    setattr(holder, attr, nn.ModuleList(list(full_layers)[: exit_layer_1idx]))
    try:
        yield
    finally:
        setattr(holder, attr, full_layers)


def _activate_for_exit(model, mode: str, exit_k: int):
    if mode == "joint":
        return
    try:
        from GeneralizableSelfDistilationEarlyExitTraining.backends.vision import (
            adapters as _va,
        )
        _va.activate(model, exit_k)
    except Exception as e:
        print(f"[vision.benchmark.trained] activate(exit_{exit_k}) failed: {e}")


def _trained_forward_exit(model, pixel_values, exit_k: int):
    """Run truncated ViT forward + heads[exit_k]. Caller manages adapter."""
    out = model.backbone(pixel_values=pixel_values, output_hidden_states=True)
    hs = out.hidden_states
    # Under truncation, hs has len(encoder.layer)+1 entries. Take the last
    # (== hs[exit_layer_1idx]) which matches model.exit_layers[exit_k].
    feat = model.dropout(hs[-1][:, 0])
    return model.heads[exit_k](feat)


# ---- loader ------------------------------------------------------------------
def _load_loader_trained(dataset: str, batch: int = 1, model_id: str = "google/vit-large-patch16-224"):
    """Eval loader using HF datasets + AutoImageProcessor — same as train side."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor  # type: ignore

    from GeneralizableSelfDistilationEarlyExitTraining.backends.vision.data import (
        _ALIASES, _IMG_KEY, _LBL_KEY,
    )

    name = _ALIASES.get(dataset, dataset)
    img_key, lbl_key = _IMG_KEY[name], _LBL_KEY[name]
    # ImageNet-1k 'test' split is UNLABELED — only 'validation' has labels.
    # CIFAR etc. ship labels in 'test'.
    split = "validation" if name == "imagenet-1k" else "test"
    # bare "imagenet-1k" is deprecated on HF; pull the gated canonical repo while
    # keeping `name` for the key/split maps above.
    load_id = "ILSVRC/imagenet-1k" if name == "imagenet-1k" else name
    ds = load_dataset(load_id, split=split)
    proc = AutoImageProcessor.from_pretrained(model_id)

    def _collate(rows):
        imgs = [r[img_key].convert("RGB") for r in rows]
        px = proc(imgs, return_tensors="pt")["pixel_values"]
        labels = torch.tensor([r[lbl_key] for r in rows], dtype=torch.long)
        return px, labels

    return DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=_collate)


# ---- HW sweep ----------------------------------------------------------------
def _run_hw_pass_trained(
    model,
    loader,
    force_exit: int,
    out_path: Path,
    *,
    mode: str,
    dataset: str,
    weight_source: str,
    model_id: str,
    warmup_steps: int,
    max_samples: Optional[int] = None,
) -> Path:
    _activate_for_exit(model, mode, force_exit)
    exit_layer_1idx = model.exit_layers[force_exit]
    dummy = torch.zeros((1, 3, 224, 224), device="cuda")
    try:
        with _exit_at_trained(model, exit_layer_1idx):
            mm = model_metrics(model.backbone, dummy)
    except Exception as e:
        print(f"[vision.benchmark.trained] model_metrics skipped: {e}")
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
            "mode": mode,
            "model_id": model_id,
            "exit_layer_1idx": exit_layer_1idx,
            **mm,
        },
    ) as prof:
        n_done = 0
        for batch in tqdm(loader, desc=f"HW {dataset}/{mode} exit={force_exit} ({weight_source})"):
            inputs = batch[0].cuda(non_blocking=True)
            with prof.timer() as t:
                with torch.no_grad(), _exit_at_trained(model, exit_layer_1idx):
                    _ = _trained_forward_exit(model, inputs, force_exit)
            prof.log_sample(
                prediction=None,
                label=None,
                forward_sec=t.elapsed_s,
                end_to_end_sec=t.elapsed_s,
                exit_layer=force_exit,
            )
            n_done += 1
            if max_samples is not None and n_done >= max_samples:
                break
    return out_path


def sweep_hw_trained(
    repo_id: str,
    dataset: str,
    mode: str,
    exits,
    n_exits: int,
    num_labels: int,
    out_root: Union[str, Path],
    *,
    weight_source: str = "trained",
    bench_batch: int = 1,
    warmup_steps: int = 3,
    use_torch_compile: bool = False,
    max_samples: Optional[int] = None,
):
    from shared import has_valid_result

    out_root = Path(out_root)
    model = _load_trained_vit(
        repo_id=repo_id,
        mode=mode,
        dataset=dataset,
        n_exits=n_exits,
        num_labels=num_labels,
        compile_model=use_torch_compile,
    )
    loader = _load_loader_trained(dataset, batch=bench_batch)
    paths = []
    for k in exits:
        run_dir = out_root / f"exit_{k}"
        out_path = run_dir / "hw_results.json"
        if has_valid_result(out_path):
            print(f"[skip] hw exists: {out_path}")
            paths.append(out_path)
            continue
        _run_hw_pass_trained(
            model, loader, k, out_path,
            mode=mode, dataset=dataset, weight_source=weight_source, model_id=repo_id,
            warmup_steps=warmup_steps, max_samples=max_samples,
        )
        paths.append(out_path)
    return paths


# ---- Quality -----------------------------------------------------------------
def evaluate_quality_trained(
    repo_id: str,
    dataset: str,
    mode: str,
    force_exit: int,
    n_exits: int,
    num_labels: int,
    out_dir: Union[str, Path],
    *,
    weight_source: str = "trained",
    bench_batch: int = 1,
    max_samples: Optional[int] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "quality_results.json"

    model = _load_trained_vit(
        repo_id=repo_id,
        mode=mode,
        dataset=dataset,
        n_exits=n_exits,
        num_labels=num_labels,
        compile_model=False,
    )
    loader = _load_loader_trained(dataset, batch=bench_batch)
    _activate_for_exit(model, mode, force_exit)
    exit_layer_1idx = model.exit_layers[force_exit]

    import numpy as np
    from shared import compute_ece
    from sklearn.metrics import f1_score as sk_f1

    preds, labels, confidences, corrects = [], [], [], []
    top5_hits = 0
    n = 0
    for inputs, lbl in tqdm(loader, desc=f"Q  {dataset}/{mode} exit={force_exit} ({weight_source})"):
        inputs = inputs.cuda(non_blocking=True)
        lbl = lbl.cuda(non_blocking=True)
        with torch.no_grad(), _exit_at_trained(model, exit_layer_1idx):
            logits = _trained_forward_exit(model, inputs, force_exit)
        pred = logits.argmax(-1)
        preds.extend(pred.cpu().tolist())
        labels.extend(lbl.cpu().tolist())
        if logits.shape[-1] > 1:
            probs = torch.softmax(logits.float(), dim=-1)
            conf = probs.max(-1).values.cpu().tolist()
            top5 = probs.topk(min(5, logits.shape[-1]), dim=-1).indices
            top5_hits += (top5 == lbl.unsqueeze(-1)).any(-1).sum().item()
            confidences.extend(conf)
            corrects.extend((pred == lbl).cpu().tolist())
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
            "mode": mode,
            "weight_source": weight_source,
            "force_exit": force_exit,
            "model_id": repo_id,
            "n_samples": len(preds),
            "top1": round(acc, 6),
            "top5": round(top5_acc, 6),
            "acc": round(acc, 6),
            "f1": round(f1, 6),
            "ece": round(ece, 6),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[evaluate_quality_trained] top1={acc:.4f} top5={top5_acc:.4f} f1={f1:.4f} ece={ece:.4f}")
    return out_path
