"""Trained MultiExitYolo (gelan-m-ee) per-exit benchmark.

Loads from HF repo pushed by GeneralizableSelfDistilationEarlyExitTraining.
Per-mode storage layout in repo:
    joint     : joint/full_model.pt
    pairwise  : teacher/head_<deep>.pt + pair_e<k>/head_<k>.pt
    cascade   : cascade_teacher/head_<deep>.pt + cascade_e<k>/head_<k>.pt

Emits identical hw_results.json / quality_results.json schema as the legacy
pretrained-weights benchmark.

Per-exit fair HW timing: NOTE — backbone forward isn't truncated here. Full
backbone + head[k] + scale[s] is timed, mirroring how the trained net is
actually called. Pure layer-by-layer truncation needs the EXIT_MAX_DEPTH map
from the legacy path and is left for a follow-up if needed.
"""

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

from shared import BenchmarkProfiler, load_env  # noqa: E402

load_env()

SUB_EXIT_NAMES = ["P3", "P4", "P5"]


# ---- stage label resolution -------------------------------------------------
def _stage_label_for_exit(mode: str, exit_k: int, deepest: int) -> Optional[str]:
    if mode == "joint":
        return "joint"
    if mode == "pairwise":
        return "teacher" if exit_k == deepest else f"pair_e{exit_k}"
    if mode == "cascade":
        return "cascade_teacher" if exit_k == deepest else f"cascade_e{exit_k}"
    return None


# ---- model load -------------------------------------------------------------
def _load_trained_yolo(
    repo_id: str,
    mode: str,
    n_exits: int,
    ee_yaml: Path,
    weights: Optional[Path] = None,
    nc: int = 80,
    *,
    hf_token: Optional[str] = None,
    compile_model: bool = False,
):
    from huggingface_hub import snapshot_download

    from GeneralizableSelfDistilationEarlyExitTraining.backends.yolo.config import (
        YoloCfg,
    )
    from GeneralizableSelfDistilationEarlyExitTraining.backends.yolo.model import (
        build_model,
    )
    from GeneralizableSelfDistilationEarlyExitTraining.backends.yolo import adapters as _ya

    token = hf_token or os.environ.get("HF_TOKEN")
    local = Path(snapshot_download(repo_id=repo_id, token=token, repo_type="model"))

    cfg = YoloCfg(mode=mode, n_exits=n_exits, ee_yaml=Path(ee_yaml), weights=weights, nc=nc)
    model = build_model(cfg, cfg.ee_yaml, cfg.weights, cfg.nc)

    if mode == "joint":
        full_path = local / "joint" / "full_model.pt"
        if not full_path.exists():
            raise FileNotFoundError(f"missing joint/full_model.pt in {repo_id}")
        sd = torch.load(full_path, map_location="cpu")
        model.net.load_state_dict(sd, strict=False)
    else:
        _ya.attach(model, cfg)
        deepest = n_exits - 1
        for k in range(n_exits):
            stage_label = _stage_label_for_exit(mode, k, deepest)
            stage_dir = local / stage_label
            head_pt = stage_dir / f"head_{k}.pt"
            if head_pt.exists():
                model.heads[k].load_state_dict(torch.load(head_pt, map_location="cpu"))
            else:
                print(f"[yolo.benchmark.trained] missing {stage_label}/head_{k}.pt — exit {k} left untrained")

    model.cuda().eval()
    if compile_model and hasattr(torch, "compile"):
        try:
            from early_exit.model import _DETECT_TYPES  # type: ignore

            n_compiled = 0
            for idx, m in enumerate(model.net.model):
                if isinstance(m, _DETECT_TYPES):
                    continue
                model.net.model[idx] = torch.compile(m)
                n_compiled += 1
            print(f"[yolo.benchmark.trained] torch.compile enabled on {n_compiled} backbone submodules")
        except Exception as e:
            print(f"[yolo.benchmark.trained] compile failed: {e}")
    return model


def _set_exit_enabled(model, exit_k: int, n_exits: int):
    """For LoRA modes: enable only exit_k's adapters."""
    try:
        from GeneralizableSelfDistilationEarlyExitTraining.backends.yolo import (
            adapters as _ya,
        )
        for i in range(n_exits):
            _ya.set_exit(model, i, enabled=(i == exit_k), trainable=False)
    except Exception:
        pass  # joint mode (no LoRA attached) — silently no-op


def _trained_forward(model, imgs, exit_k: int, sub_s: Optional[int] = None):
    """Backbone + head[exit_k]; optionally only scale s of head[exit_k]."""
    y = model.backbone_feats(imgs)
    head = model.heads[exit_k]
    head_input = model.net._resolve_input(head, None, y)
    if sub_s is None:
        return head(head_input)
    return torch.cat(
        (head.cv2[sub_s](head_input[sub_s]), head.cv3[sub_s](head_input[sub_s])),
        dim=1,
    )


def _load_loader(dataset: str, data_dir: Union[str, Path], img_size: int, batch: int):
    """Reuse legacy YOLO loader."""
    from .benchmark import _load_loader as _ll

    return _ll(dataset, data_dir, img_size, batch)


# ---- HW sweep ----------------------------------------------------------------
def _run_hw_pass_trained(
    model,
    loader,
    exit_k: int,
    sub_s: Optional[int],
    out_path: Path,
    *,
    mode: str,
    dataset: str,
    weight_source: str,
    model_id: str,
    warmup_steps: int,
    n_samples: int,
) -> Path:
    sub_tag = f"_{SUB_EXIT_NAMES[sub_s]}" if sub_s is not None else "_all"
    with BenchmarkProfiler(
        out_path=out_path,
        task=dataset,
        strategy=weight_source,
        threshold=exit_k,
        warmup_steps=warmup_steps,
        meta={
            "force_exit": exit_k,
            "sub_exit": sub_s,
            "sub_exit_name": SUB_EXIT_NAMES[sub_s] if sub_s is not None else "all",
            "weight_source": weight_source,
            "mode": mode,
            "model_id": model_id,
        },
    ) as prof:
        n_done = 0
        for batch in tqdm(loader, desc=f"HW {dataset}/{mode} exit={exit_k}{sub_tag} ({weight_source})"):
            imgs = batch[0].cuda(non_blocking=True).float() / 255.0
            with prof.timer() as t:
                with torch.no_grad():
                    _ = _trained_forward(model, imgs, exit_k, sub_s)
            prof.log_sample(
                prediction=None,
                label=None,
                forward_sec=t.elapsed_s,
                end_to_end_sec=t.elapsed_s,
                exit_layer=exit_k,
                sub_exit=sub_s,
            )
            n_done += 1
            if n_samples is not None and n_done >= n_samples:
                break
    return out_path


def sweep_hw_trained(
    repo_id: str,
    dataset: str,
    mode: str,
    exits: List[int],
    sub_exits: List[Optional[int]],
    n_exits: int,
    ee_yaml: Path,
    data_dir: Union[str, Path],
    out_root: Union[str, Path],
    *,
    pretrained_weights: Optional[Path] = None,
    weight_source: str = "trained",
    img_size: int = 640,
    bench_batch: int = 1,
    warmup_steps: int = 3,
    use_torch_compile: bool = False,
    n_samples: int = 200,
):
    from shared import has_valid_result

    out_root = Path(out_root)
    model = _load_trained_yolo(
        repo_id=repo_id,
        mode=mode,
        n_exits=n_exits,
        ee_yaml=ee_yaml,
        weights=pretrained_weights,
        compile_model=use_torch_compile,
    )
    loader = _load_loader(dataset, data_dir, img_size, bench_batch)
    paths = []
    for ei in exits:
        _set_exit_enabled(model, ei, n_exits)
        for s in sub_exits:
            sub_tag = SUB_EXIT_NAMES[s] if s is not None else "all"
            run_dir = out_root / f"exit_{ei}_{sub_tag}"
            out_path = run_dir / "hw_results.json"
            if has_valid_result(out_path):
                print(f"[skip] hw exists: {out_path}")
                paths.append(out_path)
                continue
            _run_hw_pass_trained(
                model, loader, ei, s, out_path,
                mode=mode, dataset=dataset, weight_source=weight_source,
                model_id=repo_id, warmup_steps=warmup_steps, n_samples=n_samples,
            )
            paths.append(out_path)
    return paths


def evaluate_quality_trained(
    repo_id: str,
    dataset: str,
    mode: str,
    force_exit: int,
    sub_exit: Optional[int],
    n_exits: int,
    ee_yaml: Path,
    data_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    pretrained_weights: Optional[Path] = None,
    weight_source: str = "trained",
    img_size: int = 640,
    bench_batch: int = 1,
    valid_classes: Optional[List[int]] = None,
    max_samples: Optional[int] = None,
) -> Path:
    """Defer to legacy AnyTimeYolo.evaluate_quality after loading trained model.

    The legacy quality path takes the model file from weights_path. We dump the
    trained model.net state_dict to a temp .pt so the legacy loader picks it up.
    """
    out_dir = Path(out_dir)
    out_path = out_dir / "quality_results.json"

    model = _load_trained_yolo(
        repo_id=repo_id, mode=mode, n_exits=n_exits, ee_yaml=ee_yaml,
        weights=pretrained_weights, compile_model=False,
    )
    _set_exit_enabled(model, force_exit, n_exits)

    import tempfile

    from .benchmark import evaluate_quality as _legacy_eval

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        torch.save({"model": model.net.state_dict()}, tmp.name)
        tmp_path = Path(tmp.name)
    try:
        _legacy_eval(
            ee_yaml=ee_yaml,
            weights_path=tmp_path,
            dataset=dataset,
            force_exit=force_exit,
            data_dir=data_dir,
            out_dir=out_dir,
            sub_exit=sub_exit,
            weight_source=weight_source,
            img_size=img_size,
            bench_batch=bench_batch,
            valid_classes=valid_classes,
            max_samples=max_samples,
        )
        # patch model_id + mode into emitted json
        if out_path.exists():
            d = json.loads(out_path.read_text())
            d["mode"] = mode
            d["model_id"] = repo_id
            out_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)
    return out_path
