"""YOLOv9 gelan-s-ee per-exit benchmark. TWO passes:

profile_hw(...)        -> hw_results.json       (latency + memory + energy)
evaluate_quality(...)  -> quality_results.json  (mAP@0.5 and mAP@0.5:0.95 via ap_per_class)
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
from typing import List, Optional, Tuple, Union

import torch
from tqdm import tqdm

_HERE  = Path(__file__).resolve().parent     # AnyTimeYolo/src/
_MODEL = _HERE.parent
_REPO  = _MODEL.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_MODEL / "model" / "yolov9"))
sys.path.insert(0, str(_HERE))  # for `early_exit` package at AnyTimeYolo/src/early_exit/

from shared import BenchmarkProfiler, load_env, model_metrics  # noqa: E402

load_env()

HF_TOKEN = os.environ.get("HF_TOKEN")

# ---- Architectural facts of gelan-m-ee.yaml ---------------------------------
# (these are properties of the EE yaml file, not user-tunable knobs)
# 6 exit heads at modules 22..27 (E0..E5). EXIT_MAX_DEPTH[k] = deepest backbone
# module a head's inputs need. E5 (idx 27, inputs [15,18,21]) is the full-FPN
# final head — the native gelan-m head that yields real mAP. Missing key 5 used
# to KeyError out the native exit, leaving only untrained shallow heads.
EXIT_MAX_DEPTH = {0: 8, 1: 9, 2: 15, 3: 18, 4: 21, 5: 21}
EXIT_HEAD_OFFSET = 22
SUB_EXIT_NAMES = ["P3", "P4", "P5"]


def _download_pretrained(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[yolo.benchmark] downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def _broadcast_upstream_head(state, model):
    """Upstream gelan-s.pt ships ONE DDetect at the end of the model list. EE adds
    five DDetects at indices EXIT_HEAD_OFFSET..EXIT_HEAD_OFFSET+N_EXITS-1, where
    only the final exit shares input topology with upstream. With strict=False,
    only the upstream head's slot loads; the rest stay random — quality is zero
    for those exits and meaningless for the one that loads (feature mismatch).

    Broadcast the upstream head's params into every EE head slot, per-tensor with
    a shape check: copy where shapes agree (DFL, classifier convs at matching
    scales), skip silently where channel counts diverge. Result: every exit gets
    a *plausible* initialization derived from the upstream head, so quality eval
    is not just measuring random noise.
    """
    from early_exit.model import _DETECT_TYPES  # type: ignore

    head_idxs_in_ckpt = set()
    for k in state.keys():
        parts = k.split(".")
        if len(parts) >= 4 and parts[0] == "model" and parts[2] in ("cv2", "cv3"):
            try:
                head_idxs_in_ckpt.add(int(parts[1]))
            except ValueError:
                pass
    if not head_idxs_in_ckpt:
        return state
    upstream_idx = max(head_idxs_in_ckpt)
    prefix = f"model.{upstream_idx}."
    upstream_head_params = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}

    ee_head_idxs = [i for i, m in enumerate(model.model) if isinstance(m, _DETECT_TYPES)]
    new_state = dict(state)
    n_copied_total = 0
    n_skipped_total = 0
    for target_idx in ee_head_idxs:
        if target_idx == upstream_idx:
            continue
        target_sd = model.model[target_idx].state_dict()
        for subkey, val in upstream_head_params.items():
            if subkey in target_sd and tuple(target_sd[subkey].shape) == tuple(val.shape):
                new_state[f"model.{target_idx}.{subkey}"] = val.clone()
                n_copied_total += 1
            else:
                n_skipped_total += 1
    print(
        f"[yolo.benchmark] broadcast upstream head model.{upstream_idx} -> "
        f"{[i for i in ee_head_idxs if i != upstream_idx]} "
        f"(params copied={n_copied_total}, shape-skipped={n_skipped_total})"
    )
    return new_state


def _load_ee_model(ee_yaml: Path, weights_path: Path, weight_source: str, compile_model: bool = False):
    """Load EarlyExitModel from yaml + weights."""
    # Force re-import: `models` and `utils` names collide with Vision's reference/ when both run in same session
    for _mod in list(sys.modules):
        if _mod == "models" or _mod.startswith("models.") or _mod == "utils" or _mod.startswith("utils."):
            del sys.modules[_mod]
    from early_exit.model import EarlyExitModel  # type: ignore

    model = EarlyExitModel(str(ee_yaml), ch=3)
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    if weight_source == "pretrained":
        state = _broadcast_upstream_head(state, model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[yolo.benchmark] load (ws={weight_source}) "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    model.cuda().eval()
    # Per-submodule compile: benchmark forward bypasses model.__call__ and dispatches
    # submodules directly via _forward_to_exit's Python loop. Wrapping outer model
    # would be a no-op, so compile each backbone module individually. Heads skipped
    # because sub_exit indexing path (m.cv2[s], m.cv3[s]) is awkward to trace.
    if compile_model and hasattr(torch, "compile"):
        try:
            n_compiled = 0
            for idx in range(min(EXIT_HEAD_OFFSET, len(model.model))):
                model.model[idx] = torch.compile(model.model[idx])
                n_compiled += 1
            print(f"[yolo.benchmark] torch.compile enabled on {n_compiled} backbone submodules")
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
    val_path = next(
        (base / p for p in ("val", "valid/images", "valid", "images/val2017", "val2017") if (base / p).exists()),
        None,
    )
    if val_path is None:
        # Roboflow nested: datasets/<ds>/<slug>-<ver>/valid/images/
        _cands = list(base.rglob("images/val2017")) + list(base.rglob("val2017")) + list(base.rglob("valid/images")) + list(base.rglob("valid"))
        val_path = _cands[0] if _cands else base / "val"
    val_dataset = LoadImagesAndLabels(
        str(val_path),
        img_size=img_size,
        batch_size=batch,
        augment=False,
        hyp=None,
        rect=True,
    )
    return torch.utils.data.DataLoader(val_dataset, batch_size=batch, shuffle=False)


def _decode_sub_exit(raw: torch.Tensor, detect_head, img_h: int) -> torch.Tensor:
    """Decode single-scale raw head output (B, no, H, W) -> (B, 4+nc, H*W) for NMS.

    raw = torch.cat((cv2[s](feat), cv3[s](feat)), dim=1) from _forward_to_exit sub_exit path.
    Applies DFL softmax, dist2bbox, and stride scaling using the detect head's DFL module.
    """
    from utils.tal.anchor_generator import dist2bbox, make_anchors  # type: ignore

    B, no, H, W = raw.shape
    reg_max4 = detect_head.reg_max * 4
    flat = raw.view(B, no, H * W)
    box_flat = flat[:, :reg_max4, :]   # (B, reg_max*4, H*W)
    cls_flat = flat[:, reg_max4:, :]   # (B, nc, H*W)

    box_decoded = detect_head.dfl(box_flat)  # (B, 4, H*W)

    stride_t = torch.tensor([float(img_h) / H], device=raw.device)
    anchors, strides = (t.T for t in make_anchors([raw], stride_t, 0.5))
    # anchors: (2, H*W), strides: (1, H*W)

    dbox = dist2bbox(box_decoded, anchors.unsqueeze(0), xywh=True, dim=1) * strides
    return torch.cat((dbox, cls_flat.sigmoid()), 1)  # (B, 4+nc, H*W)


def _process_batch(detections, labels, iouv):
    """IoU-matching for mAP stats (inlined from yolov9/val.py)."""
    import numpy as np
    from utils.metrics import box_iou  # type: ignore

    correct = np.zeros((detections.shape[0], iouv.shape[0]), dtype=bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


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

    try:
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
                    forward_sec=t.elapsed_s,
                    end_to_end_sec=t.elapsed_s,   # one-shot backend: e2e == forward
                    exit_layer=force_exit,
                    sub_exit=sub_exit,
                )
                n += 1
                if n >= n_samples:
                    break
    except Exception as exc:
        import traceback
        from shared import has_valid_result
        if not has_valid_result(out_path):
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps({
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "dataset": dataset,
                    "weight_source": weight_source,
                    "force_exit": force_exit,
                    "sub_exit": sub_exit,
                }, indent=2),
                encoding="utf-8",
            )
        print(f"[profile_hw] ERROR exit={force_exit} sub={sub_exit} {dataset}: {exc}")
    return out_path


# =============================================================================
# Sweep — load model + per-submodule compile ONCE, iterate (exit, sub_exit).
# =============================================================================
def sweep_hw_all_exits(
    ee_yaml: Union[str, Path],
    weights_path: Union[str, Path],
    dataset: str,
    exits: List[int],
    sub_exits: List[Optional[int]],
    data_dir: Union[str, Path],
    out_root: Union[str, Path],
    *,
    weight_source: str = "pretrained",
    img_size: int = 640,
    bench_batch: int = 1,
    warmup_steps: int = 3,
    use_torch_compile: bool = True,
    n_samples: int = 200,
) -> List[Path]:
    """Per-submodule compile cost paid once across the entire (exit x sub_exit) grid.

    profile_hw still works but reloads + recompiles per call; prefer this for sweeps.
    """
    from shared import has_valid_result

    out_root = Path(out_root)
    model = _load_ee_model(Path(ee_yaml), Path(weights_path), weight_source, compile_model=use_torch_compile)
    loader = _load_loader(dataset, data_dir, img_size, bench_batch)
    device = next(model.parameters()).device

    dummy = torch.zeros((1, 3, img_size, img_size), device=device)
    try:
        mm = model_metrics(model, dummy)
    except Exception as e:
        print(f"[yolo.benchmark] model_metrics skipped: {e}")
        mm = {}

    paths: List[Path] = []
    for ei in exits:
        for s in sub_exits:
            sub_tag = f"_{SUB_EXIT_NAMES[s]}" if s is not None else "_all"
            sub_name = SUB_EXIT_NAMES[s] if s is not None else "all"
            run_dir = out_root / f"exit_{ei}{sub_tag}"
            out_path = run_dir / "hw_results.json"
            if has_valid_result(out_path):
                print(f"[skip] hw exists: {out_path}")
                paths.append(out_path)
                continue
            try:
                n = 0
                with BenchmarkProfiler(
                    out_path=out_path,
                    task=dataset,
                    strategy=weight_source,
                    threshold=f"E{ei}{sub_tag}",
                    warmup_steps=warmup_steps,
                    meta={
                        "force_exit": ei,
                        "sub_exit": s,
                        "sub_exit_name": sub_name,
                        "weight_source": weight_source,
                        **mm,
                    },
                ) as prof:
                    desc = f"HW {dataset} E{ei}{sub_tag} ({weight_source})"
                    for batch in tqdm(loader, desc=desc):
                        imgs = batch[0].to(device).float() / 255.0
                        with prof.timer() as t:
                            with torch.no_grad():
                                _ = _forward_to_exit(model, imgs, ei, s)
                        prof.log_sample(
                            prediction=None,
                            label=None,
                            forward_sec=t.elapsed_s,
                            end_to_end_sec=t.elapsed_s,
                    end_to_end_sec=t.elapsed_s,   # one-shot backend: e2e == forward
                            exit_layer=ei,
                            sub_exit=s,
                        )
                        n += 1
                        if n >= n_samples:
                            break
                paths.append(out_path)
            except Exception as exc:
                import traceback
                if not has_valid_result(out_path):
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(
                        json.dumps({
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "dataset": dataset,
                            "weight_source": weight_source,
                            "force_exit": ei,
                            "sub_exit": s,
                        }, indent=2),
                        encoding="utf-8",
                    )
                print(f"[sweep_hw_all_exits] ERROR exit={ei} sub={s} {dataset}: {exc}")
                paths.append(out_path)
    return paths


# =============================================================================
# Quality pass — mAP@0.5 and mAP@0.5:0.95 per (exit, sub_exit).
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
    bench_batch: int = 8,
    valid_classes: Optional[List[int]] = None,
    max_samples: Optional[int] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "quality_results.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_tag = SUB_EXIT_NAMES[sub_exit] if sub_exit is not None else "all"
    try:
        from utils.dataloaders import LoadImagesAndLabels  # type: ignore
        from utils.general import non_max_suppression, xywh2xyxy  # type: ignore
        from utils.metrics import ap_per_class  # type: ignore

        model = _load_ee_model(
            Path(ee_yaml), Path(weights_path), weight_source, compile_model=False
        )
        device = next(model.parameters()).device
        real = model._orig_mod if hasattr(model, "_orig_mod") else model
        detect_head = real.model[EXIT_HEAD_OFFSET + force_exit]

        _base = Path(data_dir)
        _val_path = next(
            (_base / p for p in ("val", "valid/images", "valid", "images/val2017", "val2017") if (_base / p).exists()),
            None,
        )
        if _val_path is None:
            _cands = list(_base.rglob("images/val2017")) + list(_base.rglob("val2017")) + list(_base.rglob("valid/images")) + list(_base.rglob("valid"))
            _val_path = _cands[0] if _cands else _base / "val"
        val_dataset = LoadImagesAndLabels(
            str(_val_path),
            img_size=img_size,
            batch_size=bench_batch,
            augment=False,
            hyp=None,
            rect=False,
        )
        loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=bench_batch,
            shuffle=False,
            collate_fn=LoadImagesAndLabels.collate_fn,
            num_workers=0,
        )

        iouv = torch.linspace(0.5, 0.95, 10, device=device)
        vc_t = (
            torch.tensor(valid_classes, device=device, dtype=torch.float32)
            if valid_classes is not None else None
        )
        stats = []
        n_batches_done = 0

        for batch in tqdm(loader, desc=f"mAP {dataset} E{force_exit}/{sub_tag}"):
            imgs, targets, _paths, _shapes = batch
            imgs = imgs.to(device).float() / 255.0
            targets = targets.to(device)
            _, _, H_img, W_img = imgs.shape

            with torch.no_grad():
                raw = _forward_to_exit(model, imgs, force_exit, sub_exit)

            if sub_exit is not None:
                pred_tensor = _decode_sub_exit(raw, detect_head, H_img)
            else:
                pred_tensor = raw  # tuple (y, x) — non_max_suppression handles it

            preds = non_max_suppression(
                pred_tensor, conf_thres=0.001, iou_thres=0.6, multi_label=True
            )

            # scale normalized targets to pixel space
            scale = torch.tensor(
                [W_img, H_img, W_img, H_img], device=device, dtype=torch.float32
            )
            targets[:, 2:] = targets[:, 2:] * scale

            for si, pred in enumerate(preds):
                labels = targets[targets[:, 0] == si, 1:]  # (nl, 5): [cls, x, y, w, h]

                # filter to dataset-valid classes for cross-dataset generalization eval
                if vc_t is not None:
                    pred_mask = (pred[:, 5:6] == vc_t).any(1)
                    pred = pred[pred_mask]
                    lbl_mask = (labels[:, 0:1] == vc_t).any(1)
                    labels = labels[lbl_mask]

                nl, npr = labels.shape[0], pred.shape[0]
                correct = torch.zeros(npr, len(iouv), dtype=torch.bool, device=device)
                if npr == 0:
                    if nl:
                        stats.append(
                            (correct, *torch.zeros((2, 0), device=device), labels[:, 0])
                        )
                    continue
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # (nl, 5): [cls, xyxy]
                    correct = _process_batch(pred, labelsn, iouv)
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))

            n_batches_done += imgs.shape[0]
            if max_samples is not None and n_batches_done >= max_samples:
                break

        if not stats:
            result = {
                "main_metric": "map",
                "dataset": dataset, "weight_source": weight_source,
                "force_exit": force_exit, "sub_exit": sub_exit, "sub_exit_name": sub_tag,
                "valid_classes": valid_classes,
                "map50": 0.0, "map": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "ece": 0.0, "n_images": 0,
            }
        else:
            import numpy as np
            from shared import compute_ece

            agg = [torch.cat(x, 0) for x in zip(*stats)]
            tp, conf, pred_cls, target_cls = agg
            tp_np = tp.cpu().numpy()
            conf_np = conf.cpu().numpy()
            ap_out = ap_per_class(
                tp_np, conf_np, pred_cls.cpu().numpy(), target_cls.cpu().numpy(),
                names={},
            )
            # ap_out = (tp_count, fp_count, precision, recall, f1, ap, unique_classes)
            p_arr, r_arr, f1_arr, ap = ap_out[2], ap_out[3], ap_out[4], ap_out[5]
            map50 = float(ap[:, 0].mean()) if len(ap) else 0.0
            map_val = float(ap.mean()) if len(ap) else 0.0
            mprecision = float(p_arr.mean()) if len(p_arr) else 0.0
            mrecall = float(r_arr.mean()) if len(r_arr) else 0.0
            mf1 = float(f1_arr.mean()) if len(f1_arr) else 0.0
            # ECE: when detection confidence = X, is it actually TP@0.5 X% of the time?
            ece = compute_ece(conf_np, tp_np[:, 0].astype(bool))
            result = {
                "main_metric": "map",
                "dataset": dataset, "weight_source": weight_source,
                "force_exit": force_exit, "sub_exit": sub_exit, "sub_exit_name": sub_tag,
                "valid_classes": valid_classes,
                "map50": round(map50, 6), "map": round(map_val, 6),
                "precision": round(mprecision, 6), "recall": round(mrecall, 6),
                "f1": round(mf1, 6), "ece": round(ece, 6),
            }
            print(
                f"[evaluate_quality] {dataset} exit={force_exit} sub={sub_tag} "
                f"mAP50={map50:.4f} mAP={map_val:.4f} P={mprecision:.4f} R={mrecall:.4f} "
                f"F1={mf1:.4f} ECE={ece:.4f}"
            )

        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as exc:
        import traceback
        from shared import has_valid_result
        if not has_valid_result(out_path):
            out_path.write_text(
                json.dumps({
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "dataset": dataset,
                    "weight_source": weight_source,
                    "force_exit": force_exit,
                    "sub_exit": sub_exit,
                }, indent=2),
                encoding="utf-8",
            )
        print(f"[evaluate_quality] ERROR exit={force_exit} sub={sub_exit} {dataset}: {exc}")
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
    valid_classes: Optional[List[int]] = None,
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
        valid_classes=valid_classes,
    )
    return hw, q
