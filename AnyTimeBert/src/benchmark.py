"""ElasticBERT per-exit benchmark. TWO passes:

profile_hw(...)        -> hw_results.json       (latency + memory + energy, NO quality)
evaluate_quality(...)  -> quality_results.json  (accuracy/F1, NO HW measurement)
benchmark(...)         -> runs both

Per-exit isolation: load model with num_hidden_layers=force_exit+1 so only that
many transformer blocks run. num_output_layers=1 -> single classifier head at
final layer. Fair latency comparison across exits.
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

_HERE  = Path(__file__).resolve().parent     # AnyTimeBert/src/
_MODEL = _HERE.parent                         # AnyTimeBert/
_REPO  = _MODEL.parent                        # spd/
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_MODEL / "reference"))
sys.path.insert(0, str(_MODEL / "reference" / "finetune-static"))
sys.path.insert(0, str(_MODEL / "reference" / "finetune-dynamic"))

from transformers import BertTokenizer, glue_processors, glue_compute_metrics

from models.configuration_elasticbert import ElasticBertConfig  # type: ignore
from load_data import load_and_cache_examples_glue  # type: ignore

from shared import BenchmarkProfiler, load_env  # model_metrics imported inline below

load_env()


def _load_model(
    model_id: str,
    num_labels: int,
    compile_model: bool = False,
    num_hidden_layers: Optional[int] = None,
):
    """Load ElasticBert with FULL layer stack. force_exit is applied at forward time
    via _exit_at, not at instantiation -- so per-layer compiled artifacts can be
    reused across every exit in a sweep instead of recompiling per (force_exit+1)
    architecture.

    num_hidden_layers=None -> use the checkpoint's NATIVE depth (12 for base, 24
    for large). Hardcoding 12 truncated elasticbert-large and broke exits >=12.
    """
    native = ElasticBertConfig.from_pretrained(model_id)
    n_layers = num_hidden_layers or native.num_hidden_layers
    cfg = ElasticBertConfig.from_pretrained(
        model_id,
        num_labels=num_labels,
        num_hidden_layers=n_layers,
        num_output_layers=1,
    )
    from models.modeling_elasticbert import (  # type: ignore
        ElasticBertForSequenceClassification,
    )
    model = ElasticBertForSequenceClassification.from_pretrained(
        model_id, config=cfg, ignore_mismatched_sizes=True
    )
    _broadcast_pooler(model)
    model.cuda().eval()
    if compile_model and hasattr(torch, "compile"):
        try:
            enc = model.elasticbert.encoder
            for i in range(len(enc.layer)):
                enc.layer[i] = torch.compile(enc.layer[i])
            print(f"[bert.benchmark] torch.compile enabled per-layer ({len(enc.layer)} layers)")
        except Exception as e:
            print(f"[bert.benchmark] torch.compile failed: {e}")
    return model


def _broadcast_pooler(model):
    """ElasticBert with num_output_layers=1 trains a single pooler at
    current_pooler_num (= num_hidden_layers - 1); all other pooler slots are None.
    Clone the trained pooler into every slot so any force_exit value has a
    plausible pooler (same weights as the final-layer pooler) -- standard
    anytime-eval baseline when no per-layer pooler is trained.
    """
    enc = model.elasticbert.encoder
    if not hasattr(enc, "pooler") or enc.pooler is None:
        return
    src_idx = enc.current_pooler_num
    if src_idx is None:
        return
    src = enc.pooler[src_idx]
    if src is None:
        return
    src_sd = src.state_dict()
    n_filled = 0
    for i in range(len(enc.pooler)):
        if i == src_idx:
            continue
        if enc.pooler[i] is None:
            new_p = type(src)(model.config)
            new_p.load_state_dict(src_sd)
            enc.pooler[i] = new_p
            n_filled += 1
    if n_filled:
        print(f"[bert.benchmark] broadcast trained pooler (slot {src_idx}) -> {n_filled} other slots")


@contextlib.contextmanager
def _exit_at(model, force_exit: int):
    """Non-destructively expose only the first (force_exit+1) encoder layers so
    the encoder's `i == num_hidden_layers - 1` pooler branch fires at the truncated
    tail. Assumes _broadcast_pooler has filled every pooler slot with the trained
    weights, so `enc.pooler[force_exit]` is always a real module.
    """
    enc = model.elasticbert.encoder
    full_layers = enc.layer
    full_n = enc.num_hidden_layers
    full_pooler_num = enc.current_pooler_num

    enc.layer = nn.ModuleList(list(full_layers)[: force_exit + 1])
    enc.num_hidden_layers = force_exit + 1
    enc.current_pooler_num = force_exit
    try:
        yield
    finally:
        enc.layer = full_layers
        enc.num_hidden_layers = full_n
        enc.current_pooler_num = full_pooler_num


def _load_loader(
    model_id: str,
    task: str,
    data_dir: Union[str, Path],
    out_dir: Path,
    max_seq_length: int,
    bench_batch: int = 1,
):
    tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=True)
    eval_args = argparse.Namespace(
        task_name=task.lower(),
        data_dir=str(data_dir),
        output_dir=str(out_dir),
        max_seq_length=max_seq_length,
        per_gpu_eval_batch_size=bench_batch,
        n_gpu=1,
        local_rank=-1,
        model_name_or_path=model_id,
        overwrite_cache=False,
        model_type="elasticbert",
    )
    eval_dataset = load_and_cache_examples_glue(
        eval_args, eval_args.task_name, tokenizer, data_type="dev"
    )
    return tokenizer, DataLoader(
        eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=bench_batch
    )


# =============================================================================
# HW pass — pure latency + memory + energy. NO quality.
# =============================================================================
def profile_hw(
    model_id: str,
    task: str,
    force_exit: int,
    data_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    weight_source: str = "trained",
    max_seq_length: int = 128,
    warmup_steps: int = 3,
    use_torch_compile: bool = True,
    bench_batch: int = 1,
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "hw_results.json"

    processor = glue_processors[task.lower()]()
    num_labels = len(processor.get_labels())
    model = _load_model(model_id, num_labels, compile_model=use_torch_compile)
    _, loader = _load_loader(model_id, task, data_dir, out_dir, max_seq_length, bench_batch=bench_batch)
    _run_hw_pass(
        model, loader, force_exit, out_path,
        task=task, weight_source=weight_source, model_id=model_id,
        max_seq_length=max_seq_length, warmup_steps=warmup_steps,
    )
    return out_path


def _run_hw_pass(
    model,
    loader,
    force_exit: int,
    out_path: Path,
    *,
    task: str,
    weight_source: str,
    model_id: str,
    max_seq_length: int,
    warmup_steps: int,
    max_samples: Optional[int] = None,
) -> Path:
    dummy = (
        torch.zeros((1, max_seq_length), dtype=torch.long, device="cuda"),
        torch.ones((1, max_seq_length), dtype=torch.long, device="cuda"),
        torch.zeros((1, max_seq_length), dtype=torch.long, device="cuda"),
    )
    try:
        from shared.model_metrics import _param_count_bytes
        with _exit_at(model, force_exit):
            n, nb = _param_count_bytes(model)
            mm = {
                "params_count": n,
                "params_M": round(n / 1e6, 3),
                "model_size_mb": round(nb / (1024 ** 2), 3),
                "dtype": str(next(model.parameters()).dtype),
            }
            try:
                from thop import profile as _thop_profile
                with torch.no_grad():
                    macs, _ = _thop_profile(model, inputs=dummy, verbose=False)
                mm["flops_G"] = round(2 * macs / 1e9, 4)
                mm["macs_G"] = round(macs / 1e9, 4)
            except Exception as ee:
                print(f"[bert.benchmark] FLOPs count skipped: {ee}")
    except Exception as e:
        print(f"[bert.benchmark] model_metrics skipped: {e}")
        mm = {}

    with BenchmarkProfiler(
        out_path=out_path,
        task=task,
        strategy=weight_source,
        threshold=force_exit,
        warmup_steps=warmup_steps,
        meta={"force_exit": force_exit, "weight_source": weight_source, "model_id": model_id, **mm},
    ) as prof:
        n_done = 0
        for batch in tqdm(loader, desc=f"HW {task} exit={force_exit} ({weight_source})"):
            ids, mask, types = [b.cuda() for b in batch[:3]]
            with prof.timer() as t:
                with torch.no_grad(), _exit_at(model, force_exit):
                    _ = model(input_ids=ids, attention_mask=mask, token_type_ids=types)
            prof.log_sample(
                prediction=None,
                label=None,
                forward_sec=t.elapsed_s,
                end_to_end_sec=t.elapsed_s,   # one-shot backend: e2e == forward
                exit_layer=force_exit,
            )
            n_done += 1
            if max_samples is not None and n_done >= max_samples:
                break
    return out_path


# =============================================================================
# Quality pass — pure correctness eval. NO HW sampling.
# =============================================================================
def evaluate_quality(
    model_id: str,
    task: str,
    force_exit: int,
    data_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    weight_source: str = "trained",
    max_seq_length: int = 128,
    max_samples: Optional[int] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "quality_results.json"

    processor = glue_processors[task.lower()]()
    num_labels = len(processor.get_labels())
    model = _load_model(model_id, num_labels, compile_model=False)
    _, loader = _load_loader(model_id, task, data_dir, out_dir, max_seq_length)

    import numpy as np
    from shared import compute_ece

    preds, labels = [], []
    confidences, corrects = [], []
    for batch in tqdm(loader, desc=f"Q  {task} exit={force_exit} ({weight_source})"):
        ids, mask, types, label = [b.cuda() for b in batch[:4]]
        with torch.no_grad(), _exit_at(model, force_exit):
            _, logits = model(input_ids=ids, attention_mask=mask, token_type_ids=types)
        pred = logits.argmax(-1).item()
        lbl = label.item()
        preds.append(pred)
        labels.append(lbl)
        # ECE: softmax confidence of top prediction
        if logits.shape[-1] > 1:
            conf = torch.softmax(logits.float(), dim=-1).max(-1).values.item()
            confidences.append(conf)
            corrects.append(pred == lbl)
        if max_samples is not None and len(preds) >= max_samples:
            break

    from sklearn.metrics import f1_score as sk_f1, matthews_corrcoef

    preds_np = np.array(preds)
    labels_np = np.array(labels)

    acc = float((preds_np == labels_np).mean()) if len(preds_np) else 0.0
    n_classes = len(set(labels_np.tolist()))
    f1_avg = "binary" if n_classes <= 2 else "weighted"
    f1 = float(sk_f1(labels_np, preds_np, average=f1_avg, zero_division=0)) if len(preds_np) else 0.0
    mcc = float(matthews_corrcoef(labels_np, preds_np)) if len(preds_np) else 0.0

    metrics = glue_compute_metrics(
        task.lower(), torch.tensor(preds).numpy(), torch.tensor(labels).numpy()
    )
    _GLUE_KEY = {"cola": "mcc", "mrpc": "f1", "qqp": "f1", "mnli": "acc"}
    main_metric = _GLUE_KEY.get(task.lower(), "acc")
    glue_score = mcc if task.lower() == "cola" else (f1 if task.lower() in ("mrpc", "qqp") else acc)
    ece = compute_ece(np.array(confidences), np.array(corrects)) if confidences else 0.0

    # task-specific extras (mnli/acc, acc_and_f1, etc.) kept as bonus
    extras = {k: v for k, v in metrics.items() if k not in ("acc", "f1", "mcc")}

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "main_metric": main_metric,
                "task": task,
                "weight_source": weight_source,
                "force_exit": force_exit,
                "model_id": model_id,
                "n_samples": len(preds),
                "acc": round(acc, 6),
                "f1": round(f1, 6),
                "mcc": round(mcc, 6),
                "glue_score": round(glue_score, 6),
                "ece": round(ece, 6),
                **extras,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[evaluate_quality] acc={acc:.4f} f1={f1:.4f} mcc={mcc:.4f} ece={ece:.4f}")
    return out_path


# =============================================================================
# Sweep — load model ONCE, loop force_exit. Per-layer compile reused across k.
# =============================================================================
def sweep_hw(
    model_id: str,
    task: str,
    exits,
    data_dir: Union[str, Path],
    out_root: Union[str, Path],
    *,
    weight_source: str = "pretrained",
    max_seq_length: int = 128,
    warmup_steps: int = 3,
    use_torch_compile: bool = True,
    max_samples: Optional[int] = None,
    bench_batch: int = 1,
):
    """One model load + one per-layer compile pass shared by every exit in `exits`.

    `out_root` is the directory containing `exit_{k}/` subdirs.
    Returns list of hw_results.json paths (or None for skipped).
    """
    from shared import has_valid_result

    out_root = Path(out_root)
    processor = glue_processors[task.lower()]()
    num_labels = len(processor.get_labels())
    model = _load_model(model_id, num_labels, compile_model=use_torch_compile)
    _, loader = _load_loader(model_id, task, data_dir, out_root, max_seq_length, bench_batch=bench_batch)

    paths = []
    for k in exits:
        run_dir = out_root / f"exit_{k}"
        out_path = run_dir / "hw_results.json"
        if has_valid_result(out_path):
            print(f"[skip] hw exists: {out_path}")
            paths.append(out_path)
            continue
        _run_hw_pass(
            model, loader, k, out_path,
            task=task, weight_source=weight_source, model_id=model_id,
            max_seq_length=max_seq_length, warmup_steps=warmup_steps,
            max_samples=max_samples,
        )
        paths.append(out_path)
    return paths


# =============================================================================
# Trained-model sweep (multi-exit self-distill) — loads HF repo pushed by
# GeneralizableSelfDistilationEarlyExitTraining.sync.push_ckpts_to_hf
# Layout per mode (in the HF repo):
#   joint    : joint/full_model.pt + joint/head_<k>.pt
#   pairwise : teacher/{adapter/exit_<deep>, head_<deep>.pt}; pair_e<k>/...
#   segd  : segd_teacher/...; segd_e<k>/...
# =============================================================================


def _stage_label_for_exit(mode: str, exit_k: int, deepest: int) -> Optional[str]:
    if mode == "joint":
        return "joint"
    if mode == "pairwise":
        return "teacher" if exit_k == deepest else f"pair_e{exit_k}"
    if mode == "segd":
        return "segd_teacher" if exit_k == deepest else f"segd_e{exit_k}"
    return None


def _get_inner_encoder(model):
    """Unwrap peft (pairwise/segd) to the real ElasticBertEncoder."""
    bb = model.backbone
    if hasattr(bb, "base_model") and hasattr(bb.base_model, "model"):
        return bb.base_model.model.encoder
    return bb.encoder


@contextlib.contextmanager
def _exit_at_trained(model, exit_k: int):
    enc = _get_inner_encoder(model)
    full_layers = enc.layer
    full_n = enc.num_hidden_layers
    enc.layer = nn.ModuleList(list(full_layers)[: exit_k + 1])
    enc.num_hidden_layers = exit_k + 1
    try:
        yield
    finally:
        enc.layer = full_layers
        enc.num_hidden_layers = full_n


def _trained_forward_exit(model, ids, mask, types, exit_k: int):
    """Run truncated forward + heads[exit_k]. Caller must activate adapter."""
    _, pooled_tuple = model.backbone(
        input_ids=ids, attention_mask=mask, token_type_ids=types
    )
    pooled = pooled_tuple[-1]
    pooled = model.dropout(pooled)
    return model.heads[exit_k](pooled)


def _load_trained_model(
    repo_id: str,
    mode: str,
    n_exits: int,
    num_labels: int,
    *,
    hf_token: Optional[str] = None,
    compile_model: bool = False,
):
    """Build MultiExitElasticBert and hydrate from a HF repo pushed by the
    self-distill trainer. Returns model on cuda, eval mode."""
    import os as _os

    from huggingface_hub import snapshot_download

    from GeneralizableSelfDistilationEarlyExitTraining.backends.bert.config import (
        Cfg as _BCfg,
    )
    from GeneralizableSelfDistilationEarlyExitTraining.backends.bert.model import (
        build_model,
    )
    from GeneralizableSelfDistilationEarlyExitTraining.backends.bert import adapters as _ba

    token = hf_token or _os.environ.get("HF_TOKEN")
    local = Path(snapshot_download(repo_id=repo_id, token=token, repo_type="model"))

    bert_cfg = _BCfg(n_exits=n_exits)
    model = build_model(bert_cfg, num_labels)

    if mode == "joint":
        full_path = local / "joint" / "full_model.pt"
        if not full_path.exists():
            raise FileNotFoundError(f"missing joint/full_model.pt in {repo_id}")
        sd = torch.load(full_path, map_location="cpu")
        model.load_state_dict(sd)
    else:
        _ba.attach(model, bert_cfg)
        deepest = n_exits - 1
        for k in range(n_exits):
            stage_label = _stage_label_for_exit(mode, k, deepest)
            stage_dir = local / stage_label
            if not stage_dir.exists():
                print(f"[bert.benchmark.trained] missing stage {stage_label}; exit {k} left untrained")
                continue
            try:
                _ba.load_adapter(model, k, stage_dir / "adapter")
            except Exception as e:
                print(f"[bert.benchmark.trained] adapter load failed for exit {k}: {e}")
                continue
            head_pt = stage_dir / f"head_{k}.pt"
            if head_pt.exists():
                model.heads[k].load_state_dict(torch.load(head_pt, map_location="cpu"))
            else:
                print(f"[bert.benchmark.trained] missing head_{k}.pt in {stage_label}")

    model.cuda().eval()
    if compile_model and hasattr(torch, "compile"):
        try:
            enc = _get_inner_encoder(model)
            for i in range(len(enc.layer)):
                enc.layer[i] = torch.compile(enc.layer[i])
            print(f"[bert.benchmark.trained] torch.compile enabled per-layer ({len(enc.layer)} layers)")
        except Exception as e:
            print(f"[bert.benchmark.trained] torch.compile failed: {e}")
    return model


def _activate_for_exit(model, mode: str, exit_k: int):
    if mode == "joint":
        return
    # peft active adapter -> exit_k
    try:
        from GeneralizableSelfDistilationEarlyExitTraining.backends.bert import (
            adapters as _ba,
        )
        _ba.activate(model, exit_k)
    except Exception as e:
        print(f"[bert.benchmark.trained] activate(exit_{exit_k}) failed: {e}")


def _run_hw_pass_trained(
    model,
    loader,
    force_exit: int,
    out_path: Path,
    *,
    mode: str,
    task: str,
    weight_source: str,
    model_id: str,
    max_seq_length: int,
    warmup_steps: int,
    max_samples: Optional[int] = None,
) -> Path:
    _activate_for_exit(model, mode, force_exit)
    dummy = (
        torch.zeros((1, max_seq_length), dtype=torch.long, device="cuda"),
        torch.ones((1, max_seq_length), dtype=torch.long, device="cuda"),
        torch.zeros((1, max_seq_length), dtype=torch.long, device="cuda"),
    )
    try:
        from shared.model_metrics import _param_count_bytes

        with _exit_at_trained(model, force_exit):
            n, nb = _param_count_bytes(model)
            mm = {
                "params_count": n,
                "params_M": round(n / 1e6, 3),
                "model_size_mb": round(nb / (1024 ** 2), 3),
                "dtype": str(next(model.parameters()).dtype),
            }
            try:
                from thop import profile as _thop_profile

                with torch.no_grad():
                    # thop expects positional args matching forward signature
                    macs, _ = _thop_profile(
                        model.backbone, inputs=dummy, verbose=False
                    )
                mm["flops_G"] = round(2 * macs / 1e9, 4)
                mm["macs_G"] = round(macs / 1e9, 4)
            except Exception as ee:
                print(f"[bert.benchmark.trained] FLOPs count skipped: {ee}")
    except Exception as e:
        print(f"[bert.benchmark.trained] model_metrics skipped: {e}")
        mm = {}

    with BenchmarkProfiler(
        out_path=out_path,
        task=task,
        strategy=weight_source,
        threshold=force_exit,
        warmup_steps=warmup_steps,
        meta={
            "force_exit": force_exit,
            "weight_source": weight_source,
            "mode": mode,
            "model_id": model_id,
            **mm,
        },
    ) as prof:
        n_done = 0
        for batch in tqdm(loader, desc=f"HW {task}/{mode} exit={force_exit} ({weight_source})"):
            ids, mask, types = [b.cuda() for b in batch[:3]]
            with prof.timer() as t:
                with torch.no_grad(), _exit_at_trained(model, force_exit):
                    _ = _trained_forward_exit(model, ids, mask, types, force_exit)
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
    task: str,
    mode: str,
    exits,
    n_exits: int,
    data_dir: Union[str, Path],
    out_root: Union[str, Path],
    *,
    weight_source: str = "trained",
    max_seq_length: int = 128,
    warmup_steps: int = 3,
    use_torch_compile: bool = False,
    max_samples: Optional[int] = None,
    bench_batch: int = 1,
    pretrained_tokenizer_id: str = "OpenMOSS-Team/elasticbert-large",
):
    """Trained-model HW sweep. Loads model ONCE; loops force_exit."""
    from shared import has_valid_result

    out_root = Path(out_root)
    processor = glue_processors[task.lower()]()
    num_labels = len(processor.get_labels())

    model = _load_trained_model(
        repo_id=repo_id,
        mode=mode,
        n_exits=n_exits,
        num_labels=num_labels,
        compile_model=use_torch_compile,
    )
    _, loader = _load_loader(
        pretrained_tokenizer_id, task, data_dir, out_root, max_seq_length, bench_batch=bench_batch
    )

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
            mode=mode, task=task, weight_source=weight_source, model_id=repo_id,
            max_seq_length=max_seq_length, warmup_steps=warmup_steps,
            max_samples=max_samples,
        )
        paths.append(out_path)
    return paths


def evaluate_quality_trained(
    repo_id: str,
    task: str,
    mode: str,
    force_exit: int,
    n_exits: int,
    data_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    weight_source: str = "trained",
    max_seq_length: int = 128,
    max_samples: Optional[int] = None,
    pretrained_tokenizer_id: str = "OpenMOSS-Team/elasticbert-large",
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "quality_results.json"

    processor = glue_processors[task.lower()]()
    num_labels = len(processor.get_labels())
    model = _load_trained_model(
        repo_id=repo_id, mode=mode, n_exits=n_exits, num_labels=num_labels,
        compile_model=False,
    )
    _, loader = _load_loader(pretrained_tokenizer_id, task, data_dir, out_dir, max_seq_length)
    _activate_for_exit(model, mode, force_exit)

    import numpy as np
    from shared import compute_ece

    preds, labels = [], []
    confidences, corrects = [], []
    for batch in tqdm(loader, desc=f"Q  {task}/{mode} exit={force_exit} ({weight_source})"):
        ids, mask, types, label = [b.cuda() for b in batch[:4]]
        with torch.no_grad(), _exit_at_trained(model, force_exit):
            logits = _trained_forward_exit(model, ids, mask, types, force_exit)
        pred = logits.argmax(-1).item()
        lbl = label.item()
        preds.append(pred)
        labels.append(lbl)
        if logits.shape[-1] > 1:
            conf = torch.softmax(logits.float(), dim=-1).max(-1).values.item()
            confidences.append(conf)
            corrects.append(pred == lbl)
        if max_samples is not None and len(preds) >= max_samples:
            break

    from sklearn.metrics import f1_score as sk_f1, matthews_corrcoef

    preds_np = np.array(preds)
    labels_np = np.array(labels)

    acc = float((preds_np == labels_np).mean()) if len(preds_np) else 0.0
    n_classes = len(set(labels_np.tolist()))
    f1_avg = "binary" if n_classes <= 2 else "weighted"
    f1 = float(sk_f1(labels_np, preds_np, average=f1_avg, zero_division=0)) if len(preds_np) else 0.0
    mcc = float(matthews_corrcoef(labels_np, preds_np)) if len(preds_np) else 0.0

    metrics = glue_compute_metrics(
        task.lower(), torch.tensor(preds).numpy(), torch.tensor(labels).numpy()
    )
    _GLUE_KEY = {"cola": "mcc", "mrpc": "f1", "qqp": "f1", "mnli": "acc"}
    main_metric = _GLUE_KEY.get(task.lower(), "acc")
    glue_score = mcc if task.lower() == "cola" else (f1 if task.lower() in ("mrpc", "qqp") else acc)
    ece = compute_ece(np.array(confidences), np.array(corrects)) if confidences else 0.0
    extras = {k: v for k, v in metrics.items() if k not in ("acc", "f1", "mcc")}

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "main_metric": main_metric,
                "task": task,
                "mode": mode,
                "weight_source": weight_source,
                "force_exit": force_exit,
                "model_id": repo_id,
                "n_samples": len(preds),
                "acc": round(acc, 6),
                "f1": round(f1, 6),
                "mcc": round(mcc, 6),
                "glue_score": round(glue_score, 6),
                "ece": round(ece, 6),
                **extras,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[evaluate_quality_trained] acc={acc:.4f} f1={f1:.4f} mcc={mcc:.4f} ece={ece:.4f}")
    return out_path


# =============================================================================
# Combined: run both passes
# =============================================================================
def benchmark(
    model_id: str,
    task: str,
    force_exit: int,
    data_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    weight_source: str = "trained",
    max_seq_length: int = 128,
    warmup_steps: int = 3,
    use_torch_compile: bool = True,
) -> Tuple[Path, Path]:
    hw = profile_hw(
        model_id, task, force_exit, data_dir, out_dir,
        weight_source=weight_source,
        max_seq_length=max_seq_length,
        warmup_steps=warmup_steps,
        use_torch_compile=use_torch_compile,
    )
    q = evaluate_quality(
        model_id, task, force_exit, data_dir, out_dir,
        weight_source=weight_source,
        max_seq_length=max_seq_length,
    )
    return hw, q
