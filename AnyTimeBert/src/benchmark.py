"""ElasticBERT per-exit benchmark. TWO passes:

profile_hw(...)        -> hw_results.json       (latency + memory + energy, NO quality)
evaluate_quality(...)  -> quality_results.json  (accuracy/F1, NO HW measurement)
benchmark(...)         -> runs both

Per-exit isolation: load model with num_hidden_layers=force_exit+1 so only that
many transformer blocks run. num_output_layers=1 -> single classifier head at
final layer. Fair latency comparison across exits.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Union

import torch
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
    force_exit: int,
    compile_model: bool = False,
):
    """Load ElasticBert truncated to (force_exit+1) layers, single exit head."""
    n_layers = int(force_exit) + 1
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
    model.cuda().eval()
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print(f"[bert.benchmark] torch.compile enabled (exit={force_exit})")
        except Exception as e:
            print(f"[bert.benchmark] torch.compile failed: {e}")
    return model


def _load_loader(
    model_id: str,
    task: str,
    data_dir: Union[str, Path],
    out_dir: Path,
    max_seq_length: int,
):
    tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=True)
    eval_args = argparse.Namespace(
        task_name=task.lower(),
        data_dir=str(data_dir),
        output_dir=str(out_dir),
        max_seq_length=max_seq_length,
        per_gpu_eval_batch_size=1,
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
        eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=1
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
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "hw_results.json"

    processor = glue_processors[task.lower()]()
    num_labels = len(processor.get_labels())
    model = _load_model(model_id, num_labels, force_exit, compile_model=use_torch_compile)
    _, loader = _load_loader(model_id, task, data_dir, out_dir, max_seq_length)

    # Static model metrics (params, FLOPs)
    dummy = (
        torch.zeros((1, max_seq_length), dtype=torch.long, device="cuda"),
        torch.ones((1, max_seq_length), dtype=torch.long, device="cuda"),
        torch.zeros((1, max_seq_length), dtype=torch.long, device="cuda"),
    )
    try:
        # thop profile uses *inputs -> model(*inputs); ElasticBert positional fwd accepts ids/mask/types
        from shared.model_metrics import _param_count_bytes
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
        for batch in tqdm(loader, desc=f"HW {task} exit={force_exit} ({weight_source})"):
            ids, mask, types = [b.cuda() for b in batch[:3]]
            with prof.timer() as t:
                with torch.no_grad():
                    _ = model(input_ids=ids, attention_mask=mask, token_type_ids=types)
            prof.log_sample(
                prediction=None,
                label=None,
                ttft_sec=t.elapsed_s,
                end_to_end_sec=t.elapsed_s,
                exit_layer=force_exit,
            )
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
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "quality_results.json"

    processor = glue_processors[task.lower()]()
    num_labels = len(processor.get_labels())
    model = _load_model(model_id, num_labels, force_exit, compile_model=False)
    _, loader = _load_loader(model_id, task, data_dir, out_dir, max_seq_length)

    import numpy as np
    from shared import compute_ece

    preds, labels = [], []
    confidences, corrects = [], []
    for batch in tqdm(loader, desc=f"Q  {task} exit={force_exit} ({weight_source})"):
        ids, mask, types, label = [b.cuda() for b in batch[:4]]
        with torch.no_grad():
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

    metrics = glue_compute_metrics(
        task.lower(), torch.tensor(preds).numpy(), torch.tensor(labels).numpy()
    )
    _GLUE_KEY = {"cola": "mcc", "mrpc": "f1", "qqp": "f1", "mnli": "mnli/acc"}
    glue_score = metrics.get(_GLUE_KEY.get(task.lower(), "acc"), 0.0)
    _GLUE_MAIN = {"cola": "mcc", "mrpc": "f1", "qqp": "f1", "mnli": "acc"}
    main_metric = _GLUE_MAIN.get(task.lower(), "acc")
    ece = compute_ece(np.array(confidences), np.array(corrects)) if confidences else 0.0
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
                "ece": round(ece, 6),
                "glue_score": round(glue_score, 6),
                **metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[evaluate_quality] {metrics} ece={ece:.4f}")
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
