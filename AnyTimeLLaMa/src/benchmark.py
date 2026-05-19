"""LLaMa per-layer benchmark. TWO passes:

profile_hw(...)        -> hw_results.json       (TTFT, per-token latency, energy, VRAM)
evaluate_quality(...)  -> quality_results.json  (perplexity per layer)
sweep_exit(...)        -> runs HW + ALL quality datasets; loads model once per exit
benchmark(...)         -> runs both (legacy single-dataset wrapper)

Per-layer isolation: truncate base.model.layers to first (force_exit+1) blocks,
run forward, apply head (trained if force_exit in EXIT_LAYERS else base.lm_head).
This gives fair latency at every transformer layer for plotting curves.

weight_source=trained    -> base + your exit heads (where trained), base.lm_head elsewhere
weight_source=pretrained -> base only + base.lm_head at every layer
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm

_HERE  = Path(__file__).resolve().parent     # AnyTimeLLaMa/src/
_MODEL = _HERE.parent
_REPO  = _MODEL.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE))

from shared import BenchmarkProfiler, load_env, model_metrics

load_env()

HF_TOKEN = os.environ.get("HF_TOKEN")


def _load_base(base_model_id: str, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto",
        token=HF_TOKEN,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    for p in base.parameters():
        p.requires_grad = False
    return tokenizer, base


def _load_trained_heads(exit_heads_id: str, exit_layers: List[int]):
    """Returns dict[int, nn.Module] keyed by trained layer index."""
    from huggingface_hub import snapshot_download
    from ee.hub import load_exit_heads

    heads_dir = (
        exit_heads_id
        if Path(exit_heads_id).is_dir()
        else snapshot_download(exit_heads_id, token=HF_TOKEN)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    heads, _ = load_exit_heads(heads_dir, device=device)
    return {int(k): v for k, v in heads.items()}


_DATASET_LOADERS = {
    "cnn_dailymail": "load_cnn_dailymail",
    "arc_challenge": "load_arc",
    "arc_easy": "load_arc",
    "gsm8k": "load_gsm8k",
    "hellaswag": "load_hellaswag",
    "mmlu": "load_mmlu",
}


def _load_samples(n_samples: int, dataset: str = "cnn_dailymail") -> list:
    import ee.benchmark as _eb
    if dataset not in _DATASET_LOADERS:
        raise ValueError(f"Unknown dataset '{dataset}'. Options: {list(_DATASET_LOADERS)}")
    fn = getattr(_eb, _DATASET_LOADERS[dataset])
    if dataset == "arc_easy":
        return fn(n_samples, challenge=False)
    return fn(n_samples)


def _score_mcq(base, head, tokenizer, prompt: str, choices: List[str], max_length: int = 512):
    """Log-prob scoring: returns (pred_idx, confidence, scores, n_tokens_list)."""
    import torch.nn.functional as F
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]
    scores = []
    n_tokens_list = []
    for choice in choices:
        full = prompt + " " + choice
        ids = tokenizer(
            full, return_tensors="pt", truncation=True, max_length=max_length
        ).input_ids.to(base.device)
        with torch.no_grad():
            logits = _forward_partial(base, ids, head)
        cont_start = min(prompt_len - 1, ids.shape[1] - 1)
        shift_logits = logits[..., cont_start:-1, :].contiguous()
        shift_labels = ids[..., cont_start + 1 :].contiguous()
        if shift_labels.numel() == 0:
            scores.append(float("-inf"))
            n_tokens_list.append(0)
            continue
        lp = F.log_softmax(shift_logits.float(), dim=-1)
        token_lps = lp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        scores.append(token_lps.sum().item())
        n_tokens_list.append(shift_labels.numel())
    pred_idx = int(scores.index(max(scores)))
    scores_t = torch.tensor(scores, dtype=torch.float32)
    confidence = float(F.softmax(scores_t, dim=0)[pred_idx].item())
    return pred_idx, confidence, scores, n_tokens_list


def _head_for(force_exit: int, weight_source: str, trained_heads, base):
    """Return (head_callable, is_trained_head)."""
    if weight_source == "trained" and force_exit in trained_heads:
        head = trained_heads[force_exit].to(base.device)
        return head, True
    return base.lm_head, False


def _truncate_in_place(base, force_exit: int):
    """Permanently truncate base.model to first (force_exit+1) layers. NOT reversible."""
    base.model.layers = nn.ModuleList(base.model.layers[: force_exit + 1])
    base.config.num_hidden_layers = force_exit + 1


def _forward_partial(base, input_ids, head):
    """Run base.model (already truncated), project via head."""
    out = base.model(input_ids=input_ids)
    return head(out.last_hidden_state)


# =============================================================================
# Inner loops — model already loaded, truncated, compiled
# =============================================================================

def _run_hw_pass(
    base,
    head,
    is_trained: bool,
    tokenizer,
    force_exit: int,
    out_path: Path,
    *,
    dataset: str,
    weight_source: str,
    n_samples: int,
    warmup_steps: int,
    base_model_id: str,
    n_layers_total: int,
    mm: dict,
) -> Path:
    samples = _load_samples(n_samples, dataset)

    for s in samples[:warmup_steps]:
        ids = tokenizer(s["prompt"], return_tensors="pt").input_ids.to(base.device)
        with torch.no_grad():
            _forward_partial(base, ids, head)

    with BenchmarkProfiler(
        out_path=out_path,
        task=dataset,
        strategy=weight_source,
        threshold=force_exit,
        warmup_steps=0,
        meta={
            "force_exit": force_exit,
            "dataset": dataset,
            "weight_source": weight_source,
            "head_type": "trained" if is_trained else "base_lm_head",
            "base_model": base_model_id,
            "n_layers_total": n_layers_total,
            **mm,
        },
    ) as prof:
        for s in samples:
            ids = tokenizer(s["prompt"], return_tensors="pt").input_ids.to(base.device)
            with prof.timer() as t:
                with torch.no_grad():
                    _ = _forward_partial(base, ids, head)
            prof.log_sample(
                prediction=None,
                label=None,
                ttft_sec=t.elapsed_s,
                end_to_end_sec=t.elapsed_s,
                exit_layer=force_exit,
                head_type="trained" if is_trained else "base_lm_head",
            )
    return out_path


def _run_quality_pass(
    base,
    head,
    is_trained: bool,
    tokenizer,
    force_exit: int,
    out_path: Path,
    *,
    dataset: str,
    weight_source: str,
    n_samples: int,
    max_length: int,
    base_model_id: str,
) -> Path:
    samples = _load_samples(n_samples, dataset)
    task_type = samples[0].get("task_type", "generation") if samples else "generation"

    meta = {
        "base_model": base_model_id,
        "dataset": dataset,
        "weight_source": weight_source,
        "force_exit": force_exit,
        "head_type": "trained" if is_trained else "base_lm_head",
        "n_samples": len(samples),
        "task_type": task_type,
    }

    if task_type == "mcq":
        import numpy as np
        from shared import compute_ece
        from sklearn.metrics import f1_score as sk_f1
        correct = 0
        confidences, corrects, ppls = [], [], []
        pred_idxs, true_idxs = [], []
        for s in tqdm(samples, desc=f"Q  {dataset} exit={force_exit}"):
            pred, conf, scores, n_toks = _score_mcq(base, head, tokenizer, s["prompt"], s["choices"], max_length)
            is_correct = pred == s["correct_idx"]
            if is_correct:
                correct += 1
            confidences.append(conf)
            corrects.append(is_correct)
            pred_idxs.append(pred)
            true_idxs.append(s["correct_idx"])
            correct_lp = scores[s["correct_idx"]]
            correct_ntok = n_toks[s["correct_idx"]]
            ppl_correct = math.exp(-correct_lp / correct_ntok) if correct_ntok > 0 else float("inf")
            ppls.append(ppl_correct)
        accuracy = round(correct / len(samples), 6) if samples else 0.0
        f1 = round(float(sk_f1(true_idxs, pred_idxs, average="weighted", zero_division=0)), 6) if samples else 0.0
        ece = compute_ece(np.array(confidences), np.array(corrects)) if samples else 0.0
        avg_ppl = round(sum(p for p in ppls if p != float("inf")) / max(sum(1 for p in ppls if p != float("inf")), 1), 4)
        result = {**meta, "main_metric": "accuracy", "accuracy": accuracy, "f1": f1, "perplexity": avg_ppl, "ece": round(ece, 6)}
        print(f"[evaluate_quality] {dataset} layer={force_exit} acc={accuracy:.4f} f1={f1:.4f} ppl={avg_ppl:.4f} ece={ece:.4f}")
    else:
        total_nll = 0.0
        total_tokens = 0
        for s in tqdm(samples, desc=f"Q  {dataset} exit={force_exit}"):
            ids = tokenizer(
                s["prompt"], return_tensors="pt", truncation=True, max_length=max_length
            ).input_ids.to(base.device)
            if ids.shape[-1] < 2:
                continue
            with torch.no_grad():
                logits = _forward_partial(base, ids, head)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = ids[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_nll += float(loss.item())
            total_tokens += int(shift_labels.numel())
        ppl = float(torch.exp(torch.tensor(total_nll / total_tokens)).item()) if total_tokens else float("inf")
        result = {**meta, "main_metric": "perplexity", "n_tokens": total_tokens, "nll_sum": total_nll, "perplexity": round(ppl, 4)}
        print(f"[evaluate_quality] {dataset} layer={force_exit} ppl={ppl:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out_path


# =============================================================================
# sweep_exit — load model ONCE, run HW + all quality datasets
# =============================================================================

def sweep_exit(
    base_model_id: str,
    exit_heads_id: Optional[str],
    exit_layers: List[int],
    force_exit: int,
    *,
    hw_out_dir: Optional[Union[str, Path]] = None,
    hw_dataset: str = "cnn_dailymail",
    quality_out_dirs: Optional[Dict[str, Union[str, Path]]] = None,
    weight_source: str = "pretrained",
    n_samples: int = 100,
    max_new_tokens: int = 128,
    warmup_steps: int = 3,
    max_length: int = 512,
    use_torch_compile: bool = True,
    hw_quality_datasets: bool = False,
    dtype=torch.bfloat16,
) -> Dict[str, Path]:
    """Load model once for force_exit, then run:
      - HW pass on hw_dataset  (if hw_out_dir given)
      - quality pass per dataset in quality_out_dirs  {dataset: out_dir}

    Returns dict of {label: out_path} for all runs completed.
    """
    tokenizer, base = _load_base(base_model_id, dtype)
    trained_heads = {}
    if weight_source == "trained" and exit_heads_id is not None:
        trained_heads = _load_trained_heads(exit_heads_id, exit_layers)

    head, is_trained = _head_for(force_exit, weight_source, trained_heads, base)
    n_layers_total = base.config.num_hidden_layers
    if not (0 <= force_exit < n_layers_total):
        raise ValueError(f"force_exit={force_exit} out of [0, {n_layers_total})")

    _truncate_in_place(base, force_exit)

    try:
        mm = model_metrics(base, dummy_input=None)
    except Exception as e:
        print(f"[sweep_exit] model_metrics skipped: {e}")
        mm = {}

    if use_torch_compile and hasattr(torch, "compile"):
        try:
            base.model = torch.compile(base.model, mode="reduce-overhead")
            print(f"[sweep_exit] torch.compile enabled (exit={force_exit})")
        except Exception as e:
            print(f"[sweep_exit] torch.compile failed: {e}")

    results: Dict[str, Path] = {}

    if hw_out_dir is not None:
        hw_path = Path(hw_out_dir) / "hw_results.json"
        hw_path.parent.mkdir(parents=True, exist_ok=True)
        results["hw"] = _run_hw_pass(
            base, head, is_trained, tokenizer, force_exit, hw_path,
            dataset=hw_dataset,
            weight_source=weight_source,
            n_samples=n_samples,
            warmup_steps=warmup_steps,
            base_model_id=base_model_id,
            n_layers_total=n_layers_total,
            mm=mm,
        )

    for ds, q_dir in (quality_out_dirs or {}).items():
        q_path = Path(q_dir) / "quality_results.json"
        q_path.parent.mkdir(parents=True, exist_ok=True)
        if hw_quality_datasets:
            ds_hw_path = Path(q_dir) / "hw_results.json"
            try:
                results[f"hw_{ds}"] = _run_hw_pass(
                    base, head, is_trained, tokenizer, force_exit, ds_hw_path,
                    dataset=ds,
                    weight_source=weight_source,
                    n_samples=n_samples,
                    warmup_steps=warmup_steps,
                    base_model_id=base_model_id,
                    n_layers_total=n_layers_total,
                    mm=mm,
                )
            except Exception as e:
                print(f"[sweep_exit] hw pass failed for {ds}: {e}")
        try:
            results[ds] = _run_quality_pass(
                base, head, is_trained, tokenizer, force_exit, q_path,
                dataset=ds,
                weight_source=weight_source,
                n_samples=n_samples,
                max_length=max_length,
                base_model_id=base_model_id,
            )
        except Exception as e:
            print(f"[sweep_exit] quality pass failed for {ds}: {e}")

    return results


# =============================================================================
# Public API — thin wrappers kept for backward compat
# =============================================================================

def profile_hw(
    base_model_id: str,
    exit_heads_id: Optional[str],
    exit_layers: List[int],
    force_exit: int,
    out_dir: Union[str, Path],
    *,
    dataset: str = "cnn_dailymail",
    weight_source: str = "trained",
    n_samples: int = 100,
    max_new_tokens: int = 128,
    warmup_steps: int = 3,
    use_torch_compile: bool = True,
    dtype=torch.bfloat16,
) -> Path:
    res = sweep_exit(
        base_model_id, exit_heads_id, exit_layers, force_exit,
        hw_out_dir=out_dir,
        hw_dataset=dataset,
        quality_out_dirs=None,
        weight_source=weight_source,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        warmup_steps=warmup_steps,
        use_torch_compile=use_torch_compile,
        dtype=dtype,
    )
    return res["hw"]


def evaluate_quality(
    base_model_id: str,
    exit_heads_id: Optional[str],
    exit_layers: List[int],
    force_exit: int,
    out_dir: Union[str, Path],
    *,
    dataset: str = "cnn_dailymail",
    weight_source: str = "trained",
    n_samples: int = 100,
    max_length: int = 512,
    dtype=torch.bfloat16,
) -> Path:
    res = sweep_exit(
        base_model_id, exit_heads_id, exit_layers, force_exit,
        hw_out_dir=None,
        quality_out_dirs={dataset: out_dir},
        weight_source=weight_source,
        n_samples=n_samples,
        max_length=max_length,
        use_torch_compile=False,
        dtype=dtype,
    )
    return res[dataset]


def benchmark(
    base_model_id: str,
    exit_heads_id: Optional[str],
    exit_layers: List[int],
    force_exit: int,
    out_dir: Union[str, Path],
    *,
    dataset: str = "cnn_dailymail",
    weight_source: str = "trained",
    n_samples: int = 100,
    max_new_tokens: int = 128,
    warmup_steps: int = 3,
    use_torch_compile: bool = True,
    dtype=torch.bfloat16,
) -> Tuple[Path, Path]:
    res = sweep_exit(
        base_model_id, exit_heads_id, exit_layers, force_exit,
        hw_out_dir=out_dir,
        hw_dataset=dataset,
        quality_out_dirs={dataset: out_dir},
        weight_source=weight_source,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        warmup_steps=warmup_steps,
        use_torch_compile=use_torch_compile,
        dtype=dtype,
    )
    return res["hw"], res[dataset]
