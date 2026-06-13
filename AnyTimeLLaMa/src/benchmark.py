"""LLaMa per-layer benchmark. TWO passes:

profile_hw(...)        -> hw_results.json       (TTFT, per-token latency, energy, VRAM)
evaluate_quality(...)  -> quality_results.json  (per-task metric: ROUGE-L for
                          cnn_dailymail, exact-match for gsm8k, accuracy/f1 for
                          MCQ sets, perplexity only as a last-resort fallback)
sweep_exit(...)        -> runs HW + ALL quality datasets; loads model once per exit
benchmark(...)         -> runs both (legacy single-dataset wrapper)

Per-layer isolation: truncate base.model.layers to first (force_exit+1) blocks,
run forward, apply head (trained if force_exit in EXIT_LAYERS else base.lm_head).
This gives fair latency at every transformer layer for plotting curves.

weight_source=trained    -> base + your exit heads (where trained), base.lm_head elsewhere
weight_source=pretrained -> base only + base.lm_head at every layer
"""

import contextlib
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


def _score_mcq(base, head, tokenizer, prompt: str, choices: List[str], force_exit: int, max_length: int = 512):
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
            logits = _forward_partial(base, ids, head, force_exit)
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
    """Return (head_callable, is_trained_head).

    Broadcast policy:
    - weight_source=trained + force_exit has trained head: use it directly.
    - weight_source=trained + no head at this layer but trained_heads non-empty:
      use the trained head from the NEAREST trained layer. Better quality proxy
      than base.lm_head for in-between exits (heads trained at layer L share
      semantics with adjacent layers' hidden states).
    - otherwise: base.lm_head (the single upstream head broadcast across all exits).
    """
    if weight_source == "trained" and trained_heads:
        if force_exit in trained_heads:
            return trained_heads[force_exit].to(base.device), True
        trained_layers = sorted(trained_heads.keys())
        nearest = min(trained_layers, key=lambda l: abs(l - force_exit))
        return trained_heads[nearest].to(base.device), True
    return base.lm_head, False


@contextlib.contextmanager
def _exit_at(base, force_exit: int):
    """Non-destructively expose only first (force_exit+1) layers to base.model.forward.

    Swap base.model.layers + config.num_hidden_layers in/out around forward so each
    exit reuses the SAME per-layer compiled artifacts (no Dynamo recompile per k).
    Compare to the old _truncate_in_place which mutated permanently and forced a
    fresh model load + recompile for every force_exit value in a sweep.
    """
    full_layers = base.model.layers
    full_n = base.config.num_hidden_layers
    base.model.layers = nn.ModuleList(list(full_layers)[: force_exit + 1])
    base.config.num_hidden_layers = force_exit + 1
    try:
        yield
    finally:
        base.model.layers = full_layers
        base.config.num_hidden_layers = full_n


def _compile_per_layer(base, enable: bool) -> int:
    """Wrap each LlamaDecoderLayer in torch.compile so compiled artifacts are
    shared across every force_exit in a sweep. Outer base.model.forward stays
    eager (Python loop over compiled layers); embed/norm cheap so left eager."""
    if not enable or not hasattr(torch, "compile"):
        return 0
    try:
        n = len(base.model.layers)
        for i in range(n):
            base.model.layers[i] = torch.compile(
                base.model.layers[i]
            )
        print(f"[llama.benchmark] torch.compile enabled per-layer ({n} layers)")
        return n
    except Exception as e:
        print(f"[llama.benchmark] per-layer compile failed: {e}")
        return 0


def _forward_partial(base, input_ids, head, force_exit: int):
    """Run base.model with layers view truncated to first (force_exit+1), project via head."""
    with _exit_at(base, force_exit):
        out = base.model(input_ids=input_ids)
    return head(out.last_hidden_state)


# Generation datasets -> their REAL task metric (no perplexity proxy).
_GEN_METRIC = {"cnn_dailymail": "rougeL", "gsm8k": "exact_match"}
# Decode budget per task: a summary is short; GSM8K needs room for the reasoning
# chain before the final number.
_GEN_MAX_NEW = {"cnn_dailymail": 64, "gsm8k": 256}


@torch.no_grad()
def _gen_partial_recompute(base, head, tokenizer, ids, prompt_len, force_exit, max_new_tokens, eos):
    """Cache-free fallback: re-run the (truncated) prefix every step. Correct but
    O(L^2); used only if the KV-cache path raises (Cache-API drift / odd model)."""
    for _ in range(max_new_tokens):
        logits = _forward_partial(base, ids, head, force_exit)
        nxt = int(logits[0, -1].argmax().item())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        if eos is not None and nxt == eos:
            break
    return tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True)


@torch.no_grad()
def _greedy_generate_partial(base, head, tokenizer, prompt, force_exit, *, max_new_tokens, max_length):
    """Greedy decode through exit `force_exit` (truncated decoder + head + lm_head).

    KV cache is valid here because the truncated-layer view is held FIXED for the
    whole loop (sliced layers keep their original layer_idx 0..k, so the cache
    indexes them consistently). O(L) instead of O(L^2). Falls back to a cache-free
    recompute if the runtime's Cache API differs."""
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(base.device)
    prompt_len = ids.shape[1]
    eos = tokenizer.eos_token_id
    try:
        toks = []
        with _exit_at(base, force_exit):
            past, cur = None, ids
            attn = torch.ones_like(ids)
            for _ in range(max_new_tokens):
                out = base.model(input_ids=cur, attention_mask=attn,
                                 past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                logits = head(out.last_hidden_state[:, -1:, :])
                nxt = int(logits[0, -1].argmax().item())
                if eos is not None and nxt == eos:
                    break
                toks.append(nxt)
                cur = torch.tensor([[nxt]], device=ids.device)
                attn = torch.cat([attn, torch.ones((1, 1), device=ids.device, dtype=attn.dtype)], dim=1)
        return tokenizer.decode(toks, skip_special_tokens=True)
    except Exception as e:  # noqa: BLE001 — robustness across transformers versions
        print(f"[llama.benchmark] KV-cache decode fell back to recompute ({e})")
        return _gen_partial_recompute(base, head, tokenizer, ids, prompt_len, force_exit, max_new_tokens, eos)


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
            _forward_partial(base, ids, head, force_exit)

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
                    _ = _forward_partial(base, ids, head, force_exit)
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
            pred, conf, scores, n_toks = _score_mcq(base, head, tokenizer, s["prompt"], s["choices"], force_exit, max_length)
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
    elif dataset in _GEN_METRIC:
        # Real generation metric: ROUGE-L (summarization) / exact-match (math).
        import ee.benchmark as _eb
        gen_metric = _GEN_METRIC[dataset]
        max_new = _GEN_MAX_NEW.get(dataset, 128)
        preds, refs = [], []
        for s in tqdm(samples, desc=f"Q  {dataset} exit={force_exit} (gen)"):
            gen = _greedy_generate_partial(
                base, head, tokenizer, s["prompt"], force_exit,
                max_new_tokens=max_new, max_length=max_length,
            )
            preds.append(gen)
            refs.append(s["reference"])
        if gen_metric == "rougeL":
            r = _eb.compute_rouge(preds, refs)
            result = {**meta, "main_metric": "rougeL", "rougeL": r["rougeL_f1"],
                      "rouge2_f1": r["rouge2_f1"], "max_new_tokens": max_new}
            print(f"[evaluate_quality] {dataset} layer={force_exit} rougeL={r['rougeL_f1']:.4f}")
        else:  # exact_match
            n_correct = sum(_eb.gsm8k_exact_match(p, r) for p, r in zip(preds, refs))
            em = round(n_correct / max(len(preds), 1), 6)
            result = {**meta, "main_metric": "exact_match", "exact_match": em,
                      "n_correct": n_correct, "max_new_tokens": max_new}
            print(f"[evaluate_quality] {dataset} layer={force_exit} exact_match={em:.4f} ({n_correct}/{len(preds)})")
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
                logits = _forward_partial(base, ids, head, force_exit)
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

def _load_sweep_state(
    base_model_id: str,
    exit_heads_id: Optional[str],
    exit_layers: List[int],
    weight_source: str,
    use_torch_compile: bool,
    dtype,
):
    """Load model + (optional) trained heads + per-layer compile. Returns state
    reusable across many force_exit values."""
    tokenizer, base = _load_base(base_model_id, dtype)
    trained_heads = {}
    if weight_source == "trained" and exit_heads_id is not None:
        trained_heads = _load_trained_heads(exit_heads_id, exit_layers)
    _compile_per_layer(base, use_torch_compile)
    return tokenizer, base, trained_heads


def _run_one_exit(
    tokenizer,
    base,
    trained_heads,
    force_exit: int,
    *,
    weight_source: str,
    hw_out_dir: Optional[Union[str, Path]],
    hw_dataset: str,
    quality_out_dirs: Optional[Dict[str, Union[str, Path]]],
    n_samples: int,
    warmup_steps: int,
    max_length: int,
    base_model_id: str,
    hw_quality_datasets: bool,
) -> Dict[str, Path]:
    """One force_exit value over pre-loaded model. Used by sweep_exit (legacy)
    and sweep_all_exits (loops k)."""
    n_layers_total = base.config.num_hidden_layers
    if not (0 <= force_exit < n_layers_total):
        raise ValueError(f"force_exit={force_exit} out of [0, {n_layers_total})")
    head, is_trained = _head_for(force_exit, weight_source, trained_heads, base)

    try:
        with _exit_at(base, force_exit):
            mm = model_metrics(base, dummy_input=None)
    except Exception as e:
        print(f"[run_exit] model_metrics skipped: {e}")
        mm = {}

    results: Dict[str, Path] = {}

    if hw_out_dir is not None:
        hw_path = Path(hw_out_dir) / "hw_results.json"
        hw_path.parent.mkdir(parents=True, exist_ok=True)
        results["hw"] = _run_hw_pass(
            base, head, is_trained, tokenizer, force_exit, hw_path,
            dataset=hw_dataset, weight_source=weight_source,
            n_samples=n_samples, warmup_steps=warmup_steps,
            base_model_id=base_model_id, n_layers_total=n_layers_total, mm=mm,
        )

    for ds, q_dir in (quality_out_dirs or {}).items():
        q_path = Path(q_dir) / "quality_results.json"
        q_path.parent.mkdir(parents=True, exist_ok=True)
        if hw_quality_datasets:
            ds_hw_path = Path(q_dir) / "hw_results.json"
            try:
                results[f"hw_{ds}"] = _run_hw_pass(
                    base, head, is_trained, tokenizer, force_exit, ds_hw_path,
                    dataset=ds, weight_source=weight_source,
                    n_samples=n_samples, warmup_steps=warmup_steps,
                    base_model_id=base_model_id, n_layers_total=n_layers_total, mm=mm,
                )
            except Exception as e:
                print(f"[run_exit] hw pass failed for {ds}: {e}")
        try:
            results[ds] = _run_quality_pass(
                base, head, is_trained, tokenizer, force_exit, q_path,
                dataset=ds, weight_source=weight_source,
                n_samples=n_samples, max_length=max_length,
                base_model_id=base_model_id,
            )
        except Exception as e:
            print(f"[run_exit] quality pass failed for {ds}: {e}")

    return results


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
    """Load model once for force_exit, then run HW + quality datasets. For sweeps
    across multiple force_exit values, prefer sweep_all_exits -- it loads + compiles
    once across the whole sweep instead of per-k."""
    tokenizer, base, trained_heads = _load_sweep_state(
        base_model_id, exit_heads_id, exit_layers, weight_source, use_torch_compile, dtype,
    )
    return _run_one_exit(
        tokenizer, base, trained_heads, force_exit,
        weight_source=weight_source,
        hw_out_dir=hw_out_dir, hw_dataset=hw_dataset,
        quality_out_dirs=quality_out_dirs,
        n_samples=n_samples, warmup_steps=warmup_steps, max_length=max_length,
        base_model_id=base_model_id, hw_quality_datasets=hw_quality_datasets,
    )


def sweep_all_exits(
    base_model_id: str,
    exit_heads_id: Optional[str],
    exit_layers: List[int],
    exits: List[int],
    *,
    hw_out_dir_factory=None,         # callable(k) -> Optional[Path]
    hw_dataset: str = "cnn_dailymail",
    quality_out_dir_factories=None,  # {ds: callable(k) -> Optional[Path]}
    weight_source: str = "pretrained",
    n_samples: int = 100,
    warmup_steps: int = 3,
    max_length: int = 512,
    use_torch_compile: bool = True,
    hw_quality_datasets: bool = False,
    dtype=torch.bfloat16,
) -> Dict[int, Dict[str, Path]]:
    """Load model + per-layer compile ONCE, iterate force_exit -- compiled
    artifacts are reused across all k. Factories return None to skip a given
    (exit, dataset) so the caller can dedupe against existing results."""
    tokenizer, base, trained_heads = _load_sweep_state(
        base_model_id, exit_heads_id, exit_layers, weight_source, use_torch_compile, dtype,
    )
    all_results: Dict[int, Dict[str, Path]] = {}
    for k in exits:
        hw_out_dir = hw_out_dir_factory(k) if hw_out_dir_factory else None
        q_dirs = {}
        for ds, factory in (quality_out_dir_factories or {}).items():
            d = factory(k)
            if d is not None:
                q_dirs[ds] = d
        if hw_out_dir is None and not q_dirs:
            print(f"[sweep_all_exits] skip exit_{k}: nothing to do")
            continue
        try:
            all_results[k] = _run_one_exit(
                tokenizer, base, trained_heads, k,
                weight_source=weight_source,
                hw_out_dir=hw_out_dir, hw_dataset=hw_dataset,
                quality_out_dirs=q_dirs,
                n_samples=n_samples, warmup_steps=warmup_steps, max_length=max_length,
                base_model_id=base_model_id, hw_quality_datasets=hw_quality_datasets,
            )
        except Exception as e:
            print(f"[sweep_all_exits] exit {k} failed: {e}")
    return all_results


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
