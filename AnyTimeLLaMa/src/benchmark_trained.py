"""Trained MultiExitLM per-exit benchmark.

Loads from HF repo pushed by GeneralizableSelfDistilationEarlyExitTraining.
Per-mode storage layout in repo:
    joint     : joint/full_model.pt + head_<k>.pt
    pairwise  : teacher/{adapter, head_<deep>.pt} + pair_e<k>/{adapter, head_<k>.pt}
    segd   : segd_teacher/... + segd_e<k>/...

Emits identical hw_results.json / quality_results.json schema as the legacy
per-layer-heads benchmark.
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

from shared import BenchmarkProfiler, load_env  # noqa: E402

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
def _load_trained_lm(
    repo_id: str,
    mode: str,
    n_exits: int,
    *,
    base_model_id: str = "meta-llama/Llama-3.2-1B",
    dtype: str = "bfloat16",
    hf_token: Optional[str] = None,
    compile_model: bool = False,
):
    from huggingface_hub import snapshot_download

    from GeneralizableSelfDistilationEarlyExitTraining.backends.llama.config import (
        Cfg as _LCfg,
    )
    from GeneralizableSelfDistilationEarlyExitTraining.backends.llama.model import (
        build_model,
    )
    from GeneralizableSelfDistilationEarlyExitTraining.backends.llama import adapters as _la

    token = hf_token or os.environ.get("HF_TOKEN")
    local = Path(snapshot_download(repo_id=repo_id, token=token, repo_type="model"))

    lm_cfg = _LCfg(mode=mode, n_exits=n_exits, model_id=base_model_id, torch_dtype=dtype)
    model = build_model(lm_cfg)

    if mode == "joint":
        full_path = local / "joint" / "full_model.pt"
        if not full_path.exists():
            raise FileNotFoundError(f"missing joint/full_model.pt in {repo_id}")
        sd = torch.load(full_path, map_location="cpu")
        model.load_state_dict(sd, strict=False)
    else:
        _la.attach(model, lm_cfg)
        deepest = n_exits - 1
        for k in range(n_exits):
            stage_label = _stage_label_for_exit(mode, k, deepest)
            stage_dir = local / stage_label
            if not stage_dir.exists():
                print(f"[llama.benchmark.trained] missing stage {stage_label}; exit {k} skipped")
                continue
            try:
                _la.load_adapter(model, k, stage_dir / "adapter")
            except Exception as e:
                print(f"[llama.benchmark.trained] adapter load failed for exit {k}: {e}")
                continue
            head_pt = stage_dir / f"head_{k}.pt"
            if head_pt.exists():
                model.heads[k].load_state_dict(torch.load(head_pt, map_location="cpu"))
            else:
                print(f"[llama.benchmark.trained] missing head_{k}.pt in {stage_label}")

    model.cuda().eval()
    if compile_model and hasattr(torch, "compile"):
        try:
            dec = _get_inner_decoder(model)
            for i in range(len(dec.layers)):
                dec.layers[i] = torch.compile(dec.layers[i])
            print(f"[llama.benchmark.trained] torch.compile enabled per-layer ({len(dec.layers)} layers)")
        except Exception as e:
            print(f"[llama.benchmark.trained] torch.compile failed: {e}")
    return model


def _get_inner_decoder(model):
    """Unwrap peft + locate LLaMA-family decoder.layers list."""
    bb = model.backbone
    if hasattr(bb, "base_model") and hasattr(bb.base_model, "model"):
        bb = bb.base_model.model
    if hasattr(bb, "model") and hasattr(bb.model, "layers"):
        return bb.model
    if hasattr(bb, "transformer") and hasattr(bb.transformer, "h"):
        # GPT-2 family
        return type("Shim", (), {"layers": bb.transformer.h})()
    raise RuntimeError("could not locate decoder.layers")


@contextlib.contextmanager
def _exit_at_trained(model, exit_layer_1idx: int):
    """Truncate decoder.layers to first exit_layer_1idx blocks for HW timing."""
    dec = _get_inner_decoder(model)
    full_layers = dec.layers
    dec.layers = nn.ModuleList(list(full_layers)[: exit_layer_1idx])
    try:
        yield
    finally:
        dec.layers = full_layers


def _activate_for_exit(model, mode: str, exit_k: int):
    if mode == "joint":
        return
    try:
        from GeneralizableSelfDistilationEarlyExitTraining.backends.llama import (
            adapters as _la,
        )
        _la.activate(model, exit_k)
    except Exception as e:
        print(f"[llama.benchmark.trained] activate(exit_{exit_k}) failed: {e}")


def _trained_forward_exit(model, input_ids, attention_mask, exit_k: int):
    """HW-TIMING forward only. Meant to run under _exit_at_trained (truncated
    stack): times the layers an early exit actually executes. NOT for quality --
    truncation makes HF apply its final norm here, which `heads[k]` then norms
    again (double-norm). Harmless for latency; wrong for the output distribution.
    Use _quality_forward_exit for any metric."""
    out = model.decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    hs_last = out.hidden_states[-1]
    feat = model.heads[exit_k](hs_last)
    return model.lm_head(feat)


def _quality_forward_exit(model, input_ids, attention_mask, exit_k: int,
                          *, past=None, use_cache: bool = False):
    """TRAINING-FAITHFUL forward for quality. Runs the FULL decoder and taps the
    PRE-norm hidden_states[exit_layers[k]] -- exactly what MultiExitLM.forward
    feeds heads[k] at train time -- so the per-exit norm is applied once. Extra
    layers beyond k don't affect hs[k], so the exit distribution is identical to
    a truncated run minus the spurious double-norm. (No truncation: quality isn't
    timed.) Returns logits, or (logits, past) when use_cache for KV-cached decode."""
    out = model.decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past,
        use_cache=use_cache,
        output_hidden_states=True,
        return_dict=True,
    )
    hs_L = out.hidden_states[model.exit_layers[exit_k]]
    logits = model.lm_head(model.heads[exit_k](hs_L))
    if use_cache:
        return logits, out.past_key_values
    return logits


# Generation datasets -> their REAL task metric (no perplexity proxy).
_GEN_DATASETS = {"cnn_dailymail": "rougeL", "gsm8k": "exact_match"}
_GEN_MAX_NEW = {"cnn_dailymail": 64, "gsm8k": 256}


@torch.no_grad()
def _gen_trained_recompute(model, tokenizer, ids, prompt_len, exit_k, max_new_tokens, eos):
    """Cache-free fallback (O(L^2)); used only if the KV-cache path raises. Uses
    the training-faithful (full, single-norm) forward."""
    for _ in range(max_new_tokens):
        attn = torch.ones_like(ids)
        logits = _quality_forward_exit(model, ids, attn, exit_k)
        nxt = int(logits[0, -1].argmax().item())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        if eos is not None and nxt == eos:
            break
    return tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True)


@torch.no_grad()
def _greedy_generate_trained(model, tokenizer, prompt: str, exit_k: int,
                             *, max_new_tokens: int, seq_len: int) -> str:
    """Greedy decode through exit_k, training-faithful (full decoder, pre-norm
    hidden_states[exit_layers[k]], single norm). KV cache reuses past_key_values
    across steps; falls back to cache-free recompute on Cache-API mismatch."""
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq_len).input_ids.cuda()
    prompt_len = ids.shape[1]
    eos = tokenizer.eos_token_id
    try:
        toks = []
        past, cur = None, ids
        attn = torch.ones_like(ids)
        for _ in range(max_new_tokens):
            logits, past = _quality_forward_exit(model, cur, attn, exit_k, past=past, use_cache=True)
            nxt = int(logits[0, -1].argmax().item())
            if eos is not None and nxt == eos:
                break
            toks.append(nxt)
            cur = torch.tensor([[nxt]], device=ids.device)
            attn = torch.cat([attn, torch.ones((1, 1), device=ids.device, dtype=attn.dtype)], dim=1)
        return tokenizer.decode(toks, skip_special_tokens=True)
    except Exception as e:  # noqa: BLE001 — robustness across transformers versions
        print(f"[llama.benchmark.trained] KV-cache decode fell back to recompute ({e})")
        return _gen_trained_recompute(model, tokenizer, ids, prompt_len, exit_k, max_new_tokens, eos)


# ---- loader -----------------------------------------------------------------
def _load_eval_samples(dataset: str, n_samples: int, seq_len: int, tokenizer):
    """Reuse legacy sample loader."""
    from .benchmark import _load_samples as _ls

    return _ls(n_samples, dataset)


# ---- HW sweep ----------------------------------------------------------------
def _run_hw_pass_trained(
    model,
    tokenizer,
    samples,
    exit_k: int,
    out_path: Path,
    *,
    mode: str,
    dataset: str,
    weight_source: str,
    model_id: str,
    warmup_steps: int,
    seq_len: int,
) -> Path:
    _activate_for_exit(model, mode, exit_k)
    exit_layer_1idx = model.exit_layers[exit_k]
    with BenchmarkProfiler(
        out_path=out_path,
        task=dataset,
        strategy=weight_source,
        threshold=exit_k,
        warmup_steps=warmup_steps,
        meta={
            "force_exit": exit_k,
            "weight_source": weight_source,
            "mode": mode,
            "model_id": model_id,
            "exit_layer_1idx": exit_layer_1idx,
        },
    ) as prof:
        for sample in tqdm(samples, desc=f"HW {dataset}/{mode} exit={exit_k} ({weight_source})"):
            prompt = sample["prompt"] if isinstance(sample, dict) else sample
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq_len).to("cuda")
            with prof.timer() as t:
                with torch.no_grad(), _exit_at_trained(model, exit_layer_1idx):
                    _ = _trained_forward_exit(model, enc.input_ids, enc.attention_mask, exit_k)
            prof.log_sample(
                prediction=None,
                label=None,
                forward_sec=t.elapsed_s,
                end_to_end_sec=t.elapsed_s,
                exit_layer=exit_k,
            )
    return out_path


def sweep_hw_trained(
    repo_id: str,
    dataset: str,
    mode: str,
    exits,
    n_exits: int,
    out_root: Union[str, Path],
    *,
    base_model_id: str = "meta-llama/Llama-3.2-1B",
    dtype: str = "bfloat16",
    weight_source: str = "trained",
    seq_len: int = 256,
    n_samples: int = 100,
    warmup_steps: int = 3,
    use_torch_compile: bool = False,
):
    from shared import has_valid_result
    from transformers import AutoTokenizer  # type: ignore

    out_root = Path(out_root)
    model = _load_trained_lm(
        repo_id=repo_id,
        mode=mode,
        n_exits=n_exits,
        base_model_id=base_model_id,
        dtype=dtype,
        compile_model=use_torch_compile,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples = _load_eval_samples(dataset, n_samples, seq_len, tokenizer)
    paths = []
    for k in exits:
        run_dir = out_root / f"exit_{k}"
        out_path = run_dir / "hw_results.json"
        if has_valid_result(out_path):
            print(f"[skip] hw exists: {out_path}")
            paths.append(out_path)
            continue
        _run_hw_pass_trained(
            model, tokenizer, samples, k, out_path,
            mode=mode, dataset=dataset, weight_source=weight_source,
            model_id=repo_id, warmup_steps=warmup_steps, seq_len=seq_len,
        )
        paths.append(out_path)
    return paths


# ---- MCQ accuracy (acc_norm via log-likelihood over choices) ----------------
# Standard metric for arc_challenge / hellaswag / mmlu. No generation: score each
# answer choice by the LM log-prob of its tokens given the context, pick argmax.
# acc = raw-LL argmax; acc_norm = length-normalized argmax (lm-eval convention).
_MCQ_DATASETS = {"arc_challenge", "hellaswag", "mmlu"}


def _load_mcq(dataset: str, n_samples: int):
    """-> list[(context, choices:list[str], gold_idx:int)] for an MCQ dataset."""
    from datasets import load_dataset  # type: ignore

    if dataset == "arc_challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

        def fmt(ex):
            labels, texts = ex["choices"]["label"], ex["choices"]["text"]
            key = ex["answerKey"]
            gold = labels.index(key) if key in labels else 0
            return f"Question: {ex['question']}\nAnswer:", texts, gold

    elif dataset == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")  # test is unlabeled

        def fmt(ex):
            gold = int(ex["label"]) if str(ex["label"]) != "" else 0
            return ex["ctx"], [e.strip() for e in ex["endings"]], gold

    elif dataset == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")

        def fmt(ex):
            return f"Question: {ex['question']}\nAnswer:", list(ex["choices"]), int(ex["answer"])

    else:
        raise ValueError(f"not an MCQ dataset: {dataset}")

    # Seeded shuffle -> representative subset (esp. MMLU, which is subject-ordered)
    # and identical across all exits/modes (comparable).
    ds = ds.shuffle(seed=42)
    n = min(n_samples, len(ds))
    return [fmt(ds[i]) for i in range(n)]


def _choice_loglikelihood(model, tokenizer, context: str, continuation: str,
                          exit_k: int, seq_len: int):
    """Sum log-prob of `continuation` tokens given `context`, scored at exit_k
    (truncated decoder + head_k + lm_head). -> (ll, n_cont_tokens, n_cont_chars)."""
    ctx_ids = tokenizer(context, add_special_tokens=True).input_ids
    cont_ids = tokenizer(continuation, add_special_tokens=False).input_ids
    if len(cont_ids) == 0:
        return -1e30, 0, 1
    full = (ctx_ids + cont_ids)[-seq_len:]
    n_cont = min(len(cont_ids), len(full) - 1)  # keep >=1 context token
    ids = torch.tensor([full], device="cuda")
    attn = torch.ones_like(ids)
    with torch.no_grad():
        logits = _quality_forward_exit(model, ids, attn, exit_k)  # [1, L, V]
    logprobs = torch.log_softmax(logits[0].float(), dim=-1)        # [L, V]
    L = ids.shape[1]
    pos = torch.arange(L - n_cont, L, device=ids.device)           # continuation positions
    tgt = ids[0, pos]                                              # gold tokens
    ll = logprobs[pos - 1, tgt].sum().item()                       # token i predicted by pos i-1
    return ll, n_cont, len(continuation)


def evaluate_mcq_trained(
    repo_id: str,
    dataset: str,
    mode: str,
    force_exit: int,
    n_exits: int,
    out_dir: Union[str, Path],
    *,
    base_model_id: str = "meta-llama/Llama-3.2-1B",
    dtype: str = "bfloat16",
    weight_source: str = "trained",
    seq_len: int = 256,
    n_samples: int = 100,
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "quality_results.json"

    from transformers import AutoTokenizer  # type: ignore

    model = _load_trained_lm(
        repo_id=repo_id, mode=mode, n_exits=n_exits,
        base_model_id=base_model_id, dtype=dtype, compile_model=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _activate_for_exit(model, mode, force_exit)
    items = _load_mcq(dataset, n_samples)

    correct_acc = correct_norm = 0
    for context, choices, gold in tqdm(items, desc=f"Q(MCQ) {dataset}/{mode} exit={force_exit} ({weight_source})"):
        lls, norms = [], []
        for ch in choices:
            ll, _, nchar = _choice_loglikelihood(
                model, tokenizer, context, " " + ch.strip(), force_exit, seq_len)
            lls.append(ll)
            norms.append(ll / max(nchar, 1))   # char-length-normalized (acc_norm)
        if max(range(len(lls)), key=lambda i: lls[i]) == gold:
            correct_acc += 1
        if max(range(len(norms)), key=lambda i: norms[i]) == gold:
            correct_norm += 1

    n = len(items)
    acc = correct_acc / max(n, 1)
    acc_norm = correct_norm / max(n, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "main_metric": "acc_norm",
            "task": dataset,
            "mode": mode,
            "weight_source": weight_source,
            "force_exit": force_exit,
            "model_id": repo_id,
            "n_samples": n,
            "acc": round(acc, 6),
            "acc_norm": round(acc_norm, 6),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[evaluate_mcq_trained] {dataset} exit={force_exit} acc={acc:.4f} acc_norm={acc_norm:.4f}")
    return out_path


# ---- Generation quality (ROUGE-L / exact-match — real task metrics) ---------
def evaluate_generation_trained(
    repo_id: str,
    dataset: str,
    mode: str,
    force_exit: int,
    n_exits: int,
    out_dir: Union[str, Path],
    *,
    base_model_id: str = "meta-llama/Llama-3.2-1B",
    dtype: str = "bfloat16",
    weight_source: str = "trained",
    seq_len: int = 256,
    n_samples: int = 100,
) -> Path:
    """cnn_dailymail -> ROUGE-L F1, gsm8k -> exact-match accuracy. Greedy-decodes
    at the forced exit, then scores with the dataset's standard metric."""
    out_dir = Path(out_dir)
    out_path = out_dir / "quality_results.json"

    from transformers import AutoTokenizer  # type: ignore
    import ee.benchmark as _eb

    model = _load_trained_lm(
        repo_id=repo_id, mode=mode, n_exits=n_exits,
        base_model_id=base_model_id, dtype=dtype, compile_model=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _activate_for_exit(model, mode, force_exit)
    samples = _load_eval_samples(dataset, n_samples, seq_len, tokenizer)
    metric = _GEN_DATASETS[dataset]
    max_new = _GEN_MAX_NEW.get(dataset, 128)

    preds, refs = [], []
    for s in tqdm(samples, desc=f"Q(gen) {dataset}/{mode} exit={force_exit} ({weight_source})"):
        prompt = s["prompt"] if isinstance(s, dict) else s
        ref = s["reference"] if isinstance(s, dict) else ""
        preds.append(_greedy_generate_trained(
            model, tokenizer, prompt, force_exit, max_new_tokens=max_new, seq_len=seq_len))
        refs.append(ref)

    base_out = {
        "task": dataset, "mode": mode, "weight_source": weight_source,
        "force_exit": force_exit, "model_id": repo_id, "n_samples": len(preds),
        "max_new_tokens": max_new,
    }
    if metric == "rougeL":
        r = _eb.compute_rouge(preds, refs)
        out = {"main_metric": "rougeL", "rougeL": r["rougeL_f1"], "rouge2_f1": r["rouge2_f1"], **base_out}
        print(f"[evaluate_generation_trained] {dataset} exit={force_exit} rougeL={r['rougeL_f1']:.4f}")
    else:  # exact_match
        n_correct = sum(_eb.gsm8k_exact_match(p, rf) for p, rf in zip(preds, refs))
        em = round(n_correct / max(len(preds), 1), 6)
        out = {"main_metric": "exact_match", "exact_match": em, "n_correct": n_correct, **base_out}
        print(f"[evaluate_generation_trained] {dataset} exit={force_exit} exact_match={em:.4f} ({n_correct}/{len(preds)})")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out_path


# ---- Quality (dispatch: MCQ acc_norm / ROUGE-L / exact-match / perplexity) ---
def evaluate_quality_trained(
    repo_id: str,
    dataset: str,
    mode: str,
    force_exit: int,
    n_exits: int,
    out_dir: Union[str, Path],
    *,
    base_model_id: str = "meta-llama/Llama-3.2-1B",
    dtype: str = "bfloat16",
    weight_source: str = "trained",
    seq_len: int = 256,
    n_samples: int = 100,
) -> Path:
    # Each dataset gets its STANDARD metric: MCQ -> acc_norm, summarization ->
    # ROUGE-L, math -> exact-match. Perplexity only for anything left over.
    if dataset in _MCQ_DATASETS:
        return evaluate_mcq_trained(
            repo_id=repo_id, dataset=dataset, mode=mode, force_exit=force_exit,
            n_exits=n_exits, out_dir=out_dir, base_model_id=base_model_id,
            dtype=dtype, weight_source=weight_source, seq_len=seq_len, n_samples=n_samples,
        )
    if dataset in _GEN_DATASETS:
        return evaluate_generation_trained(
            repo_id=repo_id, dataset=dataset, mode=mode, force_exit=force_exit,
            n_exits=n_exits, out_dir=out_dir, base_model_id=base_model_id,
            dtype=dtype, weight_source=weight_source, seq_len=seq_len, n_samples=n_samples,
        )

    out_dir = Path(out_dir)
    out_path = out_dir / "quality_results.json"

    from transformers import AutoTokenizer  # type: ignore

    model = _load_trained_lm(
        repo_id=repo_id, mode=mode, n_exits=n_exits,
        base_model_id=base_model_id, dtype=dtype, compile_model=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples = _load_eval_samples(dataset, n_samples, seq_len, tokenizer)
    _activate_for_exit(model, mode, force_exit)

    total_loss = 0.0
    n_tokens = 0
    for sample in tqdm(samples, desc=f"Q  {dataset}/{mode} exit={force_exit} ({weight_source})"):
        text = sample["prompt"] if isinstance(sample, dict) else sample
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).to("cuda")
        input_ids = enc.input_ids
        attn = enc.attention_mask
        with torch.no_grad():
            logits = _quality_forward_exit(model, input_ids, attn, force_exit)
        # next-token LM loss
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        n_tokens += shift_labels.numel()

    ppl = float(torch.exp(torch.tensor(total_loss / max(n_tokens, 1))).item()) if n_tokens else float("inf")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "main_metric": "perplexity",
            "task": dataset,
            "mode": mode,
            "weight_source": weight_source,
            "force_exit": force_exit,
            "model_id": repo_id,
            "n_samples": len(samples),
            "perplexity": round(ppl, 6),
            "loss_per_token": round(total_loss / max(n_tokens, 1), 6),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[evaluate_quality_trained] ppl={ppl:.4f}")
    return out_path
