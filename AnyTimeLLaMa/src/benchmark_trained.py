"""Trained MultiExitLM per-exit benchmark.

Loads from HF repo pushed by GeneralizableSelfDistilationEarlyExitTraining.
Per-mode storage layout in repo:
    joint     : joint/full_model.pt + head_<k>.pt
    pairwise  : teacher/{adapter, head_<deep>.pt} + pair_e<k>/{adapter, head_<k>.pt}
    cascade   : cascade_teacher/... + cascade_e<k>/...

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
    if mode == "cascade":
        return "cascade_teacher" if exit_k == deepest else f"cascade_e{exit_k}"
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
    """Decoder up to truncated depth + head[exit_k] + lm_head."""
    out = model.decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    hs_last = out.hidden_states[-1]  # under truncation, this == hs[exit_layers[exit_k]]
    feat = model.heads[exit_k](hs_last)
    return model.lm_head(feat)


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
            enc = tokenizer(sample, return_tensors="pt", truncation=True, max_length=seq_len).to("cuda")
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


# ---- Quality (perplexity for generation tasks) ------------------------------
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
    exit_layer_1idx = model.exit_layers[force_exit]

    total_loss = 0.0
    n_tokens = 0
    for sample in tqdm(samples, desc=f"Q  {dataset}/{mode} exit={force_exit} ({weight_source})"):
        enc = tokenizer(sample, return_tensors="pt", truncation=True, max_length=seq_len).to("cuda")
        input_ids = enc.input_ids
        attn = enc.attention_mask
        with torch.no_grad(), _exit_at_trained(model, exit_layer_1idx):
            logits = _trained_forward_exit(model, input_ids, attn, force_exit)
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
