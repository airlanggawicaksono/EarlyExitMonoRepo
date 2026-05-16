"""LLaMa per-layer benchmark. TWO passes:

profile_hw(...)        -> hw_results.json       (TTFT, per-token latency, energy, VRAM)
evaluate_quality(...)  -> quality_results.json  (perplexity per layer)
benchmark(...)         -> runs both

Per-layer isolation: truncate base.model.layers to first (force_exit+1) blocks,
run forward, apply head (trained if force_exit in EXIT_LAYERS else base.lm_head).
This gives fair latency at every transformer layer for plotting curves.

weight_source=trained    -> base + your exit heads (where trained), base.lm_head elsewhere
weight_source=pretrained -> base only + base.lm_head at every layer
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

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


def _load_samples(n_samples: int):
    from ee.benchmark import load_cnn_dailymail
    return load_cnn_dailymail(n_samples)


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
# HW pass — per layer
# =============================================================================
def profile_hw(
    base_model_id: str,
    exit_heads_id: Optional[str],
    exit_layers: List[int],
    force_exit: int,
    out_dir: Union[str, Path],
    *,
    weight_source: str = "trained",
    n_samples: int = 100,
    max_new_tokens: int = 128,
    warmup_steps: int = 3,
    use_torch_compile: bool = True,
    dtype=torch.bfloat16,
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "hw_results.json"

    tokenizer, base = _load_base(base_model_id, dtype)
    trained_heads = {}
    if weight_source == "trained" and exit_heads_id is not None:
        trained_heads = _load_trained_heads(exit_heads_id, exit_layers)

    head, is_trained = _head_for(force_exit, weight_source, trained_heads, base)
    n_layers_total = base.config.num_hidden_layers
    if not (0 <= force_exit < n_layers_total):
        raise ValueError(f"force_exit={force_exit} out of [0,{n_layers_total})")

    _truncate_in_place(base, force_exit)

    # Model metrics (params only — thop on full Llama autoregressive is heavy)
    try:
        mm = model_metrics(base, dummy_input=None)  # params + size, no FLOPs
    except Exception as e:
        print(f"[llama.benchmark] model_metrics skipped: {e}")
        mm = {}

    if use_torch_compile and hasattr(torch, "compile"):
        try:
            base.model = torch.compile(base.model, mode="reduce-overhead")
            print(f"[llama.benchmark] torch.compile enabled (exit={force_exit})")
        except Exception as e:
            print(f"[llama.benchmark] torch.compile failed: {e}")

    samples = _load_samples(n_samples)

    # Warmup
    for s in samples[:warmup_steps]:
        ids = tokenizer(s["prompt"], return_tensors="pt").input_ids.to(base.device)
        with torch.no_grad():
            _forward_partial(base, ids, head)

    with BenchmarkProfiler(
        out_path=out_path,
        task="cnn_dailymail",
        strategy=weight_source,
        threshold=force_exit,
        warmup_steps=0,
        meta={
            "force_exit": force_exit,
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


# =============================================================================
# Quality pass — perplexity per layer
# =============================================================================
def evaluate_quality(
    base_model_id: str,
    exit_heads_id: Optional[str],
    exit_layers: List[int],
    force_exit: int,
    out_dir: Union[str, Path],
    *,
    weight_source: str = "trained",
    n_samples: int = 100,
    max_length: int = 512,
    dtype=torch.bfloat16,
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "quality_results.json"

    tokenizer, base = _load_base(base_model_id, dtype)
    trained_heads = {}
    if weight_source == "trained" and exit_heads_id is not None:
        trained_heads = _load_trained_heads(exit_heads_id, exit_layers)
    head, is_trained = _head_for(force_exit, weight_source, trained_heads, base)
    _truncate_in_place(base, force_exit)

    samples = _load_samples(n_samples)

    total_nll = 0.0
    total_tokens = 0
    for s in samples:
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

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "base_model": base_model_id,
                "weight_source": weight_source,
                "force_exit": force_exit,
                "head_type": "trained" if is_trained else "base_lm_head",
                "n_samples": n_samples,
                "n_tokens": total_tokens,
                "nll_sum": total_nll,
                "perplexity": ppl,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[evaluate_quality] layer={force_exit} ppl={ppl:.4f}")
    return out_path


def benchmark(
    base_model_id: str,
    exit_heads_id: Optional[str],
    exit_layers: List[int],
    force_exit: int,
    out_dir: Union[str, Path],
    *,
    weight_source: str = "trained",
    n_samples: int = 100,
    max_new_tokens: int = 128,
    warmup_steps: int = 3,
    use_torch_compile: bool = True,
    dtype=torch.bfloat16,
) -> Tuple[Path, Path]:
    hw = profile_hw(
        base_model_id, exit_heads_id, exit_layers, force_exit, out_dir,
        weight_source=weight_source,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        warmup_steps=warmup_steps,
        use_torch_compile=use_torch_compile,
        dtype=dtype,
    )
    q = evaluate_quality(
        base_model_id, exit_heads_id, exit_layers, force_exit, out_dir,
        weight_source=weight_source,
        n_samples=n_samples,
        dtype=dtype,
    )
    return hw, q
