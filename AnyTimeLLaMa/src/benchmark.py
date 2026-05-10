"""LLaMa early-exit benchmark. TWO passes:

profile_hw(...)        -> hw_results.json       (TTFT, per-token latency, energy, VRAM)
evaluate_quality(...)  -> quality_results.json  (perplexity, accuracy, ROUGE per exit)
benchmark(...)         -> runs both
"""

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch

_HERE  = Path(__file__).resolve().parent     # AnyTimeLLaMa/src/
_MODEL = _HERE.parent
_REPO  = _MODEL.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_MODEL))
sys.path.insert(0, str(_HERE))   # so `from ee.*` still works

import config as C  # type: ignore
from shared import BenchmarkProfiler


def _load(base_model_id: str, exit_heads_id: str, exit_layers: List[int], dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    from ee.model_wrapper import EarlyExitLlamaWrapper
    from ee.hub import load_exit_heads
    from ee.utils import freeze_base_model

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=C.HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto",
        token=C.HF_TOKEN,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    freeze_base_model(base)

    heads_dir = (
        exit_heads_id
        if Path(exit_heads_id).is_dir()
        else snapshot_download(exit_heads_id, token=C.HF_TOKEN)
    )
    head_device = "cuda" if torch.cuda.is_available() else "cpu"
    exit_heads, _ = load_exit_heads(heads_dir, device=head_device)

    wrapper = EarlyExitLlamaWrapper(
        base_model=base,
        exit_layer_indices=exit_layers,
        hidden_size=base.config.hidden_size,
        vocab_size=base.config.vocab_size,
        norm_eps=base.config.rms_norm_eps,
        init_from_base=False,
    )
    for idx, head in exit_heads.items():
        wrapper.exit_heads[str(idx)].load_state_dict(head.state_dict())
    return tokenizer, base, wrapper


def _load_samples(n_samples: int):
    from ee.benchmark import load_cnn_dailymail

    return load_cnn_dailymail(n_samples)


# =============================================================================
# HW pass — pure latency + memory + energy. NO quality.
# =============================================================================
def profile_hw(
    base_model_id: str,
    exit_heads_id: str,
    exit_layers: List[int],
    out_dir: Union[str, Path],
    *,
    confidence_threshold: float = 0.9,
    force_exit_layer: Optional[int] = None,
    n_samples: int = 100,
    max_new_tokens: int = 128,
    warmup_steps: int = 3,
    use_torch_compile: bool = True,
    dtype=torch.bfloat16,
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "hw_results.json"

    tokenizer, base, wrapper = _load(base_model_id, exit_heads_id, exit_layers, dtype)

    from ee.inference import EarlyExitGenerator

    gen = EarlyExitGenerator(
        base_model=base,
        exit_heads={int(k): wrapper.exit_heads[k] for k in wrapper.exit_heads},
        tokenizer=tokenizer,
        confidence_threshold=confidence_threshold,
        use_kv_cache=True,
        force_exit_layer=force_exit_layer,
    )
    if use_torch_compile and hasattr(torch, "compile"):
        try:
            gen.base_model = torch.compile(gen.base_model, mode="reduce-overhead")
        except Exception as e:
            print(f"[llama.benchmark] compile failed: {e}")

    samples = _load_samples(n_samples)

    # Warmup
    for s in samples[:warmup_steps]:
        gen.generate(s["prompt"], max_new_tokens=32)

    with BenchmarkProfiler(
        out_path=out_path,
        task="cnn_dailymail",
        strategy="confidence" if force_exit_layer is None else "force_exit",
        threshold=confidence_threshold
        if force_exit_layer is None
        else force_exit_layer,
        warmup_steps=0,  # already warmed manually
    ) as prof:
        for s in samples:
            with prof.timer() as t:
                out = gen.generate(s["prompt"], max_new_tokens=max_new_tokens)
            # Per-token timing if available from generator metadata
            ttft = (
                out.get("ttft_sec") if isinstance(out, dict) else None
            ) or t.elapsed_s
            prof.log_sample(
                prediction=None,
                label=None,
                ttft_sec=ttft,
                end_to_end_sec=t.elapsed_s,
                exit_layer=(out.get("exit_layer") if isinstance(out, dict) else None),
                n_new_tokens=(
                    out.get("n_tokens") if isinstance(out, dict) else max_new_tokens
                ),
            )
    return out_path


# =============================================================================
# Quality pass — perplexity + accuracy per exit. NO HW.
# =============================================================================
def evaluate_quality(
    base_model_id: str,
    exit_heads_id: str,
    exit_layers: List[int],
    out_dir: Union[str, Path],
    *,
    n_samples: int = 100,
    max_length: int = 512,
    dtype=torch.bfloat16,
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "quality_results.json"

    tokenizer, _, wrapper = _load(base_model_id, exit_heads_id, exit_layers, dtype)

    from ee.benchmark import benchmark_quality

    samples = _load_samples(n_samples)

    results = benchmark_quality(wrapper, samples, tokenizer, max_length=max_length)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "base_model": base_model_id,
                "exit_heads": exit_heads_id,
                "exit_layers": exit_layers,
                "n_samples": n_samples,
                "per_exit": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[evaluate_quality] {results}")
    return out_path


def benchmark(
    base_model_id: str,
    exit_heads_id: str,
    exit_layers: List[int],
    out_dir: Union[str, Path],
    *,
    confidence_threshold: float = 0.9,
    force_exit_layer: Optional[int] = None,
    n_samples: int = 100,
    max_new_tokens: int = 128,
    warmup_steps: int = 3,
    use_torch_compile: bool = True,
    dtype=torch.bfloat16,
) -> Tuple[Path, Path]:
    hw = profile_hw(
        base_model_id,
        exit_heads_id,
        exit_layers,
        out_dir,
        confidence_threshold=confidence_threshold,
        force_exit_layer=force_exit_layer,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        warmup_steps=warmup_steps,
        use_torch_compile=use_torch_compile,
        dtype=dtype,
    )
    q = evaluate_quality(
        base_model_id,
        exit_heads_id,
        exit_layers,
        out_dir,
        n_samples=n_samples,
        dtype=dtype,
    )
    return hw, q
