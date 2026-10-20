"""LLaMA-3.2-1B family adapter (K6 FamilyAdapter).

One request = a fixed 128-token prompt, then 32 greedy tokens through the first (exit + 1) layers.
Token 1 is the prefill, tokens 2..32 are single-token decode steps with a KV cache.
A checkpoint is a token count, so `advance(state, k)` runs tokens until k are done.
There is no separate head: every token step ends in lm_head, so `finish` only reads the tokens back.
EOS is ignored (always 32 tokens). A failing decode step raises: there is no recompute fallback.

Ported from AnyTimeLLaMa/src/benchmark.py (`_exit_at`, `_compile_per_layer`, the KV-cache decode loop);
that module is not imported. torch, transformers and datasets are imported inside functions only.
The head is base.lm_head at every exit (weight_source "pretrained"); the cost equals ExitHead up to one RMSNorm.
"""
import contextlib
import os
import time

from exp2 import contracts
from exp2.contracts import EXIT_RANGE, LLAMA_PROMPT_TOKENS, LLAMA_TOKENS
from exp2.serve import families

MODEL_ID = "meta-llama/Llama-3.2-1B"
DTYPE = "bfloat16"
PREPROCESS_MODES = ("per_request", "preload")
INPUT_MODES = ("synthetic", "dataset")
SPECIAL_TOKEN_START = 128000  # Llama-3 special tokens start here; synthetic ids stay below


@contextlib.contextmanager
def _exit_at(base, force_exit):
    """Expose only the first (force_exit + 1) layers to base.model.forward, then restore.

    Swapping base.model.layers and config.num_hidden_layers keeps one set of (compiled) layer
    modules for every exit, so changing exit does not recompile. The sliced layers keep their
    original layer_idx 0..k, so a KV cache built under this view stays valid across calls.
    """
    from torch import nn

    full_layers = base.model.layers
    full_n = base.config.num_hidden_layers
    base.model.layers = nn.ModuleList(list(full_layers)[: force_exit + 1])
    base.config.num_hidden_layers = force_exit + 1
    try:
        yield
    finally:
        base.model.layers = full_layers
        base.config.num_hidden_layers = full_n


def _compile_per_layer(base, enable):
    """Wrap each decoder layer in torch.compile (shared by every exit). Returns the layer count."""
    if not enable:
        return 0
    import torch

    n = len(base.model.layers)
    for i in range(n):
        base.model.layers[i] = torch.compile(base.model.layers[i])
    return n


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


class LlamaState:
    """Per-request state: prompt ids, KV cache, last token, last logits, generated tokens."""

    __slots__ = ("exit", "ids", "n_done", "past", "last", "logits", "tokens")

    def __init__(self, exit, ids):
        self.exit = exit
        self.ids = ids
        self.n_done = 0
        self.past = None
        self.last = None
        self.logits = None
        self.tokens = []


class LlamaAdapter:
    family = "llama"

    def __init__(self):
        self._base = None
        self._n_layers = None
        self._device = None
        self._model_id = MODEL_ID
        self._preprocess = "per_request"
        self._inputs = None
        self._sync = None

    # ---- helpers
    def _check_exit(self, exit):
        hi = EXIT_RANGE["llama"][1] if self._n_layers is None else self._n_layers - 1
        if not _is_int(exit) or not EXIT_RANGE["llama"][0] <= exit <= hi:
            raise ValueError(f"llama exit {exit!r} outside {EXIT_RANGE['llama'][0]}..{hi}")

    def _require_loaded(self):
        if self._base is None:
            raise RuntimeError("load() must run first")

    # ---- model access (overridden by tests)
    def _build_model(self, cfg):
        import torch
        from transformers import AutoModelForCausalLM

        kwargs = {"torch_dtype": getattr(torch, DTYPE), "token": os.environ.get("HF_TOKEN")}
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": 0}  # straight onto the GPU: no CPU staging, no offload
            # device_map triggers a one-shot allocation of the whole model; with other servers resident it fails
            # on the board as "NVML_SUCCESS == r INTERNAL ASSERT FAILED" (2026-10-20). Load tensor by tensor instead.
            import transformers.modeling_utils as _mu
            if hasattr(_mu, "caching_allocator_warmup"):
                _mu.caching_allocator_warmup = lambda *a, **k: None
        base = AutoModelForCausalLM.from_pretrained(self._model_id, **kwargs).eval()
        for p in base.parameters():
            p.requires_grad = False
        return base

    def _dataset_ids(self, n, seed):
        """n rows of exactly LLAMA_PROMPT_TOKENS real-text token ids (CNN/DailyMail), shape (n, 128)."""
        import torch
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self._model_id, token=os.environ.get("HF_TOKEN"))
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test").shuffle(seed=seed)
        rows = []
        for ex in ds:
            ids = tok(ex["article"], truncation=True, max_length=LLAMA_PROMPT_TOKENS).input_ids
            if len(ids) == LLAMA_PROMPT_TOKENS:
                rows.append(ids)
            if len(rows) == n:
                return torch.tensor(rows, dtype=torch.int64)
        raise RuntimeError(f"only {len(rows)} articles with {LLAMA_PROMPT_TOKENS}+ tokens, need {n}")

    def _view(self, exit):
        return _exit_at(self._base, exit)

    def _step(self, state):
        """Token n_done + 1: prefill on the first call, one decode token after that."""
        import torch

        with torch.no_grad():
            cur = state.ids if state.n_done == 0 else state.last
            out = self._base.model(input_ids=cur, past_key_values=state.past, use_cache=True, return_dict=True)
            state.past = out.past_key_values
            state.logits = self._base.lm_head(out.last_hidden_state[:, -1:, :])[:, -1, :]
            state.last = state.logits.argmax(dim=-1, keepdim=True)  # stays on device: no host sync per token
        state.tokens.append(state.last)
        state.n_done += 1

    # ---- FamilyAdapter
    def load(self, exits, cfg):
        if not exits:
            raise ValueError("load() needs at least one exit")
        for e in exits:
            self._check_exit(e)
        preprocess = cfg.get("preprocess", "per_request")
        if preprocess not in PREPROCESS_MODES:
            raise ValueError(f"preprocess {preprocess!r} not in {PREPROCESS_MODES}")
        t0 = time.perf_counter()
        self._model_id = cfg.get("model_id", MODEL_ID)
        self._preprocess = preprocess
        self._base = self._build_model(cfg)
        self._n_layers = len(self._base.model.layers)
        for e in exits:
            self._check_exit(e)
        self._device = next(self._base.parameters()).device
        compiled = _compile_per_layer(self._base, bool(cfg.get("compile", False)))
        self._sync = families.make_sync(cfg.get("sync", "event"))
        return {"load_s": time.perf_counter() - t0, "n_layers": self._n_layers, "compiled_layers": compiled}

    def make_inputs(self, n, seed, mode):
        if mode not in INPUT_MODES:
            raise ValueError(f"input mode {mode!r} not in {INPUT_MODES}")
        if not _is_int(n) or n < 1:
            raise ValueError(f"n must be a positive int, got {n!r}")
        self._require_loaded()
        import torch

        if mode == "synthetic":
            gen = torch.Generator().manual_seed(seed)
            high = min(self._base.config.vocab_size, SPECIAL_TOKEN_START)
            ids = torch.randint(0, high, (n, LLAMA_PROMPT_TOKENS), generator=gen)
        else:
            ids = self._dataset_ids(n, seed)
        if tuple(ids.shape) != (n, LLAMA_PROMPT_TOKENS):
            raise ValueError(f"prompt ids shape {tuple(ids.shape)}, expected {(n, LLAMA_PROMPT_TOKENS)}")
        if self._preprocess == "preload":
            ids = ids.to(self._device)
        self._inputs = list(ids.view(n, 1, LLAMA_PROMPT_TOKENS))

    def checkpoints(self, exit):
        self._check_exit(exit)
        return contracts.default_checkpoints("llama", exit)

    def begin(self, input_idx, exit):
        if self._inputs is None:
            raise RuntimeError("make_inputs() must run before begin()")
        if not _is_int(input_idx) or not 0 <= input_idx < len(self._inputs):
            raise ValueError(f"input_idx {input_idx!r} outside 0..{len(self._inputs) - 1}")
        self._check_exit(exit)
        ids = self._inputs[input_idx]
        if self._preprocess == "per_request":
            ids = ids.to(self._device)  # the per-request host work: H2D copy of the prompt
        return LlamaState(exit, ids)

    def advance(self, state, ckpt):
        if not _is_int(ckpt) or not state.n_done < ckpt <= LLAMA_TOKENS:
            raise ValueError(f"ckpt {ckpt!r} must be an int in ({state.n_done}, {LLAMA_TOKENS}]")
        with self._view(state.exit):
            while state.n_done < ckpt:
                self._step(state)

    def sync(self):
        if self._sync is None:
            raise RuntimeError("load() must run before sync()")
        self._sync()

    def finish(self, state, reached):
        """Read the first `reached` generated token ids back to the host (small D2H). None: no-op."""
        if reached is None:
            return None
        if not _is_int(reached) or not 1 <= reached <= state.n_done:
            raise ValueError(f"reached {reached!r} outside 1..{state.n_done}")
        import torch

        return torch.cat(state.tokens[:reached], dim=1)[0].tolist()

    def mem_stats(self):
        return families.mem_stats()

    def reset_peak(self):
        families.reset_peak()
