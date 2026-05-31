"""Multi-exit causal LM. Taps hidden_states at chosen layer indices, applies a
per-exit (cloned) final-norm, then a SHARED lm_head (EE-LLM / LITE pattern).

Works for GPT-2-family and LLaMA-family by locating the decoder + norm + head
generically (see `_resolve_lm`).
"""

import copy

import torch
import torch.nn as nn

from . import bootstrap  # noqa: F401
from transformers import AutoModelForCausalLM  # type: ignore


_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def _resolve_lm(lm_model):
    """Return (decoder_module, final_norm, lm_head)."""
    # GPT-2 family: lm_model.transformer.ln_f
    if hasattr(lm_model, "transformer") and hasattr(lm_model.transformer, "ln_f"):
        return lm_model.transformer, lm_model.transformer.ln_f, lm_model.lm_head
    # LLaMA / Mistral family: lm_model.model.norm
    if hasattr(lm_model, "model") and hasattr(lm_model.model, "norm"):
        return lm_model.model, lm_model.model.norm, lm_model.lm_head
    raise RuntimeError(
        f"could not locate decoder + final_norm + lm_head on {type(lm_model).__name__}"
    )


def _evenly_spaced_exits(n_blocks: int, n_exits: int):
    if n_exits >= n_blocks:
        return tuple(range(1, n_blocks + 1))
    step = n_blocks / n_exits
    return tuple(int(round((i + 1) * step)) for i in range(n_exits))


class MultiExitLM(nn.Module):
    def __init__(self, lm_model, exit_layers):
        super().__init__()
        self.backbone = lm_model
        self.exit_layers = tuple(exit_layers)
        self.n_exits = len(self.exit_layers)
        self.decoder, _norm, self.lm_head = _resolve_lm(lm_model)
        self.heads = nn.ModuleList(
            [copy.deepcopy(_norm) for _ in self.exit_layers]
        )

    def forward(self, input_ids, attention_mask=None):
        out = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states  # tuple len num_hidden_layers + 1
        return [self.lm_head(self.heads[i](hs[L])) for i, L in enumerate(self.exit_layers)]


def build_model(cfg) -> MultiExitLM:
    dtype = _DTYPES.get(cfg.torch_dtype, torch.float32)
    lm_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=dtype)
    n_blocks = lm_model.config.num_hidden_layers
    exits = _evenly_spaced_exits(n_blocks, cfg.n_exits)
    return MultiExitLM(lm_model, exits)
