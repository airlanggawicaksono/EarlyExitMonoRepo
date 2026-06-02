"""Multi-exit ViT: HF ViTModel + one classifier head per chosen layer.

Mirrors MultiExitElasticBert — `output_hidden_states=True` gives the per-block
hidden states; we tap CLS token at each exit layer and project through its own
nn.Linear head.
"""

import torch.nn as nn

from . import bootstrap  # noqa: F401
from transformers import ViTModel  # type: ignore


def _evenly_spaced_exits(n_blocks: int, n_exits: int):
    """Tuple of layer indices (1-indexed: hidden_states[i] is after block i-1)."""
    if n_exits >= n_blocks:
        return tuple(range(1, n_blocks + 1))
    step = n_blocks / n_exits
    return tuple(int(round((i + 1) * step)) for i in range(n_exits))


class MultiExitViT(nn.Module):
    def __init__(self, backbone, hidden_size: int, num_labels: int, exit_layers, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.exit_layers = tuple(exit_layers)
        self.n_exits = len(self.exit_layers)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, num_labels) for _ in self.exit_layers]
        )

    def forward(self, pixel_values, return_features: bool = False):
        out = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        hs = out.hidden_states  # tuple len num_hidden_layers + 1
        feats = [self.dropout(hs[L][:, 0]) for L in self.exit_layers]
        logits = [self.heads[i](feats[i]) for i in range(self.n_exits)]
        if return_features:
            return logits, feats
        return logits


def build_model(cfg, num_labels: int) -> MultiExitViT:
    backbone = ViTModel.from_pretrained(cfg.model_id, add_pooling_layer=False)
    n_blocks = backbone.config.num_hidden_layers
    exits = _evenly_spaced_exits(n_blocks, cfg.n_exits)
    return MultiExitViT(backbone, backbone.config.hidden_size, num_labels, exits)
