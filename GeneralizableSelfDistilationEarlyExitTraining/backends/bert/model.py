"""Multi-exit ElasticBERT wrapper.

ElasticBertModel(num_output_layers=N) emits a tuple of N pooled outputs
(one per exit) in ONE forward. We attach one classifier head per exit and
return a list of N logits. Backbone warm-starts from pretrained weights.

Heads live OUTSIDE any peft wrapper (see adapters.py) so they stay trainable
and are saved/loaded by storage.py independently of LoRA adapters.
"""

import torch.nn as nn

from . import bootstrap  # noqa: F401  (injects sys.path)
from models.configuration_elasticbert import ElasticBertConfig  # type: ignore
from models.modeling_elasticbert import ElasticBertModel  # type: ignore


class MultiExitElasticBert(nn.Module):
    """Frozen-or-trainable ElasticBERT backbone + per-exit linear heads."""

    def __init__(self, backbone, hidden_size: int, num_labels: int, n_exits: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, num_labels) for _ in range(n_exits)]
        )
        self.n_exits = n_exits

    def forward(self, input_ids, attention_mask, token_type_ids, return_features: bool = False):
        # num_output_layers>1 -> backbone returns (seq_tuple, pooled_tuple).
        _, pooled_tuple = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        feats = [self.dropout(pooled_tuple[i]) for i in range(self.n_exits)]
        logits = [self.heads[i](feats[i]) for i in range(self.n_exits)]
        if return_features:
            return logits, feats
        return logits


def build_model(cfg, num_labels: int) -> MultiExitElasticBert:
    """Construct a multi-exit ElasticBERT, backbone warm-started from cfg.model_id."""
    config = ElasticBertConfig.from_pretrained(
        cfg.model_id,
        num_labels=num_labels,
        num_hidden_layers=cfg.n_exits,
        num_output_layers=cfg.n_exits,
        max_output_layers=cfg.n_exits,
    )
    backbone = ElasticBertModel.from_pretrained(
        cfg.model_id,
        config=config,
        add_pooling_layer=True,
        ignore_mismatched_sizes=True,
    )
    return MultiExitElasticBert(
        backbone=backbone,
        hidden_size=config.hidden_size,
        num_labels=num_labels,
        n_exits=cfg.n_exits,
        dropout=config.hidden_dropout_prob,
    )
