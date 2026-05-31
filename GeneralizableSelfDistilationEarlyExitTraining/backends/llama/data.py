"""LLM data IO. Tokenize + concat + chunk into fixed-length sequences.

Dry-run default = wikitext-2-raw-v1. Real LLaMA pretraining would swap to a
streaming C4 loader (allenai/c4) — same surface (build_loader returns a
DataLoader yielding (input_ids, attention_mask, labels)).
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset

from . import bootstrap  # noqa: F401
from datasets import load_dataset                # type: ignore
from transformers import AutoTokenizer           # type: ignore


def _limit(dataset, n: Optional[int]):
    if n is None:
        return dataset
    return Subset(dataset, list(range(min(n, len(dataset)))))


def _tokenize(ds, tokenizer):
    return ds.map(
        lambda batch: tokenizer(batch["text"], add_special_tokens=False,
                                return_attention_mask=False),
        batched=True,
        remove_columns=ds.column_names,
    )


def _chunk(ds, seq_len: int):
    def _group(examples):
        all_ids = sum(examples["input_ids"], [])
        n = (len(all_ids) // seq_len) * seq_len
        chunks = [all_ids[i : i + seq_len] for i in range(0, n, seq_len)]
        return {"input_ids": chunks}
    return ds.map(_group, batched=True, batch_size=1000, remove_columns=ds.column_names)


def _split_name(data_type: str) -> str:
    return {"train": "train", "dev": "validation", "test": "test"}[data_type]


def build_loader(cfg, data_type: str = "train") -> DataLoader:
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    raw = load_dataset(cfg.dataset, cfg.dataset_config, split=_split_name(data_type))
    raw = _tokenize(raw, tok)
    raw = _chunk(raw, cfg.seq_len)
    raw = _limit(raw, cfg.max_train_samples if data_type == "train" else None)

    def _collate(rows):
        ids = torch.tensor([r["input_ids"] for r in rows], dtype=torch.long)
        mask = torch.ones_like(ids)
        return ids, mask, ids  # labels == input_ids; step.py shifts

    return DataLoader(
        raw, batch_size=cfg.batch_size,
        shuffle=(data_type == "train"), collate_fn=_collate,
    )
