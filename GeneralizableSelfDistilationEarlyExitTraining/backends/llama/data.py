"""LLM data IO. Three paths, picked by cfg:

  from_disk  cfg.dataset_path set -> load_from_disk on a pre-tokenized HF arrow.
             Fastest (no network, no re-tokenize) — used after preload-to-Drive.
  streaming  huge / web datasets (C4) — tokenize per example w/ pad+truncate
             to seq_len, .take() for max_train_samples. Each session re-streams.
  static     small / indexed datasets (wikitext) — tokenize + concat + chunk
             into fixed-length sequences, Subset for max_train_samples.

All paths yield (input_ids, attention_mask, labels) per batch; labels mirror
input_ids and step.py does the causal shift.
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset

from . import bootstrap  # noqa: F401
from datasets import load_dataset, load_from_disk  # type: ignore
from transformers import AutoTokenizer             # type: ignore


_DATASET_ALIASES = {"wikitext": "Salesforce/wikitext"}


def _resolve(name: str) -> str:
    return _DATASET_ALIASES.get(name, name)


def _split_name(data_type: str) -> str:
    return {"train": "train", "dev": "validation", "test": "validation"}[data_type]


# ---- static (chunked) path --------------------------------------------------
def _limit(dataset, n: Optional[int]):
    if n is None:
        return dataset
    return Subset(dataset, list(range(min(n, len(dataset)))))


def _tokenize_static(ds, tok):
    return ds.map(
        lambda batch: tok(batch["text"], add_special_tokens=False,
                          return_attention_mask=False),
        batched=True, remove_columns=ds.column_names,
    )


def _chunk_static(ds, seq_len: int):
    def _group(examples):
        all_ids = sum(examples["input_ids"], [])
        n = (len(all_ids) // seq_len) * seq_len
        chunks = [all_ids[i : i + seq_len] for i in range(0, n, seq_len)]
        return {"input_ids": chunks}
    return ds.map(_group, batched=True, batch_size=1000, remove_columns=ds.column_names)


def _collate_static(rows):
    ids = torch.tensor([r["input_ids"] for r in rows], dtype=torch.long)
    mask = torch.ones_like(ids)
    return ids, mask, ids


def _build_static(cfg, tok, data_type):
    raw = load_dataset(_resolve(cfg.dataset), cfg.dataset_config, split=_split_name(data_type))
    raw = _tokenize_static(raw, tok)
    raw = _chunk_static(raw, cfg.seq_len)
    raw = _limit(raw, cfg.max_train_samples if data_type == "train" else None)
    return DataLoader(
        raw, batch_size=cfg.batch_size,
        shuffle=(data_type == "train"), collate_fn=_collate_static,
    )


# ---- streaming (pad/truncate) path -----------------------------------------
def _tokenize_stream(tok, seq_len):
    def fn(ex):
        out = tok(ex["text"], add_special_tokens=False, truncation=True,
                  padding="max_length", max_length=seq_len,
                  return_attention_mask=True)
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}
    return fn


def _collate_stream(rows):
    ids = torch.tensor([r["input_ids"] for r in rows], dtype=torch.long)
    mask = torch.tensor([r["attention_mask"] for r in rows], dtype=torch.long)
    return ids, mask, ids


def _build_stream(cfg, tok, data_type):
    raw = load_dataset(_resolve(cfg.dataset), cfg.dataset_config,
                       split=_split_name(data_type), streaming=True)
    raw = raw.map(_tokenize_stream(tok, cfg.seq_len),
                  remove_columns=raw.column_names)
    if cfg.max_train_samples is not None and data_type == "train":
        raw = raw.take(cfg.max_train_samples)
    return DataLoader(raw, batch_size=cfg.batch_size, collate_fn=_collate_stream)


# ---- from-disk path (preloaded + tokenized) --------------------------------
def _build_from_disk(cfg, tok, data_type):
    """Load a pre-tokenized arrow dataset from cfg.dataset_path.
    Expects rows with input_ids + attention_mask columns, seq_len fixed.
    """
    ds = load_from_disk(str(cfg.dataset_path))
    if cfg.max_train_samples is not None and data_type == "train":
        ds = ds.select(range(min(cfg.max_train_samples, len(ds))))
    return DataLoader(
        ds, batch_size=cfg.batch_size,
        shuffle=(data_type == "train"), collate_fn=_collate_stream,
    )


_BUILDERS = {True: _build_stream, False: _build_static}


def build_loader(cfg, data_type: str = "train") -> DataLoader:
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if getattr(cfg, "dataset_path", None):
        return _build_from_disk(cfg, tok, data_type)
    return _BUILDERS[getattr(cfg, "streaming", False)](cfg, tok, data_type)
