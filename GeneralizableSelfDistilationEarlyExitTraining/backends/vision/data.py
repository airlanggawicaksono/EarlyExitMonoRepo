"""Vision data IO: HF datasets + AutoImageProcessor. cifar10 / cifar100 /
imagenet-1k supported by key-mapping the differing field names.
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset

from . import bootstrap  # noqa: F401
from datasets import load_dataset                # type: ignore
from transformers import AutoImageProcessor      # type: ignore


_IMG_KEY = {"cifar10": "img", "cifar100": "img", "imagenet-1k": "image"}
_LBL_KEY = {"cifar10": "label", "cifar100": "fine_label", "imagenet-1k": "label"}
_NUM_LABELS = {"cifar10": 10, "cifar100": 100, "imagenet-1k": 1000}


def count_labels(dataset_name: str) -> int:
    return _NUM_LABELS[dataset_name]


def _limit(dataset, n: Optional[int]):
    if n is None:
        return dataset
    return Subset(dataset, list(range(min(n, len(dataset)))))


def _split_name(dataset_name: str, data_type: str) -> str:
    return {"train": "train", "dev": "test", "test": "test"}[data_type]


def build_loader(cfg, data_type: str = "train") -> DataLoader:
    img_key, lbl_key = _IMG_KEY[cfg.dataset], _LBL_KEY[cfg.dataset]
    split = _split_name(cfg.dataset, data_type)
    ds = load_dataset(cfg.dataset, split=split)
    proc = AutoImageProcessor.from_pretrained(cfg.model_id)

    def _collate(rows):
        imgs = [r[img_key].convert("RGB") for r in rows]
        px = proc(imgs, return_tensors="pt")["pixel_values"]
        labels = torch.tensor([r[lbl_key] for r in rows], dtype=torch.long)
        return px, labels

    ds = _limit(ds, cfg.max_train_samples if data_type == "train" else None)
    return DataLoader(ds, batch_size=cfg.batch_size,
                      shuffle=(data_type == "train"), collate_fn=_collate)
