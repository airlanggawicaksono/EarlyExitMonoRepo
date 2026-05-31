"""Vision data IO: HF datasets + AutoImageProcessor. cifar10 / cifar100 /
imagenet-1k supported by key-mapping the differing field names.
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset

from . import bootstrap  # noqa: F401
from datasets import load_dataset                # type: ignore
from transformers import AutoImageProcessor      # type: ignore


# Legacy aliases for HF datasets v3 (which requires namespaced repo ids).
# Accept both forms so older configs / notebooks keep working.
_ALIASES = {
    "cifar10": "uoft-cs/cifar10",
    "cifar100": "uoft-cs/cifar100",
}


def _resolve(name: str) -> str:
    return _ALIASES.get(name, name)


_IMG_KEY = {
    "uoft-cs/cifar10": "img", "uoft-cs/cifar100": "img",
    "imagenet-1k": "image",
}
_LBL_KEY = {
    "uoft-cs/cifar10": "label", "uoft-cs/cifar100": "fine_label",
    "imagenet-1k": "label",
}
_NUM_LABELS = {
    "uoft-cs/cifar10": 10, "uoft-cs/cifar100": 100, "imagenet-1k": 1000,
}


def count_labels(dataset_name: str) -> int:
    return _NUM_LABELS[_resolve(dataset_name)]


def _limit(dataset, n: Optional[int]):
    if n is None:
        return dataset
    return Subset(dataset, list(range(min(n, len(dataset)))))


def _split_name(dataset_name: str, data_type: str) -> str:
    return {"train": "train", "dev": "test", "test": "test"}[data_type]


def build_loader(cfg, data_type: str = "train") -> DataLoader:
    name = _resolve(cfg.dataset)
    img_key, lbl_key = _IMG_KEY[name], _LBL_KEY[name]
    split = _split_name(name, data_type)
    ds = load_dataset(name, split=split)
    proc = AutoImageProcessor.from_pretrained(cfg.model_id)

    def _collate(rows):
        imgs = [r[img_key].convert("RGB") for r in rows]
        px = proc(imgs, return_tensors="pt")["pixel_values"]
        labels = torch.tensor([r[lbl_key] for r in rows], dtype=torch.long)
        return px, labels

    ds = _limit(ds, cfg.max_train_samples if data_type == "train" else None)
    return DataLoader(ds, batch_size=cfg.batch_size,
                      shuffle=(data_type == "train"), collate_fn=_collate)
