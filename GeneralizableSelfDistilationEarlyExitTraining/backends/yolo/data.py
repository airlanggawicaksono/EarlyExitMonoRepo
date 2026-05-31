"""COCO dataloader via yolov9's create_dataloader. All data IO here.

yolov9 collate yields (imgs, targets, paths, shapes); we consume imgs + targets.

Hyp = inlined copy of yolov9's `hyp.scratch-high.yaml`. The vendored yolov9
checkout in this repo doesn't ship the `data/hyps/` dir, so we keep the dict
here (single source, no file IO). Canonical values from official
WongKinYiu/yolov9.
"""

import yaml

from . import bootstrap  # noqa: F401  (injects sys.path)
from .hyp import HYP
from utils.dataloaders import create_dataloader  # type: ignore

_STRIDE = 32


def build_loader(cfg):
    from pathlib import Path

    with open(cfg.data_yaml) as f:
        data = yaml.safe_load(f)
    # yolov9's create_dataloader receives an absolute / repo-relative path; the
    # `path` key from the data yaml is NOT auto-joined by the loader (yolov5
    # does that join in train.py, which we don't use). Join here.
    root = Path(data.get("path", "."))
    train_path = str(root / data["train"])
    loader, _dataset = create_dataloader(
        train_path,
        cfg.img_size,
        cfg.batch_size,
        _STRIDE,
        hyp=HYP,
        augment=True,
        rect=False,
        workers=4,
        shuffle=True,
        prefix="selfdistill: ",
    )
    return loader
