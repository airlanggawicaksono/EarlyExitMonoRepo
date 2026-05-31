"""COCO dataloader via yolov9's create_dataloader. All data IO here.

yolov9 collate yields (imgs, targets, paths, shapes); we consume imgs + targets.
Hyp (augment + loss gains) loaded from a yolov9 hyp yaml.

TODO(colab): verify create_dataloader signature + hyp path against the pinned
yolov9 checkout; both live under AnyTimeYolo/src/model/yolov9.
"""

from pathlib import Path

import yaml

from . import bootstrap  # noqa: F401  (injects sys.path)
from utils.dataloaders import create_dataloader  # type: ignore

_YOLOV9 = bootstrap._YOLO_SRC / "model" / "yolov9"
_HYP = _YOLOV9 / "data" / "hyps" / "hyp.scratch-high.yaml"
_STRIDE = 32


def _load_hyp():
    with open(_HYP) as f:
        return yaml.safe_load(f)


def build_loader(cfg):
    with open(cfg.data_yaml) as f:
        data = yaml.safe_load(f)
    train_path = data["train"]
    loader, _dataset = create_dataloader(
        train_path,
        cfg.img_size,
        cfg.batch_size,
        _STRIDE,
        hyp=_load_hyp(),
        augment=True,
        rect=False,
        workers=4,
        shuffle=True,
        prefix="selfdistill: ",
    )
    return loader
