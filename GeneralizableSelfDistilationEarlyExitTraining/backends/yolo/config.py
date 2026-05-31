"""YOLO self-distill config. IO-free dataclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .bootstrap import REPO_ROOT

_YOLO = REPO_ROOT / "AnyTimeYolo"
_EE_YAML = _YOLO / "src" / "early_exit" / "configs" / "gelan-s-ee.yaml"
_OUT = REPO_ROOT / "logs" / "selfdistill_yolo"


@dataclass
class YoloCfg:
    mode: str = "cascade"                 # joint | pairwise | cascade
    dataset: str = "coco"
    ee_yaml: Path = field(default=_EE_YAML)
    weights: Optional[Path] = None        # pretrained gelan-s.pt (None -> from scratch)
    data_yaml: Optional[Path] = None      # COCO data.yaml (set by cli/notebook)

    n_exits: int = 5
    nc: int = 80
    reg_max: int = 16                     # DDetect DFL bins/side (verify on model)
    img_size: int = 640

    # distill
    tau: float = 0.5                      # teacher-foreground confidence threshold
    alpha_kd: float = 0.9                 # KD weight; (1-alpha) on supervised TAL

    # loss-component gains (mirror yolov9 hyp.scratch-high; KD scaled to match TAL)
    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5                 # currently unused (TAL owns dfl)

    # lora (head convs)
    lora_r: int = 8
    lora_alpha: int = 16

    # optim
    epochs: int = 1
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 5e-4
    max_grad_norm: float = 10.0
    max_train_batches: Optional[int] = None   # dry-run cap

    out_root: Path = field(default=_OUT)
    device: str = "cuda"

    @property
    def deepest(self) -> int:
        return self.n_exits - 1

    @property
    def run_dir(self) -> Path:
        return self.out_root / self.mode / self.dataset
