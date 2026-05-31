"""ViT self-distill config. IO-free dataclass — same shape as BERT's Cfg."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from .bootstrap import REPO_ROOT

_OUT = REPO_ROOT / "logs" / "selfdistill_vision"


@dataclass
class Cfg:
    dataset: str = "uoft-cs/cifar10"                      # HF datasets v3 requires namespaced repo ids
    mode: str = "joint"                                   # joint | pairwise | cascade
    model_id: str = "google/vit-base-patch16-224"
    n_exits: int = 12                                     # one per ViT block (ViT-base = 12)

    # distill
    temperature: float = 2.0
    alpha_kd: float = 0.9
    use_true_labels: bool = True

    # lora
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_targets: Tuple[str, ...] = ("query", "value")    # HF ViT attention names

    # optim
    image_size: int = 224
    epochs: int = 3
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_train_samples: Optional[int] = None
    seed: int = 42

    out_root: Path = field(default=_OUT)
    device: str = "cuda"

    @property
    def deepest(self) -> int:
        return self.n_exits - 1

    @property
    def run_dir(self) -> Path:
        return self.out_root / self.mode / self.dataset
