"""YOLO self-distillation early-exit backend (gelan-m-ee).

    from GeneralizableSelfDistilationEarlyExitTraining.backends.yolo import YoloCfg, train
    train(YoloCfg(mode="segd", data_yaml=..., weights=...))

Shares plan/Stage/MODE_BUILDERS with the classification framework; only model /
loss / data are YOLO-specific. Heads train fully (no LoRA — they are seeded from
the upstream gelan head, see model._load_weights). See methodology.md.
"""

from .config import YoloCfg
from .train import train

__all__ = ["YoloCfg", "train"]
