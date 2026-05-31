"""YOLO self-distillation early-exit backend (gelan-s-ee).

    from GeneralizableSelfDistilationEarlyExitTraining.backends.yolo import YoloCfg, train
    train(YoloCfg(mode="cascade", data_yaml=..., weights=...))

Shares plan/Stage/MODE_BUILDERS with the classification framework; only model /
loss / adapters / data are YOLO-specific. See methodology.md.
"""

from .config import YoloCfg
from .train import train

__all__ = ["YoloCfg", "train"]
