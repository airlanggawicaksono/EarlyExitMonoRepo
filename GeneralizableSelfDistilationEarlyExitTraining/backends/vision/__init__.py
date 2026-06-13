"""Vision self-distillation early-exit backend (HuggingFace ViT).

    from GeneralizableSelfDistilationEarlyExitTraining.backends.vision import Cfg, train
    train(Cfg(dataset="cifar10", mode="segd"))

Shares plan/Stage/MODE_BUILDERS with the rest of the framework; only model /
losses / adapters / data are vision-specific.
"""

from .config import Cfg
from .train import train

__all__ = ["Cfg", "train"]
