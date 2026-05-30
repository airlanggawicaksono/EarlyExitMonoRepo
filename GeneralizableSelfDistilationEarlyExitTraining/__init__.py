"""Generalizable self-distillation early-exit training.

Public API:
    from GeneralizableSelfDistilationEarlyExitTraining import Cfg, train
    train(Cfg(task="SST-2", mode="cascade"))

Modes (plan.MODE_BUILDERS):
    joint    — BYOT, one forward, all exits, deepest=teacher. No LoRA.
    pairwise — deepest=fixed teacher; each shallower exit distilled (own adapter).
    cascade  — exit k distilled from k+1, walking down. Per-exit adapter.
"""

from .config import Cfg
from .plan import MODE_BUILDERS
from .train import train

__all__ = ["Cfg", "train", "MODE_BUILDERS"]
