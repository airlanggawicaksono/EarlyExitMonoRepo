"""BERT self-distillation early-exit backend (ElasticBERT).

    from GeneralizableSelfDistilationEarlyExitTraining.backends.bert import Cfg, train
    train(Cfg(task="SST-2", mode="segd"))

Shares plan/Stage/MODE_BUILDERS with the rest of the framework; only model /
losses / adapters / data are BERT-specific.
"""

from .config import Cfg
from .train import train

__all__ = ["Cfg", "train"]
