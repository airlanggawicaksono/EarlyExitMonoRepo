"""Decoder-LM self-distillation early-exit backend (GPT-2 default; LLaMA-ready).

    from GeneralizableSelfDistilationEarlyExitTraining.backends.llama import Cfg, train
    train(Cfg(model_id="gpt2", mode="cascade"))

For real LLaMA training: swap `model_id="meta-llama/Llama-3.2-1B"` and
`lora_targets=("q_proj","v_proj")`. The model wrapper auto-locates the right
decoder + final-norm + lm_head for either family.
"""

from .config import Cfg
from .train import train

__all__ = ["Cfg", "train"]
