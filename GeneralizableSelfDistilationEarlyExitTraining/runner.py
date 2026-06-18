"""Cross-backend training grid runner. Train one (backend, cfg) at a time,
push checkpoints to HF between iterations.

Example:

    from GeneralizableSelfDistilationEarlyExitTraining.backends.bert import Cfg as BertCfg, train as bert_train
    from GeneralizableSelfDistilationEarlyExitTraining.runner import GridItem, run_grid

    items = [
        GridItem("bert-sst2-segd",   bert_train, BertCfg(task="SST-2", mode="segd")),
        GridItem("bert-mnli-segd",   bert_train, BertCfg(task="MNLI", mode="segd")),
    ]
    run_grid(items, hf_user=os.environ["HF_USER"], hf_token=os.environ["HF_TOKEN"])

Each item runs to completion before the next starts. The train fn writes to
cfg.out_root; the notebook rsyncs that tree to Drive on its own (background
thread), so the runner only handles training + the per-item HF push.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from . import sync


@dataclass
class GridItem:
    """One training run + its HF repo label."""
    label: str           # unique slug; used as HF repo name suffix
    train_fn: Callable   # backend's train(cfg) function
    cfg: Any             # backend's Cfg dataclass instance


def _maybe_push_ckpts(run_dir, item, hf_user, hf_token, repo_prefix, private):
    if not (hf_user and hf_token):
        return
    repo_id = f"{hf_user}/{repo_prefix}-{item.label}"
    print(f"[runner] push ckpts -> {repo_id}")
    sync.push_ckpts_to_hf(run_dir, repo_id, hf_token, private=private)


def run_grid(
    items,
    *,
    hf_user: Optional[str] = None,
    hf_token: Optional[str] = None,
    repo_prefix: str = "selfdistill",
    private: bool = True,
):
    """Run every item, pushing its checkpoints to HF after each (HF push is
    opt-in: skipped when hf_user/hf_token are None)."""
    for item in items:
        print(f"\n[runner] start: {item.label}")
        run_dir = Path(item.train_fn(item.cfg))
        _maybe_push_ckpts(run_dir, item, hf_user, hf_token, repo_prefix, private)
        print(f"[runner] done: {item.label}")
