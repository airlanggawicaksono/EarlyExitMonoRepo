"""All disk IO: checkpoints + metrics. Compute modules never touch the disk.

Layout per run:
    <run_dir>/<stage.label>/
        adapter/                 # peft adapter weights (lora stages)
        head_<exit>.pt           # per-exit classifier head
        full_model.pt            # whole model (joint stage)
        metrics.json             # resume marker (has_valid_result)

Resume: has_stage() reuses shared.has_valid_result — a stage with a valid
metrics.json is skipped, matching the benchmark continuation contract.
"""

import json

import torch

from . import adapters, bootstrap  # noqa: F401
from shared import has_valid_result  # type: ignore


def stage_dir(cfg, label: str):
    return cfg.run_dir / label


def _head_path(d, exit_idx: int):
    return d / f"head_{exit_idx}.pt"


def _save_lora_stage(model, stage, d):
    for e in stage.student_exits:
        adapters.save_adapter(model, e, d / "adapter")
        torch.save(model.heads[e].state_dict(), _head_path(d, e))


def _save_full_stage(model, stage, d):
    torch.save(model.state_dict(), d / "full_model.pt")
    for e in stage.student_exits:
        torch.save(model.heads[e].state_dict(), _head_path(d, e))


_SAVERS = {True: _save_lora_stage, False: _save_full_stage}


def save_stage(model, stage, cfg, metrics: dict):
    d = stage_dir(cfg, stage.label)
    d.mkdir(parents=True, exist_ok=True)
    _SAVERS[stage.use_lora](model, stage, d)
    write_metrics(d / "metrics.json", metrics)


def write_metrics(path, metrics: dict):
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def has_stage(cfg, stage) -> bool:
    return has_valid_result(stage_dir(cfg, stage.label) / "metrics.json")


def load_teacher(model, stage, cfg):
    """Reload teacher adapter + head (from teacher_ckpt stage) into the model so
    distill stages survive process restarts. No-op when there is no teacher."""
    d = stage_dir(cfg, stage.teacher_ckpt)
    adapters.load_adapter(model, stage.teacher_exit, d / "adapter")
    head_sd = torch.load(_head_path(d, stage.teacher_exit), map_location=cfg.device)
    model.heads[stage.teacher_exit].load_state_dict(head_sd)
