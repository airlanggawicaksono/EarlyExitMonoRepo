"""Disk IO for YOLO self-distill: head checkpoints + metrics + resume marker.

lora stage  -> save each trained exit's head state_dict (incl LoRAConv2d A/B).
joint stage -> save the whole net.
Resume reuses shared.has_valid_result (same contract as the rest of the repo).
"""

import json

import torch

from . import bootstrap  # noqa: F401
from shared import has_valid_result  # type: ignore


def stage_dir(cfg, label: str):
    return cfg.run_dir / label


def _head_path(d, exit_idx: int):
    return d / f"head_{exit_idx}.pt"


def _save_lora_stage(model, stage, d):
    for e in stage.student_exits:
        torch.save(model.heads[e].state_dict(), _head_path(d, e))


def _save_full_stage(model, stage, d):
    torch.save(model.net.state_dict(), d / "full_model.pt")


_SAVERS = {True: _save_lora_stage, False: _save_full_stage}


def save_stage(model, stage, cfg, metrics: dict):
    d = stage_dir(cfg, stage.label)
    d.mkdir(parents=True, exist_ok=True)
    _SAVERS[stage.use_lora](model, stage, d)
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def has_stage(cfg, stage) -> bool:
    return has_valid_result(stage_dir(cfg, stage.label) / "metrics.json")


def load_teacher(model, stage, cfg):
    """Reload the teacher exit's head (from its prior stage) so distill stages
    survive process restarts. No-op if absent (teacher still in memory)."""
    if stage.teacher_ckpt is None:
        return
    p = _head_path(stage_dir(cfg, stage.teacher_ckpt), stage.teacher_exit)
    if p.exists():
        model.heads[stage.teacher_exit].load_state_dict(
            torch.load(p, map_location=cfg.device)
        )
