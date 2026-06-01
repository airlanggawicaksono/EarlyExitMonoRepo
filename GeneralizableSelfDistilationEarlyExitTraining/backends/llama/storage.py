"""Disk IO for decoder-LM self-distill: adapter + per-exit norm head + metrics."""

import json
import shutil

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
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def has_stage(cfg, stage) -> bool:
    return has_valid_result(stage_dir(cfg, stage.label) / "metrics.json")


def load_teacher(model, stage, cfg):
    d = stage_dir(cfg, stage.teacher_ckpt)
    adapters.load_adapter(model, stage.teacher_exit, d / "adapter")
    head_sd = torch.load(_head_path(d, stage.teacher_exit), map_location=cfg.device)
    model.heads[stage.teacher_exit].load_state_dict(head_sd)


# ---- mid-stage resume ------------------------------------------------------
def _resume_dir(cfg, stage):
    return stage_dir(cfg, stage.label) / "_resume"


def _load_lora_stage(model, stage, cfg, d):
    for e in stage.student_exits:
        adapters.load_adapter(model, e, d / "adapter")
        head_sd = torch.load(_head_path(d, e), map_location=cfg.device)
        model.heads[e].load_state_dict(head_sd)


def _load_full_stage(model, stage, cfg, d):
    sd = torch.load(d / "full_model.pt", map_location=cfg.device)
    model.load_state_dict(sd)


_LOADERS = {True: _load_lora_stage, False: _load_full_stage}


def save_step_ckpt(model, stage, cfg, step: int, trainer_state: dict):
    d = _resume_dir(cfg, stage)
    d.mkdir(parents=True, exist_ok=True)
    _SAVERS[stage.use_lora](model, stage, d)
    torch.save({"step": step, **trainer_state}, d / "trainer.pt")
    (d / "progress.json").write_text(json.dumps({"step": step}))


def load_step_ckpt(model, stage, cfg):
    d = _resume_dir(cfg, stage)
    pf = d / "progress.json"
    if not pf.exists():
        return 0, {}
    _LOADERS[stage.use_lora](model, stage, cfg, d)
    trainer = torch.load(d / "trainer.pt", map_location=cfg.device)
    step = int(trainer.pop("step"))
    return step, trainer


def clear_resume(cfg, stage):
    d = _resume_dir(cfg, stage)
    if d.exists():
        shutil.rmtree(d)
