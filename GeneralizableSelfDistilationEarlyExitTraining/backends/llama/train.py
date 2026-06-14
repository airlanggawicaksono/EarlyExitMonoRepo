"""Decoder-LM orchestration. Shared plan.MODE_BUILDERS; per-stage setup +
step + saver dispatched via dicts. No mode branches.
"""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup  # type: ignore

from . import adapters, storage
from .data import build_loader
from .model import build_model
from ...plan import MODE_BUILDERS
from .step import STEP_FNS
from shared import TrainingProfiler  # type: ignore


def _set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _to_device(batch, cfg):
    return tuple(t.to(cfg.device) for t in batch)


def _noop_attach(model, cfg):
    return model


_ATTACH = {True: adapters.attach, False: _noop_attach}


# ---- per-stage trainable + adapter setup ------------------------------------
def _freeze_heads(model):
    for p in model.heads.parameters():
        p.requires_grad_(False)


def _enable_head(model, exit_idx):
    for p in model.heads[exit_idx].parameters():
        p.requires_grad_(True)


def setup_single(model, stage, cfg):
    exit_idx = stage.student_exits[0]
    adapters.set_adapter_trainable(model, exit_idx)
    adapters.activate(model, exit_idx)
    _freeze_heads(model)
    _enable_head(model, exit_idx)


def setup_full(model, stage, cfg):
    for p in model.parameters():
        p.requires_grad_(True)


def setup_distill(model, stage, cfg):
    setup_single(model, stage, cfg)
    storage.load_teacher(model, stage, cfg)


_SETUP = {
    "supervise": setup_single,
    "joint": setup_full,
    "distill": setup_distill,
}


# ---- optim ------------------------------------------------------------------
def _trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def _est_total_steps(loader, cfg):
    """len(loader) for static; for streaming (IterableDataset), estimate from
    max_train_samples // batch_size, falling back to a safe constant."""
    try:
        return len(loader) * cfg.epochs
    except TypeError:
        if cfg.max_train_samples is not None:
            return max(1, cfg.max_train_samples // cfg.batch_size) * cfg.epochs
        return 1000 * cfg.epochs


def _build_optim(model, loader, cfg):
    params = _trainable(model)
    optim = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = _est_total_steps(loader, cfg)
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(cfg.warmup_ratio * total), num_training_steps=total
    )
    return optim, sched


def _metrics(stage, cfg, final_loss):
    return {
        "mode": cfg.mode,
        "dataset": f"{cfg.dataset}/{cfg.dataset_config}",
        "kind": stage.kind,
        "label": stage.label,
        "student_exits": list(stage.student_exits),
        "teacher_exit": stage.teacher_exit,
        "epochs": cfg.epochs,
        "final_loss": round(final_loss, 6),
    }


# ---- stage execution --------------------------------------------------------
def run_stage(model, stage, loader, cfg):
    if storage.has_stage(cfg, stage):
        print(f"[skip] stage done: {stage.label}")
        return

    _SETUP[stage.kind](model, stage, cfg)
    optim, sched = _build_optim(model, loader, cfg)
    step_fn = STEP_FNS[stage.kind]
    trainable = _trainable(model)

    sd = storage.stage_dir(cfg, stage.label)
    sd.mkdir(parents=True, exist_ok=True)

    resume_step, trainer_state = storage.load_step_ckpt(model, stage, cfg)
    if "optim" in trainer_state:
        optim.load_state_dict(trainer_state["optim"])
    if "sched" in trainer_state:
        sched.load_state_dict(trainer_state["sched"])
    if resume_step:
        print(f"[{stage.label}] resuming from step {resume_step}")

    model.train()
    last = float(trainer_state.get("last", 0.0))
    try:
        steps_per_epoch = len(loader)
    except TypeError:
        steps_per_epoch = 0  # streaming IterableDataset
    if steps_per_epoch:
        start_epoch = resume_step // steps_per_epoch
        skip_in_first = resume_step % steps_per_epoch
        global_step = start_epoch * steps_per_epoch
    else:
        start_epoch = 0
        skip_in_first = resume_step
        global_step = 0
    with TrainingProfiler(
        str(sd / "train_metrics.json"),
        batch_size=cfg.batch_size,
        seq_length=cfg.seq_len,
    ) as prof:
        for epoch in range(start_epoch, cfg.epochs):
            prof.begin_epoch(epoch)
            pbar = tqdm(loader, desc=f"[{stage.label}] ep{epoch + 1}/{cfg.epochs}", leave=False)
            for i, batch in enumerate(pbar):
                if epoch == start_epoch and i < skip_in_first:
                    global_step += 1
                    continue
                prof.step_begin()
                loss, components = step_fn(model, stage, _to_device(batch, cfg), cfg)
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
                optim.step(); sched.step(); optim.zero_grad()
                last = float(loss.detach())
                prof.log_step(global_step, loss=last, lr=optim.param_groups[0]["lr"], **components)
                global_step += 1
                _post = {"loss": f"{last:.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                _ce = next((v for k, v in components.items() if k.startswith("ce_raw_e")), None)
                _mse = next((v for k, v in components.items() if k.startswith("feat_raw_e")), None)
                if _ce is not None: _post["ce"] = f"{_ce:.4f}"     # raw CE only (not the mixed loss)
                if _mse is not None: _post["mse"] = f"{_mse:.4f}"  # raw feature MSE
                pbar.set_postfix(**_post)
                if cfg.save_every_steps and global_step % cfg.save_every_steps == 0:
                    storage.save_step_ckpt(model, stage, cfg, global_step,
                                           {"optim": optim.state_dict(), "sched": sched.state_dict(), "last": last})
            prof.end_epoch(epoch)
            print(f"[{stage.label}] epoch {epoch + 1}/{cfg.epochs} loss={last:.4f}")

    storage.save_stage(model, stage, cfg, _metrics(stage, cfg, last))
    storage.clear_resume(cfg, stage)
    print(f"[save] stage {stage.label} -> {storage.stage_dir(cfg, stage.label)}")


def train(cfg):
    _set_seed(cfg.seed)
    model = build_model(cfg).to(cfg.device)

    plan = MODE_BUILDERS[cfg.mode](cfg)
    _ATTACH[any(s.use_lora for s in plan)](model, cfg)

    loader = build_loader(cfg, "train")
    for stage in plan:
        run_stage(model, stage, loader, cfg)
    return cfg.run_dir
