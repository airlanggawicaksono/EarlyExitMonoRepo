"""YOLO self-distill orchestration. Reuses the shared plan (MODE_BUILDERS) +
Stage; only setup / step / save are YOLO-specific. No mode branches — dict
dispatch on stage.kind / stage.use_lora.
"""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from tqdm.auto import tqdm

from . import adapters, storage
from .data import build_loader
from .model import build_model
from .step import STEP_FNS
from .tal import build_sup_loss
from ...plan import MODE_BUILDERS
from shared import TrainingProfiler  # type: ignore


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch, cfg):
    # yolov9 collate -> (imgs, targets, paths, shapes); we use imgs + targets
    imgs, targets = batch[0], batch[1]
    return imgs.to(cfg.device).float() / 255.0, targets.to(cfg.device)


# ---- per-stage adapter setup ------------------------------------------------
def setup_full(model, stage, cfg):
    for p in model.parameters():
        p.requires_grad_(True)


def setup_supervise(model, stage, cfg):
    adapters.freeze_all(model)
    adapters.set_exit(model, stage.student_exits[0], enabled=True, trainable=True)


def setup_distill(model, stage, cfg):
    adapters.freeze_all(model)
    adapters.set_exit(model, stage.student_exits[0], enabled=True, trainable=True)
    adapters.set_exit(model, stage.teacher_exit, enabled=True, trainable=False)
    storage.load_teacher(model, stage, cfg)


_SETUP = {
    "supervise": setup_supervise,
    "joint": setup_full,
    "distill": setup_distill,
}


def _trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def _metrics(stage, cfg, last_loss):
    return {
        "mode": cfg.mode,
        "dataset": cfg.dataset,
        "kind": stage.kind,
        "label": stage.label,
        "student_exits": list(stage.student_exits),
        "teacher_exit": stage.teacher_exit,
        "epochs": cfg.epochs,
        "final_loss": round(last_loss, 6),
    }


def run_stage(model, stage, loader, cfg, sup_loss):
    if storage.has_stage(cfg, stage):
        print(f"[skip] stage done: {stage.label}")
        return

    _SETUP[stage.kind](model, stage, cfg)
    optim = SGD(_trainable(model), lr=cfg.lr, momentum=0.937, weight_decay=cfg.weight_decay)
    step_fn = STEP_FNS[stage.kind]
    trainable = _trainable(model)

    sd = storage.stage_dir(cfg, stage.label)
    sd.mkdir(parents=True, exist_ok=True)

    resume_step, trainer_state = storage.load_step_ckpt(model, stage, cfg)
    if "optim" in trainer_state:
        optim.load_state_dict(trainer_state["optim"])
    if resume_step:
        print(f"[{stage.label}] resuming from step {resume_step}")

    model.train()
    last = float(trainer_state.get("last", 0.0))
    steps_per_epoch = len(loader)
    start_epoch = resume_step // steps_per_epoch if steps_per_epoch else 0
    skip_in_first = resume_step % steps_per_epoch if steps_per_epoch else 0
    global_step = start_epoch * steps_per_epoch
    with TrainingProfiler(str(sd / "train_metrics.json"), batch_size=cfg.batch_size) as prof:
        for epoch in range(start_epoch, cfg.epochs):
            prof.begin_epoch(epoch)
            pbar = tqdm(loader, desc=f"[{stage.label}] ep{epoch + 1}/{cfg.epochs}", leave=False)
            for bi, batch in enumerate(pbar):
                if epoch == start_epoch and bi < skip_in_first:
                    global_step += 1
                    continue
                prof.step_begin()
                loss, components = step_fn(model, stage, _to_device(batch, cfg), cfg, sup_loss)
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
                optim.step()
                optim.zero_grad()
                last = float(loss.detach())
                prof.log_step(global_step, loss=last, lr=optim.param_groups[0]["lr"], **components)
                global_step += 1
                _post = {"loss": f"{last:.4f}", "step": global_step}
                _sup = next((v for k, v in components.items() if k.startswith("sup_e")), None)
                _mse = next((v for k, v in components.items() if k.startswith("feat_raw_e")), None)
                if _sup is not None: _post["sup"] = f"{_sup:.4f}"  # raw TAL sup (yolo's label loss)
                if _mse is not None: _post["mse"] = f"{_mse:.4f}"  # raw feature MSE
                pbar.set_postfix(**_post)
                if cfg.save_every_steps and global_step % cfg.save_every_steps == 0:
                    storage.save_step_ckpt(model, stage, cfg, global_step,
                                           {"optim": optim.state_dict(), "last": last})
                if cfg.max_train_batches is not None and bi + 1 >= cfg.max_train_batches:
                    break
            prof.end_epoch(epoch)
            print(f"[{stage.label}] epoch {epoch + 1}/{cfg.epochs} loss={last:.4f}")

    storage.save_stage(model, stage, cfg, _metrics(stage, cfg, last))
    storage.clear_resume(cfg, stage)
    print(f"[save] stage {stage.label} -> {storage.stage_dir(cfg, stage.label)}")


def train(cfg):
    _set_seed(42)
    model = build_model(cfg, cfg.ee_yaml, cfg.weights, cfg.nc).to(cfg.device)

    plan = MODE_BUILDERS[cfg.mode](cfg)
    if any(s.use_lora for s in plan):
        adapters.attach(model, cfg)
        model.to(cfg.device)        # LoRAConv2d creates new modules on CPU; re-pin to device

    loader = build_loader(cfg)
    sup_loss = build_sup_loss(model, cfg)
    for stage in plan:
        run_stage(model, stage, loader, cfg, sup_loss)
    return cfg.run_dir
