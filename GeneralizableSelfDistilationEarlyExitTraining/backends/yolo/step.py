"""Per-batch compute for YOLO self-distill. Returns scalar loss. No IO.

sup_loss(model, exit_idx, exit_out, targets, imgs) -> scalar TAL term (injected;
see tal.py). detection_kd detaches the teacher internally.

Backbone is frozen for lora stages -> cache feats once (no_grad) then run heads.
joint trains the backbone -> one grad-enabled exit_outputs pass.
"""

import torch

from .loss import detection_kd


def _kd(student_out, teacher_out, cfg, bs: int):
    """KD scaled to TAL magnitude: per-component gains * batch_size, mirror yolov9 hyp."""
    kd_box, kd_cls = detection_kd(student_out, teacher_out, reg_max=cfg.reg_max, nc=cfg.nc, tau=cfg.tau)
    return bs * (cfg.box_gain * kd_box + cfg.cls_gain * kd_cls)


def _student_loss(model, exit_idx, student_out, teacher_out, targets, imgs, cfg, sup_loss):
    kd = _kd(student_out, teacher_out, cfg, imgs.shape[0])
    sup = sup_loss(model, exit_idx, student_out, targets, imgs)
    return cfg.alpha_kd * kd + (1.0 - cfg.alpha_kd) * sup


def joint_step(model, stage, batch, cfg, sup_loss):
    """Full finetune. One forward, all exits. Deepest = TAL + soft-label source."""
    imgs, targets = batch
    outs = model.exit_outputs(imgs)
    deep = stage.teacher_exit
    loss = sup_loss(model, deep, outs[deep], targets, imgs)
    for i in stage.student_exits:
        if i == deep:
            continue
        loss = loss + _student_loss(model, i, outs[i], outs[deep], targets, imgs, cfg, sup_loss)
    return loss


def supervise_step(model, stage, batch, cfg, sup_loss):
    """Train one exit's head adapter on TAL (build the teacher)."""
    imgs, targets = batch
    with torch.no_grad():
        y = model.backbone_feats(imgs)
    e = stage.student_exits[0]
    out = model.head_output(e, y)
    return sup_loss(model, e, out, targets, imgs)


def distill_step(model, stage, batch, cfg, sup_loss):
    """One student vs frozen teacher (both heads on cached frozen backbone)."""
    imgs, targets = batch
    with torch.no_grad():
        y = model.backbone_feats(imgs)
        teacher = model.head_output(stage.teacher_exit, y)
    s = stage.student_exits[0]
    student = model.head_output(s, y)
    return _student_loss(model, s, student, teacher, targets, imgs, cfg, sup_loss)


def cascade_step(model, stage, batch, cfg, sup_loss):
    """All head adapters at once. Deepest = TAL; EVERY shallower exit learns
    from the deepest head's detached output. One backward updates every head."""
    imgs, targets = batch
    with torch.no_grad():
        y = model.backbone_feats(imgs)
    n = model.n_exits
    outs = [model.head_output(i, y) for i in range(n)]
    teacher = [t.detach() for t in outs[n - 1]]  # 3-scale list, each detached
    loss = sup_loss(model, n - 1, outs[n - 1], targets, imgs)
    for i in range(n - 1):
        loss = loss + _student_loss(model, i, outs[i], teacher, targets, imgs, cfg, sup_loss)
    return loss


STEP_FNS = {
    "supervise": supervise_step,
    "joint": joint_step,
    "distill": distill_step,
    "cascade": cascade_step,
}
