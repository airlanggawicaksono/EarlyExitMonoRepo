"""Per-batch compute for YOLO self-distill. Returns (scalar_loss, components_dict).

sup_loss(model, exit_idx, exit_out, targets, imgs) -> scalar TAL term (injected;
see tal.py). detection_kd detaches the teacher internally.

Backbone is frozen for head-only stages -> cache feats once (no_grad) then run heads.
joint trains the backbone -> one grad-enabled exit_outputs pass.
"""

import torch

from .loss import detection_kd, feature_hint_loss


def _kd_components(student_out, teacher_out, cfg, bs: int):
    """KD scaled to TAL magnitude. Returns (combined_kd, kd_box_unscaled, kd_cls_unscaled)."""
    kd_box, kd_cls = detection_kd(student_out, teacher_out, reg_max=cfg.reg_max, nc=cfg.nc, tau=cfg.tau)
    combined = bs * (cfg.box_gain * kd_box + cfg.cls_gain * kd_cls)
    return combined, kd_box, kd_cls


def _student_loss(model, exit_idx, student_out, teacher_out, targets, imgs, cfg, sup_loss):
    bs = imgs.shape[0]
    kd, kd_box, kd_cls = _kd_components(student_out, teacher_out, cfg, bs)
    sup = sup_loss(model, exit_idx, student_out, targets, imgs)
    total = cfg.alpha_kd * kd + (1.0 - cfg.alpha_kd) * sup
    comp = {
        f"loss_e{exit_idx}":   float(total.detach()),
        f"sup_e{exit_idx}":    float(sup.detach()),
        f"kd_e{exit_idx}":     float(kd.detach()),
        f"kd_box_e{exit_idx}": float(kd_box.detach()),
        f"kd_cls_e{exit_idx}": float(kd_cls.detach()),
    }
    return total, comp


def joint_step(model, stage, batch, cfg, sup_loss):
    """Full finetune. One forward, all exits. Deepest = TAL + soft-label source."""
    imgs, targets = batch
    outs = model.exit_outputs(imgs)
    deep = stage.teacher_exit
    teacher_sup = sup_loss(model, deep, outs[deep], targets, imgs)
    components = {"teacher_sup": float(teacher_sup.detach())}
    total = teacher_sup
    for i in stage.student_exits:
        if i == deep:
            continue
        ld, comp = _student_loss(model, i, outs[i], outs[deep], targets, imgs, cfg, sup_loss)
        components.update(comp)
        total = total + ld
    return total, components


def supervise_step(model, stage, batch, cfg, sup_loss):
    """Train one exit's head adapter on TAL (build the teacher)."""
    imgs, targets = batch
    with torch.no_grad():
        y = model.backbone_feats(imgs)
    e = stage.student_exits[0]
    out = model.head_output(e, y)
    loss = sup_loss(model, e, out, targets, imgs)
    return loss, {"teacher_sup": float(loss.detach())}


def distill_step(model, stage, batch, cfg, sup_loss):
    """One student vs frozen teacher (both heads on cached frozen backbone).

    segd stays FAITHFUL (output KD only: kd_box + kd_cls). `pairwise` is our own
    variant -> also adds a BYOT penultimate-feature L2 (student vs deepest teacher),
    gated on cfg.mode so segd is untouched."""
    imgs, targets = batch
    want_feat = cfg.mode == "pairwise"
    with torch.no_grad():
        y = model.backbone_feats(imgs)
        teacher = model.head_output(stage.teacher_exit, y)
        teacher_pen = model.head_penult(stage.teacher_exit, y) if want_feat else None
    s = stage.student_exits[0]
    student = model.head_output(s, y)
    total, comp = _student_loss(model, s, student, teacher, targets, imgs, cfg, sup_loss)
    if want_feat:
        student_pen = model.head_penult(s, y)
        mse = feature_hint_loss(student_pen, teacher_pen)
        lf = cfg.lambda_feat * mse
        total = total + lf
        comp[f"feat_e{s}"] = float(lf.detach())          # scaled (λ·MSE)
        comp[f"feat_raw_e{s}"] = float(mse.detach())     # raw MSE, comparable
    return total, comp


STEP_FNS = {
    "supervise": supervise_step,
    "joint": joint_step,
    "distill": distill_step,
}
