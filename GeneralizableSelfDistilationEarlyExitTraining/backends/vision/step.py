"""Per-batch compute for vision. Returns (scalar_loss, components_dict).

Identical topology to BERT step.py — STEP_FNS dict on stage.kind. Only the
batch unpack differs: vision batch = (pixel_values, labels) instead of
BERT's 4-tuple.
"""

import torch

from . import adapters
from .losses import ce_loss, distill_loss, feature_hint_loss, kd_loss


def _inputs(batch):
    pixel_values, labels = batch
    return {"pixel_values": pixel_values}, labels


def forward_logits(model, batch):
    inputs, _ = _inputs(batch)
    return model(**inputs)


def supervise_step(model, stage, batch, cfg):
    exit_idx = stage.student_exits[0]
    adapters.activate(model, exit_idx)
    _, labels = _inputs(batch)
    logits = forward_logits(model, batch)[exit_idx]
    loss = ce_loss(logits, labels)
    return loss, {"teacher_ce": float(loss.detach())}


def joint_step(model, stage, batch, cfg):
    inputs, labels = _inputs(batch)
    logits, feats = model(**inputs, return_features=True)
    deep = stage.teacher_exit
    teacher = logits[deep].detach()
    teacher_feat = feats[deep].detach()
    teacher_ce = ce_loss(logits[deep], labels)
    components = {"teacher_ce": float(teacher_ce.detach())}
    total = teacher_ce
    for j in [i for i in stage.student_exits if i != deep]:
        ld = distill_loss(
            logits[j], teacher, labels,
            temperature=cfg.temperature, alpha_kd=cfg.alpha_kd,
            use_true_labels=cfg.use_true_labels,
        )
        lf = cfg.lambda_feat * feature_hint_loss(feats[j], teacher_feat)
        components[f"loss_e{j}"] = float(ld.detach())
        components[f"feat_e{j}"] = float(lf.detach())
        total = total + ld + lf
    return total, components


def distill_step(model, stage, batch, cfg):
    """SEGD faithful to LoRAExit (KD + CE only). `pairwise` is our star-topology
    variant -> adds a BYOT feature-hint L2 (student vs deepest-teacher feature)."""
    t_exit, s_exit = stage.teacher_exit, stage.student_exits[0]
    inputs, labels = _inputs(batch)
    want_feat = cfg.mode == "pairwise"

    adapters.activate(model, t_exit)
    with torch.no_grad():
        if want_feat:
            t_logits, t_feats = model(**inputs, return_features=True)
            teacher, teacher_feat = t_logits[t_exit], t_feats[t_exit]
        else:
            teacher = forward_logits(model, batch)[t_exit]
    adapters.activate(model, s_exit)
    if want_feat:
        s_logits, s_feats = model(**inputs, return_features=True)
        student, student_feat = s_logits[s_exit], s_feats[s_exit]
    else:
        student = forward_logits(model, batch)[s_exit]
    loss = distill_loss(
        student, teacher, labels,
        temperature=cfg.temperature, alpha_kd=cfg.alpha_kd,
        use_true_labels=cfg.use_true_labels,
    )
    comps = {f"loss_e{s_exit}": float(loss.detach())}
    # raw UNWEIGHTED components — mixed loss_e isn't comparable across tasks/modes.
    comps[f"ce_raw_e{s_exit}"] = float(ce_loss(student, labels).detach())
    comps[f"kd_raw_e{s_exit}"] = float(kd_loss(student, teacher, cfg.temperature).detach())
    if want_feat:
        mse = feature_hint_loss(student_feat, teacher_feat.detach())
        lf = cfg.lambda_feat * mse
        loss = loss + lf
        comps[f"feat_e{s_exit}"] = float(lf.detach())
        comps[f"feat_raw_e{s_exit}"] = float(mse.detach())
    return loss, comps


STEP_FNS = {
    "supervise": supervise_step,
    "joint": joint_step,
    "distill": distill_step,
}
