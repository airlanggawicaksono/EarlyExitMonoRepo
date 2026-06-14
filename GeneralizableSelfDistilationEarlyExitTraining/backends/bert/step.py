"""Per-batch compute. Returns (scalar_loss, components_dict). No IO, no optimizer.

The training loop backwards `scalar_loss` and logs every key in `components_dict`
per-step (teacher_ce, loss_e{j}, etc) so metric analysis has full breakdown.
"""

import torch

from . import adapters
from .losses import ce_loss, distill_loss, feature_hint_loss, kd_loss


def _inputs(batch):
    ids, mask, types, labels = batch
    return dict(input_ids=ids, attention_mask=mask, token_type_ids=types), labels


def forward_logits(model, batch):
    """All exit logits for one batch -> list[Tensor]."""
    inputs, _ = _inputs(batch)
    return model(**inputs)


def supervise_step(model, stage, batch, cfg):
    """CE only — train one exit (the teacher) on true labels."""
    exit_idx = stage.student_exits[0]
    adapters.activate(model, exit_idx)
    _, labels = _inputs(batch)
    logits = forward_logits(model, batch)[exit_idx]
    loss = ce_loss(logits, labels)
    return loss, {"teacher_ce": float(loss.detach())}


def joint_step(model, stage, batch, cfg):
    """BYOT: one forward, deepest exit supervised, all shallower distilled from it.
    Adds feature-hint L2 (MSE between shallow pooled feat and deepest pooled feat,
    detached) — third term of original Zhang et al. 2019 loss."""
    inputs, labels = _inputs(batch)
    logits, feats = model(**inputs, return_features=True)
    deep = stage.teacher_exit
    teacher = logits[deep].detach()
    teacher_feat = feats[deep].detach()

    teacher_ce = ce_loss(logits[deep], labels)
    components = {"teacher_ce": float(teacher_ce.detach())}
    total = teacher_ce
    for j in stage.student_exits:
        if j == deep:
            continue
        ld = distill_loss(
            logits[j], teacher, labels,
            temperature=cfg.temperature,
            alpha_kd=cfg.alpha_kd,
            use_true_labels=cfg.use_true_labels,
        )
        lf = cfg.lambda_feat * feature_hint_loss(feats[j], teacher_feat)
        components[f"loss_e{j}"] = float(ld.detach())
        components[f"feat_e{j}"] = float(lf.detach())
        total = total + ld + lf
    return total, components


def distill_step(model, stage, batch, cfg):
    """One student vs one frozen teacher. Two forwards (different adapters).

    SEGD is kept FAITHFUL to LoRAExit (eq 4): KD + CE only, no feature term.
    `pairwise` is our own star-topology variant (every student <- deepest), so it
    ADDITIONALLY gets a BYOT feature-hint L2 between the student feature and the
    deepest-teacher feature. Gated on cfg.mode so segd is untouched.
    """
    t_exit = stage.teacher_exit
    s_exit = stage.student_exits[0]
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
        temperature=cfg.temperature,
        alpha_kd=cfg.alpha_kd,
        use_true_labels=cfg.use_true_labels,
    )
    comps = {f"loss_e{s_exit}": float(loss.detach())}
    # raw UNWEIGHTED components — the mixed loss_e (α·KD+(1-α)·CE[+λ·MSE]) isn't
    # comparable across tasks/modes; log each term standalone for analysis.
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
