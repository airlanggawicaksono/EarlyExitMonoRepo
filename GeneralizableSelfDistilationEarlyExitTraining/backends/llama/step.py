"""Per-batch compute for decoder LM. Returns (scalar_loss, components_dict).

Identical topology to BERT/Vision step.py — STEP_FNS dict on stage.kind. Two
differences:
  1. batch unpack: (input_ids, attention_mask, labels).
  2. shift: causal LM predicts next token, so we shift logits/labels/mask once
     here; the loss module receives shifted tensors and just masks-and-means.
"""

import torch

from . import adapters
from .losses import ce_loss, distill_loss, feature_hint_loss, kd_loss


def _inputs(batch):
    input_ids, attn_mask, labels = batch
    return {"input_ids": input_ids, "attention_mask": attn_mask}, labels, attn_mask


def forward_logits(model, batch):
    """List of full-length [B, T, V] logits, one per exit."""
    inputs, _, _ = _inputs(batch)
    return model(**inputs)


def _shift(logits, labels, mask):
    return (
        logits[..., :-1, :].contiguous(),
        labels[..., 1:].contiguous(),
        mask[..., 1:].contiguous().float(),
    )


def supervise_step(model, stage, batch, cfg):
    exit_idx = stage.student_exits[0]
    adapters.activate(model, exit_idx)
    _, labels, mask = _inputs(batch)
    logits = forward_logits(model, batch)[exit_idx]
    s, l, m = _shift(logits, labels, mask)
    loss = ce_loss(s, l, m)
    return loss, {"teacher_ce": float(loss.detach())}


def joint_step(model, stage, batch, cfg):
    inputs, labels, mask = _inputs(batch)
    logits, feats = model(**inputs, return_features=True)
    feat_mask = mask.float()                                         # [B, T] full-seq
    deep = stage.teacher_exit
    s_d, l_d, m_d = _shift(logits[deep], labels, mask)
    teacher_shifted = s_d.detach()
    teacher_feat = feats[deep].detach()
    teacher_ce = ce_loss(s_d, l_d, m_d)
    components = {"teacher_ce": float(teacher_ce.detach())}
    total = teacher_ce
    for j in [i for i in stage.student_exits if i != deep]:
        s_j, l_j, m_j = _shift(logits[j], labels, mask)
        ld = distill_loss(
            s_j, teacher_shifted, l_j, m_j,
            temperature=cfg.temperature, alpha_kd=cfg.alpha_kd,
            use_true_labels=cfg.use_true_labels,
        )
        lf = cfg.lambda_feat * feature_hint_loss(feats[j], teacher_feat, feat_mask)
        components[f"loss_e{j}"] = float(ld.detach())
        components[f"feat_e{j}"] = float(lf.detach())
        total = total + ld + lf
    return total, components


def distill_step(model, stage, batch, cfg):
    """SEGD faithful to LoRAExit (KD + CE only). `pairwise` is our star-topology
    variant -> adds a BYOT feature-hint L2 (student vs deepest-teacher feature,
    padding-masked)."""
    t_exit, s_exit = stage.teacher_exit, stage.student_exits[0]
    inputs, labels, mask = _inputs(batch)
    want_feat = cfg.mode == "pairwise"

    adapters.activate(model, t_exit)
    with torch.no_grad():
        if want_feat:
            t_logits, t_feats = model(**inputs, return_features=True)
            teacher_full, teacher_feat = t_logits[t_exit], t_feats[t_exit]
        else:
            teacher_full = forward_logits(model, batch)[t_exit]
    adapters.activate(model, s_exit)
    if want_feat:
        s_logits, s_feats = model(**inputs, return_features=True)
        student_full, student_feat = s_logits[s_exit], s_feats[s_exit]
    else:
        student_full = forward_logits(model, batch)[s_exit]
    s_s, l_s, m_s = _shift(student_full, labels, mask)
    s_t, _, _ = _shift(teacher_full, labels, mask)
    loss = distill_loss(
        s_s, s_t, l_s, m_s,
        temperature=cfg.temperature, alpha_kd=cfg.alpha_kd,
        use_true_labels=cfg.use_true_labels,
    )
    comps = {f"loss_e{s_exit}": float(loss.detach())}
    # raw UNWEIGHTED, padding-masked components — mixed loss_e isn't comparable.
    comps[f"ce_raw_e{s_exit}"] = float(ce_loss(s_s, l_s, m_s).detach())
    comps[f"kd_raw_e{s_exit}"] = float(kd_loss(s_s, s_t, cfg.temperature, m_s).detach())
    if want_feat:
        mse = feature_hint_loss(student_feat, teacher_feat.detach(), mask.float())
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
