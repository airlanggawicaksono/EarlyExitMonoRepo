"""Per-batch compute for decoder LM. Returns (scalar_loss, components_dict).

Identical topology to BERT/Vision step.py — STEP_FNS dict on stage.kind. Two
differences:
  1. batch unpack: (input_ids, attention_mask, labels).
  2. shift: causal LM predicts next token, so we shift logits/labels/mask once
     here; the loss module receives shifted tensors and just masks-and-means.
"""

import torch

from . import adapters
from .losses import ce_loss, distill_loss, feature_hint_loss


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
    t_exit, s_exit = stage.teacher_exit, stage.student_exits[0]
    _, labels, mask = _inputs(batch)
    adapters.activate(model, t_exit)
    with torch.no_grad():
        teacher_full = forward_logits(model, batch)[t_exit]
    adapters.activate(model, s_exit)
    student_full = forward_logits(model, batch)[s_exit]
    s_s, l_s, m_s = _shift(student_full, labels, mask)
    s_t, _, _ = _shift(teacher_full, labels, mask)
    loss = distill_loss(
        s_s, s_t, l_s, m_s,
        temperature=cfg.temperature, alpha_kd=cfg.alpha_kd,
        use_true_labels=cfg.use_true_labels,
    )
    return loss, {f"loss_e{s_exit}": float(loss.detach())}


def _logit_per_adapter(model, batch, n_exits):
    out = []
    for i in range(n_exits):
        adapters.activate(model, i)
        out.append(forward_logits(model, batch)[i])
    return out


def cascade_step(model, stage, batch, cfg):
    """All adapters at once. Deepest anchored on labels; EVERY shallower exit
    learns from the deepest (detached). One backward updates every adapter."""
    _, labels, mask = _inputs(batch)
    n = model.n_exits
    logits = _logit_per_adapter(model, batch, n)
    s_d, l_d, m_d = _shift(logits[n - 1], labels, mask)
    teacher_shifted = s_d.detach()
    teacher_ce = ce_loss(s_d, l_d, m_d)
    components = {"teacher_ce": float(teacher_ce.detach())}
    total = teacher_ce
    for i in range(n - 1):
        s_i, l_i, m_i = _shift(logits[i], labels, mask)
        ld = distill_loss(
            s_i, teacher_shifted, l_i, m_i,
            temperature=cfg.temperature, alpha_kd=cfg.alpha_kd,
            use_true_labels=cfg.use_true_labels,
        )
        components[f"loss_e{i}"] = float(ld.detach())
        total = total + ld
    return total, components


STEP_FNS = {
    "supervise": supervise_step,
    "joint": joint_step,
    "distill": distill_step,
    "cascade": cascade_step,
}
