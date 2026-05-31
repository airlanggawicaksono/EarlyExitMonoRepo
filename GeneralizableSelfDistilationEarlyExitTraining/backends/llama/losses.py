"""Token-level distillation losses for causal LM. Padding-aware.

Inputs to every loss are ALREADY shifted by the caller (step.py):
    logits : [B, T-1, V]   prediction for next token
    labels : [B, T-1]      ground-truth next token
    mask   : [B, T-1]      1 where labels valid, 0 where padded
"""

import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, temperature: float, mask):
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(s, t, reduction="none").sum(dim=-1)             # [B, T-1]
    denom = mask.sum().clamp_min(1.0)
    return (kl * mask).sum() / denom * (temperature ** 2)


def ce_loss(logits, labels, mask):
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).reshape(labels.shape)                                       # [B, T-1]
    denom = mask.sum().clamp_min(1.0)
    return (nll * mask).sum() / denom


def distill_loss(
    student_logits,
    teacher_logits,
    labels,
    mask,
    *,
    temperature: float,
    alpha_kd: float,
    use_true_labels: bool,
):
    loss = alpha_kd * kd_loss(student_logits, teacher_logits, temperature, mask)
    ce_weight = (1.0 - alpha_kd) * float(use_true_labels)
    return loss + ce_weight * ce_loss(student_logits, labels, mask)
