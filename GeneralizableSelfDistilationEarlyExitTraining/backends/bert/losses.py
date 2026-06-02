"""Distillation losses. Pure functions — no model, no IO, no state.

kd_loss   : temperature-scaled KL(student || teacher).
ce_loss   : standard cross-entropy vs true labels.
distill_loss : alpha * KD + (1-alpha) * CE, CE term optional.
"""

import torch
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, temperature: float):
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


def ce_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def distill_loss(
    student_logits,
    teacher_logits,
    labels,
    *,
    temperature: float,
    alpha_kd: float,
    use_true_labels: bool,
):
    loss = alpha_kd * kd_loss(student_logits, teacher_logits, temperature)
    ce_weight = (1.0 - alpha_kd) * float(use_true_labels)
    return loss + ce_weight * ce_loss(logits=student_logits, labels=labels)


def feature_hint_loss(student_feat, teacher_feat):
    """BYOT feature L2 hint: MSE between shallow exit's pooled feature and the
    deepest exit's (detached) pooled feature. Same hidden_size across exits
    (shared backbone) so no projector needed."""
    return F.mse_loss(student_feat, teacher_feat)
