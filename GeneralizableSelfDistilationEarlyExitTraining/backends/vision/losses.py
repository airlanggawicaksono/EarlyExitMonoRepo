"""Classification distillation losses. Same shape as BERT's — KL on logits + CE.

Vision logits are [B, num_labels] like BERT, so this is a verbatim copy of the
BERT loss module. Kept as a per-backend file because the user wants each
backend to own its own loss surface, even when the math is identical.
"""

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
