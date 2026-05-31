"""Detection knowledge-distillation losses for YOLO early exits. Pure functions.

Teacher = a deeper exit's DDetect output; student = a shallower exit's. Both are
raw per-scale feature maps (training mode). KD = two logit-KDs per scale:

    cls_KD : soft-BCE, student class logits vs teacher sigmoid probs.
    box_LD : Localization Distillation = KL on the DFL box distribution per side.

Applied per scale (P3/P4/P5), masked to teacher-foreground cells, teacher detached.
"""

import torch.nn.functional as F

from .split import box_distribution, split_ddetect


def box_ld(student_box, teacher_box, reg_max: int):
    """KL(student_dist || teacher_dist) over DFL bins, per side, per cell -> [B,H,W]."""
    s = box_distribution(student_box, reg_max)
    t = box_distribution(teacher_box, reg_max)
    s_logp = F.log_softmax(s, dim=2)
    t_p = F.softmax(t, dim=2)
    kl = F.kl_div(s_logp, t_p, reduction="none").sum(dim=2)  # [B,4,H,W]
    return kl.mean(dim=1)  # [B,H,W]  mean over 4 box sides


def cls_kd(student_cls, teacher_cls):
    """Soft-BCE, teacher sigmoid as target, per class, per cell -> [B,H,W]."""
    target = teacher_cls.sigmoid()
    bce = F.binary_cross_entropy_with_logits(student_cls, target, reduction="none")
    return bce.mean(dim=1)  # [B,H,W]  mean over classes


def fg_mask(teacher_cls, tau: float):
    """Teacher-foreground cells: max class prob > tau -> [B,H,W] in {0,1}."""
    conf = teacher_cls.sigmoid().amax(dim=1)
    return (conf > tau).float()


def kd_scale(student_feat, teacher_feat, *, reg_max: int, nc: int, tau: float):
    """One scale -> (box_term, cls_term) FG-masked means. teacher detached by caller."""
    s_box, s_cls = split_ddetect(student_feat, reg_max, nc)
    t_box, t_cls = split_ddetect(teacher_feat, reg_max, nc)
    box_per = box_ld(s_box, t_box, reg_max)   # [B,H,W]
    cls_per = cls_kd(s_cls, t_cls)            # [B,H,W]
    mask = fg_mask(t_cls, tau)
    norm = mask.sum().clamp_min(1.0)
    return (box_per * mask).sum() / norm, (cls_per * mask).sum() / norm


def detection_kd(student_exit, teacher_exit, *, reg_max: int, nc: int, tau: float):
    """student_exit / teacher_exit = list of 3 scale feats [P3, P4, P5].
    Returns (kd_box, kd_cls) averaged over scales. Teacher detached here."""
    pairs = [
        kd_scale(s, t.detach(), reg_max=reg_max, nc=nc, tau=tau)
        for s, t in zip(student_exit, teacher_exit)
    ]
    n = len(pairs)
    kd_box = sum(p[0] for p in pairs) / n
    kd_cls = sum(p[1] for p in pairs) / n
    return kd_box, kd_cls
