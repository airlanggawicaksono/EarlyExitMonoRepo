"""Per-batch compute. Returns (scalar_loss, components_dict). No IO, no optimizer.

The training loop backwards `scalar_loss` and logs every key in `components_dict`
per-step (teacher_ce, loss_e{j}, etc) so metric analysis has full breakdown.
"""

import torch

from . import adapters
from .losses import ce_loss, distill_loss


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
    """BYOT: one forward, deepest exit supervised, all shallower distilled from it."""
    _, labels = _inputs(batch)
    logits = forward_logits(model, batch)
    deep = stage.teacher_exit
    teacher = logits[deep].detach()

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
        components[f"loss_e{j}"] = float(ld.detach())
        total = total + ld
    return total, components


def distill_step(model, stage, batch, cfg):
    """One student vs one frozen teacher. Two forwards (different adapters)."""
    t_exit = stage.teacher_exit
    s_exit = stage.student_exits[0]
    _, labels = _inputs(batch)

    adapters.activate(model, t_exit)
    with torch.no_grad():
        teacher = forward_logits(model, batch)[t_exit]

    adapters.activate(model, s_exit)
    student = forward_logits(model, batch)[s_exit]

    loss = distill_loss(
        student, teacher, labels,
        temperature=cfg.temperature,
        alpha_kd=cfg.alpha_kd,
        use_true_labels=cfg.use_true_labels,
    )
    return loss, {f"loss_e{s_exit}": float(loss.detach())}


def _logit_per_adapter(model, batch, n_exits):
    """Exit i's logit computed through its OWN adapter -> list[Tensor], all under
    grad. Each adapter is active exactly during its own forward, so gradients
    flow back to every adapter from one shared backward."""
    out = []
    for i in range(n_exits):
        adapters.activate(model, i)
        out.append(forward_logits(model, batch)[i])
    return out


def cascade_step(model, stage, batch, cfg):
    """All adapters at once. Deepest anchored on labels; EVERY shallower exit
    learns from the deepest (detached). One backward updates every adapter."""
    _, labels = _inputs(batch)
    n = model.n_exits
    logits = _logit_per_adapter(model, batch, n)
    teacher = logits[n - 1].detach()

    teacher_ce = ce_loss(logits[n - 1], labels)
    components = {"teacher_ce": float(teacher_ce.detach())}
    total = teacher_ce
    for i in range(n - 1):
        ld = distill_loss(
            logits[i], teacher, labels,
            temperature=cfg.temperature,
            alpha_kd=cfg.alpha_kd,
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
