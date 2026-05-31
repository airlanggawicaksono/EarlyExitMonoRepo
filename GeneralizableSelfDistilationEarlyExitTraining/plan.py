"""Training plan: each mode emits a list of Stage objects. The executor runs
stages uniformly — NO `if mode ==` anywhere downstream.

Stage.kind selects the per-batch step function (see step.py STEP_FNS):
    supervise : CE(student, labels) only           -> establishes a teacher
    joint     : all exits, one forward, KD+CE       -> BYOT, no LoRA
    distill   : KD(student, frozen teacher) + CE    -> one student vs one teacher

teacher_ckpt names the prior stage whose adapter+head supply the teacher logits
(reloaded by storage so runs resume across process restarts).
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Stage:
    kind: str                          # supervise | joint | distill
    label: str                         # ckpt/log id, unique within a run
    student_exits: Tuple[int, ...]     # exits trained this stage
    teacher_exit: Optional[int]        # source of soft labels; None for supervise
    teacher_ckpt: Optional[str]        # prior stage label holding teacher weights
    use_lora: bool


def build_joint(cfg):
    """One stage. Deepest exit = supervised teacher; all shallower distilled
    from it in the same forward. Full fine-tune, no LoRA."""
    return [
        Stage(
            kind="joint",
            label="joint",
            student_exits=tuple(range(cfg.n_exits)),
            teacher_exit=cfg.deepest,
            teacher_ckpt=None,
            use_lora=False,
        )
    ]


def build_pairwise(cfg):
    """Deepest exit trained once (CE) as fixed teacher; then every shallower
    exit distilled independently from it. n-1 student stages."""
    teacher = Stage(
        kind="supervise",
        label="teacher",
        student_exits=(cfg.deepest,),
        teacher_exit=None,
        teacher_ckpt=None,
        use_lora=True,
    )
    students = [
        Stage(
            kind="distill",
            label=f"pair_e{j}",
            student_exits=(j,),
            teacher_exit=cfg.deepest,
            teacher_ckpt="teacher",
            use_lora=True,
        )
        for j in range(cfg.deepest)
    ]
    return [teacher] + students


def build_cascade(cfg):
    """All per-exit adapters trained TOGETHER in one pass.

    Loss topology = joint, but with per-exit LoRA + frozen backbone: deepest
    exit anchored on true labels; EVERY shallower exit learns from the
    deepest (detached). Same forward graph; one shared backward updates every
    adapter. One stage, `epochs` data passes (NOT n separate trainings).

    Difference vs joint: joint = full fine-tune, single backbone forward;
    cascade = per-exit LoRA, per-adapter forward × n, backbone frozen."""
    return [
        Stage(
            kind="cascade",
            label="cascade",
            student_exits=tuple(range(cfg.n_exits)),
            teacher_exit=None,        # chain: teacher of exit k is k+1, set in step
            teacher_ckpt=None,
            use_lora=True,
        )
    ]


MODE_BUILDERS = {
    "joint": build_joint,
    "pairwise": build_pairwise,
    "cascade": build_cascade,
}
