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
    """LoRAExit SEGD (Superior-Exit Guided Distillation, Liu et al. EMNLP-F 2024,
    eqs 3-5). Sequential chain: deepest exit trained first on CE; each exit k
    then learns from exit k+1 (its 'superior exit', not the deepest). Smaller
    teacher-student gap than pairwise, at cost of n-1 sequential stages.

    Difference vs pairwise: pairwise = star topology, every student ← deepest
    (one teacher reused). Cascade = chain topology, student k ← k+1 (teacher
    refreshed each stage). Both use LoRA-per-exit on q/v projections."""
    teacher = Stage(
        kind="supervise",
        label="cascade_teacher",
        student_exits=(cfg.deepest,),
        teacher_exit=None,
        teacher_ckpt=None,
        use_lora=True,
    )
    chain = []
    prev = "cascade_teacher"
    for k in range(cfg.deepest - 1, -1, -1):
        chain.append(
            Stage(
                kind="distill",
                label=f"cascade_e{k}",
                student_exits=(k,),
                teacher_exit=k + 1,         # superior exit, not deepest
                teacher_ckpt=prev,
                use_lora=True,
            )
        )
        prev = f"cascade_e{k}"
    return [teacher] + chain


MODE_BUILDERS = {
    "joint": build_joint,
    "pairwise": build_pairwise,
    "cascade": build_cascade,
}
