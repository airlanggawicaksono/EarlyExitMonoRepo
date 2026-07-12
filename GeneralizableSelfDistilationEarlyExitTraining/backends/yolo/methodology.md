# YOLO Backend — Self-Distillation Early-Exit (gelan-m-ee)

Detection backend for the shared self-distill framework (`plan / train / storage /
step dispatch`). Only the compute is YOLO-specific.

## Target model

YOLOv9 **gelan-m-ee**: 6 early-exit DDetect heads (E0..E5), each over 3 scales
(P3/P4/P5) = 18 sub-exits. `EarlyExitModel` already emits all 6 exits in one
training forward. Teacher = **E5** (full FPN, deepest).

```
E0 [4,6,7]    L7   pre-P5-ELAN (cheapest stride-aligned)
E1 [4,6,8]    L8   backbone only
E2 [4,6,9]    L9   + SPPELAN
E3 [15,12,9]  L15  + top-down FPN
E4 [15,18,9]  L18  + P4 bottom-up
E5 [15,18,21] L21  full FPN   <- teacher
```

## Goal

Raise quality at **all 18 anytime points** via self-distillation. Profiling
(untrained baseline + trained) is the benchmark's separate job — this backend only
produces the trained weights.

## Head training (full, no LoRA)

- **Heads train FULLY** (`cv2`/`cv3` convs). Backbone + FPN **frozen** in
  head-only stages (`Stage.use_lora=True` means "head-only" for this backend).
- Why no LoRA here: unlike BERT/ViT/LLaMA (adapters on a *pretrained* base),
  EE detection heads have no pretrained-in-place base — a low-rank delta on a
  frozen random/mismatched conv cannot recover a working detector. That exact
  configuration produced 0 mAP at every exit.
- **Head init**: every exit taps (P3,P4,P5) at channels 240/360/480, identical
  to the upstream gelan-m head — so `model._load_weights` **broadcasts the
  upstream `model.22` DDetect tensors into all 6 head slots**. Each exit starts
  from a trained detector, not noise.
- `DDetect.dfl` is a fixed arange projection — always frozen (`_freeze_dfl`).
- **Efficiency:** backbone+FPN frozen → forward it **once** (no_grad), cache
  feature maps `y[]`, then run each head on the cache. Per batch = 1 backbone
  pass + N head passes (not N full forwards). Teacher and student share the
  cached feats.
- `joint` = full finetune (all params except dfl).

## Detection KD loss (LOCKED)

DDetect output per scale (training mode) = `cat(cv2_box, cv3_cls)`,
shape `[B, 4*reg_max + nc, H, W]` (reg_max=16). Box is a **DFL distribution** ->
KD = KL. So detection KD = two logit-KDs, per scale:

```
cls_KD = soft-BCE(student_cls, teacher_cls.sigmoid())     # per class
box_LD = KL(softmax(student_box_dist), softmax(teacher_box_dist))  # per side, Localization Distillation
```

- **Same-scale** teacher matching (S.P3 <- T.P3, etc.); grids align (same stride).
- **Teacher-foreground masked**: keep cells where `teacher_cls.sigmoid().max(c) > tau`.
- Summed over P3/P4/P5 → per-(exit,scale) KD terms across the run.
- Teacher always `.detach()`.

## Student supervision (LOCKED)

```
KD_scaled  = batch_size * (box_gain * kd_box + cls_gain * kd_cls)
student_loss = alpha * KD_scaled + (1 - alpha) * TAL(E_i, gt)
```

- `kd_box`, `kd_cls` from `detection_kd` (FG-masked means, scales averaged).
- `box_gain`, `cls_gain` mirror yolov9 hyp.scratch-high (7.5, 0.5); `* batch_size`
  matches yolov9 `ComputeLoss` returning `loss * bs`. KD now lives in same
  numeric range as TAL, so `alpha` is a real mix knob (default **0.9**, soft
  mimicry dominant; TAL = small GT anchor).
- TAL owns DFL (`dfl_gain`); KD owns its own DFL signal via `box_LD` already.
  Double-supervision on box is intentional — teacher's box dist is itself
  GT-anchored (trained on TAL), so KD's soft target ≈ smoothed GT.
- Teacher/anchor exit (E5) trained on pure TAL.
- `pairwise` (our variant) adds a penultimate-feature L2 hint vs the teacher
  (`lambda_feat`); `segd` stays output-KD only.

## Modes (shared framework, unchanged)

- **joint** — 1 forward, all exits; E5 = TAL(gt) + soft-label source; E_i<5 =
  KD(detach E5) + TAL. Full finetune.
- **pairwise** — E5 trained (TAL) as fixed teacher; each E_i distilled in its own
  stage, own head.
- **segd** — sequential chain: E5 (TAL) then E_k <- detach(E_{k+1}).

## Backend files

```
model.py    wrap EarlyExitModel; upstream-head broadcast into all EE heads;
            split forward (backbone-once + per-head); exit_outputs()
loss.py     detection_kd (cls_KD + box_LD, FG-masked) + feature_hint_loss
tal.py      per-exit supervised TAL via _HeadProxy
data.py     COCO loader (reuse yolov9 utils/dataloaders)
split.py    DDetect feat -> (box_dfl, cls_logits) per scale
```

## Handoff to profiler

Stages save plain per-head state dicts (`<stage>/head_<k>.pt`) — standard convs,
no adapter merge needed. `AnyTimeYolo/src/benchmark_trained.py` rebuilds
MultiExitYolo from `gelan-m-ee.yaml` + gelan-m.pt and loads each trained head.

## Caveats

- Needs `AnyTimeYolo/src/model/yolov9` (Colab-only, not in local tree) — written
  here, smoke-tested on Colab.
- `reg_max` / `nc` read from the model at runtime, not hardcoded.
