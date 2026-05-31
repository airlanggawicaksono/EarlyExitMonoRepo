# YOLO Backend — Self-Distillation Early-Exit (gelan-s-ee)

Detection backend for the shared self-distill framework (`plan / train / storage /
step dispatch`). Only the compute is YOLO-specific.

## Target model

YOLOv9 **gelan-s-ee**: 5 early-exit DDetect heads (E0..E4), each over 3 scales
(P3/P4/P5) = 15 sub-exits. `EarlyExitModel` already emits all 5 exits in one
training forward. Teacher = **E4** (full FPN, deepest).

```
E0 [4,6,8]    L8   backbone only
E1 [4,6,9]    L9   + SPPELAN
E2 [15,12,9]  L15  + top-down FPN
E3 [15,18,9]  L18  + P4 bottom-up
E4 [15,18,21] L21  full FPN   <- teacher
```

## Goal

Raise quality at **all 15 anytime points** via self-distillation. Profiling
(untrained baseline + trained) is the benchmark's separate job — this backend only
produces the trained weights.

## LoRA application (LOCKED)

- **Target = DDetect head convs only** (`cv2`/`cv3`). Backbone + FPN **frozen**.
- One named adapter per exit (`exit_0..exit_4`); head_i is unique to E_i.
- **Efficiency:** backbone+FPN is frozen → forward it **once** (no_grad), cache
  feature maps `y[]`, then run each head on the cache with its adapter active.
  Per batch = 1 backbone pass + N head passes (not N full forwards). Teacher and
  student share the cached feats.
- `joint` = full finetune (no LoRA); `pairwise`/`cascade` = head-only LoRA.
- Tradeoff: lightest scope. If E0/E1 underfit (raw backbone feats), widen target
  to FPN/backbone later — one flag, framework untouched.

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
- Summed over P3/P4/P5 → 15 per-(exit,scale) KD terms total across the run.
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
- Teacher/anchor exit (E4) trained on pure TAL.

## Modes (shared framework, unchanged)

- **joint** — 1 forward, all exits; E4 = TAL(gt) + soft-label source; E_i<4 =
  KD(detach E4) + TAL. Full finetune.
- **pairwise** — E4 trained (TAL) as fixed teacher; each E_i distilled in its own
  run, own head adapter.
- **cascade** — all head adapters jointly; E_i (i<4) <- detach(E4); E4 anchored TAL.

## Backend files

```
model.py    wrap EarlyExitModel; split forward (backbone-once + per-head); exit_outputs()
loss.py     detection_kd (cls_KD + box_LD, FG-masked) + single-exit TAL wrapper   [done]
adapters.py peft Conv2d LoRA on DDetect head convs; activate/freeze
data.py     COCO loader (reuse yolov9 utils/dataloaders)
split.py    DDetect feat -> (box_dfl, cls_logits) per scale
```

## Handoff to profiler

Trained model carries LoRA adapters. Before benchmarking, `merge_and_unload` each
exit's adapter into a plain gelan-s-ee checkpoint so the 15-sub-exit profiler eats
standard convs. One step at handoff.

## Caveats

- Needs `AnyTimeYolo/src/model/yolov9` (Colab-only, not in local tree) — written
  here, smoke-tested on Colab.
- `reg_max` / `nc` read from the model at runtime, not hardcoded.
