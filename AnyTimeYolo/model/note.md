 How Early Exits Work in YOLOv9

  YOLOv9 base structure (what you're working with):

  Backbone (GELAN blocks)
      ↓
  Neck / FPN (feature pyramid — 3 scales: small/med/large)
      ↓
  Detection Head(s) → predictions

  PGI already has auxiliary branches during training (removed at inference). The
  trick: don't remove them — repurpose as early exits.

  ---
  What AnytimeYOLO Actually Does

  1. Sub-exits = detection heads at intermediate GELAN blocks
  Not one exit per layer — one exit = tuple of 3 heads (one per scale):
  Exit N = (block_small, block_med, block_large)
  e.g.   = (15, 18, 21)
  Must be 3-tuple because you need multi-scale for decent mAP.

  2. Files to modify (standard YOLOv9 repo):

  ┌────────────────────┬─────────────────────────────────────────────────────────┐
  │        File        │                      What changes                       │
  ├────────────────────┼─────────────────────────────────────────────────────────┤
  │ models/yolov9.yaml │ Add extra Detect heads at intermediate blocks           │
  ├────────────────────┼─────────────────────────────────────────────────────────┤
  │ models/common.py   │ No change usually                                       │
  ├────────────────────┼─────────────────────────────────────────────────────────┤
  │ models/yolo.py     │ Model.forward() — return list of outputs from all exits │
  ├────────────────────┼─────────────────────────────────────────────────────────┤
  │ utils/loss.py      │ Extend distillation loss to handle N exits              │
  ├────────────────────┼─────────────────────────────────────────────────────────┤
  │ train.py           │ Exit sampling strategy during training                  │
  └────────────────────┴─────────────────────────────────────────────────────────┘

  3. YAML interface — where exits plug in

  In yolov9.yaml, exits are just extra Detect nodes pointing to earlier feature maps:

  # Normal final head
  - [[P3, P4, P5], 1, Detect, [nc]]   # exit at end

  # Early exit — attach to intermediate feature maps
  - [[P3_early, P4_early, P5_early], 1, Detect, [nc]]  # exit 1
  - [[P3_mid,   P4_mid,   P5_mid],   1, Detect, [nc]]  # exit 2

  4. Transposed architecture (GELAN-tT) — key innovation

  Standard GELAN: processes scales sequentially → large-scale features depend on
  finishing previous scales → early exit = poor quality

  Transposed GELAN-T: scales processed as parallel branches → earlier exits get all 3
  scales immediately → much better early-exit quality

  Standard:   Backbone → Scale1 → Scale2 → Scale3 → Head
  Transposed: Backbone → Scale1 ↘
                       → Scale2 → merge → Head
                       → Scale3 ↗

  5. Loss modification

  Standard: distillation loss on final head only
  AnytimeYOLO: loss summed over all active exits, with exit sampling (randomly pick
  which exits are "active" each batch to avoid overfitting to specific exits).

  ---
  Your Current Project State

  modules/model.py has only a predict() wrapper — no architecture yet. You need the
  full YOLOv9 repo as base. Options:

  1. Clone WongKinYiu/yolov9 into your project — gives you models/, utils/, train.py
  2. Use Ultralytics YOLOv9 — cleaner API but less hackable for custom exits
  3. Clone AnytimeYOLO repo if they released code (paper is March 2025, check arxiv
  page)

  ---
  Feasibility verdict

  Fully doable. The hard parts:
  - YAML edit for exit placement: easy
  - forward() returning multiple outputs: medium
  - Loss extension: medium-hard
  - Transposed architecture: hard (restructures backbone)

  Start without transposed arch — just add extra Detect heads in YAML → easier and
  still functional.

  ---
  Sources:
  - https://arxiv.org/abs/2503.17497
  - https://arxiv.org/html/2503.17497v1
  - https://github.com/WongKinYiu/yolov9
  - https://link.springer.com/chapter/10.1007/978-3-032-10185-3_35
  - https://www.researchgate.net/publication/390142533_You_Only_Look_Once_at_Anytime_A
  nytimeYOLO_Analysis_and_Optimization_of_Early-Exits_for_Object-Detection