# idea/ — paper draft

**Working title:** Measure, Don't Search: Measurement-Driven Exit Selection for
Hardware-Constrained Early-Exit Networks Across Architectures and Devices

- `main.tex` — full draft skeleton: motivation, RQ1–RQ4, MDES method,
  contribution claims, execution plan, venue targets. `\todo{}` marks = fill
  from real sweep numbers (`compare_devices.py` xlsx).
- `references.bib` — NACHOS, ElasticBERT, MSDNet, LayerSkip, AnytimeYOLO,
  LoRA-Exit/SEGD, BranchyNet, EdgeBERT. Verify the LoRA-Exit entry.

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`

## One-paragraph pitch

NACHOS does NAS with *estimated* hardware costs, CNNs only, one device class.
We already have what nobody publishes: **measured** per-exit
quality/latency/energy/memory tables for 4 architecture families on BOTH an
A100 and a Jetson Orin Nano, under 2 uniform distillation protocols, at
sub-exit granularity. So: (1) publish the cross-architecture cross-device
Pareto characterization (show exit cost rankings INVERT between devices →
estimate-based NAS cost models are structurally wrong on edge); (2) MDES —
exact exit+threshold selection over measured tables, seconds instead of
GPU-days, with an A100→Jetson transfer-regret experiment as the headline
figure. Venue: JSA / TECS / FGCS (expected Q2, floor Q3).

## Blocking before numbers can go in

1. YOLO retrain (fixed pipeline) + full Jetson sweep all 4 backends
2. nvpmodel 7W/15W/MAXN ablation sweep
3. MDES script (~300 lines over existing benchmark JSONs)
