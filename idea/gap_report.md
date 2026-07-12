# Gap Report — Hardware-Aware Early-Exit Inference

Literature scan July 2026 (9 web queries, ~30 papers triaged, 2020–2026).
Purpose: find an **original, defensible gap** for a Q2 journal paper built on
the monorepo's existing assets. Companion to `main.tex`.

---

## 0. Executive summary

Every method that designs early-exit networks (EENNs) for hardware optimizes
against **estimated or single-device cost models** (NACHOS, Zniber+, AEBNAS).
Every cost-prediction work that crosses devices or power modes is
**latency-only, NAS-oriented, or training-workload-oriented** (HELP,
Multi-Predict, PowerTrain). Every EE deployment/selection work is
**single-family, single-device, no energy** (AnytimeYOLO subset selection,
EERO, HAPI). Nobody has: (a) measured per-exit
quality/latency/**energy**/memory tables across architecture *families* and
across a datacenter-GPU/edge-SoC pair; (b) tested whether per-exit cost
**rankings** survive device or nvpmodel power-mode changes; (c) formulated
**exit-design translation** — predicting a target device/power-mode's per-exit
table from a source table + K calibration runs using the *nested-prefix
structure* unique to EENNs — evaluated by **selection regret** rather than
predictor error. That triple is the paper.

---

## 1. Landscape by cluster

### A. Hardware-aware EENN *design* (the direct competitors)

| Paper | Year | What it does | What it does NOT do |
|---|---|---|---|
| [HAPI](https://arxiv.org/pdf/2008.03997) | 2020 | Exit-placement search, fine granularity, measured latency | 1 CNN family, 1 device, no energy, design-time only |
| [NACHOS](https://arxiv.org/pdf/2401.13330) | 2024 | NAS over backbone+exits under HW constraints | costs are differentiable **estimates**; CNN cls only; search repeated per constraint/device |
| [Zniber et al.](https://arxiv.org/abs/2512.04705) | 2025 | HW-algo co-design (quantization × exit config × multi-core mapping), constrained multi-objective | **analytical** DSE on simulated accelerator; CNN cls; own finding — tiny arch changes → disproportionate HW-efficiency swings — undermines analytic-cost reliability but they never validate against measurement |
| [AEBNAS](https://arxiv.org/abs/2512.10671) | 2025 | HW-aware NAS strengthening exit branches | estimates again; cls only |

**Takeaway:** design side = estimated-cost monoculture, CNN classifiers. None
measures whether their cost model's *orderings* hold on real silicon.

### B. Per-family EE training (we build on, don't compete)

ElasticBERT (NAACL'22), MSDNet (ICLR'18), LayerSkip (ACL'24),
[AnytimeYOLO](https://arxiv.org/abs/2503.17497) (2025), LoRA-Exit/SEGD
(EMNLP-F'24), [CalexNet](https://arxiv.org/pdf/2509.08318) (2025, branch
training + calibration + measured GPU latency/energy for **one** cls family).
No work trains ≥2 families under one protocol or profiles them under one
harness. Both surveys ([Laskaridis 2021](https://arxiv.org/pdf/2106.05022),
[ACM CSUR 2024](https://dl.acm.org/doi/10.1145/3698767)) explicitly flag
detection/other tasks as "largely unexplored".

⚠ **Threat**: [AnytimeYOLO](https://arxiv.org/abs/2503.17497) §deployment
proposes algorithms for optimal exit execution order **and optimal early-exit
subset selection** for low-resource deployment. So "exit subset selection"
per se is NOT novel. Their selection: single family (YOLO), single device,
quality–GPU-time only (no energy/memory), design-time analysis, no confidence
thresholds, no cross-device question. Our selection must be positioned as the
**multi-resource, measured, cross-device/cross-power-mode generalization**
with threshold co-optimization + per-sample confidence replay — not as the
first exit selection ever.

### C. Deployment-time control (adjacent, not competing)

[EERO](https://arxiv.org/pdf/2402.03779) (budgeted exiting w/ reject option,
statistical guarantees, FLOPs proxy), [Performance Control in Early
Exiting](https://arxiv.org/html/2412.19325v1) (threshold control for quality
targets), [Pruning EE Networks](https://arxiv.org/pdf/2207.03644),
[DREX](https://arxiv.org/abs/2512.15705) (serving-side dynamic rebatching),
[E4](https://arxiv.org/pdf/2503.04865) (AAAI'25; EE + per-layer DVFS scaling
for edge video, measured power, CNN only — uses DVFS as a *control knob*,
never characterizes how power modes reorder exit costs),
[EdgeServing](https://arxiv.org/pdf/2605.05527), model-distributed EE
inference ([2408.05247](https://arxiv.org/pdf/2408.05247)). All single-family;
none owns measured multi-resource tables.

### D. Edge energy benchmarking / characterization (venue precedent)

[DeepEdgeBench](https://arxiv.org/pdf/2108.09457),
[Jetson end-to-end benchmarking](https://arxiv.org/pdf/2307.16834),
[Unveiling Energy Efficiency in DL](https://arxiv.org/html/2310.18329v2)
(measurement+prediction+scoring across edge devices — but whole-model, no EE),
[EdgeReasoning](https://arxiv.org/html/2511.01866v1) (LLM edge-GPU
characterization), [SLM energy footprint on edges](https://arxiv.org/pdf/2511.11624),
[CNN-optimization comparative incl. EE](https://arxiv.org/pdf/2604.14789)
(2026, CNN-only). Proves characterization papers publish well (FGCS, JSA,
IoT-J) — none does per-exit granularity across families.

### E. Cross-device / cross-mode cost prediction (nearest methodological prior)

| Paper | Predicts | Across | Granularity | Not covered |
|---|---|---|---|---|
| [HELP](https://arxiv.org/abs/2106.08630) (NeurIPS'21) | latency | unseen devices (meta-learning, ~10 samples) | whole architecture (NAS space) | energy, memory, power modes, EE structure |
| [Multi-Predict](https://arxiv.org/pdf/2306.02459) (2023) | latency+acc few-shot | devices | NAS space | energy on SoC, EE, selection downstream |
| [PowerTrain](https://arxiv.org/html/2407.13944) (**FGCS'24**) | power + time | Jetson **power modes** (transfer-learned) | whole **training** workload | **inference**, per-exit granularity, cross-device (A100→Jetson), any selection/decision layer |
| [DVFS-aware GPU latency modeling](https://arxiv.org/pdf/2502.06295) (2025) | latency under DVFS | frequencies | whole model | energy, EE, selection |

⚠ **Threat**: PowerTrain proves power-mode cost transfer is publishable at
FGCS — and occupies "Jetson power-mode prediction". Our translation must be
differentiated on ALL of: inference (not training), **per-exit** tables (their
unit is a whole workload), the **nested-prefix structural prior** (exit cost =
shared backbone prefix + head; prefixes rescale with SM/EMC clock ratios —
a prior no generic predictor exploits), energy+latency+memory jointly, A100→
Jetson in addition to mode→mode, and **selection-regret** evaluation (ranking
fidelity, not MAPE). That set survives.

### F. EE for LLMs (scope threat)

LayerSkip (ACL'24), [Diminishing Returns of EE Decoding in Modern
LLMs](https://arxiv.org/html/2603.23701) (2026): EE decoding helps less on
modern deep-trained LLMs. Datacenter, decoding-quality only, ≥7B focus. Our
1B-edge, task-suite (MCQ vs generation), energy-per-token scope is disjoint;
convergent pessimism on generation would corroborate, not contradict.

---

## 2. Gap analysis — kill or keep

| Candidate gap | Verdict | Why |
|---|---|---|
| "First exit subset selection" | **KILL** | AnytimeYOLO does subset selection (single-device, quality-time) |
| "First HW-aware EE work" | **KILL** | cluster A exists |
| "First power-mode cost prediction on Jetson" | **KILL** | PowerTrain (training workloads) |
| "First cross-family EE benchmark, one protocol, one harness, measured energy, 2 device classes" | **KEEP** | surveys confirm; cluster D has nothing per-exit cross-family |
| "First measurement of per-exit cost-**ranking** stability across devices AND power modes; quantify estimated-cost-model error" | **KEEP — core empirical claim** | nobody measures ranking stability; directly attacks cluster A's foundational assumption |
| "Exit-design **translation**: few-shot per-exit table prediction (latency+energy+memory) via nested-prefix prior, evaluated by selection regret" | **KEEP — core methodological claim** | HELP/Multi-Predict = latency/NAS; PowerTrain = training/whole-workload; nobody uses EE structure or regret metric |
| "Multi-resource measured selection w/ threshold co-opt + confidence replay" | **KEEP as supporting contribution** (positioned against AnytimeYOLO §deployment + EERO) | generalization, not first-ever |

**Original core (one sentence):** *early-exit design translation — measured,
structure-aware, few-shot prediction of per-exit latency/energy/memory tables
across devices and power modes, validated by the regret of the exit
configurations it selects — plus the first cross-family, cross-device measured
evidence that such translation is necessary (rankings do not transfer) and
sufficient (K≪N calibration recovers near-oracle selections).*

---

## 3. Novelty checklist (claim → nearest prior → what's still ours)

1. Cross-family measured per-exit Pareto (4 families, 2 devices, energy
   counters) → CalexNet / CNN-comparative'26 (1 family, 1 device) → breadth +
   per-exit + energy + detector/LLM inclusion.
2. Ranking-stability across devices/power-modes → none found → whole claim.
3. Estimated-vs-measured cost-model error quantification → Zniber+ note
   sensitivity but never validate → whole claim.
4. Exit-design translation w/ nested-prefix prior → HELP/PowerTrain →
   inference + per-exit + energy + structure + regret metric.
5. Multi-resource selection + thresholds + replay → AnytimeYOLO/EERO →
   multi-resource, measured, cross-device, replay-based.

## 4. Risks

- **R1**: rankings turn out stable across devices/modes → translation trivial.
  *Mitigation*: even a null is a finding (validates cheap A100-only design);
  Zniber+ sensitivity result + unified-memory physics make full stability
  unlikely; detector sub-exits (memory-heavy P3 vs cheap P5) are the likeliest
  inversion site — check first.
- **R2**: reviewer says "PowerTrain did power-mode transfer" → pre-empted in
  related work with the 6-point differentiation (see §E).
- **R3**: LLM EE skepticism (Diminishing Returns) → scope disjoint; cite and
  engage, drop llama to secondary evidence if needed.
- **R4**: YOLO retrain quality insufficient for credible detector fronts →
  training fix landed (upstream-head broadcast + full-head training); verify
  teacher ≈ vanilla gelan-m mAP before committing to detector claims.

## 5. Required work (delta from today)

1. YOLO retrain + full sweeps (A100 + Jetson) — in progress, blocking.
2. Power-mode sweep — **feature landed** (`bench_jetson.py all --power-modes
   maxn,15w,7w`, logs to `logs/benchmark[.mode]/`, jtop-verified switching).
3. MDES selector (~300 lines over existing JSONs).
4. Translation model: 2-param clock-ratio prior + ridge residual on K exits;
   regret evaluation harness (pure post-processing).
5. Baselines: FLOPs-proportional costs, HELP-style black-box predictor
   (no structure), PowerTrain-style whole-model scaling, uniform spacing,
   AnytimeYOLO-style quality-time selection.
6. Paper: fill tables from `compare_devices.py` xlsx.

## 6. Venue

JSA (primary — PowerTrain-adjacent scope published at FGCS, sister venue),
FGCS, ACM TECS, Sustainable Computing; floor: Micro&Micro / IEEE Access.

---

## Sources

Design: [NACHOS](https://arxiv.org/pdf/2401.13330) · [Zniber+ 2512.04705](https://arxiv.org/abs/2512.04705) · [AEBNAS](https://arxiv.org/abs/2512.10671) · [HAPI](https://arxiv.org/pdf/2008.03997)
Training/families: [AnytimeYOLO](https://arxiv.org/abs/2503.17497) · [CalexNet](https://arxiv.org/pdf/2509.08318) · [LayerSkip ACL'24] · [ElasticBERT NAACL'22] · [MSDNet ICLR'18]
Control/serving: [EERO](https://arxiv.org/pdf/2402.03779) · [Perf-Control](https://arxiv.org/html/2412.19325v1) · [Pruning EE](https://arxiv.org/pdf/2207.03644) · [DREX](https://arxiv.org/abs/2512.15705) · [E4 AAAI'25](https://arxiv.org/pdf/2503.04865) · [EdgeServing](https://arxiv.org/pdf/2605.05527) · [MDI-EE](https://arxiv.org/pdf/2408.05247)
Prediction/translation: [HELP](https://arxiv.org/abs/2106.08630) · [Multi-Predict](https://arxiv.org/pdf/2306.02459) · [PowerTrain FGCS'24](https://arxiv.org/html/2407.13944) · [DVFS latency modeling](https://arxiv.org/pdf/2502.06295)
Benchmarking: [DeepEdgeBench](https://arxiv.org/pdf/2108.09457) · [Jetson bench](https://arxiv.org/pdf/2307.16834) · [Unveiling Energy Efficiency](https://arxiv.org/html/2310.18329v2) · [EdgeReasoning](https://arxiv.org/html/2511.01866v1) · [SLM edge energy](https://arxiv.org/pdf/2511.11624) · [CNN-opt comparative](https://arxiv.org/pdf/2604.14789)
Surveys/LLM: [Laskaridis'21](https://arxiv.org/pdf/2106.05022) · [ACM CSUR](https://dl.acm.org/doi/10.1145/3698767) · [Diminishing Returns](https://arxiv.org/html/2603.23701)
