# Generalizable Self-Distillation Early-Exit Training

Train multi-exit ElasticBERT with three self-distillation recipes. Deepest exit is
the teacher; shallower exits learn from it. Backbone is warm-started from
pretrained ElasticBERT.

## Modes

| mode | recipe | teacher | LoRA | stages | epochs* |
|------|--------|---------|------|--------|---------|
| `joint` | BYOT — one forward, all exits, deepest supervised + soft-labels all shallower | deepest (in-graph, detached) | no (full fine-tune) | 1 | 3 |
| `pairwise` | deepest trained once as fixed teacher; each shallower exit distilled in its **own separate run** (combinatorial ablation) | deepest (frozen) | yes (per-exit adapter) | 1 + (n−1) | 36 |
| `cascade` | **LoRAExit** — all per-exit adapters trained **jointly in one pass**; exit `k` learns from `k+1` (detached), deepest anchored on labels | neighbor `k+1` (in-graph, detached) | yes (per-exit adapter) | 1 | 3 |

`n = n_exits = 12`. *total data-pass epochs for one task at `epochs=3`. Cascade is
**one** pass like the original LoRAExit (n forwards/batch, one shared backward) —
NOT n sequential trainings. Pairwise is deliberately n−1 separate runs (that is the
ablation); it is the expensive mode.

## Why this shape

- **Frozen teacher.** Self-distillation needs a stable target. LoRA freezes the
  backbone so the teacher exit never drifts while students train. Joint is the
  no-LoRA BYOT baseline (trains everything at once).
- **Per-exit adapter (LoRAExit).** Each student gets its own low-rank adapter →
  no cross-exit gradient conflict; decoupled training.
- **Swappable adapter.** Only `adapters.py` knows LoRA/peft. Swap it for IA3 /
  heads-only / full without touching plan / step / train. Per-architecture
  generalization = change `lora_targets` (transformer: `query,value`; conv nets:
  Conv2d targets).

## Module map (clean IO / no mode branches)

```
config.py    @dataclass Cfg — all knobs, IO-free
data.py      GLUE dataloaders (reuse repo load_data) — all data IO
storage.py   ckpt + metrics save/load, resume marker — all disk IO
model.py     MultiExitElasticBert: backbone + per-exit heads -> N logits/forward
adapters.py  peft LoRA attach/activate/freeze — model-agnostic via target_modules
losses.py    kd_loss / ce_loss / distill_loss — pure
plan.py      Stage + build_{joint,pairwise,cascade}; MODE_BUILDERS dict
step.py      supervise/joint/distill step fns; STEP_FNS dict — pure compute
train.py     cfg -> plan -> run stages -> save — thin orchestration
cli.py       argparse -> Cfg -> train
```

Dispatch is dict-based end to end — no `if mode ==`:

```python
plan   = MODE_BUILDERS[cfg.mode](cfg)      # mode -> stages
_SETUP[stage.kind](model, stage, cfg)      # kind -> trainable params + adapters
STEP_FNS[stage.kind](model, stage, batch)  # kind -> per-batch loss
_SAVERS[stage.use_lora](model, stage, d)   # lora? -> checkpoint shape
```

## Resume

Each finished stage writes `metrics.json`. `storage.has_stage` reuses
`shared.has_valid_result` (same contract as the benchmark): a stage with a valid
metrics file is skipped. Distill stages reload their teacher from disk, so skips
never break the teacher chain — safe to interrupt and rerun.

## Usage

```bash
# from repo root
python -m GeneralizableSelfDistilationEarlyExitTraining.cli --task SST-2 --mode joint
python -m GeneralizableSelfDistilationEarlyExitTraining.cli --task MRPC --mode pairwise --epochs 3
python -m GeneralizableSelfDistilationEarlyExitTraining.cli --task RTE  --mode cascade  --lora-r 16
```

```python
from GeneralizableSelfDistilationEarlyExitTraining import Cfg, train
train(Cfg(task="SST-2", mode="cascade", epochs=3, lora_r=16))
```

Output: `logs/selfdistill/<mode>/<task>/<stage>/` holding adapter weights, per-exit
heads, and `metrics.json`.

## Requirements

`peft` (pairwise/cascade only), `transformers`, `torch`. GPU. Joint never imports
peft. Runs on Colab; not exercised locally (no GPU/peft).

## Notes / TODO

- ElasticBERT encoder applies `gradient_rescale` per exit when `training`
  (gradient-equilibrium trick). Identity in forward, scales grads only — left on.
  Revisit if it perturbs single-exit LoRA training.
- `distill_step` (pairwise) runs two full backbone forwards (teacher + student,
  different adapters). `cascade_step` runs n full forwards/batch (one per adapter).
  Both could truncate each exit's forward to `exit+1` layers (≈ n²/2 layer-evals
  instead of n² ) — deferred; correctness first.
- Verify peft per-exit adapter save/load layout on first Colab run
  (`adapters.save_adapter` writes one subdir per exit under `<stage>/adapter/`).
- Eval/quality pass not wired here; feed trained adapters+heads back through the
  benchmark for accuracy/latency.
