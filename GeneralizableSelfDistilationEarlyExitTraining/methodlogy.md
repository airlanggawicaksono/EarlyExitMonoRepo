# Methodology

In self-distillation the deepest exit produces the soft labels (teacher); all
shallower exits learn from it. Three recipes are implemented:

1. **Joint (BYOT)** — one forward over all exits. Deepest exit gets cross-entropy
   on true labels and simultaneously supplies detached soft labels to every
   shallower exit. Full fine-tune, no LoRA. One stage, one backward.

2. **Pairwise (combinatorial)** — the deepest exit is trained once as a fixed
   teacher, then each shallower exit is distilled from it in its own **separate
   run**, each with its own LoRA adapter. `1 + (n-1)` stages. This is the
   combinatorial ablation (which single teacher→student pair helps which exit);
   it is the expensive mode.

3. **Cascade (LoRAExit)** — all per-exit LoRA adapters are trained **together in
   one pass**. The teacher chain is loss topology only: exit `k` learns from the
   detached output of exit `k+1`, and the deepest exit is anchored on true labels.
   Every adapter updates from one shared backward. `1` stage, `epochs` data passes
   (n forwards/batch) — exactly the original LoRAExit, not n sequential trainings.

See `README.md` for the module map, dispatch design, resume contract, and usage.
