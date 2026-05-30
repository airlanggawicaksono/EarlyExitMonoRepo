"""LoRA adapter strategy (peft), behind a swappable interface.

The rest of the framework only calls: attach / activate / set_adapter_trainable.
It never references peft directly. To swap LoRA for IA3 / heads-only / full,
replace this module's internals — plan/step/train stay untouched.

One named adapter per exit ("exit_{k}") lives on the backbone. peft freezes the
base weights; we additionally gate requires_grad so only ONE adapter trains at a
time (decoupled per-exit training, the LoRAExit recipe).

peft is imported lazily: only pairwise/cascade need it; joint never imports it.
"""


def exit_name(exit_idx: int) -> str:
    return f"exit_{exit_idx}"


def _lora_config(cfg):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_targets),
        bias="none",
    )


def attach(model, cfg):
    """Wrap backbone with one LoRA adapter per exit. Base weights frozen by peft."""
    from peft import get_peft_model

    names = [exit_name(i) for i in range(cfg.n_exits)]
    model.backbone = get_peft_model(model.backbone, _lora_config(cfg), adapter_name=names[0])
    for n in names[1:]:
        model.backbone.add_adapter(n, _lora_config(cfg))
    return model


def activate(model, exit_idx: int):
    """Route forward through this exit's adapter."""
    model.backbone.set_adapter(exit_name(exit_idx))


def set_adapter_trainable(model, exit_idx: int):
    """requires_grad True ONLY for this exit's adapter params; everything else off."""
    name = exit_name(exit_idx)
    for pname, p in model.backbone.named_parameters():
        p.requires_grad_(name in pname)


def load_adapter(model, exit_idx: int, adapter_root):
    """Reload one exit's adapter from its per-exit subdir (resume / teacher)."""
    src = adapter_root / exit_name(exit_idx)
    model.backbone.load_adapter(str(src), adapter_name=exit_name(exit_idx))


def save_adapter(model, exit_idx: int, adapter_root):
    """Persist one exit's adapter into its own subdir (collision-free for the
    multi-adapter cascade stage)."""
    dst = adapter_root / exit_name(exit_idx)
    dst.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(str(dst), selected_adapters=[exit_name(exit_idx)])
