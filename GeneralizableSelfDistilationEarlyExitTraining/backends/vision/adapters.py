"""peft LoRA adapter strategy — verbatim mirror of BERT's adapters.py. Same
interface (attach / activate / set_adapter_trainable / load_adapter /
save_adapter); only cfg.lora_targets differs (ViT uses "query"/"value").
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
    from peft import get_peft_model

    names = [exit_name(i) for i in range(cfg.n_exits)]
    model.backbone = get_peft_model(model.backbone, _lora_config(cfg), adapter_name=names[0])
    for n in names[1:]:
        model.backbone.add_adapter(n, _lora_config(cfg))
    return model


def activate(model, exit_idx: int):
    model.backbone.set_adapter(exit_name(exit_idx))


def set_adapter_trainable(model, exit_idx: int):
    name = exit_name(exit_idx)
    for pname, p in model.backbone.named_parameters():
        p.requires_grad_(name in pname)


def load_adapter(model, exit_idx: int, adapter_root):
    src = adapter_root / exit_name(exit_idx)
    model.backbone.load_adapter(str(src), adapter_name=exit_name(exit_idx))


def save_adapter(model, exit_idx: int, adapter_root):
    adapter_root.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(str(adapter_root), selected_adapters=[exit_name(exit_idx)])
