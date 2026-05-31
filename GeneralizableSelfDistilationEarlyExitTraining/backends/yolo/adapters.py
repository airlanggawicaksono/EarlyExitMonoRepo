"""Hand-rolled LoRA on YOLO DDetect head convs. peft-free (avoids wrapping a
custom detection model). Each head is exit-unique, so each head conv carries
exactly ONE adapter — no per-conv multiplexing.

Two independent controls per adapter:
    enabled       -> forward adds the low-rank delta (teacher + student both ON)
    requires_grad -> the adapter trains (student ON, teacher OFF)
"""

import torch.nn as nn


class LoRAConv2d(nn.Module):
    """Frozen base Conv2d + one low-rank delta: B(A(x)) * scale, B init 0."""

    def __init__(self, base: nn.Conv2d, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        cin, cout = base.in_channels, base.out_channels
        self.A = nn.Conv2d(cin, r, kernel_size=1, bias=False)
        self.B = nn.Conv2d(
            r, cout, kernel_size=base.kernel_size, stride=base.stride,
            padding=base.padding, dilation=base.dilation, groups=1, bias=False,
        )
        nn.init.zeros_(self.B.weight)  # delta starts at 0 -> identity init
        self.scale = alpha / r
        self.enabled = True

    def forward(self, x):
        out = self.base(x)
        if self.enabled:
            out = out + self.scale * self.B(self.A(x))
        return out


def _wrap_convs(module, r, alpha):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child, r, alpha))
        else:
            _wrap_convs(child, r, alpha)


def attach(model, cfg):
    """Replace every Conv2d inside each DDetect head with a LoRAConv2d."""
    for head in model.heads:
        _wrap_convs(head, cfg.lora_r, cfg.lora_alpha)
    return model


def _head_adapters(head):
    return [m for m in head.modules() if isinstance(m, LoRAConv2d)]


def set_exit(model, exit_idx: int, *, enabled: bool, trainable: bool):
    """Configure one exit's head adapters for forward + grad."""
    for m in _head_adapters(model.heads[exit_idx]):
        m.enabled = enabled
        for p in (*m.A.parameters(), *m.B.parameters()):
            p.requires_grad_(trainable)


def freeze_all(model):
    """All adapters OFF (disabled + frozen). Stage setup re-enables what it needs."""
    for i in range(model.n_exits):
        set_exit(model, i, enabled=False, trainable=False)
