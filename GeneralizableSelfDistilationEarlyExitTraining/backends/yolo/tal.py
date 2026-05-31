"""Supervised TAL loss, one per exit head.

yolov9's ComputeLoss binds to a single Detect head and reads:
    model.model[-1]            (the detect head — for stride / nc / reg_max)
    model.hyp                  (box / cls / dfl gains + cls_pw / fl_gamma)
    next(model.parameters())   (to get device)

So the proxy must be a real `nn.Module` (so `.parameters()` works), with a
`model` ModuleList whose last element is the head, plus a `hyp` dict.

build_sup_loss(model, cfg) -> sup_loss(model, exit_idx, exit_out, targets, imgs)
    exit_out = the head's raw 3-scale training output.
    returns scalar TAL (box + cls + dfl) * batch_size.
"""

import torch.nn as nn

from . import bootstrap  # noqa: F401
from utils.loss_tal import ComputeLoss  # type: ignore


class _HeadProxy(nn.Module):
    """nn.Module that ComputeLoss can bind to a single EE detect head."""

    def __init__(self, net, head):
        super().__init__()
        self.model = nn.ModuleList([head])      # ComputeLoss reads model[-1]
        self.hyp = net.hyp
        self.nc = head.nc
        self.names = getattr(net, "names", None)


def build_sup_loss(model, cfg):
    net = model.net
    losses = [ComputeLoss(_HeadProxy(net, h)) for h in model.heads]

    def sup_loss(_model, exit_idx, exit_out, targets, imgs):
        loss, _items = losses[exit_idx](exit_out, targets)
        return loss

    return sup_loss
