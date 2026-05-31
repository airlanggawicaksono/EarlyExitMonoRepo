"""Supervised TAL loss, one per exit head. THE integration seam to discuss.

yolov9's ComputeLoss binds to a single Detect head (reads model.model[-1] for
stride / nc / reg_max + model.hyp + device). Each EE head has its own stride/anchors
(set by EarlyExitModel._reinit_exit_heads), so we build ONE ComputeLoss per exit,
each pointed at that head via a thin proxy.

build_sup_loss(model, cfg) -> sup_loss(model, exit_idx, exit_out, targets, imgs)
    exit_out = the head's raw 3-scale training output.
    returns scalar TAL (box + cls + dfl).

TODO(colab): ComputeLoss reads a handful of model attrs — confirm the exact set
against the pinned yolov9 (utils/loss_tal.py). Proxy below exposes the known ones;
extend if it asks for more.
"""

from types import SimpleNamespace

from . import bootstrap  # noqa: F401
from utils.loss_tal import ComputeLoss  # type: ignore


def _head_proxy(net, head):
    """Minimal model-like object ComputeLoss can bind to a single head."""
    proxy = SimpleNamespace()
    proxy.model = [head]            # ComputeLoss reads model.model[-1]
    proxy.hyp = net.hyp
    proxy.nc = head.nc
    proxy.names = getattr(net, "names", None)
    return proxy


def build_sup_loss(model, cfg):
    net = model.net
    losses = [ComputeLoss(_head_proxy(net, h)) for h in model.heads]

    def sup_loss(_model, exit_idx, exit_out, targets, imgs):
        loss, _items = losses[exit_idx](exit_out, targets)
        return loss

    return sup_loss
