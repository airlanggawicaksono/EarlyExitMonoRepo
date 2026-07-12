"""Multi-exit YOLO wrapper around AnyTimeYolo's EarlyExitModel.

Key trick (head-only training + frozen backbone): run the backbone+FPN ONCE, cache
the feature maps, then apply each DDetect head to the cache. Per batch = 1 backbone
pass + N head passes, not N full forwards. Teacher and student share the cache.

exit_outputs(imgs) -> list of N exit outputs; each = the DDetect raw training-mode
output (list of 3 scale feature maps).
"""

import torch.nn as nn

from . import bootstrap  # noqa: F401  (injects sys.path)
from early_exit.model import EarlyExitModel, _DETECT_TYPES  # type: ignore


class MultiExitYolo(nn.Module):
    def __init__(self, net: "EarlyExitModel"):
        super().__init__()
        self.net = net
        self._head_idx = [i for i, m in enumerate(net.model) if isinstance(m, _DETECT_TYPES)]
        self.n_exits = len(self._head_idx)

    @property
    def heads(self):
        return [self.net.model[i] for i in self._head_idx]

    # -- split forward --------------------------------------------------------
    def backbone_feats(self, imgs):
        """Run every non-head layer once -> y[] cache (index-aligned to layers).
        Caller wraps in torch.no_grad() when the backbone is frozen."""
        y, x = [], imgs
        for m in self.net.model:
            if isinstance(m, _DETECT_TYPES):
                y.append(None)  # placeholder: keep y[i] aligned to layer i
                continue
            x = self.net._resolve_input(m, x, y)
            x = m(x)
            y.append(x if m.i in self.net.save else None)
        return y

    def head_output(self, exit_idx: int, y):
        head = self.heads[exit_idx]
        inp = self.net._resolve_input(head, None, y)  # head.f is a list of layer idxs
        return head(inp)

    def head_penult(self, exit_idx: int, y):
        """Per-scale PENULTIMATE head features for the pairwise feature-hint:
        [(box_pen, cls_pen), ...] = each scale's cv2[i]/cv3[i] conv stack WITHOUT
        its final 1x1 prediction conv (`[:-1]`). Recomputed on the cached trunk
        y (pairwise-only, so the extra head conv pass is acceptable)."""
        head = self.heads[exit_idx]
        inp = self.net._resolve_input(head, None, y)
        return [
            (head.cv2[i][:-1](inp[i]), head.cv3[i][:-1](inp[i]))
            for i in range(head.nl)
        ]

    def exit_outputs(self, imgs):
        """All exits, one backbone pass. Used by joint (backbone trainable) and as
        the cache source for per-exit head runs."""
        y = self.backbone_feats(imgs)
        return [self.head_output(i, y) for i in range(self.n_exits)]


def _load_weights(net, weights_path):
    """Load pretrained gelan weights: backbone+FPN direct, upstream detection
    head BROADCAST into every EE head slot.

    yolov9 checkpoints pickle a full DetectionModel object, so PyTorch 2.6+'s
    default `weights_only=True` rejects them. We trust the source (official
    yolov9 release), so disable the safe-pickle gate.

    The vanilla gelan-{s,m,c,e}.pt ckpts ship ONE DDetect head at the deepest
    FPN level (ckpt keys `model.22.*`). Every gelan-m-ee exit taps (P3,P4,P5)
    feature maps with the SAME channel widths (240/360/480), so the upstream
    head's tensors fit every EE head slot. Seeding all heads with it gives each
    exit a trained detector as starting point; from-scratch heads on a frozen
    backbone converge far slower and shallow exits often not at all.
    """
    import torch

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"].float().state_dict() if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = dict(sd)

    model_sd = net.state_dict()

    # upstream head = highest model.N index carrying cv2/cv3 tensors
    head_idxs = set()
    for k in sd:
        parts = k.split(".")
        if len(parts) >= 3 and parts[0] == "model" and parts[2] in ("cv2", "cv3"):
            head_idxs.add(int(parts[1]))
    ee_heads = [i for i, m in enumerate(net.model) if isinstance(m, _DETECT_TYPES)]
    upstream = max(head_idxs) if head_idxs else None
    if upstream is not None:
        prefix = f"model.{upstream}."
        head_params = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        for t in ee_heads:
            for sub, v in head_params.items():
                key = f"model.{t}.{sub}"
                if key in model_sd and model_sd[key].shape == v.shape:
                    sd[key] = v.clone()

    filtered = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    print(
        f"[yolo._load_weights] loaded {len(filtered)}/{len(model_sd)} model tensors "
        f"(upstream head model.{upstream} broadcast to EE heads {ee_heads})"
    )
    net.load_state_dict(filtered, strict=False)


def build_model(cfg, ee_yaml, weights_path, nc: int) -> MultiExitYolo:
    from .hyp import HYP

    net = EarlyExitModel(str(ee_yaml), ch=3, nc=nc)
    net.hyp = HYP                        # yolov9 ComputeLoss reads model.hyp
    if weights_path is not None:
        _load_weights(net, weights_path)
    return MultiExitYolo(net)
