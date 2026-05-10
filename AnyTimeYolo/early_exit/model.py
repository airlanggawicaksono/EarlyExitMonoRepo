import sys
import torch
import torch.nn as nn
from pathlib import Path

_YOLOV9 = Path(__file__).resolve().parent.parent / 'model' / 'yolov9'
sys.path.insert(0, str(_YOLOV9))

from models.yolo import DetectionModel, Detect, DDetect, DualDetect, TripleDetect, DualDDetect, TripleDDetect

_DETECT_TYPES = (Detect, DDetect, DualDetect, TripleDetect, DualDDetect, TripleDDetect)


class EarlyExitModel(DetectionModel):
    """
    YOLOv9 with N early-exit detection heads.

    Exits are defined in YAML as sequential DDetect/Detect layers.
    Each exit = one DDetect node referencing 3 feature map indices (P3, P4, P5).
    The final DDetect must be last in the YAML head section.

    Training forward returns: List[exit_0_feats, exit_1_feats, ..., exit_N_feats]
    Inference forward returns: exit_N_feats (final exit only)

    Compatible with loss_tal_triple.ComputeLoss when n_exits == 3.
    """

    def __init__(self, cfg, ch=3, nc=None, anchors=None):
        self._init_mode = True
        super().__init__(cfg, ch, nc, anchors)
        self._init_mode = False
        self._reinit_exit_heads(ch)

    # ------------------------------------------------------------------ helpers

    def _exit_heads(self):
        return [m for m in self.model if isinstance(m, _DETECT_TYPES)]

    def _compute_stride(self, feats, s):
        return torch.tensor([s / f.shape[-2] for f in feats])

    def _resolve_input(self, m, x, y):
        if m.f == -1:
            return x
        if isinstance(m.f, int):
            return y[m.f]
        return [x if j == -1 else y[j] for j in m.f]

    # ------------------------------------------------------------------ init

    def _reinit_exit_heads(self, ch):
        """Init strides and biases for all early exit heads (parent only handles last)."""
        s = 256
        self.train()
        with torch.no_grad():
            all_exit_feats = self._forward_once(torch.zeros(1, ch, s, s))
        for head, feats in zip(self._exit_heads()[:-1], all_exit_feats[:-1]):
            head.inplace = self.inplace
            head.stride = self._compute_stride(feats, s)
            head.bias_init()

    # ------------------------------------------------------------------ overrides

    def _forward_once(self, x, profile=False, visualize=False):
        if self._init_mode:
            return super()._forward_once(x, profile, visualize)

        y, exits = [], []
        for m in self.model:
            x = self._resolve_input(m, x, y)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if isinstance(m, _DETECT_TYPES):
                exits.append(x)

        return exits if self.training else exits[-1]

    def _apply(self, fn):
        self = nn.Module._apply(self, fn)
        for head in self._exit_heads():
            head.stride = fn(head.stride)
            head.anchors = fn(head.anchors)
            head.strides = fn(head.strides)
        return self
