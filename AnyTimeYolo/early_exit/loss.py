import random
import sys
from pathlib import Path

_YOLOV9 = Path(__file__).resolve().parent.parent / 'model' / 'yolov9'
sys.path.insert(0, str(_YOLOV9))

from utils.loss_tal_triple import ComputeLoss


class EarlyExitLoss(ComputeLoss):
    """
    Extends triple TAL loss with exit sampling (AnytimeYOLO training strategy).

    With sample_exits=True:  randomly pick one early exit per batch + always final.
    With sample_exits=False: train all exits every batch (standard triple behavior).

    Triple loss expects p = [feats_aux1, feats_aux2, feats_final].
    Weights: aux1=0.25, aux2=0.25, final=1.0.

    When sampling: p becomes [sampled_early, sampled_early, final].
    The sampled early exit receives 0.5 total weight that batch.
    """

    def __init__(self, model, use_dfl=True, sample_exits=False):
        super().__init__(model, use_dfl)
        self.sample_exits = sample_exits

    def _sample_to_triple(self, exits):
        """Pick one early exit randomly, duplicate it to fill both aux slots."""
        idx = random.randint(0, len(exits) - 2)
        return [exits[idx], exits[idx], exits[-1]]

    def __call__(self, p, targets, img=None, epoch=0):
        if self.sample_exits:
            p = self._sample_to_triple(p)
        return super().__call__(p, targets, img, epoch)
