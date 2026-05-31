"""DDetect raw output -> (box DFL logits, cls logits). Pure, no IO.

DDetect training output per scale = cat(cv2_box, cv3_cls) along channels:
    [B, 4*reg_max + nc, H, W]
Box = first 4*reg_max channels (DFL: 4 sides x reg_max bins).
Cls = last nc channels (sigmoid, multi-label).
"""


def split_ddetect(feat, reg_max: int, nc: int):
    box = feat[:, : 4 * reg_max]
    cls = feat[:, 4 * reg_max:]
    return box, cls


def box_distribution(box, reg_max: int):
    """[B, 4*reg_max, H, W] -> [B, 4, reg_max, H, W] (per-side bin logits)."""
    b, _, h, w = box.shape
    return box.view(b, 4, reg_max, h, w)
