import numpy as np


def iou(box1, box2):
    """IoU between two boxes in xywh format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    ox1 = max(x1, x2)
    oy1 = max(y1, y2)
    ox2 = min(x1 + w1, x2 + w2)
    oy2 = min(y1 + h1, y2 + h2)

    if ox2 <= ox1 or oy2 <= oy1:
        return 0.0

    overlap = (ox2 - ox1) * (oy2 - oy1)
    union = w1 * h1 + w2 * h2 - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) xywh vs (M,4) xywh -> (N,M) IoU matrix."""
    ax1, ay1, aw, ah = a.T
    bx1, by1, bw, bh = b.T

    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    inter_w = np.maximum(
        0.0, np.minimum(ax2[:, None], bx2[None, :]) - np.maximum(ax1[:, None], bx1[None, :])
    )
    inter_h = np.maximum(
        0.0, np.minimum(ay2[:, None], by2[None, :]) - np.maximum(ay1[:, None], by1[None, :])
    )
    inter = inter_w * inter_h
    union = (aw[:, None] * ah[:, None]) + (bw[None, :] * bh[None, :]) - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
