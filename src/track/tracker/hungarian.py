from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from .iou import iou_matrix


@dataclass
class Track:
    id: int
    box: np.ndarray  # xywh, float32
    cls: int
    hits: int = 1
    age: int = 0


class HungarianTracker:
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 10, min_hits: int = 3):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[Track] = []
        self.next_id = 0

    def update(self, boxes: np.ndarray, classes: np.ndarray) -> list[Track]:
        boxes = np.asarray(boxes, dtype=np.float32)
        classes = np.asarray(classes, dtype=int)

        prev = list(self.tracks)
        matched_track_rows: set[int] = set()
        matched_det_cols: set[int] = set()
        updated: list[Track] = []

        # 1. Associate
        if prev and len(boxes):
            track_boxes = np.array([t.box for t in prev], dtype=np.float32)
            ious = iou_matrix(track_boxes, boxes)
            rows, cols = linear_sum_assignment(-ious)

            for r, c in zip(rows, cols):
                if ious[r, c] >= self.iou_thresh:
                    t = prev[r]
                    updated.append(
                        Track(
                            id=t.id,
                            box=boxes[c].copy(),
                            cls=int(classes[c]),
                            hits=t.hits + 1,
                            age=0,
                        )
                    )
                    matched_track_rows.add(r)
                    matched_det_cols.add(c)

        # 2. Age unmatched existing tracks
        for r, t in enumerate(prev):
            if r not in matched_track_rows:
                if t.age + 1 <= self.max_age:
                    updated.append(
                        Track(
                            id=t.id,
                            box=t.box.copy(),
                            cls=t.cls,
                            hits=t.hits,
                            age=t.age + 1,
                        )
                    )

        # 3. Spawn new tracks
        for c in range(len(boxes)):
            if c not in matched_det_cols:
                updated.append(
                    Track(
                        id=self.next_id,
                        box=boxes[c].copy(),
                        cls=int(classes[c]),
                        hits=1,
                        age=0,
                    )
                )
                self.next_id += 1

        self.tracks = updated
        return [t for t in self.tracks if t.hits >= self.min_hits]