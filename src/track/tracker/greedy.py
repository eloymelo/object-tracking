from dataclasses import dataclass, field
from typing import List, Optional
from .iou import iou


@dataclass
class Track:
    id: int
    box: tuple[float, float, float, float]  # xywh
    cls: int
    hits: int = 1
    age: int = 0


class GreedyTracker:
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 10, min_hits: int = 1):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[Track] = []
        self.next_id = 0

    def update(self, boxes, classes) -> list[Track]:
        prev = list(self.tracks)
        matched_tracks = set()
        matched_dets = set()

        for i, det in enumerate(zip(boxes, classes)):
            if i in matched_dets:
                continue
            det_box, det_cls = det
            best_score, best_t = -1.0, None
            for t in prev:
                if t.id in matched_tracks:
                    continue
                score = iou(t.box, det_box)
                if score > best_score:
                    best_score, best_t = score, t

            if best_t is not None and best_score >= self.iou_thresh:
                best_t.box = det_box
                best_t.cls = det_cls
                best_t.hits += 1
                best_t.age = 0
                matched_tracks.add(best_t.id)
                matched_dets.add(i)

        # New tracks for unmatched detections
        for i, (det_box, det_cls) in enumerate(zip(boxes, classes)):
            if i not in matched_dets:
                self.tracks.append(
                    Track(id=self.next_id, box=det_box, cls=det_cls, hits=1, age=0)
                )
                self.next_id += 1

        # Age unmatched existing tracks
        updated = []
        for t in prev:
            if t.id in matched_tracks:
                updated.append(t)
            else:
                t.age += 1
                if t.age <= self.max_age:
                    updated.append(t)

        self.tracks = updated

        # Only return tracks confirmed by min_hits
        return [t for t in self.tracks if t.hits >= self.min_hits]
