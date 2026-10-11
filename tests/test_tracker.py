# tests/test_tracker.py
import unittest
import numpy as np
from track.tracker.iou import iou, iou_matrix
from track.tracker.greedy import GreedyTracker
from track.tracker.hungarian import HungarianTracker


class TestIoU(unittest.TestCase):
    def test_identical_boxes(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[0, 0, 10, 10]], dtype=np.float32)
        self.assertAlmostEqual(iou_matrix(a, b)[0, 0], 1.0, places=5)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0, places=5)

    def test_no_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[20, 20, 10, 10]], dtype=np.float32)
        self.assertEqual(iou_matrix(a, b)[0, 0], 0.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 10, 10)), 0.0)


class TestHungarianTracker(unittest.TestCase):
    def make(self, **kwargs):
        defaults = dict(iou_thresh=0.3, max_age=5, min_hits=1)
        defaults.update(kwargs)
        return HungarianTracker(**defaults)

    def test_persists_id_across_frames(self):
        tr = self.make()
        out1 = tr.update(np.array([[10, 10, 20, 20]], dtype=np.float32), np.array([0]))
        self.assertEqual(len(out1), 1)
        id1 = out1[0].id

        out2 = tr.update(np.array([[12, 12, 20, 20]], dtype=np.float32), np.array([0]))
        self.assertEqual(len(out2), 1)
        self.assertEqual(out2[0].id, id1)

    def test_min_hits_gate(self):
        tr = self.make(min_hits=3)
        box = np.array([[0, 0, 10, 10]], dtype=np.float32)
        cls = np.array([0])

        self.assertEqual(len(tr.update(box, cls)), 0)  # hits=1
        self.assertEqual(len(tr.update(box, cls)), 0)  # hits=2
        out = tr.update(box, cls)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].hits, 3)

    def test_reappears_within_max_age(self):
        tr = self.make(max_age=3)
        box = np.array([[5, 5, 10, 10]], dtype=np.float32)
        cls = np.array([0])
        empty_box = np.zeros((0, 4), dtype=np.float32)
        empty_cls = np.zeros(0, dtype=int)

        id0 = tr.update(box, cls)[0].id
        for _ in range(3):  # age becomes 1, 2, 3 — still <= max_age
            tr.update(empty_box, empty_cls)

        out = tr.update(box, cls)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].id, id0)

    def test_new_id_after_max_age(self):
        tr = self.make(max_age=3)
        box = np.array([[5, 5, 10, 10]], dtype=np.float32)
        cls = np.array([0])
        empty_box = np.zeros((0, 4), dtype=np.float32)
        empty_cls = np.zeros(0, dtype=int)

        id0 = tr.update(box, cls)[0].id
        for _ in range(4):  # age becomes 4 — deleted
            tr.update(empty_box, empty_cls)

        out = tr.update(box, cls)
        self.assertEqual(len(out), 1)
        self.assertNotEqual(out[0].id, id0)

    def test_two_detections_cannot_share_one_track(self):
        tr = self.make()
        tr.update(np.array([[0, 0, 10, 10]], dtype=np.float32), np.array([0]))

        # Two detections overlapping the same track
        out = tr.update(
            np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32),
            np.array([0, 0]),
        )
        self.assertEqual(len(out), 2)
        ids = {t.id for t in out}
        self.assertEqual(len(ids), 2)


class TestGreedyTracker(TestHungarianTracker):
    def make(self, **kwargs):
        defaults = dict(iou_thresh=0.3, max_age=5, min_hits=1)
        defaults.update(kwargs)
        return GreedyTracker(**defaults)


if __name__ == "__main__":
    unittest.main()
