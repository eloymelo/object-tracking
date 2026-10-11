# src/track/run_yolo.py
import argparse
import cv2
import time
import numpy as np
from track.metrics import Perf
from track.tracker.hungarian import HungarianTracker


FLAG_CLASSES = {2, 3, 7}  # car, motorcycle, truck
PERSON_CLASS = 0

COLORS = {
    "flag": (0, 0, 255),     # RED
    "person": (0, 255, 0),   # GREEN
    "default": (255, 0, 0),  # BLUE
}


def get_color(cls):
    if cls in FLAG_CLASSES:
        return COLORS["flag"]
    if cls == PERSON_CLASS:
        return COLORS["person"]
    return COLORS["default"]


def get_label(model, cls):
    return model.names[int(cls)]


def main(
    source: str = "1",
    conf: float = 0.5,
    iou_thresh: float = 0.3,
    max_age: int = 10,
    min_hits: int = 3,
) -> None:
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    tracker = HungarianTracker(
        iou_thresh=iou_thresh, max_age=max_age, min_hits=min_hits
    )
    perf = Perf(warmup=30)

    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Inference timing (render excluded)
        t0 = time.perf_counter()
        results = model(frame, conf=conf, verbose=False)[0]
        t1 = time.perf_counter()

        # Batch-convert YOLO boxes to xywh
        if results.boxes is not None and len(results.boxes):
            xyxy = results.boxes.xyxy.cpu().numpy()
            dets_cls = results.boxes.cls.cpu().numpy().astype(int)
            dets_box = np.column_stack([
                xyxy[:, 0],
                xyxy[:, 1],
                xyxy[:, 2] - xyxy[:, 0],
                xyxy[:, 3] - xyxy[:, 1],
            ]).astype(np.float32)
        else:
            dets_box = np.zeros((0, 4), dtype=np.float32)
            dets_cls = np.zeros(0, dtype=int)

        tracks = tracker.update(dets_box, dets_cls)

        # Draw tracks
        for t in tracks:
            x, y, w, h = map(int, t.box)
            color = get_color(t.cls)
            label = get_label(model, t.cls)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"ID:{t.id} {label}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            if t.cls in FLAG_CLASSES:
                cv2.putText(
                    frame,
                    "!! FLAG !!",
                    (x, y + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

        # Full-loop timing (includes drawing + imshow)
        t2 = time.perf_counter()
        perf.record(t1 - t0, t2 - t0)

        stats = perf.stats()
        fps = stats.get("fps_render", 0.0)
        infer_ms = stats.get("infer_p50_ms", 0.0)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} infer={infer_ms:.1f}ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.imshow("YOLO Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def _parse_args():
    p = argparse.ArgumentParser(description="Track objects with YOLOv8 + Hungarian IoU.")
    p.add_argument("--source", type=str, default="1", help="Camera index or video file path")
    p.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.3, help="IoU threshold for matching")
    p.add_argument("--max-age", type=int, default=10, help="Frames to keep a lost track")
    p.add_argument("--min-hits", type=int, default=3, help="Frames before a track is reported")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        source=args.source,
        conf=args.conf,
        iou_thresh=args.iou,
        max_age=args.max_age,
        min_hits=args.min_hits,
    )