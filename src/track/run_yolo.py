import argparse
import cv2
import time


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


def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ox1, oy1 = max(x1, x2), max(y1, y2)
    ox2, oy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    if ox2 <= ox1 or oy2 <= oy1:
        return 0.0
    overlap = (ox2 - ox1) * (oy2 - oy1)
    union = w1 * h1 + w2 * h2 - overlap
    return overlap / union


def main(
    source: str = "1",
    conf: float = 0.5,
    iou_thresh: float = 0.3,
    max_age: int = 10,
) -> None:
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    next_id = 0
    tracks = {}

    frame_count = 0
    fps_start = time.time()
    FPS = 0.0

    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # DETECT
        results = model(frame, conf=conf, verbose=False)

        # FPS
        frame_count += 1
        FPS = frame_count / (time.time() - fps_start)

        # Convert YOLO boxes to xywh
        dets = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            cls = int(box.cls[0])
            dets.append((x1, y1, x2 - x1, y2 - y1, cls))

        # MATCHING
        matched_ids = set()
        for det in dets:
            x, y, w, h, cls = det
            best_iou, best_id = 0.0, -1
            for tid, tr in tracks.items():
                score = iou((x, y, w, h), (tr["x"], tr["y"], tr["w"], tr["h"]))
                if score > best_iou:
                    best_iou, best_id = score, tid
            if best_iou > iou_thresh:
                tracks[best_id].update(x=x, y=y, w=w, h=h, age=0, cls=cls)
                matched_ids.add(best_id)
            else:
                next_id += 1
                tracks[next_id] = {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "age": 0,
                    "cls": cls,
                }
                matched_ids.add(next_id)

        # AGING
        for tid in tracks:
            if tid not in matched_ids:
                tracks[tid]["age"] += 1

        # DELETE
        to_delete = [t for t, tr in tracks.items() if tr["age"] > max_age]
        for tid in to_delete:
            del tracks[tid]

        # DRAW
        for tid, tr in tracks.items():
            x, y = int(tr["x"]), int(tr["y"])
            w, h = int(tr["w"]), int(tr["h"])
            color = get_color(tr["cls"])
            label = get_label(model, tr["cls"])

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"ID:{tid} {label}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

            if tr["cls"] in FLAG_CLASSES:
                cv2.putText(
                    frame,
                    "!! FLAG !!",
                    (x, y + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

        cv2.putText(
            frame,
            f"FPS: {FPS:.1f}",
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
    p = argparse.ArgumentParser(description="Track objects with YOLOv8 + greedy IoU.")
    p.add_argument("--source", type=str, default="1", help="Camera index or video file path")
    p.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.3, help="Iou threshold for matching")
    p.add_argument("--max-age", type=int, default=10, help="Frames to keep a lost track")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        source=args.source,
        conf=args.conf,
        iou_thresh=args.iou,
        max_age=args.max_age,
    )