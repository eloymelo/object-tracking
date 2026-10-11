# src/track/run_haar.py
import argparse
import cv2
import numpy as np
from track.tracker.greedy import GreedyTracker


def main(
    camera_index: int = 1,
    iou_thresh: float = 0.3,
    max_age: int = 10,
    min_hits: int = 1,
) -> None:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    tracker = GreedyTracker(
        iou_thresh=iou_thresh, max_age=max_age, min_hits=min_hits
    )

    cap = cv2.VideoCapture(camera_index)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

        # Convert OpenCV faces to xywh float boxes
        if len(faces):
            boxes = np.array([[x, y, w, h] for (x, y, w, h) in faces], dtype=np.float32)
            classes = np.zeros(len(faces), dtype=int)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            classes = np.zeros(0, dtype=int)

        tracks = tracker.update(boxes, classes)

        for t in tracks:
            x, y, w, h = map(int, t.box)
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"ID:{t.id}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def _parse_args():
    p = argparse.ArgumentParser(description="Track faces with Haar + greedy IoU.")
    p.add_argument("--camera", type=int, default=1, help="Camera index (default: 1)")
    p.add_argument("--iou", type=float, default=0.3, help="IoU threshold for matching")
    p.add_argument("--max-age", type=int, default=10, help="Frames to keep a lost track")
    p.add_argument("--min-hits", type=int, default=1, help="Frames before a track is reported")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        camera_index=args.camera,
        iou_thresh=args.iou,
        max_age=args.max_age,
        min_hits=args.min_hits,
    )