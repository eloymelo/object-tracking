import cv2


def iou(box1, box2):
    """Compute Intersection over Union between two boxes in xywh format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    overlap_x1 = max(x1, x2)
    overlap_y1 = max(y1, y2)
    overlap_x2 = min(x1 + w1, x2 + w2)
    overlap_y2 = min(y1 + h1, y2 + h2)

    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
        return 0.0

    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
    union_area = w1 * h1 + w2 * h2 - overlap_area

    return overlap_area / union_area


def main(camera_index: int = 1, iou_thresh: float = 0.3, max_age: int = 10) -> None:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    next_id = 0
    tracks = {}

    cap = cv2.VideoCapture(camera_index)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

        # MATCHING
        matched_ids = set()

        for face in faces:
            best_iou = 0.0
            best_id = -1

            for track_id, track in tracks.items():
                track_box = (track["x"], track["y"], track["w"], track["h"])
                score = iou(face, track_box)
                if score > best_iou:
                    best_iou, best_id = score, track_id

            if best_iou > iou_thresh:
                x, y, w, h = face
                tracks[best_id].update(
                    {"x": x, "y": y, "w": w, "h": h, "age": 0}
                )
                matched_ids.add(best_id)
            else:
                next_id += 1
                x, y, w, h = face
                tracks[next_id] = {"x": x, "y": y, "w": w, "h": h, "age": 0}
                matched_ids.add(next_id)

        # AGING
        for track_id in tracks:
            if track_id not in matched_ids:
                tracks[track_id]["age"] += 1

        # DELETION
        to_delete = [tid for tid, t in tracks.items() if t["age"] > max_age]
        for tid in to_delete:
            del tracks[tid]

        # DRAW
        for track_id, track in tracks.items():
            x, y, w, h = track["x"], track["y"], track["w"], track["h"]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"ID:{track_id}",
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


if __name__ == "__main__":
    main()