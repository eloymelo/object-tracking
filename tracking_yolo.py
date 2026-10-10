import cv2
from ultralytics import YOLO

# ─── DETECTION LAYER (the only part that changed) ───
model = YOLO('yolov8n.pt')          # downloads ~6MB on first run
CONF = 0.5

# ─── TRACKING STATE (identical to before) ───
next_id = 0
tracks = {}


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


cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # ─── DETECT ───
    results = model(frame, conf=CONF, verbose=False)

    # Convert YOLO boxes (x1,y1,x2,y2) -> (x, y, w, h)
    dets = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        dets.append((x1, y1, x2 - x1, y2 - y1))

    # ─── MATCHING (same as your face tracker) ───
    matched_ids = set()
    for det in dets:
        best_iou, best_id = 0.0, -1
        for tid, tr in tracks.items():
            score = iou(det, (tr['x'], tr['y'], tr['w'], tr['h']))
            if score > best_iou:
                best_iou, best_id = score, tid
        if best_iou > 0.3:
            x, y, w, h = det
            tracks[best_id].update(x=x, y=y, w=w, h=h, age=0)
            matched_ids.add(best_id)
        else:
            next_id += 1
            x, y, w, h = det
            tracks[next_id] = {'x': x, 'y': y, 'w': w, 'h': h, 'age': 0}
            matched_ids.add(next_id)

    # ─── AGING ───
    for tid in tracks:
        if tid not in matched_ids:
            tracks[tid]['age'] += 1

    # ─── DELETE ───
    for tid in [t for t, tr in tracks.items() if tr['age'] > 10]:
        del tracks[tid]

    # ─── DRAW ───
    for tid, tr in tracks.items():
        x, y = int(tr['x']), int(tr['y'])
        w, h = int(tr['w']), int(tr['h'])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'ID:{tid}', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('YOLO Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()