# object-tracking

Real-time multi-object tracking from a video stream or camera.

This project demonstrates the complete detection + tracking pipeline:

1. Detect objects per frame with a chosen detector.
2. Associate detections across frames using IoU-based matching.
3. Assign persistent IDs so the same object keeps the same number.

It ships with two reference pipelines:

| Runner                       | Detector                   | Tracker       | Best for                            |
|------------------------------|----------------------------|---------------|-------------------------------------|
| `python -m track`            | YOLOv8 (80 COCO classes)   | Hungarian IoU | general objects, vehicles, people   |
| `python -m track.run_haar`   | Haar cascade (faces)       | Greedy IoU    | simple face tracking                |

---

## Quick start

```bash
# 1. Install the project in editable mode
make install

# 2. Run the YOLOv8 tracker on the default camera
make run-yolo

# 3. Or run the Haar face tracker
make run-haar

# 4. Run tests
make test
```

---

## Installation

Requires Python 3.9+.

```bash
python3 -m pip install -e ".[dev]"
```

The `ultralytics` package will be downloaded automatically.

---

## Usage

### YOLOv8 tracker

```bash
python3 -m track --source 1 --conf 0.5 --iou 0.3
```

Arguments:

| Argument    | Default | Description                            |
|-------------|---------|----------------------------------------|
| `--source`  | `1`     | Camera index or path to a video file   |
| `--conf`    | `0.5`   | YOLO detection confidence threshold    |
| `--iou`     | `0.3`   | IoU threshold used to match detections |
| `--max-age` | `10`    | Frames a lost track is kept alive      |
| `--min-hits`| `3`     | Frames before a new track is reported  |

Special visual cues:

- Persons are green, vehicles (car/motorcycle/truck) are red.
- Vehicles show an extra `!! FLAG !!` label.

### Haar face tracker

```bash
python3 -m track.run_haar --camera 1 --iou 0.3
```

Same `--iou`, `--max-age`, and `--min-hits` semantics as the YOLO runner.

---

## Project layout

```text
src/track/
├── annotate.py          # Shared drawing helpers
├── cli_cameras.py       # Utility to list available cameras
├── detector_haar.py     # Haar cascade detector wrapper
├── metrics.py           # FPS / latency metrics
├── run_haar.py          # Haar + greedy tracker demo
├── run_yolo.py          # YOLOv8 + Hungarian tracker demo
└── tracker/
    ├── greedy.py        # Greedy one-to-one IoU tracker
    ├── hungarian.py     # Hungarian-algorithm IoU tracker
    └── iou.py           # IoU utilities
tests/
└── test_tracker.py      # Deterministic tracker tests
```

---

## Design decisions

- **IoU-only association.** Both trackers rely solely on bounding-box overlap. This keeps the code small and easy to understand while being sufficient for moderate frame rates and slow-to-medium motion.
- **Two trackers.** `GreedyTracker` is the simplest possible correct implementation; `HungarianTracker` uses `scipy.optimize.linear_sum_assignment` for globally optimal one-to-one assignment. They share the same `update(boxes, classes)` interface, so they can be swapped easily.
- **min-hits gate.** A track is not reported until it has been seen for `--min-hits` consecutive frames. This suppresses most false-positive IDs from detector noise.
- **Honest metrics.** FPS numbers include drawing and `imshow` time, and a short warmup period is excluded so reported numbers reflect steady-state performance.

---

## Limitations

- Identity switches can still happen when objects overlap heavily or move fast.
- Occlusion, re-entries after `--max-age`, and similar-looking objects are the main failure modes common to all pure-IoU trackers.
- The Haar pipeline is limited to frontal faces and is sensitive to lighting.

---

## License

MIT. See [LICENSE](LICENSE).