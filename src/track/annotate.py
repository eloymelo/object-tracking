import cv2
from typing import Callable, Iterable, Optional, Tuple

_Color = Tuple[int, int, int]


def draw_tracks(
    frame,
    tracks: Iterable[object],
    color: Optional[_Color] = None,
    color_fn: Optional[Callable[[object], _Color]] = None,
    label_fmt: str = "ID:{id}",
    label_fn: Optional[Callable[[object], str]] = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> None:
    """Draw bounding boxes and labels for a list of tracks.

    A track object is expected to have ``.box`` as ``(x, y, w, h)`` and
    ``.id`` (and optionally ``.cls`` for formatting).
    """
    for t in tracks:
        x, y, w, h = map(int, t.box)

        if color is not None:
            c = color
        elif color_fn is not None:
            c = color_fn(t)
        else:
            c = (255, 255, 255)

        if label_fn is not None:
            text = label_fn(t)
        else:
            cls = getattr(t, "cls", "")
            text = label_fmt.format(id=t.id, cls=cls)

        cv2.rectangle(frame, (x, y), (x + w, y + h), c, thickness)
        cv2.putText(
            frame,
            text,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            c,
            1,
        )


def draw_metrics(
    frame,
    stats: dict,
    pos: Tuple[int, int] = (10, 30),
    color: _Color = (255, 255, 255),
) -> None:
    """Render a small FPS / inference-time overlay from ``Perf.stats()``."""
    fps = stats.get("fps_render", 0.0)
    infer_ms = stats.get("infer_p50_ms", 0.0)
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} infer={infer_ms:.1f}ms",
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )