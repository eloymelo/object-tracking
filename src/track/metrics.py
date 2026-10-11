import time
import numpy as np
from collections import deque


class Perf:
    def __init__(self, warmup: int = 30, window: int = 300):
        self.warmup = warmup
        self.n = 0
        self.infer: deque[float] = deque(maxlen=window)
        self.total: deque[float] = deque(maxlen=window)

    def record(self, infer_s: float, total_s: float) -> None:
        self.n += 1
        if self.n > self.warmup:
            self.infer.append(infer_s)
            self.total.append(total_s)

    def stats(self) -> dict:
        i = np.array(self.infer)
        t = np.array(self.total)
        return {
            "infer_p50_ms": float(np.percentile(i, 50) * 1e3) if len(i) else 0.0,
            "infer_p95_ms": float(np.percentile(i, 95) * 1e3) if len(i) else 0.0,
            "fps_render": float(1 / np.mean(t)) if len(t) else 0.0,
            "fps_infer_only": float(1 / np.mean(i)) if len(i) else 0.0,
        }
