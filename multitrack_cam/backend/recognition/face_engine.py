"""InsightFace (SCRFD detector + ArcFace) face embeddings attached to person tracks.

Runs on its own thread so face analysis never stalls video or tracking. Only
embedding vectors leave this module; face images are never stored.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

Box = tuple[float, float, float, float]


class FaceEngine:
    def __init__(
        self,
        model: str = "buffalo_l",
        *,
        det_size: int = 1280,
        det_thresh: float = 0.5,
        min_face_size: float = 20.0,
        quality_threshold: float = 0.30,
        head_region: float = 0.5,
    ) -> None:
        from insightface.app import FaceAnalysis  # optional dependency, imported only when enabled

        self.min_face_size = min_face_size
        self.quality_threshold = quality_threshold
        self.head_region = head_region
        self.app = FaceAnalysis(
            name=model,
            allowed_modules=["detection", "recognition"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=det_thresh)
        self.last_ms = 0.0

    @staticmethod
    def _frontal(kps: np.ndarray | None) -> float:
        if kps is None or len(kps) < 3:
            return 0.5
        le, re, nose = kps[0], kps[1], kps[2]
        eye_dist = float(np.hypot(*(re - le)))
        if eye_dist < 1.0:
            return 0.0
        yaw = abs(float(nose[0]) - (float(le[0]) + float(re[0])) / 2.0) / eye_dist
        return float(np.clip(1.0 - yaw / 0.6, 0.0, 1.0))

    def analyze(self, frame: np.ndarray, boxes: list[tuple[str, Box]]) -> dict[str, tuple[np.ndarray, float]]:
        """Return {track_key: (unit face embedding, quality)} for faces inside a person's head region."""
        t0 = time.perf_counter()
        faces = self.app.get(frame)
        out: dict[str, tuple[np.ndarray, float]] = {}
        for f in faces:
            x1, y1, x2, y2 = (float(v) for v in f.bbox)
            w = x2 - x1
            if w < self.min_face_size or getattr(f, "normed_embedding", None) is None:
                continue
            quality = float(f.det_score) * min(1.0, w / 40.0) * self._frontal(getattr(f, "kps", None))
            if quality < self.quality_threshold:
                continue
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            owners = [
                (key, b) for key, b in boxes
                if b[0] <= cx <= b[2] and b[1] <= cy <= b[1] + self.head_region * (b[3] - b[1])
            ]
            if not owners:
                continue
            key, _ = min(owners, key=lambda kb: (kb[1][2] - kb[1][0]) * (kb[1][3] - kb[1][1]))
            if key not in out or quality > out[key][1]:
                out[key] = (np.asarray(f.normed_embedding, dtype=np.float32), quality)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return out


class FaceWorker:
    """Background face analysis; results are collected by the pipeline each cycle."""

    def __init__(self, engine: FaceEngine, interval_seconds: float = 0.5) -> None:
        self.engine = engine
        self.interval = interval_seconds
        self._queue: queue.Queue[tuple[str, np.ndarray, list[tuple[str, Box]]] | None] = queue.Queue(maxsize=8)
        self._results: dict[str, tuple[np.ndarray, float]] = {}
        self._lock = threading.Lock()
        self._last_submit: dict[str, float] = {}
        self._stop = threading.Event()
        self.faces_found = 0
        self._thread = threading.Thread(target=self._run, name="face-worker", daemon=True)
        self._thread.start()

    def submit(self, camera_id: str, frame: np.ndarray, boxes: list[tuple[str, Box]]) -> None:
        now = time.monotonic()
        if not boxes or now - self._last_submit.get(camera_id, 0.0) < self.interval:
            return
        try:
            self._queue.put_nowait((camera_id, frame.copy(), list(boxes)))
            self._last_submit[camera_id] = now
        except queue.Full:
            pass

    def drain(self) -> dict[str, tuple[np.ndarray, float]]:
        with self._lock:
            out, self._results = self._results, {}
        return out

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def _run(self) -> None:
        frames, last_log = 0, time.monotonic()
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None:
                break
            _, frame, boxes = item
            try:
                found = self.engine.analyze(frame, boxes)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Face analysis failed: %s", exc)
                continue
            frames += 1
            if time.monotonic() - last_log >= 60.0:
                logger.info("Face worker: %d frames analysed, %d usable faces so far (%.0f ms/frame)",
                            frames, self.faces_found + len(found), self.engine.last_ms)
                last_log = time.monotonic()
            if found:
                self.faces_found += len(found)
                with self._lock:
                    self._results.update(found)

    def stats(self) -> dict[str, Any]:
        return {"face_ms": round(self.engine.last_ms, 1), "faces_found": self.faces_found}
