"""Grab one rotated frame so ROI boxes are drawn in detection coordinates."""

from __future__ import annotations

import time

import cv2
import numpy as np

from api.redact import redact_rtsp
from core.io.video_reader import ReconnectPolicy, VideoSourceError, default_capture_factory, rotate

# HEVC/H.265 live streams often emit grey mush until the first IDR keyframe.
# Real office frames from these cameras land around std ≈ 70; corrupt opens are < 15.
_MIN_FRAME_STD = 20.0


def _frame_detail(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(gray.std())


def grab_frame(source: str, rotation: int, timeout_seconds: float = 12.0) -> tuple[bytes, int, int]:
    """Return JPEG bytes and the size of the frame after ``rotation``.

    Reads until a usable frame arrives (not the first incomplete HEVC decode),
    or until ``timeout_seconds`` elapses.
    """
    policy = ReconnectPolicy(open_timeout_seconds=timeout_seconds, read_timeout_seconds=timeout_seconds)
    capture = default_capture_factory(source, policy)
    safe_source = redact_rtsp(source)
    if capture is None or not capture.isOpened():
        raise VideoSourceError(f"Could not open video source: {safe_source}")

    deadline = time.monotonic() + timeout_seconds
    best: tuple[float, np.ndarray] | None = None
    chosen: np.ndarray | None = None
    try:
        while time.monotonic() < deadline:
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                continue
            detail = _frame_detail(frame)
            if best is None or detail > best[0]:
                best = (detail, frame)
            if detail >= _MIN_FRAME_STD:
                chosen = frame
                break
    finally:
        capture.release()

    if chosen is None:
        if best is None:
            raise VideoSourceError(f"Could not read a frame from {safe_source}")
        if best[0] < 12.0:
            raise VideoSourceError(
                f"Only corrupt/grey frames from {safe_source} (detail std={best[0]:.1f}). "
                "The stream may still be waiting for a keyframe."
            )
        chosen = best[1]

    frame = rotate(chosen, rotation)
    height, width = frame.shape[:2]
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise VideoSourceError("Could not encode the snapshot frame.")
    return encoded.tobytes(), int(width), int(height)
