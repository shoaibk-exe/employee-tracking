"""RTSP / MP4 capture with skip-bad-frame and reconnect handling."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pipeline import config as pipeline_config


def _ffmpeg_options(is_rtsp: bool) -> None:
    if is_rtsp:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|"
            "fflags;nobuffer|"
            "flags;low_delay|"
            "max_delay;500000|"
            "stimeout;5000000"
        )
    else:
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)


def mask_source(source: str) -> str:
    """Hide credentials in rtsp://user:pass@host/... for UI display."""
    if not source:
        return "(not set)"
    if source.startswith("rtsp://") and "@" in source:
        prefix, rest = source.split("://", 1)
        creds, host = rest.split("@", 1)
        user = creds.split(":", 1)[0] if creds else ""
        return f"{prefix}://{user}:***@{host}"
    return source


def status_canvas(message: str, frame: Optional[np.ndarray] = None) -> np.ndarray:
    if frame is None or frame.size == 0:
        canvas = np.full((480, 854, 3), 18, dtype=np.uint8)
    else:
        canvas = frame.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 54), (16, 16, 16), -1)
    cv2.putText(
        canvas,
        message[:90],
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (70, 70, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _resize_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / float(frame.shape[1])
    new_size = (max_width, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _is_valid_frame(frame: Optional[np.ndarray]) -> bool:
    if frame is None:
        return False
    if not isinstance(frame, np.ndarray):
        return False
    if frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
        return False
    return True


@dataclass
class FrameResult:
    ok: bool
    frame: Optional[np.ndarray]
    status: str
    message: str = ""


class StreamCapture:
    """OpenCV capture that skips corrupt frames and reopens dead RTSP sockets."""

    def __init__(
        self,
        source: str,
        is_rtsp: Optional[bool] = None,
        loop_mp4: Optional[bool] = None,
        reconnect_seconds: Optional[float] = None,
        max_bad_frames: Optional[int] = None,
        open_timeout_seconds: Optional[float] = None,
        frame_max_width: Optional[int] = None,
    ) -> None:
        cfg = pipeline_config
        resolve = getattr(cfg, "resolve_video_source", lambda value: (value or "").strip())
        is_rtsp_fn = getattr(cfg, "source_is_rtsp", lambda _source, mode: mode)
        default_settings = cfg.settings
        self.source = resolve(source or "")
        mode_rtsp = default_settings.is_rtsp if is_rtsp is None else is_rtsp
        self.is_rtsp = is_rtsp_fn(self.source, mode_rtsp)
        self.last_error = ""
        self.loop_mp4 = default_settings.mp4_loop if loop_mp4 is None else loop_mp4
        self.reconnect_seconds = (
            default_settings.capture_reconnect_seconds
            if reconnect_seconds is None
            else reconnect_seconds
        )
        self.max_bad_frames = (
            default_settings.max_bad_frames if max_bad_frames is None else max_bad_frames
        )
        self.open_timeout_seconds = (
            default_settings.open_timeout_seconds
            if open_timeout_seconds is None
            else open_timeout_seconds
        )
        self.frame_max_width = (
            default_settings.frame_max_width if frame_max_width is None else frame_max_width
        )
        self._cap: Optional[cv2.VideoCapture] = None
        self._bad = 0
        self._backoff = self.reconnect_seconds
        self._next_open_at = 0.0
        self._last_good: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    @property
    def last_good(self) -> Optional[np.ndarray]:
        return self._last_good

    def open(self) -> bool:
        if not self.source:
            self.last_error = "No video source configured"
            return False
        self.release()
        _ffmpeg_options(self.is_rtsp)

        local_path = Path(self.source)
        if not self.is_rtsp:
            if not local_path.is_file():
                self.last_error = f"File not found: {self.source}"
                return False
            backends = (cv2.CAP_ANY, cv2.CAP_FFMPEG, cv2.CAP_MSMF)
            for backend in backends:
                cap = cv2.VideoCapture(self.source, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self._cap = cap
                    self._bad = 0
                    self.last_error = ""
                    return True
                cap.release()
            self.last_error = f"Could not open video: {self.source}"
            return False

        holder: dict = {"cap": None, "error": None}

        def _connect() -> None:
            try:
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                holder["cap"] = cap
            except Exception as exc:  # noqa: BLE001
                holder["error"] = exc

        worker = threading.Thread(target=_connect, daemon=True)
        worker.start()
        worker.join(self.open_timeout_seconds)
        if worker.is_alive() or holder["cap"] is None:
            self.last_error = holder["error"] and str(holder["error"]) or "Camera open timed out"
            return False
        cap: cv2.VideoCapture = holder["cap"]
        if not cap.isOpened():
            cap.release()
            self.last_error = f"Could not open stream: {self.source}"
            return False
        self._cap = cap
        self._bad = 0
        self.last_error = ""
        return True

    def _fail(self, status: str, message: str) -> FrameResult:
        self._bad += 1
        if self._bad >= self.max_bad_frames:
            self.release()
            self._next_open_at = time.monotonic() + self._backoff
            self._backoff = min(self._backoff * 1.7, 15.0)
            self._bad = 0
            return FrameResult(False, self._last_good, "reconnecting", message)
        return FrameResult(False, self._last_good, status, message)

    def read(self) -> FrameResult:
        with self._lock:
            return self._read_unlocked()

    def _read_unlocked(self) -> FrameResult:
        if not self.source:
            return FrameResult(False, None, "error", "No video source configured")

        now = time.monotonic()
        if self._cap is None or not self._cap.isOpened():
            if now < self._next_open_at:
                wait = max(0.0, self._next_open_at - now)
                return FrameResult(
                    False,
                    self._last_good,
                    "reconnecting",
                    f"Reconnecting in {wait:.1f}s",
                )
            if not self.open():
                self._next_open_at = time.monotonic() + self._backoff
                self._backoff = min(self._backoff * 1.7, 15.0)
                return FrameResult(
                    False,
                    self._last_good,
                    "reconnecting",
                    self.last_error or "Camera open failed; retrying",
                )

        assert self._cap is not None
        grabbed = self._cap.grab()
        if not grabbed:
            if not self.is_rtsp:
                if self.loop_mp4:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    grabbed = self._cap.grab()
                    if not grabbed:
                        return self._fail("eof", "Video ended")
                else:
                    return FrameResult(False, self._last_good, "eof", "Video ended")
            else:
                return self._fail("bad_frame", "Grab failed")

        retrieved, frame = self._cap.retrieve()
        if not retrieved or not _is_valid_frame(frame):
            return self._fail("bad_frame", "Corrupt or empty frame skipped")

        frame = _resize_max_width(frame, self.frame_max_width)
        self._bad = 0
        self._backoff = self.reconnect_seconds
        self._last_good = frame
        return FrameResult(True, frame, "ok")

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
