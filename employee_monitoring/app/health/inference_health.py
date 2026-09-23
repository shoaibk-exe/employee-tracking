"""Inference failures stay on the model, not on the employee."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime

from app.utils.timestamps import utc_now


@dataclass
class InferenceHealth:
    camera_id: str
    frames_processed: int
    error_count: int
    last_ok: datetime | None
    last_error: str
    last_error_at: datetime | None


class InferenceHealthMonitor:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._lock = threading.Lock()
        self._frames = 0
        self._errors = 0
        self._last_ok: datetime | None = None
        self._last_error = ""
        self._last_error_at: datetime | None = None

    def mark_ok(self) -> None:
        with self._lock:
            self._frames += 1
            self._last_ok = utc_now()

    def mark_error(self, message: str) -> None:
        with self._lock:
            self._errors += 1
            self._last_error = message
            self._last_error_at = utc_now()

    def snapshot(self) -> InferenceHealth:
        with self._lock:
            return InferenceHealth(
                camera_id=self.camera_id,
                frames_processed=self._frames,
                error_count=self._errors,
                last_ok=self._last_ok,
                last_error=self._last_error,
                last_error_at=self._last_error_at,
            )
