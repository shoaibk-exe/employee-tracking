"""Camera liveness. Failure is a camera fact, not an employee absence."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from app.utils.timestamps import utc_now


class CameraStatus(str, Enum):
    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"
    RECONNECTING = "RECONNECTING"


@dataclass
class CameraHealth:
    camera_id: str
    last_frame_time: datetime | None
    fps: float
    connection_status: CameraStatus
    reconnect_count: int
    process_fps: float = 0.0
    detail: str = ""


class CameraHealthMonitor:
    def __init__(
        self,
        camera_id: str,
        offline_after_seconds: float = 5.0,
        degraded_below_fps: float = 2.0,
    ) -> None:
        self.camera_id = camera_id
        self.offline_after_seconds = offline_after_seconds
        self.degraded_below_fps = degraded_below_fps
        self._lock = threading.Lock()
        self._last_frame_time: datetime | None = None
        self._frame_times: deque[datetime] = deque(maxlen=60)
        self._process_times: deque[datetime] = deque(maxlen=60)
        self._reconnect_count = 0
        self._reconnecting = False
        self._forced_offline = False
        self._detail = ""

    def mark_reconnecting(self, detail: str = "") -> None:
        with self._lock:
            self._reconnecting = True
            self._forced_offline = False
            self._detail = detail

    def mark_connected(self) -> None:
        with self._lock:
            self._reconnecting = False
            self._forced_offline = False
            self._detail = ""

    def mark_offline(self, detail: str = "") -> None:
        with self._lock:
            self._forced_offline = True
            self._reconnecting = False
            self._detail = detail

    def mark_reconnect_attempt(self) -> None:
        with self._lock:
            self._reconnect_count += 1
            self._reconnecting = True

    def mark_frame(self, when: datetime | None = None) -> None:
        moment = when or utc_now()
        with self._lock:
            self._last_frame_time = moment
            self._frame_times.append(moment)
            self._reconnecting = False
            self._forced_offline = False
            self._detail = ""

    def mark_processed(self, when: datetime | None = None) -> None:
        moment = when or utc_now()
        with self._lock:
            self._process_times.append(moment)

    def snapshot(self, now: datetime | None = None) -> CameraHealth:
        moment = now or utc_now()
        with self._lock:
            fps = _rate(self._frame_times, moment)
            process_fps = _rate(self._process_times, moment)
            status = self._status(moment, fps)
            return CameraHealth(
                camera_id=self.camera_id,
                last_frame_time=self._last_frame_time,
                fps=round(fps, 2),
                connection_status=status,
                reconnect_count=self._reconnect_count,
                process_fps=round(process_fps, 2),
                detail=self._detail,
            )

    def _status(self, now: datetime, fps: float) -> CameraStatus:
        if self._forced_offline:
            return CameraStatus.OFFLINE
        if self._reconnecting and self._last_frame_time is None:
            return CameraStatus.RECONNECTING
        if self._last_frame_time is None:
            return CameraStatus.OFFLINE
        age = (now - self._last_frame_time).total_seconds()
        if age > self.offline_after_seconds:
            return CameraStatus.RECONNECTING if self._reconnecting else CameraStatus.OFFLINE
        if self._reconnecting:
            return CameraStatus.RECONNECTING
        if fps < self.degraded_below_fps:
            return CameraStatus.DEGRADED
        return CameraStatus.ONLINE


def _rate(samples: deque[datetime], now: datetime) -> float:
    recent = [item for item in samples if (now - item).total_seconds() <= 2.0]
    if len(recent) < 2:
        return 0.0
    span = (recent[-1] - recent[0]).total_seconds()
    if span <= 0:
        return 0.0
    return (len(recent) - 1) / span
