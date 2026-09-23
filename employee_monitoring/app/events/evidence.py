"""Short evidence clips around an event. The NVR keeps the raw day; this process does not."""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from app.events.event_types import Event
from app.utils.timestamps import utc_now

logger = logging.getLogger(__name__)


@dataclass
class _PendingClip:
    event: Event
    frames: list[tuple[datetime, object]]
    deadline: datetime


class EvidenceBuffer:
    def __init__(
        self,
        root: Path,
        enabled: bool = False,
        pre_seconds: float = 3.0,
        post_seconds: float = 5.0,
        retention_days: int = 14,
    ) -> None:
        self.root = root
        self.enabled = enabled
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self._recent: dict[str, deque[tuple[datetime, object]]] = {}
        self._pending: list[_PendingClip] = []

    def push(self, camera_id: str, frame: object, timestamp: datetime) -> None:
        if not self.enabled:
            return
        with self._lock:
            bucket = self._recent.setdefault(camera_id, deque())
            bucket.append((timestamp, frame.copy() if hasattr(frame, "copy") else frame))
            cutoff = timestamp - timedelta(seconds=self.pre_seconds + 1)
            while bucket and bucket[0][0] < cutoff:
                bucket.popleft()

    def on_event(self, event: Event) -> None:
        if not self.enabled or event.camera_id is None:
            return
        with self._lock:
            recent = list(self._recent.get(event.camera_id, ()))
            pre_from = event.timestamp - timedelta(seconds=self.pre_seconds)
            frames = [(ts, frame) for ts, frame in recent if ts >= pre_from]
            self._pending.append(
                _PendingClip(
                    event=event,
                    frames=frames,
                    deadline=event.timestamp + timedelta(seconds=self.post_seconds),
                )
            )

    def poll(self, now: datetime | None = None) -> None:
        if not self.enabled:
            return
        moment = now or utc_now()
        ready: list[_PendingClip] = []
        with self._lock:
            still: list[_PendingClip] = []
            for clip in self._pending:
                camera_frames = self._recent.get(clip.event.camera_id or "", ())
                have = {id(frame) for _ts, frame in clip.frames}
                for ts, frame in camera_frames:
                    if ts <= clip.event.timestamp or id(frame) in have:
                        continue
                    if ts <= clip.deadline:
                        clip.frames.append((ts, frame))
                        have.add(id(frame))
                if moment >= clip.deadline:
                    ready.append(clip)
                else:
                    still.append(clip)
            self._pending = still
        for clip in ready:
            self._write(clip)

    def purge_expired(self, now: datetime | None = None) -> None:
        if not self.root.exists():
            return
        moment = now or utc_now()
        cutoff = moment - timedelta(days=self.retention_days)
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=moment.tzinfo)
            if modified < cutoff:
                path.unlink(missing_ok=True)

    def _write(self, clip: _PendingClip) -> None:
        if not clip.frames:
            return
        employee = clip.event.employee_id or "unknown"
        day = clip.event.timestamp.strftime("%Y-%m-%d")
        folder = self.root / employee / day
        folder.mkdir(parents=True, exist_ok=True)
        stamp = clip.event.timestamp.strftime("%H%M%S")
        name = f"{clip.event.event_type.value.lower()}_{stamp}.mp4"
        path = folder / name
        frame = clip.frames[0][1]
        height, width = frame.shape[:2]
        try:
            import cv2

            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                8.0,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("video writer did not open")
            for _ts, image in clip.frames:
                writer.write(image)
            writer.release()
        except Exception:
            logger.warning(
                "evidence clip failed",
                extra={"event": "EVIDENCE_FAILED", "employee": employee, "camera": clip.event.camera_id},
            )
