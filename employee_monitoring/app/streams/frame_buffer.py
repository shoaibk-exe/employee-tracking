"""Bounded latest-frame buffer.

An unbounded queue lets inference fall behind live video. Monitoring only needs the newest frame.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class FramePacket:
    camera_id: str
    frame: object
    frame_number: int
    capture_timestamp: datetime
    received_timestamp: datetime


class LatestFrameBuffer(Generic[T]):
    def __init__(self, maxlen: int = 2) -> None:
        # 1–3 frames. More than that is a latency queue, not a buffer.
        self._maxlen = max(1, min(3, int(maxlen)))
        self._items: deque[T] = deque(maxlen=self._maxlen)
        self._lock = threading.Lock()

    def put(self, item: T) -> None:
        with self._lock:
            self._items.append(item)

    def get_latest(self) -> T | None:
        """Return the newest item and drop anything older still sitting in the buffer."""
        with self._lock:
            if not self._items:
                return None
            latest = self._items[-1]
            self._items.clear()
            return latest

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
