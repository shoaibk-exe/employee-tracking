"""Backoff for dead RTSP sockets. The reader never gives up; it only slows retries."""

from __future__ import annotations


class ReconnectManager:
    def __init__(self, base_seconds: float = 2.0, max_seconds: float = 15.0) -> None:
        self.base_seconds = base_seconds
        self.max_seconds = max_seconds
        self.count = 0

    def fail(self) -> float:
        self.count += 1
        delay = self.base_seconds * (1.7 ** (self.count - 1))
        return min(delay, self.max_seconds)

    def reset(self) -> None:
        self.count = 0
