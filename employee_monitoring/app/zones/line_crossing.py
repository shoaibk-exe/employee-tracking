"""Directional entrance line.

OUTSIDE -> INSIDE is ENTRY. INSIDE -> OUTSIDE is EXIT.
A person standing on the line does not flap events: points inside the hysteresis band keep
their previous side, and the same track cannot repeat a direction inside the debounce window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.utils.geometry import Point, line_cross, line_distance


@dataclass
class CrossingEvent:
    track_id: int
    direction: str  # ENTRY or EXIT
    timestamp: datetime


class LineCrossingDetector:
    def __init__(
        self,
        p1: Point,
        p2: Point,
        inside_direction: str = "down",
        debounce_seconds: float = 3.0,
        hysteresis_px: float = 12.0,
    ) -> None:
        self.p1 = p1
        self.p2 = p2
        self.inside_direction = inside_direction
        self.debounce_seconds = debounce_seconds
        self.hysteresis_px = hysteresis_px
        self._side: dict[int, str] = {}
        self._last_fire: dict[tuple[int, str], datetime] = {}

    def update(self, track_id: int, point: Point, timestamp: datetime) -> CrossingEvent | None:
        distance = line_distance(self.p1, self.p2, point)
        if distance < self.hysteresis_px:
            return None
        side = "inside" if self._is_inside(point) else "outside"
        previous = self._side.get(track_id)
        self._side[track_id] = side
        if previous is None or previous == side:
            return None
        direction = "ENTRY" if previous == "outside" and side == "inside" else "EXIT"
        key = (track_id, direction)
        last = self._last_fire.get(key)
        if last is not None and timestamp - last < timedelta(seconds=self.debounce_seconds):
            return None
        self._last_fire[key] = timestamp
        return CrossingEvent(track_id=track_id, direction=direction, timestamp=timestamp)

    def forget(self, track_id: int) -> None:
        self._side.pop(track_id, None)

    def _is_inside(self, point: Point) -> bool:
        cross = line_cross(self.p1, self.p2, point)
        # Image y grows downward. For a left-to-right line, positive cross is below the line.
        if self.inside_direction == "down":
            return cross > 0
        if self.inside_direction == "up":
            return cross < 0
        if self.inside_direction == "right":
            return cross > 0
        return cross < 0
