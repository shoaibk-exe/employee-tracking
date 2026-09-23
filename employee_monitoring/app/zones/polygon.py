"""Desk polygons. Coordinates come from configuration, never from source code."""

from __future__ import annotations

from dataclasses import dataclass

from app.utils.geometry import Point, point_in_polygon


@dataclass(frozen=True)
class DeskZone:
    desk_id: str
    employee_id: str
    camera_id: str
    polygon: tuple[Point, ...]

    def contains(self, point: Point) -> bool:
        return point_in_polygon(point, list(self.polygon))


def smallest_containing(zones: list[DeskZone], point: Point) -> DeskZone | None:
    """If polygons overlap, the tighter one wins so a shared edge does not pick the neighbor."""
    hits = [zone for zone in zones if zone.contains(point)]
    if not hits:
        return None
    return min(hits, key=_area)


def _area(zone: DeskZone) -> float:
    points = zone.polygon
    total = 0.0
    for index, current in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        total += current[0] * nxt[1] - nxt[0] * current[1]
    return abs(total) / 2.0
