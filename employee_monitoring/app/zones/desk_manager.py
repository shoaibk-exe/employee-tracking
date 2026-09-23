"""Assigned desks for one office camera."""

from __future__ import annotations

from app.config import DeskConfig
from app.utils.geometry import Point
from app.zones.polygon import DeskZone, smallest_containing


class DeskManager:
    def __init__(self, desks: list[DeskConfig]) -> None:
        self._zones = [
            DeskZone(
                desk_id=desk.desk_id,
                employee_id=desk.employee_id,
                camera_id=desk.camera_id,
                polygon=tuple((float(x), float(y)) for x, y in desk.polygon),
            )
            for desk in desks
        ]

    def zones_for(self, camera_id: str) -> list[DeskZone]:
        return [zone for zone in self._zones if zone.camera_id == camera_id]

    def locate(self, camera_id: str, point: Point) -> DeskZone | None:
        return smallest_containing(self.zones_for(camera_id), point)
