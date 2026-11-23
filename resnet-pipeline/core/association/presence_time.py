"""Person- and employee-level seated time, derived from association and identity events.

* Seated interval: a confirmed person-to-workstation association (start backdated to the
  first supporting frame, end backdated to the last supporting frame).
* Employee interval: the part of a seated interval covered by a verified identity
  assignment of that track. Uncovered parts are attributed to ``UNKNOWN_EMPLOYEE`` and are
  never merged into any employee's totals.

This measures presence at a workstation, not productivity.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from core.association.association_engine import AssociationEvent
from core.identity.identity_manager import UNKNOWN_EMPLOYEE, IdentityEvent


@dataclass(frozen=True)
class SeatedInterval:
    composite_track_id: str
    local_track_id: int
    workstation_id: str
    start: float
    end: float
    end_reason: str
    employee_id: str = UNKNOWN_EMPLOYEE

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class PresenceTimeTracker:
    def __init__(self) -> None:
        self._open: dict[str, tuple[str, int, float]] = {}
        self._identity: dict[str, list[list]] = defaultdict(list)  # track -> [[employee, start, end|None]]
        self.seated: list[SeatedInterval] = []
        self.employee_intervals: list[SeatedInterval] = []

    def identity_event(self, event: IdentityEvent) -> None:
        windows = self._identity[event.composite_track_id]
        if event.kind == "assigned" and event.employee_id:
            windows.append([event.employee_id, event.timestamp_seconds, None])
        elif event.kind == "ended":
            for window in windows:
                if window[2] is None and window[0] == event.employee_id:
                    window[2] = event.timestamp_seconds

    def apply(self, event: AssociationEvent) -> list[SeatedInterval]:
        """Returns the employee-split intervals closed by this event (empty for starts)."""
        if event.kind == "start":
            self._open[event.composite_track_id] = (event.workstation_id, event.local_track_id, event.event_seconds)
            return []
        opened = self._open.pop(event.composite_track_id, None)
        if opened is None:
            return []
        ws_id, local_id, start = opened
        end = max(start, event.event_seconds)
        interval = SeatedInterval(event.composite_track_id, local_id, ws_id, start, end, event.reason)
        self.seated.append(interval)
        parts = self._split_by_identity(interval)
        self.employee_intervals.extend(parts)
        return parts

    def _split_by_identity(self, interval: SeatedInterval) -> list[SeatedInterval]:
        cuts: list[tuple[float, float, str]] = []
        for employee, w_start, w_end in self._identity.get(interval.composite_track_id, []):
            lo = max(interval.start, w_start)
            hi = min(interval.end, w_end if w_end is not None else interval.end)
            if hi > lo:
                cuts.append((lo, hi, employee))
        cuts.sort()
        parts: list[SeatedInterval] = []
        cursor = interval.start

        def piece(a: float, b: float, employee: str) -> None:
            if b > a:
                parts.append(SeatedInterval(interval.composite_track_id, interval.local_track_id,
                                            interval.workstation_id, a, b, interval.end_reason, employee))

        for lo, hi, employee in cuts:
            piece(cursor, lo, UNKNOWN_EMPLOYEE)
            piece(max(cursor, lo), hi, employee)
            cursor = max(cursor, hi)
        piece(cursor, interval.end, UNKNOWN_EMPLOYEE)
        return parts

    def open_duration(self, composite_track_id: str, now: float) -> float:
        opened = self._open.get(composite_track_id)
        return max(0.0, now - opened[2]) if opened else 0.0

    def workstation_totals(self) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for interval in self.seated:
            totals[interval.workstation_id] += interval.duration
        return dict(totals)

    def employee_totals(self) -> dict[str, dict[str, float]]:
        """{employee_id: {workstation_id: seconds}}; UNKNOWN_EMPLOYEE kept separate."""
        totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for interval in self.employee_intervals:
            totals[interval.employee_id][interval.workstation_id] += interval.duration
        return {emp: dict(per_ws) for emp, per_ws in totals.items()}
