"""First-seen-today attendance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pipeline.face_engine import FaceMatch
from pipeline.store import Store


@dataclass
class AttendanceEvent:
    employee_id: int
    name: str
    clock_in: str
    confidence: float
    already_present: bool


def apply_matches(
    store: Store,
    matches: list[FaceMatch],
    source: str = "kiosk",
) -> list[AttendanceEvent]:
    events: list[AttendanceEvent] = []
    seen: set[int] = set()
    for match in matches:
        if match.employee_id is None or match.employee_id in seen:
            continue
        seen.add(match.employee_id)
        event = mark_if_first_seen(
            store,
            employee_id=match.employee_id,
            name=match.name,
            confidence=match.score,
            source=source,
        )
        if event:
            events.append(event)
    return events


def mark_if_first_seen(
    store: Store,
    employee_id: int,
    name: str,
    confidence: float,
    source: str = "kiosk",
) -> Optional[AttendanceEvent]:
    clock_in = store.mark_first_seen(employee_id, confidence, source)
    if clock_in:
        return AttendanceEvent(
            employee_id=employee_id,
            name=name,
            clock_in=clock_in,
            confidence=confidence,
            already_present=False,
        )
    return AttendanceEvent(
        employee_id=employee_id,
        name=name,
        clock_in="",
        confidence=confidence,
        already_present=True,
    )
