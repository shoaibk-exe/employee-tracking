"""Attendance sessions. Presence time is exit minus entry, summed across the day."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4

from app.events.event_types import Event, EventType

logger = logging.getLogger(__name__)


class AttendanceState(str, Enum):
    OUTSIDE = "OUTSIDE"
    ENTERING = "ENTERING"
    PRESENT = "PRESENT"
    EXITING = "EXITING"
    UNKNOWN = "UNKNOWN"
    UNKNOWN_CAMERA_FAILURE = "UNKNOWN_CAMERA_FAILURE"


@dataclass
class AttendanceSessionRecord:
    employee_id: str
    entry_time: datetime
    exit_time: datetime | None = None
    entry_camera: str | None = None
    exit_camera: str | None = None
    session_id: str = field(default_factory=lambda: str(uuid4()))
    status: str = "open"


@dataclass
class _EmployeeAttendance:
    state: AttendanceState = AttendanceState.OUTSIDE
    session: AttendanceSessionRecord | None = None
    state_before_failure: AttendanceState | None = None


class AttendanceStateManager:
    def __init__(self) -> None:
        self._people: dict[str, _EmployeeAttendance] = {}

    def state_of(self, employee_id: str) -> AttendanceState:
        return self._people.get(employee_id, _EmployeeAttendance()).state

    def session_of(self, employee_id: str) -> AttendanceSessionRecord | None:
        person = self._people.get(employee_id)
        if person is None:
            return None
        return person.session

    def restore(self, record: AttendanceSessionRecord) -> None:
        """Restart keeps an open session. It must not insert another entry."""
        person = self._people.setdefault(record.employee_id, _EmployeeAttendance())
        person.state = AttendanceState.PRESENT
        person.session = record

    def mark_identity_seen(self, employee_id: str) -> None:
        person = self._people.setdefault(employee_id, _EmployeeAttendance())
        if person.state == AttendanceState.OUTSIDE:
            person.state = AttendanceState.ENTERING

    def on_crossing(
        self,
        employee_id: str | None,
        direction: str,
        timestamp: datetime,
        camera_id: str,
        confidence: float | None = None,
        track_id: int | None = None,
    ) -> list[Event]:
        if not employee_id:
            if direction != "ENTRY":
                return []
            return [
                Event(
                    event_type=EventType.UNKNOWN_PERSON_ENTERED,
                    timestamp=timestamp,
                    camera_id=camera_id,
                    confidence=confidence,
                    metadata={"track_id": track_id},
                )
            ]
        person = self._people.setdefault(employee_id, _EmployeeAttendance())
        if person.state == AttendanceState.UNKNOWN_CAMERA_FAILURE:
            # A dead entrance camera is not an exit. Keep the last real state underneath.
            return []
        if direction == "ENTRY":
            return self._enter(person, employee_id, timestamp, camera_id, confidence, track_id)
        if direction == "EXIT":
            return self._exit(person, employee_id, timestamp, camera_id, confidence, track_id)
        return []

    def camera_failed(self, timestamp: datetime, camera_id: str) -> list[Event]:
        events: list[Event] = []
        for employee_id, person in self._people.items():
            if person.state == AttendanceState.UNKNOWN_CAMERA_FAILURE:
                continue
            person.state_before_failure = person.state
            person.state = AttendanceState.UNKNOWN_CAMERA_FAILURE
        events.append(
            Event(
                event_type=EventType.CAMERA_OFFLINE,
                timestamp=timestamp,
                camera_id=camera_id,
            )
        )
        return events

    def camera_restored(self, timestamp: datetime, camera_id: str) -> list[Event]:
        for person in self._people.values():
            if person.state != AttendanceState.UNKNOWN_CAMERA_FAILURE:
                continue
            person.state = person.state_before_failure or AttendanceState.UNKNOWN
            person.state_before_failure = None
        return [
            Event(
                event_type=EventType.CAMERA_ONLINE,
                timestamp=timestamp,
                camera_id=camera_id,
            )
        ]

    def _enter(
        self,
        person: _EmployeeAttendance,
        employee_id: str,
        timestamp: datetime,
        camera_id: str,
        confidence: float | None,
        track_id: int | None,
    ) -> list[Event]:
        if person.state == AttendanceState.PRESENT or (
            person.session is not None and person.session.exit_time is None
        ):
            logger.warning(
                "duplicate entry ignored",
                extra={"event": "DUPLICATE_ENTRY", "employee": employee_id, "camera": camera_id},
            )
            return []
        person.session = AttendanceSessionRecord(
            employee_id=employee_id,
            entry_time=timestamp,
            entry_camera=camera_id,
        )
        person.state = AttendanceState.PRESENT
        return [
            Event(
                event_type=EventType.EMPLOYEE_ENTERED,
                timestamp=timestamp,
                employee_id=employee_id,
                camera_id=camera_id,
                confidence=confidence,
                metadata={"session_id": person.session.session_id, "track_id": track_id},
            )
        ]

    def _exit(
        self,
        person: _EmployeeAttendance,
        employee_id: str,
        timestamp: datetime,
        camera_id: str,
        confidence: float | None,
        track_id: int | None,
    ) -> list[Event]:
        if person.state != AttendanceState.PRESENT or person.session is None or person.session.exit_time is not None:
            logger.warning(
                "exit ignored; employee is not present",
                extra={"event": "DUPLICATE_EXIT", "employee": employee_id, "camera": camera_id},
            )
            return []
        person.state = AttendanceState.EXITING
        person.session.exit_time = timestamp
        person.session.exit_camera = camera_id
        person.session.status = "closed"
        person.state = AttendanceState.OUTSIDE
        return [
            Event(
                event_type=EventType.EMPLOYEE_EXITED,
                timestamp=timestamp,
                employee_id=employee_id,
                camera_id=camera_id,
                confidence=confidence,
                metadata={"session_id": person.session.session_id, "track_id": track_id},
            )
        ]
