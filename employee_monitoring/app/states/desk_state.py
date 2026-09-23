"""Desk presence from observations, not from frame counts.

Missing detections inside the grace window stay AT_DESK. A dead camera becomes
UNKNOWN_CAMERA_FAILURE and does not close the session as AWAY. The outage is not
added to the away timer when the camera returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4

from app.config import DeskStateConfig
from app.events.event_types import Event, EventType


class DeskState(str, Enum):
    UNKNOWN = "UNKNOWN"
    AT_DESK = "AT_DESK"
    POSSIBLY_AWAY = "POSSIBLY_AWAY"
    AWAY = "AWAY"
    POSSIBLY_RETURNED = "POSSIBLY_RETURNED"
    UNKNOWN_CAMERA_FAILURE = "UNKNOWN_CAMERA_FAILURE"


@dataclass
class DeskSessionRecord:
    employee_id: str
    desk_id: str
    start_time: datetime
    end_time: datetime | None = None
    session_id: str = field(default_factory=lambda: str(uuid4()))
    status: str = "open"

    @property
    def duration_seconds(self) -> float | None:
        if self.end_time is None:
            return None
        return max(0.0, (self.end_time - self.start_time).total_seconds())


@dataclass
class _DeskRuntime:
    employee_id: str
    desk_id: str
    state: DeskState = DeskState.UNKNOWN
    last_positive_at: datetime | None = None
    return_started_at: datetime | None = None
    session: DeskSessionRecord | None = None
    state_before_failure: DeskState | None = None
    confidence: float = 0.0


class DeskStateManager:
    def __init__(self, rules: DeskStateConfig | None = None) -> None:
        self.rules = rules or DeskStateConfig()
        self._desks: dict[str, _DeskRuntime] = {}

    def ensure(self, employee_id: str, desk_id: str) -> _DeskRuntime:
        runtime = self._desks.get(employee_id)
        if runtime is None:
            runtime = _DeskRuntime(employee_id=employee_id, desk_id=desk_id)
            self._desks[employee_id] = runtime
        return runtime

    def state_of(self, employee_id: str) -> DeskState:
        runtime = self._desks.get(employee_id)
        if runtime is None:
            return DeskState.UNKNOWN
        return runtime.state

    def session_of(self, employee_id: str) -> DeskSessionRecord | None:
        runtime = self._desks.get(employee_id)
        if runtime is None:
            return None
        return runtime.session

    def confidence_of(self, employee_id: str) -> float:
        runtime = self._desks.get(employee_id)
        if runtime is None:
            return 0.0
        return runtime.confidence

    def restore(self, record: DeskSessionRecord, now: datetime) -> None:
        """Keep the open session, but do not claim AT_DESK until the camera sees the person again."""
        runtime = self.ensure(record.employee_id, record.desk_id)
        runtime.session = record
        runtime.state = DeskState.UNKNOWN
        runtime.last_positive_at = now
        runtime.return_started_at = None

    def camera_failed(self, employee_ids: list[str]) -> None:
        for employee_id in employee_ids:
            runtime = self._desks.get(employee_id)
            if runtime is None or runtime.state == DeskState.UNKNOWN_CAMERA_FAILURE:
                continue
            runtime.state_before_failure = runtime.state
            runtime.state = DeskState.UNKNOWN_CAMERA_FAILURE

    def camera_restored(self, employee_ids: list[str], now: datetime) -> None:
        for employee_id in employee_ids:
            runtime = self._desks.get(employee_id)
            if runtime is None or runtime.state != DeskState.UNKNOWN_CAMERA_FAILURE:
                continue
            previous = runtime.state_before_failure or DeskState.UNKNOWN
            runtime.state = DeskState.UNKNOWN if previous == DeskState.UNKNOWN_CAMERA_FAILURE else previous
            runtime.state_before_failure = None
            runtime.last_positive_at = now
            runtime.return_started_at = None

    def observe(
        self,
        employee_id: str,
        desk_id: str,
        present: bool,
        timestamp: datetime,
        confidence: float = 0.0,
        camera_id: str | None = None,
        camera_ok: bool = True,
    ) -> list[Event]:
        runtime = self.ensure(employee_id, desk_id)
        if not camera_ok:
            if runtime.state != DeskState.UNKNOWN_CAMERA_FAILURE:
                runtime.state_before_failure = runtime.state
                runtime.state = DeskState.UNKNOWN_CAMERA_FAILURE
            return []
        if runtime.state == DeskState.UNKNOWN_CAMERA_FAILURE:
            self.camera_restored([employee_id], timestamp)
        if present:
            runtime.confidence = confidence
            return self._on_present(runtime, timestamp, camera_id, confidence)
        return self._on_absent(runtime, timestamp, camera_id)

    def _on_present(
        self,
        runtime: _DeskRuntime,
        timestamp: datetime,
        camera_id: str | None,
        confidence: float,
    ) -> list[Event]:
        runtime.last_positive_at = timestamp
        if runtime.state in {DeskState.AT_DESK, DeskState.POSSIBLY_AWAY}:
            runtime.state = DeskState.AT_DESK
            runtime.return_started_at = None
            return []
        if runtime.return_started_at is None:
            runtime.return_started_at = timestamp
            runtime.state = DeskState.POSSIBLY_RETURNED
            return []
        held = timestamp - runtime.return_started_at
        if held < timedelta(seconds=self.rules.return_confirm_seconds):
            runtime.state = DeskState.POSSIBLY_RETURNED
            return []
        runtime.return_started_at = None
        if runtime.session is not None and runtime.session.end_time is None:
            runtime.state = DeskState.AT_DESK
            return []
        return self._open_session(runtime, timestamp, camera_id, confidence)

    def _on_absent(
        self,
        runtime: _DeskRuntime,
        timestamp: datetime,
        camera_id: str | None,
    ) -> list[Event]:
        if runtime.last_positive_at is None:
            return []
        elapsed = timestamp - runtime.last_positive_at
        if elapsed < timedelta(seconds=self.rules.detection_grace_seconds):
            return []
        if runtime.state == DeskState.POSSIBLY_RETURNED:
            runtime.return_started_at = None
            if runtime.session is None or runtime.session.end_time is not None:
                runtime.state = DeskState.AWAY
                return []
        confirm = timedelta(seconds=self.rules.away_confirm_seconds)
        candidate = timedelta(seconds=self.rules.away_candidate_seconds)
        if elapsed >= confirm and runtime.state in {
            DeskState.AT_DESK,
            DeskState.POSSIBLY_AWAY,
            DeskState.UNKNOWN,
            DeskState.POSSIBLY_RETURNED,
        }:
            if runtime.session is not None and runtime.session.end_time is None:
                return self._close_session(runtime, timestamp, camera_id)
            runtime.state = DeskState.AWAY
            return []
        if elapsed >= candidate and runtime.state == DeskState.AT_DESK:
            runtime.state = DeskState.POSSIBLY_AWAY
        return []

    def _open_session(
        self,
        runtime: _DeskRuntime,
        timestamp: datetime,
        camera_id: str | None,
        confidence: float,
    ) -> list[Event]:
        runtime.session = DeskSessionRecord(
            employee_id=runtime.employee_id,
            desk_id=runtime.desk_id,
            start_time=timestamp,
        )
        runtime.state = DeskState.AT_DESK
        return [
            Event(
                event_type=EventType.DESK_ENTERED,
                timestamp=timestamp,
                employee_id=runtime.employee_id,
                camera_id=camera_id,
                confidence=confidence,
                metadata={"desk_id": runtime.desk_id, "session_id": runtime.session.session_id},
            )
        ]

    def _close_session(
        self,
        runtime: _DeskRuntime,
        timestamp: datetime,
        camera_id: str | None,
    ) -> list[Event]:
        if runtime.session is None or runtime.session.end_time is not None:
            runtime.state = DeskState.AWAY
            return []
        runtime.session.end_time = timestamp
        runtime.session.status = "closed"
        runtime.state = DeskState.AWAY
        runtime.return_started_at = None
        return [
            Event(
                event_type=EventType.DESK_LEFT,
                timestamp=timestamp,
                employee_id=runtime.employee_id,
                camera_id=camera_id,
                confidence=runtime.confidence or None,
                metadata={
                    "desk_id": runtime.desk_id,
                    "session_id": runtime.session.session_id,
                    "duration_seconds": runtime.session.duration_seconds,
                },
            )
        ]
