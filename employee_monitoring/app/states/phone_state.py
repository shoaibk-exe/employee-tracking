"""Phone-use timing. Not called by the V1 pipeline.

A single phone box is only a candidate. Use starts after the association holds
for confirm_seconds, which is what filters a phone lying on the desk.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from app.config import PhoneConfig
from app.events.event_types import Event, EventType


class PhoneState(str, Enum):
    NO_PHONE = "NO_PHONE"
    PHONE_CANDIDATE = "PHONE_CANDIDATE"
    PHONE_USE = "PHONE_USE"


@dataclass
class _PhoneRuntime:
    state: PhoneState = PhoneState.NO_PHONE
    candidate_since: datetime | None = None
    use_since: datetime | None = None
    confidence: float = 0.0


class PhoneStateManager:
    def __init__(self, rules: PhoneConfig | None = None) -> None:
        self.rules = rules or PhoneConfig()
        self._people: dict[str, _PhoneRuntime] = {}

    def state_of(self, employee_id: str) -> PhoneState:
        return self._people.get(employee_id, _PhoneRuntime()).state

    def observe(
        self,
        employee_id: str,
        phone_associated: bool,
        timestamp: datetime,
        confidence: float = 0.0,
        camera_id: str | None = None,
    ) -> list[Event]:
        runtime = self._people.setdefault(employee_id, _PhoneRuntime())
        if not phone_associated or confidence < self.rules.minimum_confidence:
            return self._clear(runtime, employee_id, timestamp, camera_id)
        runtime.confidence = confidence
        if runtime.state == PhoneState.NO_PHONE:
            runtime.state = PhoneState.PHONE_CANDIDATE
            runtime.candidate_since = timestamp
            return []
        if runtime.state == PhoneState.PHONE_CANDIDATE:
            started = runtime.candidate_since or timestamp
            held = timestamp - started
            if held < timedelta(seconds=self.rules.candidate_seconds):
                return []
            if held < timedelta(seconds=self.rules.confirm_seconds):
                return []
            runtime.state = PhoneState.PHONE_USE
            runtime.use_since = timestamp
            return [
                Event(
                    event_type=EventType.PHONE_USE_STARTED,
                    timestamp=timestamp,
                    employee_id=employee_id,
                    camera_id=camera_id,
                    confidence=confidence,
                )
            ]
        return []

    def _clear(
        self,
        runtime: _PhoneRuntime,
        employee_id: str,
        timestamp: datetime,
        camera_id: str | None,
    ) -> list[Event]:
        if runtime.state == PhoneState.PHONE_USE:
            runtime.state = PhoneState.NO_PHONE
            runtime.candidate_since = None
            runtime.use_since = None
            return [
                Event(
                    event_type=EventType.PHONE_USE_ENDED,
                    timestamp=timestamp,
                    employee_id=employee_id,
                    camera_id=camera_id,
                    confidence=runtime.confidence,
                )
            ]
        runtime.state = PhoneState.NO_PHONE
        runtime.candidate_since = None
        return []
