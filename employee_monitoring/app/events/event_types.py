"""Domain events. Analytics read these and the sessions they open. They do not count frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4


class EventType(str, Enum):
    EMPLOYEE_ENTERED = "EMPLOYEE_ENTERED"
    EMPLOYEE_EXITED = "EMPLOYEE_EXITED"
    DESK_ENTERED = "DESK_ENTERED"
    DESK_LEFT = "DESK_LEFT"
    PHONE_USE_STARTED = "PHONE_USE_STARTED"
    PHONE_USE_ENDED = "PHONE_USE_ENDED"
    CAMERA_OFFLINE = "CAMERA_OFFLINE"
    CAMERA_ONLINE = "CAMERA_ONLINE"
    IDENTITY_UNKNOWN = "IDENTITY_UNKNOWN"
    UNKNOWN_PERSON_ENTERED = "UNKNOWN_PERSON_ENTERED"


@dataclass
class Event:
    event_type: EventType
    timestamp: datetime
    employee_id: str | None = None
    camera_id: str | None = None
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid4()))

    def as_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "employee_id": self.employee_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
