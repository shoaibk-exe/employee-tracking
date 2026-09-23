"""In-memory view joined from the state machines. Durable time lives in sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.states.attendance_state import AttendanceState
from app.states.desk_state import DeskState
from app.states.phone_state import PhoneState


@dataclass
class EmployeeRuntimeState:
    employee_id: str
    attendance_state: AttendanceState = AttendanceState.UNKNOWN
    desk_state: DeskState = DeskState.UNKNOWN
    phone_state: PhoneState = PhoneState.NO_PHONE
    last_seen_time: datetime | None = None
    last_face_time: datetime | None = None
    last_desk_detection_time: datetime | None = None
    current_attendance_session_id: str | None = None
    current_desk_session_id: str | None = None
    current_camera_id: str | None = None
    current_track_id: int | None = None
    desk_confidence: float | None = None
    identity_confidence: float | None = None
    desk_id: str | None = None
