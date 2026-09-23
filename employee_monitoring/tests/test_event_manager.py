from datetime import datetime, timezone

from app.events.event_manager import EventManager
from app.events.event_types import Event, EventType


def test_published_event_has_required_fields():
    seen = []
    manager = EventManager(listener=seen.append)
    event = manager.publish(
        Event(
            event_type=EventType.DESK_LEFT,
            employee_id="EMP001",
            camera_id="office_01",
            timestamp=datetime(2026, 9, 22, 10, 12, 44, tzinfo=timezone.utc),
            confidence=0.94,
            metadata={"desk_id": "desk_001"},
        )
    )
    payload = event.as_dict()
    assert payload["event_type"] == "DESK_LEFT"
    assert payload["employee_id"] == "EMP001"
    assert payload["camera_id"] == "office_01"
    assert payload["confidence"] == 0.94
    assert payload["event_id"]
    assert "embedding" not in payload["metadata"]
    assert seen == [event]


def test_unknown_person_is_not_assigned_an_employee():
    from app.states.attendance_state import AttendanceStateManager

    manager = AttendanceStateManager()
    when = datetime(2026, 9, 22, 9, 1, tzinfo=timezone.utc)
    events = manager.on_crossing(None, "ENTRY", when, "entrance_01", track_id=4)
    assert len(events) == 1
    assert events[0].event_type is EventType.UNKNOWN_PERSON_ENTERED
    assert events[0].employee_id is None
    assert manager.session_of("EMP001") is None
