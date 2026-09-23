from datetime import datetime, timedelta, timezone

from app.config import DeskStateConfig
from app.events.event_types import EventType
from app.states.desk_state import DeskState, DeskStateManager
from app.utils.geometry import point_in_polygon
from app.zones.polygon import DeskZone, smallest_containing


RULES = DeskStateConfig(
    away_candidate_seconds=3,
    away_confirm_seconds=10,
    return_confirm_seconds=3,
    detection_grace_seconds=5,
)


def _at(seconds: float) -> datetime:
    return datetime(2026, 9, 22, 9, 0, tzinfo=timezone.utc) + timedelta(seconds=seconds)


def _seat(manager: DeskStateManager, present: bool, seconds: float, camera_ok: bool = True):
    return manager.observe(
        "EMP001",
        "desk_001",
        present,
        _at(seconds),
        confidence=0.94,
        camera_id="office_01",
        camera_ok=camera_ok,
    )


def _reach_desk(manager: DeskStateManager) -> list:
    events = []
    events += _seat(manager, True, 0)
    events += _seat(manager, True, 3)
    assert manager.state_of("EMP001") is DeskState.AT_DESK
    return events


def test_missing_detection_does_not_leave_desk():
    manager = DeskStateManager(RULES)
    events = _reach_desk(manager)
    events += _seat(manager, True, 4)
    events += _seat(manager, False, 5)
    events += _seat(manager, True, 6)
    assert manager.state_of("EMP001") is DeskState.AT_DESK
    assert not any(event.event_type is EventType.DESK_LEFT for event in events)


def test_absence_for_twelve_seconds_emits_one_desk_left():
    manager = DeskStateManager(RULES)
    events = _reach_desk(manager)
    for second in range(4, 16):
        events += _seat(manager, False, second)
    left = [event for event in events if event.event_type is EventType.DESK_LEFT]
    assert len(left) == 1
    assert manager.state_of("EMP001") is DeskState.AWAY
    assert manager.session_of("EMP001").end_time is not None


def test_camera_outage_is_not_away():
    manager = DeskStateManager(RULES)
    _reach_desk(manager)
    events = []
    for second in range(4, 30):
        events += _seat(manager, False, second, camera_ok=False)
    assert manager.state_of("EMP001") is DeskState.UNKNOWN_CAMERA_FAILURE
    assert events == []
    assert manager.session_of("EMP001").end_time is None


def test_restart_does_not_open_a_second_desk_session():
    manager = DeskStateManager(RULES)
    events = _reach_desk(manager)
    session = manager.session_of("EMP001")
    restored = DeskStateManager(RULES)
    restored.restore(session, _at(100))
    assert restored.state_of("EMP001") is DeskState.UNKNOWN
    again = []
    again += restored.observe("EMP001", "desk_001", True, _at(100), 0.9, "office_01")
    again += restored.observe("EMP001", "desk_001", True, _at(103), 0.9, "office_01")
    assert restored.state_of("EMP001") is DeskState.AT_DESK
    assert again == []
    assert restored.session_of("EMP001").session_id == session.session_id
    assert len([event for event in events if event.event_type is EventType.DESK_ENTERED]) == 1


def test_polygon_uses_bottom_center_inside_assigned_desk():
    zone = DeskZone(
        "desk_001",
        "EMP001",
        "office_01",
        ((0, 0), (100, 0), (100, 100), (0, 100)),
    )
    assert point_in_polygon((50, 80), list(zone.polygon))
    assert smallest_containing([zone], (50, 80)) == zone
    assert smallest_containing([zone], (150, 80)) is None


def test_away_from_desk_is_not_outside_the_office():
    from app.analytics import TimeSpan, away_seconds, spans_on_day, total_seconds

    day = _at(0).date()
    office = spans_on_day([TimeSpan(_at(0), _at(8 * 3600))], day, "UTC", _at(9 * 3600))
    desk = spans_on_day(
        [TimeSpan(_at(10 * 60), _at(90 * 60)), TimeSpan(_at(105 * 60), _at(4 * 3600))],
        day,
        "UTC",
        _at(9 * 3600),
    )
    assert total_seconds(office) == 8 * 3600
    assert away_seconds(office, desk) < total_seconds(office)
    assert away_seconds([], desk) == 0
