from datetime import datetime, timedelta, timezone

from app.zones.line_crossing import LineCrossingDetector


def _at(seconds: float) -> datetime:
    return datetime(2026, 9, 22, 9, 0, tzinfo=timezone.utc) + timedelta(seconds=seconds)


def test_outside_to_inside_is_one_entry():
    line = LineCrossingDetector((0, 100), (200, 100), "down", debounce_seconds=3, hysteresis_px=5)
    assert line.update(7, (100, 40), _at(0)) is None
    crossing = line.update(7, (100, 160), _at(1))
    assert crossing is not None
    assert crossing.direction == "ENTRY"
    assert crossing.track_id == 7


def test_inside_to_outside_is_exit():
    line = LineCrossingDetector((0, 100), (200, 100), "down", debounce_seconds=3, hysteresis_px=5)
    line.update(7, (100, 40), _at(0))
    line.update(7, (100, 160), _at(1))
    crossing = line.update(7, (100, 40), _at(2))
    assert crossing is not None
    assert crossing.direction == "EXIT"


def test_same_direction_is_debounced():
    line = LineCrossingDetector((0, 100), (200, 100), "down", debounce_seconds=3, hysteresis_px=5)
    line.update(7, (100, 40), _at(0))
    first = line.update(7, (100, 160), _at(1))
    assert first is not None and first.direction == "ENTRY"
    exit_event = line.update(7, (100, 40), _at(1.5))
    assert exit_event is not None and exit_event.direction == "EXIT"
    repeat = line.update(7, (100, 160), _at(2))
    assert repeat is None


def test_two_tracks_cross_independently():
    line = LineCrossingDetector((0, 100), (200, 100), "down", debounce_seconds=3, hysteresis_px=5)
    line.update(14, (40, 40), _at(0))
    line.update(15, (140, 40), _at(0))
    first = line.update(14, (40, 160), _at(1))
    second = line.update(15, (140, 160), _at(1))
    assert first is not None and second is not None
    assert {first.track_id, second.track_id} == {14, 15}


def test_duplicate_entry_does_not_open_another_session():
    from app.events.event_types import EventType
    from app.states.attendance_state import AttendanceState, AttendanceStateManager

    manager = AttendanceStateManager()
    first = manager.on_crossing("EMP001", "ENTRY", _at(0), "entrance_01", 0.9, track_id=17)
    second = manager.on_crossing("EMP001", "ENTRY", _at(2), "entrance_01", 0.9, track_id=17)
    assert len(first) == 1
    assert first[0].event_type is EventType.EMPLOYEE_ENTERED
    assert second == []
    assert manager.state_of("EMP001") is AttendanceState.PRESENT
    assert manager.session_of("EMP001") is not None
    assert manager.session_of("EMP001").exit_time is None


def test_exit_when_already_outside_is_ignored():
    from app.states.attendance_state import AttendanceStateManager

    manager = AttendanceStateManager()
    assert manager.on_crossing("EMP001", "EXIT", _at(0), "entrance_01") == []
    assert manager.state_of("EMP001").value == "OUTSIDE"
