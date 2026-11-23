from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from core.occupancy.state_manager import UNKNOWN, StateConfig, StateTransition, WorkstationStateManager
from core.occupancy.time_tracker import WorkstationTimeTracker, split_by_day

CLASSES = ["empty", "occupied"]


def tr(prev, new, at, reason="confirmed"):
    return StateTransition("W01", prev, new, at, at, reason)


def run(tracker, transitions):
    return [i for i in (tracker.apply(t) for t in transitions) if i is not None]


def test_exact_durations_and_multiple_transitions():
    t = WorkstationTimeTracker("W01", CLASSES)
    h = 3600.0
    intervals = run(t, [tr(None, UNKNOWN, 9 * h), tr(UNKNOWN, "occupied", 9 * h), tr("occupied", "empty", 10.25 * h),
                        tr("empty", "occupied", 10.5 * h)])
    t.observe(12 * h)
    last = t.finalize("end_of_video")
    assert [(i.state, i.duration) for i in intervals] == [("occupied", 1.25 * h), ("empty", 0.25 * h)]
    assert last.state == "occupied" and last.duration == 1.5 * h and last.end_reason == "truncated_end_of_video"
    s = t.summary()
    assert s["occupied_seconds"] == 2.75 * h and s["empty_seconds"] == 0.25 * h
    assert s["observed_seconds"] == 3 * h and s["transition_count"] == 3


def test_unknown_periods_counted_separately():
    t = WorkstationTimeTracker("W01", CLASSES)
    run(t, [tr(None, UNKNOWN, 0), tr(UNKNOWN, "occupied", 10), tr("occupied", UNKNOWN, 50, "source_outage"),
            tr(UNKNOWN, "occupied", 80)])
    t.observe(100)
    t.finalize("end_of_video")
    s = t.summary()
    assert s["unknown_seconds"] == 40 and s["occupied_seconds"] == 60 and s["observed_seconds"] == 60


def test_open_interval_never_extends_past_last_observation():
    t = WorkstationTimeTracker("W01", CLASSES)
    run(t, [tr(None, UNKNOWN, 0), tr(UNKNOWN, "occupied", 1)])
    t.observe(42.5)
    assert t.totals_at()["occupied"] == 41.5
    interval = t.finalize("end_of_video", end_ts=999)
    assert interval.end == 42.5 and interval.truncated
    assert t.finalize("again") is None


def test_no_double_counting_sum_equals_span():
    sm = WorkstationStateManager("W01", CLASSES, StateConfig())
    t = WorkstationTimeTracker("W01", CLASSES)
    pattern = ["occupied"] * 10 + ["empty"] * 10 + ["occupied"] * 3 + ["empty"] * 12
    for ts, cls in enumerate(pattern):
        for transition in sm.observe(float(ts), cls, 0.95):
            t.apply(transition)
        t.observe(float(ts))
    t.finalize("end_of_video")
    assert sum(t.totals.values()) == pytest.approx(len(pattern) - 1)


def test_out_of_order_transition_rejected():
    t = WorkstationTimeTracker("W01", CLASSES)
    run(t, [tr(None, UNKNOWN, 10), tr(UNKNOWN, "occupied", 20)])
    with pytest.raises(ValueError):
        t.apply(tr("occupied", "empty", 15))
    t.observe(30)
    t.observe(25)
    assert t.last_observation == 30


def test_seconds_by_day_includes_the_open_interval():
    origin = datetime(2026, 10, 21, 23, 0, tzinfo=timezone.utc)
    t = WorkstationTimeTracker("W01", CLASSES, origin, timezone.utc)
    run(t, [tr(None, UNKNOWN, 0), tr(UNKNOWN, "occupied", 0)])
    t.observe(7200)
    parts = t.seconds_by_day(7200)
    assert parts[date(2026, 10, 21)]["occupied"] == pytest.approx(3600)
    assert parts[date(2026, 10, 22)]["occupied"] == pytest.approx(3600)
    assert sum(day["occupied"] for day in parts.values()) == pytest.approx(t.totals_at(7200)["occupied"])


def test_day_boundary_live_session():
    origin = datetime(2026, 10, 21, 23, 0, tzinfo=timezone.utc)
    t = WorkstationTimeTracker("W01", CLASSES, origin, timezone.utc)
    run(t, [tr(None, UNKNOWN, 0), tr(UNKNOWN, "occupied", 0)])
    t.observe(7200)
    t.finalize("shutdown")
    assert t.totals["occupied"] == 7200
    assert dict(t.daily) == {(date(2026, 10, 21), "occupied"): 3600, (date(2026, 10, 22), "occupied"): 3600}


def test_split_by_day_respects_timezone():
    origin = datetime(2026, 10, 21, 18, 30, tzinfo=timezone.utc)  # 23:30 in Asia/Karachi (UTC+5)
    parts = split_by_day(0, 3600, origin, ZoneInfo("Asia/Karachi"))
    assert parts == [(date(2026, 10, 21), 1800), (date(2026, 10, 22), 1800)]


def test_reconnection_after_outage():
    sm = WorkstationStateManager("W01", CLASSES, StateConfig())
    t = WorkstationTimeTracker("W01", CLASSES)

    def apply(transitions):
        for x in transitions:
            t.apply(x)

    for ts in range(6):
        apply(sm.observe(float(ts), "occupied", 0.95))
        t.observe(float(ts))
    apply(sm.mark_unavailable(8.0))
    for ts in range(300, 306):
        apply(sm.observe(float(ts), "occupied", 0.95))
        t.observe(float(ts))
    t.finalize("end_of_stream")
    assert t.totals["occupied"] == pytest.approx(5 + 5)
    assert t.totals["empty"] == pytest.approx(295)
    assert t.totals[UNKNOWN] == pytest.approx(0)
