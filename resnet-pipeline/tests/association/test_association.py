"""Association engine scenarios: seated, walking, standing, stand-up, occlusion, ID switch, ambiguity."""

from __future__ import annotations

import pytest

from core.association.association_engine import (
    AssociationEngine, AssociationState, Posture, SeatedHeightModel,
)
from core.tracking.person_tracker import PersonTrack, TrackLifecycle
from core.scene.scene_mapper import STATUS_CONFIRMED, Workstation

SEAT = (400, 400, 500, 600)
CONTEXT = (380, 380, 520, 620)


def ws(ws_id="WS01", seat=SEAT, context=CONTEXT, ref=200.0):
    return Workstation(ws_id, seat, context, 1.0, STATUS_CONFIRMED, "admin", ref, {"dwell_persistence": 1.0})


def track(local_id, box, t, predicted=False):
    return PersonTrack(
        local_track_id=local_id, composite_track_id=f"t1:CAM01:s1:{local_id}", tenant_id="t1", camera_id="CAM01",
        tracking_session_id="s1", bbox=tuple(box), confidence=0.9,
        lifecycle=TrackLifecycle.LOST if predicted else TrackLifecycle.CONFIRMED, is_predicted=predicted,
        first_seen_seconds=0.0, last_seen_seconds=t, last_observed_seconds=t, hit_count=10, age_frames=10,
        time_since_update_frames=1 if predicted else 0, lost_age_seconds=0.0,
    )


def run(engine, script, step=0.1):
    """script: list of (t_start, t_end, callable(t) -> list[PersonTrack])."""
    events, frames = [], []
    for start, end, make in script:
        t = start
        while t < end - 1e-9:
            frame = engine.update(round(t, 3), make(round(t, 3)))
            frames.append(frame)
            events += frame.events
            t += step
    return events, frames


def test_seated_person_confirmed_with_backdated_start():
    engine = AssociationEngine([ws()])
    events, frames = run(engine, [(0.0, 5.0, lambda t: [track(1, SEAT, t)])])
    starts = [e for e in events if e.kind == "start"]
    assert len(starts) == 1
    assert starts[0].workstation_id == "WS01"
    assert starts[0].event_seconds == pytest.approx(0.0)
    assert starts[0].decided_seconds == pytest.approx(3.0)
    assert frames[-1].occupants == {"WS01": "t1:CAM01:s1:1"}
    assert frames[-1].tracks["t1:CAM01:s1:1"].posture == Posture.SEATED


def test_walking_person_is_never_associated():
    engine = AssociationEngine([ws()])
    events, frames = run(engine, [(0.0, 4.0, lambda t: [track(1, (int(100 + 600 * t), 400,
                                                                   int(200 + 600 * t), 600), t)])])
    assert not [e for e in events if e.kind == "start"]
    assert all(not f.occupants for f in frames)


def test_standing_person_near_desk_is_not_seated_and_is_reported_near():
    engine = AssociationEngine([ws()])
    standing = (420, 250, 480, 600)
    events, frames = run(engine, [(0.0, 5.0, lambda t: [track(1, standing, t)])])
    assert not events
    last = frames[-1]
    assert last.tracks["t1:CAM01:s1:1"].posture == Posture.STANDING
    assert last.standing_near == {"WS01": ["t1:CAM01:s1:1"]}
    assert last.persons_in_zone == {"WS01": ["t1:CAM01:s1:1"]}


def test_stand_up_ends_association_backdated_to_last_seated_frame():
    engine = AssociationEngine([ws()])
    standing = (420, 250, 480, 600)
    events, _ = run(engine, [
        (0.0, 6.0, lambda t: [track(1, SEAT, t)]),
        (6.0, 10.0, lambda t: [track(1, standing, t)]),
    ])
    ends = [e for e in events if e.kind == "end"]
    assert len(ends) == 1
    assert ends[0].event_seconds == pytest.approx(5.9)
    assert ends[0].decided_seconds == pytest.approx(8.0)


def test_short_occlusion_keeps_association_and_long_gap_ends_it():
    engine = AssociationEngine([ws()])
    events, frames = run(engine, [
        (0.0, 5.0, lambda t: [track(1, SEAT, t)]),
        (5.0, 8.0, lambda t: [track(1, SEAT, t, predicted=True)]),
        (8.0, 9.0, lambda t: [track(1, SEAT, t)]),
    ])
    assert not [e for e in events if e.kind == "end"]
    events, _ = run(engine, [(9.0, 15.0, lambda t: [track(1, SEAT, t, predicted=True)])])
    ends = [e for e in events if e.kind == "end"]
    assert len(ends) == 1 and ends[0].reason == "evidence_gap"
    assert ends[0].event_seconds == pytest.approx(8.9)


def test_candidate_survives_detection_flicker():
    engine = AssociationEngine([ws()])
    events, _ = run(engine, [(0.0, 5.0, lambda t: [track(1, SEAT, t, predicted=int(round(t * 10)) % 2 == 1)])])
    starts = [e for e in events if e.kind == "start"]
    assert len(starts) == 1 and starts[0].event_seconds == pytest.approx(0.0)


def test_track_removed_ends_association_at_last_support():
    engine = AssociationEngine([ws()])
    run(engine, [(0.0, 5.0, lambda t: [track(1, SEAT, t)])])
    frame = engine.update(5.0, [], removed=[track(1, SEAT, 4.9)])
    assert [(e.kind, e.reason, e.event_seconds) for e in frame.events] == [("end", "track_removed", 4.9)]


def test_id_switch_hands_over_workstation_without_overlap():
    engine = AssociationEngine([ws()])
    events, _ = run(engine, [
        (0.0, 5.0, lambda t: [track(1, SEAT, t)]),
        (5.0, 14.0, lambda t: [track(1, SEAT, t, predicted=True), track(2, SEAT, t)]),
    ])
    a_end = next(e for e in events if e.kind == "end" and e.local_track_id == 1)
    b_start = next(e for e in events if e.kind == "start" and e.local_track_id == 2)
    assert a_end.event_seconds == pytest.approx(4.9)
    assert b_start.event_seconds >= a_end.event_seconds
    assert b_start.decided_seconds >= a_end.decided_seconds


def test_person_between_two_identical_seats_is_ambiguous_not_assigned():
    engine = AssociationEngine([ws("WSA", (400, 400, 500, 600), (380, 380, 560, 620)),
                                ws("WSB", (440, 400, 540, 600), (380, 380, 560, 620))])
    events, frames = run(engine, [(0.0, 5.0, lambda t: [track(1, (420, 400, 520, 600), t)])])
    assert not events
    assert frames[-1].tracks["t1:CAM01:s1:1"].state == AssociationState.AMBIGUOUS
    assert frames[-1].ambiguous_workstations == {"WSA", "WSB"}


def test_one_person_per_workstation_two_people_two_seats():
    engine = AssociationEngine([ws("WS01"), ws("WS02", (700, 400, 800, 600), (680, 380, 820, 620))])
    events, frames = run(engine, [(0.0, 5.0, lambda t: [track(1, SEAT, t), track(2, (700, 400, 800, 600), t)])])
    assert frames[-1].occupants == {"WS01": "t1:CAM01:s1:1", "WS02": "t1:CAM01:s1:2"}


def test_sitting_detector_class_marks_seated_for_association():
    engine = AssociationEngine([ws()])
    events = []
    frame = None
    for i in range(50):
        t = round(i * 0.1, 3)
        person = track(1, SEAT, t)
        person.activity_class = "sitting"
        frame = engine.update(t, [person])
        events += frame.events
    assert frame is not None
    assert frame.tracks["t1:CAM01:s1:1"].posture == Posture.SEATED
    assert [e.kind for e in events] == ["start"]


def test_seated_height_model_linear_fit():
    model = SeatedHeightModel([ws("A", (0, 300, 10, 400), ref=100.0), ws("B", (0, 700, 10, 900), ref=300.0)])
    assert model.expected(400) == pytest.approx(100.0)
    assert model.expected(650) == pytest.approx(200.0)


def test_mark_unavailable_ends_all_associations():
    engine = AssociationEngine([ws()])
    run(engine, [(0.0, 5.0, lambda t: [track(1, SEAT, t)])])
    events = engine.mark_unavailable(6.0, "source_outage")
    assert [(e.kind, e.reason, e.event_seconds) for e in events] == [("end", "source_outage", 4.9)]
    assert engine.update(6.1, []).occupants == {}
