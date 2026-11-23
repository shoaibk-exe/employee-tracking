"""Working-time clock: YOLO sitting + ResNet-occupied ROI; desk empty/working totals."""

from __future__ import annotations

from core.tracking.person_tracker import PersonTrack, TrackLifecycle, suppress_overlapping_detections
from core.association.working_time import WorkingTimeTracker, WorkstationOccupancy


def make_track(local_id, box, activity, ts=0.0, predicted=False):
    return PersonTrack(
        local_id, f"demo:CAM01:s:{local_id}", "demo", "CAM01", "s", box, 0.9,
        TrackLifecycle.CONFIRMED, predicted, 0.0, ts, ts, 5, 5, 0, 0.0,
        activity_class=activity,
    )


def test_timer_starts_only_when_working_and_roi_occupied():
    clock = WorkingTimeTracker()
    zone = WorkstationOccupancy("WS01", (0, 0, 100, 100), state="occupied")
    t = make_track(1, (10, 10, 50, 80), "working", ts=0.0)

    clock.update(0.0, [t], [zone])
    clock.update(5.0, [make_track(1, (10, 10, 50, 80), "working", ts=5.0)], [zone])
    views = clock.snapshot(5.0)
    assert len(views) == 1 and views[0].active and views[0].total_working_seconds == 5.0
    desks = {d.workstation_id: d for d in clock.desk_snapshot(5.0)}
    assert desks["WS01"].working_seconds == 5.0 and desks["WS01"].empty_seconds == 0.0

    clock.update(8.0, [make_track(1, (10, 10, 50, 80), "not_working", ts=8.0)], [zone])
    views = clock.snapshot(8.0)
    assert not views[0].active and views[0].total_working_seconds == 8.0

    clock.update(10.0, [make_track(1, (10, 10, 50, 80), "working", ts=10.0)], [zone])
    clock.update(13.0, [make_track(1, (10, 10, 50, 80), "working", ts=13.0)], [zone])
    views = clock.snapshot(13.0)
    assert views[0].active and views[0].total_working_seconds == 11.0


def test_sitting_standing_labels_map_to_working_not_working():
    assert WorkingTimeTracker.normalize_activity("sitting") == "working"
    assert WorkingTimeTracker.normalize_activity("standing") == "not_working"

    clock = WorkingTimeTracker()
    zone = WorkstationOccupancy("WS01", (0, 0, 100, 100), state="occupied")
    clock.update(0.0, [make_track(1, (10, 10, 50, 80), "sitting", ts=0.0)], [zone])
    clock.update(4.0, [make_track(1, (10, 10, 50, 80), "sitting", ts=4.0)], [zone])
    views = clock.snapshot(4.0)
    assert views[0].active and views[0].activity_class == "working"
    assert views[0].total_working_seconds == 4.0

    clock.update(6.0, [make_track(1, (10, 10, 50, 80), "standing", ts=6.0)], [zone])
    views = clock.snapshot(6.0)
    assert not views[0].active and views[0].activity_class == "not_working"
    # Open sitting interval started at 0.0 and closes at the standing observation.
    assert views[0].total_working_seconds == 6.0


def test_empty_roi_accumulates_empty_not_working():
    clock = WorkingTimeTracker()
    empty = WorkstationOccupancy("WS01", (0, 0, 100, 100), state="empty")
    t = make_track(2, (10, 10, 50, 80), "working", ts=0.0)
    clock.update(0.0, [t], [empty])
    clock.update(4.0, [make_track(2, (10, 10, 50, 80), "working", ts=4.0)], [empty])
    views = clock.snapshot(4.0)
    assert views[0].total_working_seconds == 0.0 and not views[0].active
    desks = {d.workstation_id: d for d in clock.desk_snapshot(4.0)}
    assert desks["WS01"].empty_seconds == 4.0 and desks["WS01"].working_seconds == 0.0


def test_new_track_id_gets_separate_timer():
    clock = WorkingTimeTracker()
    zone = WorkstationOccupancy("WS01", (0, 0, 100, 100), state="occupied")
    clock.update(0.0, [make_track(1, (10, 10, 40, 70), "working", 0.0)], [zone])
    clock.update(5.0, [make_track(1, (10, 10, 40, 70), "working", 5.0)], [zone])
    clock.update(6.0, [], [zone], removed=[make_track(1, (10, 10, 40, 70), "working", 6.0)])
    clock.update(10.0, [make_track(9, (10, 10, 40, 70), "working", 10.0)], [zone])
    clock.update(15.0, [make_track(9, (10, 10, 40, 70), "working", 15.0)], [zone])
    by_id = {v.local_track_id: v for v in clock.snapshot(15.0)}
    assert by_id[1].total_working_seconds == 6.0 and not by_id[1].active
    assert by_id[9].total_working_seconds == 5.0 and by_id[9].active


def test_suppress_overlapping_working_and_not_working_boxes():
    boxes = [(10, 10, 60, 90), (12, 12, 58, 88), (200, 20, 260, 100)]
    confs = [0.7, 0.9, 0.8]
    class_ids = [0, 1, 0]
    track_ids = [1, 2, 3]
    keep = suppress_overlapping_detections(boxes, confs, class_ids, track_ids, iou_thresh=0.45)
    assert keep == [1, 2]  # highest conf of the overlap pair, plus the distant person
