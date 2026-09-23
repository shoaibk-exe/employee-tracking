from datetime import datetime, timezone

from app.tracking.tracker import TrackObservation, dedupe_tracks


def _track(track_id: int, bbox: tuple[float, float, float, float], confidence: float = 0.9) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=bbox,
        class_name="person",
        confidence=confidence,
        timestamp=datetime(2026, 9, 23, tzinfo=timezone.utc),
    )


def test_overlapping_boxes_on_one_person_keep_the_older_id():
    tracks = dedupe_tracks(
        [
            _track(4, (100, 100, 180, 320), 0.91),
            _track(19, (108, 110, 176, 310), 0.62),
        ]
    )
    assert [track.track_id for track in tracks] == [4]


def test_people_sitting_apart_keep_both_ids():
    tracks = dedupe_tracks(
        [
            _track(1, (10, 100, 80, 300)),
            _track(2, (200, 100, 280, 300)),
        ]
    )
    assert {track.track_id for track in tracks} == {1, 2}
