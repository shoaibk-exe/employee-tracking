from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from backend.tracking.botsort_tracker import BotSortCameraTracker
from backend.types import Detection, FramePacket, TrackLifecycle


def _packet(cam: str, i: int) -> FramePacket:
    return FramePacket(cam, np.zeros((720, 1280, 3), np.uint8), i, i * 0.5, datetime.now(timezone.utc))


def _det(cam: str, i: int, box, conf=0.8) -> Detection:
    return Detection(cam, i, i * 0.5, tuple(map(float, box)), conf)


def test_ids_stable_and_low_score_people_tracked():
    tr = BotSortCameraTracker("cam_x", new_track_thresh=0.4, track_high_thresh=0.35)
    ids = []
    for i in range(1, 8):
        dets = [_det("cam_x", i, (100 + i, 100, 200 + i, 400), 0.9), _det("cam_x", i, (600, 100, 700, 400), 0.45)]
        out = [t for t in tr.update(_packet("cam_x", i), dets) if t.lifecycle != TrackLifecycle.REMOVED]
        ids.append(sorted(t.local_track_id for t in out))
    assert len(ids[-1]) == 2
    assert ids[-1] == ids[-2] == ids[-3]


def test_new_tracker_does_not_reset_other_cameras_ids():
    a = BotSortCameraTracker("cam_a")
    for i in range(1, 4):
        out_a = a.update(_packet("cam_a", i), [_det("cam_a", i, (100, 100, 200, 400))])
    id_a = out_a[0].local_track_id
    b = BotSortCameraTracker("cam_b")
    out_b = []
    for i in range(1, 4):
        out_b = b.update(_packet("cam_b", i), [_det("cam_b", i, (100, 100, 200, 400))])
    assert out_b and out_b[0].local_track_id != id_a


def test_removed_tracks_reported_after_lost_seconds():
    tr = BotSortCameraTracker("cam_r", lost_seconds=1.0, min_buffer_frames=1)
    for i in range(1, 4):
        tr.update(_packet("cam_r", i), [_det("cam_r", i, (100, 100, 200, 400))])
    removed = []
    for i in range(4, 12):
        removed += [t for t in tr.update(_packet("cam_r", i), []) if t.lifecycle == TrackLifecycle.REMOVED]
    assert len(removed) == 1
