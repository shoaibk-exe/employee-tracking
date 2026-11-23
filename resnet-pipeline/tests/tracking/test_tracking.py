"""Unit tests for PersonTracker (mocked detector) and composite track identity rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from core.tracking.person_tracker import (
    DetectionConfig,
    IdentityStatus,
    PersonTracker,
    TrackLifecycle,
    TrackerError,
    TrackingConfig,
    composite_id,
)


@dataclass
class FakeBoxes:
    xyxy: torch.Tensor
    conf: torch.Tensor
    id: torch.Tensor | None
    cls: torch.Tensor | None = None


@dataclass
class FakeResult:
    boxes: FakeBoxes | None


class FakeYOLO:
    """Deterministic stand-in for ultralytics.YOLO.track()."""

    names = {0: "person", 1: "chair"}

    def __init__(self, script: list[list[tuple[int, list[float], float]]]):
        # script[frame_i] = [(local_id, xyxy, conf), ...]
        self.script = script
        self.calls = 0
        self.predictor = object()

    def track(self, **kwargs):
        assert kwargs["persist"] is True
        assert 0 in kwargs["classes"]
        assert kwargs.get("agnostic_nms") is True
        frame_i = min(self.calls, len(self.script) - 1)
        rows = self.script[frame_i]
        self.calls += 1
        if not rows:
            return [FakeResult(None)]
        xyxy = torch.tensor([r[1] for r in rows], dtype=torch.float32)
        conf = torch.tensor([r[2] for r in rows], dtype=torch.float32)
        ids = torch.tensor([r[0] for r in rows], dtype=torch.float32)
        cls = torch.zeros(len(rows), dtype=torch.float32)
        return [FakeResult(FakeBoxes(xyxy, conf, ids, cls))]


@pytest.fixture
def configs(tmp_path: Path):
    weights = tmp_path / "detector.pt"
    weights.write_bytes(b"placeholder")
    tracker = tmp_path / "botsort.yaml"
    tracker.write_text("tracker_type: botsort\nwith_reid: False\n")
    detection = DetectionConfig(model=weights, image_size=640, confidence=0.3, device="cpu")
    tracking = TrackingConfig(
        algorithm="botsort", enable_reid=False, tracker_config=tracker,
        confirm_hits=2, max_lost_seconds=2.0,
    )
    return detection, tracking


def test_composite_id_scopes_tenant_camera_session():
    a = composite_id("acme", "CAM01", "sessA", 1)
    b = composite_id("acme", "CAM02", "sessA", 1)
    c = composite_id("beta", "CAM01", "sessA", 1)
    assert a != b and a != c
    assert a == "acme:CAM01:sessA:1"


def test_person_class_must_exist(configs, tmp_path):
    detection, tracking = configs

    class Bad:
        names = {0: "car"}

        def track(self, **kwargs):
            return []

    with pytest.raises(TrackerError, match="person"):
        PersonTracker("t1", "CAM01", detection, tracking, model=Bad())


def test_reid_flag_written_into_botsort_yaml(configs):
    detection, tracking = configs
    with_reid = TrackingConfig(
        tracker_config=tracking.tracker_config, enable_reid=True, confirm_hits=2, max_lost_seconds=2.0
    )
    PersonTracker("t1", "CAM01", detection, with_reid, model=FakeYOLO([]))
    text = tracking.tracker_config.read_text()
    assert "with_reid: True" in text


def test_lifecycle_tentative_confirmed_lost_removed(configs):
    detection, tracking = configs
    model = FakeYOLO([
        [(7, [10, 10, 50, 80], 0.9)],
        [(7, [12, 10, 52, 80], 0.91)],
        [],  # lost but within max_lost_seconds
        [],
        [],  # beyond max_lost -> REMOVED
    ])
    tracker = PersonTracker("acme", "CAM01", detection, tracking, tracking_session_id="sess1", model=model)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    u0 = tracker.update(frame, 0.0, 0)
    assert len(u0.tracks) == 1 and u0.tracks[0].lifecycle == TrackLifecycle.TENTATIVE
    # Dense display ID 1 (Ultralytics raw id 7 is kept separately).
    assert u0.tracks[0].local_track_id == 1 and u0.tracks[0].raw_tracker_id == 7
    assert u0.tracks[0].composite_track_id == "acme:CAM01:sess1:1"
    assert u0.tracks[0].identity_status == IdentityStatus.UNKNOWN

    u1 = tracker.update(frame, 0.05, 1)
    assert u1.tracks[0].lifecycle == TrackLifecycle.CONFIRMED
    assert not u1.tracks[0].is_predicted

    u2 = tracker.update(frame, 0.10, 2)
    assert u2.tracks[0].lifecycle == TrackLifecycle.LOST
    assert u2.tracks[0].is_predicted

    u3 = tracker.update(frame, 2.5, 3)
    # still within 2.0s from last observe at 0.05? 2.5-0.05=2.45 > 2.0 -> removed
    assert u3.removed and u3.removed[0].lifecycle == TrackLifecycle.REMOVED
    assert u3.removed[0].local_track_id == 1 and u3.removed[0].raw_tracker_id == 7


def test_non_monotonic_timestamp_rejected(configs):
    detection, tracking = configs
    model = FakeYOLO([[(1, [1, 1, 20, 40], 0.8)], [(1, [1, 1, 20, 40], 0.8)]])
    tracker = PersonTracker("t", "C", detection, tracking, model=model)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    tracker.update(frame, 1.0, 0)
    with pytest.raises(TrackerError, match="Non-monotonic"):
        tracker.update(frame, 0.5, 1)


def test_reset_session_changes_composite_scope(configs):
    detection, tracking = configs
    model = FakeYOLO([[(3, [5, 5, 30, 60], 0.9)], [(3, [5, 5, 30, 60], 0.9)]])
    tracker = PersonTracker("t", "C", detection, tracking, tracking_session_id="aaa", model=model)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    first = tracker.update(frame, 0.0, 0).tracks[0].composite_track_id
    tracker.reset_session("bbb")
    second = tracker.update(frame, 0.1, 1).tracks[0].composite_track_id
    assert first.endswith(":aaa:1") and second.endswith(":bbb:1")
    assert first != second


def test_local_track_id_is_not_employee_identity(configs):
    detection, tracking = configs
    model = FakeYOLO([[(42, [0, 0, 40, 40], 0.99)]])
    tracker = PersonTracker("tenant", "CAM01", detection, tracking, model=model)
    track = tracker.update(np.zeros((60, 60, 3), dtype=np.uint8), 0.0).tracks[0]
    assert track.local_track_id == 1 and track.raw_tracker_id == 42
    assert track.identity_status == IdentityStatus.UNKNOWN
    assert track.composite_track_id.endswith(":1")
    # Documented invariant for SaaS: never treat local id as employee id.
    assert track.identity_status.value != str(track.local_track_id)
