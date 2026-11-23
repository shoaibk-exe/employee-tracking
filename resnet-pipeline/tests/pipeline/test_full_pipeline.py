"""End-to-end full mode with injected fakes (no GPU models): association, fusion, intervals, storage."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.occupancy.classifier import Prediction
from core.config import load_settings
from core.tracking.person_tracker import PersonTrack, TrackerUpdate, TrackLifecycle
from core.scene.scene_mapper import STATUS_CONFIRMED, SceneMap, Workstation
from main import FullApp
from storage.repository import JsonlRepository

ROOT = Path(__file__).resolve().parents[2]
SIZE = (320, 240)
SEAT1, CTX1 = (40, 60, 120, 200), (30, 50, 130, 210)
SEAT2, CTX2 = (200, 60, 280, 200), (190, 50, 290, 210)


class FakeTracker:
    """One person seated at WS01 for the whole video."""

    tracking_session_id = "sess1"

    def __init__(self):
        self.resets = 0

    def update(self, frame, ts, frame_index):
        track = PersonTrack(1, "demo:CAM01:sess1:1", "demo", "CAM01", "sess1", SEAT1, 0.9,
                            TrackLifecycle.CONFIRMED, False, 0.0, ts, ts, frame_index + 1, frame_index + 1, 0, 0.0)
        return TrackerUpdate(ts, frame_index, [track], [], 1, 1.0, 0.1)

    def reset_session(self):
        self.resets += 1


class FixedClassifier:
    class_names = ["empty", "occupied"]

    def classify(self, crops):
        out = {}
        for ws_id in crops:
            cls = "occupied" if ws_id == "WS01" else "empty"
            out[ws_id] = Prediction(cls, 0.9, {cls: 0.9, ("empty" if cls == "occupied" else "occupied"): 0.1})
        return out


def make_video(path: Path, seconds=8, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, SIZE)
    for _ in range(seconds * fps):
        writer.write(np.full((SIZE[1], SIZE[0], 3), 90, np.uint8))
    writer.release()


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def settings(tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video)
    for name in ("ckpt.pth", "det.pt"):
        (tmp_path / name).write_bytes(b"x")
    smap = SceneMap("demo", "CAM01", 1, SIZE, 0, [
        Workstation("WS01", SEAT1, CTX1, 1.0, STATUS_CONFIRMED, "admin", 140.0, {"dwell_persistence": 1.0}),
        Workstation("WS02", SEAT2, CTX2, 1.0, STATUS_CONFIRMED, "admin", 140.0, {"dwell_persistence": 1.0}),
    ], "2026-01-01T00:00:00+00:00", "test")
    smap.save(tmp_path / "scene_map.json")
    return load_settings(ROOT / "config" / "settings.yaml", {
        "pipeline": {"mode": "full"}, "tenant": {"tenant_id": "demo"},
        "camera": {"source": str(video), "camera_id": "CAM01"},
        "model": {"checkpoint": str(tmp_path / "ckpt.pth")},
        "detection": {"model": str(tmp_path / "det.pt")},
        "scene": {"map": str(tmp_path / "scene_map.json"), "use_admin_rois": False},
        "appearance_gallery": {"enabled": False},
        "activity": {"enabled": False},
        "fusion": {"min_crop_side_px": 32},
        "visualization": {"show_window": False, "save_video": True},
        "output": {"directory": str(tmp_path / "out")}, "logging": {"directory": str(tmp_path / "logs")},
        "storage": {"directory": str(tmp_path / "store")},
    })


def test_full_mode_end_to_end(settings, tmp_path):
    app = FullApp(settings, session_id="s1", classifier=FixedClassifier(), person_tracker=FakeTracker())
    result = app.run()
    out = tmp_path / "out"
    assert result.end_reason == "end_of_video"
    assert result.tracking_frames == 80 and result.inference_frames == 8
    assert result.annotated_video.is_file()

    events = read_csv(out / "association_events_s1.csv")
    assert [(e["kind"], e["workstation_id"], float(e["event_timestamp"])) for e in events] == [
        ("start", "WS01", 0.0), ("end", "WS01", 7.9)]
    seated = read_csv(out / "seated_intervals_s1.csv")
    assert len(seated) == 1 and seated[0]["employee_id"] == "UNKNOWN_EMPLOYEE"
    assert float(seated[0]["duration_seconds"]) == pytest.approx(7.9)

    summary = {r["workstation_id"]: r for r in read_csv(out / "summary.csv")}
    assert float(summary["WS01"]["occupied_seconds"]) == pytest.approx(7.0)
    assert float(summary["WS02"]["empty_seconds"]) == pytest.approx(7.0)
    fusion = read_csv(out / "fusion_s1.csv")
    assert [r["reason"] for r in fusion if r["workstation_id"] == "WS01"] == \
        ["candidate_corroborated"] * 3 + ["person_associated"] * 5
    assert {r["evidence_source"] for r in fusion if r["workstation_id"] == "WS02"} == {"resnet"}
    tracks = read_csv(out / "tracks_s1.csv")
    assert {r["employee_id"] for r in tracks} == {"UNKNOWN_EMPLOYEE"}

    repo = JsonlRepository(tmp_path / "store", "demo")
    assert len(repo.query("seated_intervals")) == 1
    assert len(repo.query("sessions")) == 1
    assert repo.query("employee_intervals") == []

    FullApp(settings, session_id="s1", classifier=FixedClassifier(), person_tracker=FakeTracker()).run()
    repo = JsonlRepository(tmp_path / "store", "demo")
    assert len(repo.query("seated_intervals")) == 1, "replaying the same session must not duplicate history"


def test_full_mode_rejects_scene_map_for_other_camera(settings, tmp_path):
    from core.scene.scene_mapper import SceneMapError

    smap = SceneMap.load(settings.scene_map_path)
    smap.camera_id = "CAM99"
    smap.save(settings.scene_map_path)
    app = FullApp(settings, session_id="s2", classifier=FixedClassifier(), person_tracker=FakeTracker())
    with pytest.raises(SceneMapError, match="camera"):
        app.run()
