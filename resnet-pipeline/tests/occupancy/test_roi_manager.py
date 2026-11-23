import json

import numpy as np
import pytest

from core.occupancy.roi_manager import ROIConfigError, ROIManager
from scripts.scene.select_rois import DisplayMapper, SelectionSession, next_free_id, save_configuration


def make_frame(width=200, height=100):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = np.arange(width, dtype=np.uint8)[None, :]
    frame[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None]
    return frame


def test_correct_cropping_returns_exact_view():
    manager = ROIManager({"W01": [10, 20, 50, 60]}, (200, 100))
    frame = make_frame()
    crop = manager.extract(frame)["W01"]
    assert crop.shape == (40, 40, 3)
    assert crop[0, 0, 0] == 10 and crop[0, 0, 1] == 20
    assert np.shares_memory(crop, frame)


def test_different_crop_sizes():
    manager = ROIManager({"W01": [0, 0, 30, 90], "W02": [100, 10, 190, 40]}, (200, 100))
    crops = manager.extract(make_frame())
    assert crops["W01"].shape[:2] == (90, 30)
    assert crops["W02"].shape[:2] == (30, 90)


@pytest.mark.parametrize("bbox", [[50, 10, 40, 30], [10, 10, 10, 30], [1, 2, 3], ["a", 0, 10, 10], [0.5, 0, 20, 20]])
def test_invalid_coordinates_rejected(bbox):
    with pytest.raises(ROIConfigError):
        ROIManager({"W01": bbox}, (200, 100))


@pytest.mark.parametrize("bbox", [[-5, 0, 40, 40], [180, 50, 210, 90], [10, 80, 50, 101]])
def test_partially_outside_frame_rejected(bbox):
    with pytest.raises(ROIConfigError, match="outside"):
        ROIManager({"W01": bbox}, (200, 100))


def test_tiny_crop_rejected():
    with pytest.raises(ROIConfigError, match="smaller"):
        ROIManager({"W01": [0, 0, 4, 40]}, (200, 100))


def test_no_workstations_rejected():
    with pytest.raises(ROIConfigError):
        ROIManager({}, (200, 100))


def test_same_aspect_resolution_is_scaled():
    manager = ROIManager({"W01": [10, 20, 50, 60]}, (200, 100))
    manager.configure_for_frame(400, 200)
    assert manager.workstations[0].bbox == (20, 40, 100, 120)
    assert manager.extract(make_frame(400, 200))["W01"].shape[:2] == (80, 80)


def test_incompatible_resolution_rejected():
    manager = ROIManager({"W01": [10, 20, 50, 60]}, (200, 100))
    with pytest.raises(ROIConfigError, match="incompatible"):
        manager.configure_for_frame(200, 200)
    with pytest.raises(ROIConfigError):
        manager.configure_for_frame(400, 200, allow_scaling=False)


def test_frame_size_change_mid_stream_rejected():
    manager = ROIManager({"W01": [10, 20, 50, 60]}, (200, 100))
    manager.extract(make_frame())
    with pytest.raises(ROIConfigError, match="changed"):
        manager.extract(make_frame(400, 200))


def test_empty_frame_rejected():
    manager = ROIManager({"W01": [10, 20, 50, 60]}, (200, 100))
    with pytest.raises(ValueError):
        manager.extract(np.zeros((0, 0, 3), dtype=np.uint8))


def test_json_roundtrip_and_rotation_check(tmp_path):
    path = tmp_path / "ws.json"
    ROIManager({"W01": [10, 20, 50, 60], "W02": [60, 0, 90, 30]}, (200, 100), rotation=180).to_json(path)
    loaded = ROIManager.from_json(path)
    assert loaded.reference_resolution == (200, 100)
    assert [ws.bbox for ws in loaded.workstations] == [(10, 20, 50, 60), (60, 0, 90, 30)]
    loaded.check_rotation(180)
    with pytest.raises(ROIConfigError, match="rotation"):
        loaded.check_rotation(0)


def test_example_format_without_rotation_loads(tmp_path):
    path = tmp_path / "ws.json"
    path.write_text(json.dumps({
        "reference_resolution": [1625, 836],
        "workstations": {"W01": {"bbox": [272, 377, 480, 562]}},
    }))
    assert ROIManager.from_json(path).rotation is None


@pytest.mark.parametrize("content", ["", "{not json", json.dumps({"workstations": {}})])
def test_missing_or_malformed_file_rejected(tmp_path, content):
    path = tmp_path / "ws.json"
    path.write_text(content)
    with pytest.raises(ROIConfigError):
        ROIManager.from_json(path)


def test_next_free_id_reuses_gaps():
    assert next_free_id([]) == "W01"
    assert next_free_id(["W01", "W03"]) == "W02"


def test_display_mapper_roundtrip():
    mapper = DisplayMapper.fit(1920, 1080, 1600, 900)
    assert mapper.scale == pytest.approx(1600 / 1920)
    assert mapper.to_frame(*mapper.to_display(960, 540)) == (960, 540)


def test_selection_session_accept_reject_undo_delete():
    session = SelectionSession(200, 100)
    session.propose(50, 60, 10, 20)
    assert session.pending == (10, 20, 50, 60)
    assert session.accept() == "W01"
    session.propose(0, 0, 3, 3)
    assert session.pending is None
    session.propose(100, 10, 150, 50)
    assert session.reject() and session.pending is None
    session.propose(100, 10, 150, 50)
    assert session.accept() == "W02"
    assert session.delete_at(20, 30) == "W01"
    session.propose(10, 20, 60, 70)
    assert session.accept() == "W01"
    assert session.undo() == "W01"
    assert list(session.boxes) == ["W02"]


def test_save_configuration_backs_up_and_writes(tmp_path):
    out = tmp_path / "workstations.json"
    out.write_text('{"old": true}')
    session = SelectionSession(200, 100, boxes={"W01": (10, 20, 50, 60)})
    reference = save_configuration(session, out, 180, "video.mp4", make_frame(), 12.5)
    assert reference.is_file()
    assert list(tmp_path.glob("workstations.json.*.bak"))
    data = json.loads(out.read_text())
    assert data["rotation"] == 180 and data["workstations"]["W01"]["bbox"] == [10, 20, 50, 60]
    assert ROIManager.from_json(out).reference_resolution == (200, 100)
