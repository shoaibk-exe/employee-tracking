"""End-to-end pipeline tests: synthetic video + mock classifier, and the real checkpoint."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.occupancy.classifier import Prediction
from core.config import load_settings
from core.occupancy.roi_manager import ROIManager
from main import MonitoringApp

ROOT = Path(__file__).resolve().parents[2]


class BrightnessClassifier:
    """Mock: bright crop -> occupied, dark crop -> empty."""

    class_names = ["empty", "occupied"]

    def classify(self, crops):
        out = {}
        for ws_id, crop in crops.items():
            occupied = float(crop.mean()) > 127
            conf = 0.95
            probs = {"occupied": conf if occupied else 1 - conf, "empty": 1 - conf if occupied else conf}
            out[ws_id] = Prediction("occupied" if occupied else "empty", conf, probs)
        return out


def make_video(path, seconds=20, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for i in range(seconds * fps):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        if i < 10 * fps:
            frame[:, :80] = 255  # W01 occupied for the first 10 s, then empty
        frame[:, 80:] = 255      # W02 occupied throughout
        writer.write(frame)
    writer.release()


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def test_pipeline_with_mock_classifier(tmp_path):
    video = tmp_path / "synthetic.mp4"
    make_video(video)
    rois = tmp_path / "ws.json"
    ROIManager({"W01": [10, 10, 70, 110], "W02": [90, 10, 150, 110]}, (160, 120), rotation=0).to_json(rois)
    out = tmp_path / "out"
    settings = load_settings(ROOT / "config" / "settings.yaml", {
        "camera": {"source": str(video)}, "rois": {"config": str(rois)},
        "visualization": {"show_window": False, "save_video": True},
        "output": {"directory": str(out)}, "logging": {"directory": str(tmp_path / "logs")},
    })
    result = MonitoringApp(settings, classifier=BrightnessClassifier(), session_id="test").run()
    assert result.end_reason == "end_of_video"
    assert result.inference_frames == 20
    assert result.annotated_video.is_file()

    events = [e for e in read_csv(out / "events.csv") if e["previous_state"]]
    w01 = [(e["previous_state"], e["new_state"], float(e["transition_timestamp"])) for e in events
           if e["workstation_id"] == "W01"]
    assert w01 == [("empty", "occupied", 0.0), ("occupied", "empty", 10.0)]

    summary = {r["workstation_id"]: r for r in read_csv(out / "summary.csv")}
    assert float(summary["W01"]["occupied_seconds"]) == pytest.approx(10.0)
    assert float(summary["W01"]["empty_seconds"]) == pytest.approx(9.0)
    assert float(summary["W02"]["occupied_seconds"]) == pytest.approx(19.0)

    intervals = read_csv(out / "intervals.csv")
    for ws in ("W01", "W02"):
        rows = [r for r in intervals if r["workstation_id"] == ws]
        assert sum(float(r["duration_seconds"]) for r in rows) == pytest.approx(19.0)
        assert rows[-1]["end_reason"] == "truncated_end_of_video"
    predictions = read_csv(out / "predictions.csv")
    assert len(predictions) == 40
    assert set(json.loads(predictions[0]["probabilities_json"])) == {"empty", "occupied"}

    second = MonitoringApp(settings, classifier=BrightnessClassifier(), session_id="test2").run(max_frames=5)
    assert second.end_reason == "max_frames"
    sessions = {r["session_id"] for r in read_csv(out / "summary.csv")}
    assert sessions == {"test", "test2"}


REAL_CHECKPOINT = ROOT / "models" / "resnet18_empty_occupied.pth"
REAL_ROIS = ROOT / "config" / "workstations.json"
REAL_VIDEO = ROOT / "testing" / "input.mp4"


@pytest.mark.skipif(not (REAL_CHECKPOINT.is_file() and REAL_ROIS.is_file() and REAL_ROIS.stat().st_size
                         and REAL_VIDEO.is_file()), reason="real checkpoint/ROIs/video not configured")
def test_pipeline_with_real_checkpoint(tmp_path):
    settings = load_settings(ROOT / "config" / "settings.yaml", {
        "camera": {"source": str(REAL_VIDEO)}, "model": {"checkpoint": str(REAL_CHECKPOINT)},
        "rois": {"config": str(REAL_ROIS)}, "visualization": {"show_window": False, "save_video": False},
        "output": {"directory": str(tmp_path)}, "logging": {"directory": str(tmp_path)},
    })
    result = MonitoringApp(settings, session_id="real").run(max_frames=5)
    assert result.inference_frames == 5
    rois = ROIManager.from_json(REAL_ROIS)
    predictions = read_csv(tmp_path / "predictions.csv")
    assert len(predictions) == 5 * len(rois.workstation_ids)
    for row in predictions:
        probs = json.loads(row["probabilities_json"])
        assert set(probs) == {"empty", "occupied"}
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)
