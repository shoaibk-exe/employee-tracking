"""Tracker boundary.

Business logic consumes TrackObservation only. The camera path uses BoT-SORT with ReID
so a person who passes through someone else keeps the same track id. Swap the yaml to
change the tracker without changing attendance or desk sessions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.detection.employee_detector import Detection
from app.utils.geometry import iou

logger = logging.getLogger(__name__)


def _resolve_tracker_config(tracker_config: str) -> str:
    """Ultralytics only finds a custom tracker file from an absolute path."""
    if tracker_config in {"bytetrack.yaml", "botsort.yaml"}:
        return tracker_config
    path = Path(tracker_config)
    if not path.is_absolute():
        root = Path(__file__).resolve().parents[2]
        path = root / tracker_config
    return str(path)


@dataclass
class TrackObservation:
    track_id: int
    bbox: tuple[float, float, float, float]
    class_name: str
    confidence: float
    timestamp: datetime


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: list[Detection], timestamp: datetime) -> list[TrackObservation]:
        """Associate already-computed detections. Track ids are temporary."""


class IoUTracker(Tracker):
    """Greedy overlap tracker used when detections are produced outside Ultralytics."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 1
        self._tracks: dict[int, tuple[tuple[float, float, float, float], int]] = {}

    def update(self, detections: list[Detection], timestamp: datetime) -> list[TrackObservation]:
        unmatched = set(self._tracks)
        assigned: list[TrackObservation] = []
        for detection in detections:
            best_id = None
            best_iou = self.iou_threshold
            for track_id in list(unmatched):
                box, _age = self._tracks[track_id]
                score = iou(box, detection.bbox)
                if score >= best_iou:
                    best_iou = score
                    best_id = track_id
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            else:
                unmatched.discard(best_id)
            self._tracks[best_id] = (detection.bbox, 0)
            assigned.append(
                TrackObservation(
                    track_id=best_id,
                    bbox=detection.bbox,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    timestamp=timestamp,
                )
            )
        expired: list[int] = []
        for track_id in unmatched:
            box, age = self._tracks[track_id]
            age += 1
            if age > self.max_age:
                expired.append(track_id)
            else:
                self._tracks[track_id] = (box, age)
        for track_id in expired:
            self._tracks.pop(track_id, None)
        return assigned


class UltralyticsByteTracker:
    """YOLO track() with BoT-SORT ReID by default.

    Appearance features keep a track id when two people pass through each other.
    That id is still temporary. It is never stored as an employee id.
    """

    def __init__(
        self,
        model_path: str,
        image_size: int = 640,
        confidence: float = 0.4,
        class_filter: set[str] | None = None,
        tracker_config: str = "configs/trackers/botsort_reid.yaml",
    ) -> None:
        self.model_path = model_path
        self.image_size = image_size
        self.confidence = confidence
        self.class_filter = {name.lower() for name in class_filter} if class_filter else None
        self.tracker_config = _resolve_tracker_config(tracker_config)
        self._model = None
        self._missing_logged = False

    def track(self, frame: object, timestamp: datetime) -> list[TrackObservation]:
        model = self._load()
        results = model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.confidence,
            iou=0.45,
            imgsz=self.image_size,
            tracker=self.tracker_config,
        )
        # A second box on the same body must not become a second person id.
        return dedupe_tracks(self._parse(results, timestamp))

    def update(self, detections: list[Detection], timestamp: datetime) -> list[TrackObservation]:
        # Kept so a future external ByteTrack can replace Ultralytics without a new call site.
        return IoUTracker().update(detections, timestamp)

    def _load(self):
        if self._model is None:
            path = Path(self.model_path)
            if self.model_path != "yolov8n.pt" and not path.is_file():
                if not self._missing_logged:
                    logger.error(
                        "tracker weights missing",
                        extra={"event": "MODEL_MISSING", "model": self.model_path},
                    )
                    self._missing_logged = True
                raise FileNotFoundError(self.model_path)
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model

    def _parse(self, results: object, timestamp: datetime) -> list[TrackObservation]:
        tracks: list[TrackObservation] = []
        if not results:
            return tracks
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return tracks
        names = getattr(result, "names", {}) or {}
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = str(names.get(cls_id, cls_id)).lower()
            if self.class_filter is not None and class_name not in self.class_filter:
                continue
            track_attr = getattr(box, "id", None)
            if track_attr is None:
                # A detection without an id is not a track. Dropping it avoids inventing a stable person.
                continue
            xyxy = box.xyxy[0].detach().cpu().numpy()
            tracks.append(
                TrackObservation(
                    track_id=int(track_attr[0]),
                    bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    class_name=class_name,
                    confidence=float(box.conf[0]),
                    timestamp=timestamp,
                )
            )
        return tracks


def dedupe_tracks(tracks: list[TrackObservation]) -> list[TrackObservation]:
    """Drop a newer box that sits on a person who already has an id.

    The detector often emits two boxes for one body. BoT-SORT then assigns two ids.
    The older id is kept. Neighboring people do not overlap this much, so they stay separate.
    """
    ordered = sorted(tracks, key=lambda track: (track.track_id, -track.confidence))
    kept: list[TrackObservation] = []
    for track in ordered:
        if any(_same_body(track.bbox, previous.bbox) for previous in kept):
            continue
        kept.append(track)
    return kept


def _same_body(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    if iou(a, b) >= 0.5:
        return True
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    smaller = min(area_a, area_b)
    return smaller > 0 and inter / smaller >= 0.7
