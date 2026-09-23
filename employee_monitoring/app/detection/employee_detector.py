"""Person-state detector. The model reports sitting or standing. It does not decide time."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    class_name: str
    confidence: float


class EmployeeDetector:
    def __init__(self, model_path: str, image_size: int = 640, confidence: float = 0.4) -> None:
        self.model_path = model_path
        self.image_size = image_size
        self.confidence = confidence
        self._model = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        model = self._load()
        results = model.predict(
            frame,
            verbose=False,
            conf=self.confidence,
            imgsz=self.image_size,
        )
        return _detections_from_results(results)

    def _load(self):
        if self._model is None:
            path = Path(self.model_path)
            if not path.is_file() and self.model_path != "yolov8n.pt":
                raise FileNotFoundError(
                    f"Detector weights not found: {self.model_path}. "
                    "Train the sitting/standing model before expecting desk state."
                )
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model


def _detections_from_results(results: object) -> list[Detection]:
    detections: list[Detection] = []
    if not results:
        return detections
    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections
    names = getattr(result, "names", {}) or {}
    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = str(names.get(cls_id, cls_id)).lower()
        xyxy = box.xyxy[0].detach().cpu().numpy()
        detections.append(
            Detection(
                bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                class_name=class_name,
                confidence=float(box.conf[0]),
            )
        )
    return detections
