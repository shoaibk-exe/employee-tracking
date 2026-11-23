"""Shared YOLO person detector with optional batched multi-camera inference."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from backend.types import Detection, FramePacket
from backend.utils.paths import resolve_path

logger = logging.getLogger(__name__)


@dataclass
class DetectorStats:
    last_inference_ms: float = 0.0
    total_inferences: int = 0


class PersonDetector:
    """Single shared YOLO instance for all cameras."""

    def __init__(
        self,
        model: str = "yolo11n.pt",
        image_size: int = 640,
        confidence: float = 0.35,
        iou: float = 0.50,
        person_class_id: int = 0,
        max_det: int = 100,
        device: str = "auto",
        half: bool = True,
        batch_size: int = 4,
    ) -> None:
        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou
        self.person_class_id = person_class_id
        self.max_det = max_det
        self.device = self._resolve_device(device)
        self.half = bool(half) and self.device.startswith("cuda")
        self.batch_size = max(1, batch_size)
        self.stats = DetectorStats()

        model_path = str(resolve_path(model)) if "/" in model or model.endswith(".pt") else model
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("ultralytics is required for person detection") from exc

        self._model = YOLO(model_path)
        # Warm-up on a tiny blank image to allocate CUDA kernels once.
        blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self._model.predict(
            blank,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            classes=[self.person_class_id],
            max_det=self.max_det,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        logger.info(
            "PersonDetector ready model=%s device=%s half=%s imgsz=%d",
            model_path, self.device, self.half, self.image_size,
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    def detect_frame(self, packet: FramePacket) -> list[Detection]:
        results = self.detect_batch([packet])
        return results.get(packet.camera_id, [])

    def detect_batch(self, packets: Sequence[FramePacket]) -> dict[str, list[Detection]]:
        if not packets:
            return {}
        out: dict[str, list[Detection]] = {p.camera_id: [] for p in packets}
        t0 = time.perf_counter()
        for start in range(0, len(packets), self.batch_size):
            chunk = list(packets[start : start + self.batch_size])
            images = [p.frame for p in chunk]
            preds = self._model.predict(
                images,
                imgsz=self.image_size,
                conf=self.confidence,
                iou=self.iou,
                classes=[self.person_class_id],
                max_det=self.max_det,
                device=self.device,
                half=self.half,
                verbose=False,
            )
            for packet, result in zip(chunk, preds):
                boxes = getattr(result, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    continue
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                for box, conf in zip(xyxy, confs):
                    x1, y1, x2, y2 = map(float, box.tolist())
                    out[packet.camera_id].append(
                        Detection(
                            camera_id=packet.camera_id,
                            frame_id=packet.frame_id,
                            timestamp=packet.timestamp,
                            bbox_xyxy=(x1, y1, x2, y2),
                            confidence=float(conf),
                        )
                    )
        self.stats.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.total_inferences += 1
        return out
