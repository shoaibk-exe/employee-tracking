"""VideoMAE desk-activity classifier for workstation ROI clips.

Trained labels (folder names under ``training/videomae/trims``):
  computer_interaction, neither, phone_using

YOLO still owns person posture (``sitting`` / ``standing``, mapped to
``working`` / ``not_working``). This module only classifies the desk ROI over a
short temporal window.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch

from core.gpu_gate import gpu_gate
from core.scalability import CoalescingWorker
from core.tracking.person_tracker import resolve_device

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path("models/videomae-large-activity")


class ActivityError(RuntimeError):
    """VideoMAE checkpoint or inference failure."""


@dataclass(frozen=True)
class ActivityConfig:
    enabled: bool = True
    checkpoint: Path = DEFAULT_CHECKPOINT
    device: str = "auto"
    num_frames: int = 16
    frame_stride: int = 2
    interval_seconds: float = 1.0
    min_side_px: int = 64
    buffer_size: int = 48
    store_size: int = 224


@dataclass(frozen=True)
class ActivityPrediction:
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "confidence": round(float(self.confidence), 4),
            "probabilities": {key: round(float(value), 4) for key, value in self.probabilities.items()},
        }


def clip_span(num_frames: int, frame_stride: int) -> int:
    return (num_frames - 1) * frame_stride + 1


def sample_indices(length: int, num_frames: int, frame_stride: int) -> list[int]:
    """Evenly cover the buffer when it is shorter than a Kinetics-style clip."""
    needed = clip_span(num_frames, frame_stride)
    if length <= 1:
        return [0] * num_frames
    if length < needed:
        return np.linspace(0, length - 1, num_frames).astype(int).tolist()
    start = length - needed
    return [start + i * frame_stride for i in range(num_frames)]


def _square_resize(frame: np.ndarray, size: int) -> np.ndarray:
    """Center-crop to 1:1 then resize — matches the square ROI used in training."""
    height, width = frame.shape[:2]
    side = min(height, width)
    if side < 1:
        return np.zeros((size, size, 3), dtype=np.uint8)
    top = (height - side) // 2
    left = (width - side) // 2
    crop = frame[top : top + side, left : left + side]
    if crop.shape[0] == size and crop.shape[1] == size:
        return np.ascontiguousarray(crop)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


class VideoMAEActivityClassifier:
    """Loads the fine-tuned VideoMAE-Large checkpoint and scores RGB clip tensors."""

    def __init__(self, config: ActivityConfig) -> None:
        self.config = config
        checkpoint = Path(config.checkpoint)
        if not checkpoint.is_dir():
            raise ActivityError(f"VideoMAE checkpoint directory not found: {checkpoint}")
        if not (checkpoint / "config.json").is_file():
            raise ActivityError(f"VideoMAE checkpoint is incomplete: {checkpoint}")
        try:
            from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
        except ImportError as exc:
            raise ActivityError("transformers is required for VideoMAE activity classification") from exc

        self.device = torch.device(resolve_device(config.device))
        logger.info("Loading VideoMAE activity model from %s on %s", checkpoint, self.device)
        self.processor = VideoMAEImageProcessor.from_pretrained(checkpoint)
        self.model = VideoMAEForVideoClassification.from_pretrained(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.class_names = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        logger.info("VideoMAE classes: %s", self.class_names)

    @torch.inference_mode()
    def predict(self, frames: Sequence[np.ndarray]) -> ActivityPrediction:
        if len(frames) != self.config.num_frames:
            raise ActivityError(f"Expected {self.config.num_frames} frames, got {len(frames)}")
        rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else frame for frame in frames]
        packed = self.processor(rgb, return_tensors="pt")
        pixels = packed["pixel_values"]
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(0)
        pixels = pixels.to(self.device, non_blocking=True)
        with gpu_gate():
            try:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ):
                    logits = self.model(pixel_values=pixels).logits.float()
                probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
                index = int(logits[0].argmax().item())
            finally:
                # Release peak activations so ReID / YOLO in other cams can allocate.
                if self.device.type == "cuda":
                    del pixels
                    torch.cuda.empty_cache()
        return ActivityPrediction(
            predicted_class=self.class_names[index],
            confidence=float(probs[index]),
            probabilities={name: float(probs[i]) for i, name in enumerate(self.class_names)},
        )


class WorkstationActivityMonitor:
    """Per-desk frame buffer + throttled VideoMAE inference.

    When ``async_infer`` is True, classify jobs go to a coalescing worker
    (1 running + 1 pending). Overlay / fusion keep using the last cached labels;
    YOLO / ResNet paths are unaffected.
    """

    def __init__(
        self,
        classifier: VideoMAEActivityClassifier,
        workstation_ids: Sequence[str],
        config: ActivityConfig | None = None,
        *,
        async_infer: bool = False,
        phase_offset_seconds: float = 0.0,
    ) -> None:
        self.classifier = classifier
        self.config = config or classifier.config
        self._buffers: dict[str, deque[np.ndarray]] = {
            ws_id: deque(maxlen=self.config.buffer_size) for ws_id in workstation_ids
        }
        self._latest: dict[str, ActivityPrediction] = {}
        self._last_run: dict[str, float] = {}
        self._needed = clip_span(self.config.num_frames, self.config.frame_stride)
        self._lock = threading.Lock()
        self._phase = max(0.0, float(phase_offset_seconds))
        self._async = bool(async_infer)
        self._worker: CoalescingWorker[tuple[float, dict[str, list[np.ndarray]]]] | None = None
        if self._async:
            self._worker = CoalescingWorker(
                "videomae-coalesce",
                self._run_async_batch,
            )

    def latest(self, workstation_id: str) -> ActivityPrediction | None:
        with self._lock:
            return self._latest.get(workstation_id)

    def snapshot(self) -> dict[str, ActivityPrediction]:
        with self._lock:
            return dict(self._latest)

    def needs_heartbeat(self, timestamp: float, max_age_seconds: float = 8.0) -> bool:
        """True when any desk label is missing or older than ``max_age_seconds``."""
        age = max(1.0, float(max_age_seconds))
        with self._lock:
            if not self._buffers:
                return False
            for ws_id in self._buffers:
                last = self._last_run.get(ws_id)
                if last is None or (timestamp - last) >= age:
                    return True
        return False

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def observe(
        self,
        timestamp: float,
        frame: np.ndarray,
        boxes: Mapping[str, tuple[int, int, int, int]],
        *,
        run_inference: bool,
    ) -> dict[str, ActivityPrediction]:
        """Append ROI crops every call; classify when enough history and the interval elapsed."""
        height, width = frame.shape[:2]
        for ws_id, box in boxes.items():
            buffer = self._buffers.get(ws_id)
            if buffer is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 - x1 < self.config.min_side_px or y2 - y1 < self.config.min_side_px:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            buffer.append(_square_resize(crop, self.config.store_size))

        if not run_inference:
            return self.snapshot()
        if self._phase > 0 and timestamp + 1e-6 < self._phase and not self._last_run:
            return self.snapshot()

        # Async reserves the interval on submit (drop-stale). Sync only advances after success.
        jobs = self._collect_due_clips(timestamp, reserve=self._worker is not None)
        if not jobs:
            return self.snapshot()

        if self._worker is not None:
            # Newest clip set wins; never queue a backlog of VideoMAE work.
            self._worker.submit((timestamp, jobs))
            return self.snapshot()

        updated: dict[str, ActivityPrediction] = {}
        for ws_id, frames in jobs.items():
            try:
                prediction = self.classifier.predict(frames)
            except Exception:
                logger.exception("VideoMAE failed for %s", ws_id)
                continue
            with self._lock:
                self._latest[ws_id] = prediction
                self._last_run[ws_id] = timestamp
            updated[ws_id] = prediction
        return updated

    def _collect_due_clips(
        self, timestamp: float, *, reserve: bool,
    ) -> dict[str, list[np.ndarray]]:
        jobs: dict[str, list[np.ndarray]] = {}
        for ws_id, buffer in self._buffers.items():
            if len(buffer) < max(8, self.config.num_frames):
                continue
            last = self._last_run.get(ws_id)
            if last is not None and timestamp - last < self.config.interval_seconds:
                continue
            indices = sample_indices(len(buffer), self.config.num_frames, self.config.frame_stride)
            # Copy frames so the async worker is not racing the deque overwrite.
            jobs[ws_id] = [np.ascontiguousarray(buffer[i]) for i in indices]
            if reserve:
                self._last_run[ws_id] = timestamp
        return jobs

    def _run_async_batch(self, job: tuple[float, dict[str, list[np.ndarray]]]) -> None:
        timestamp, clips = job
        for ws_id, frames in clips.items():
            try:
                prediction = self.classifier.predict(frames)
            except Exception:
                logger.exception("VideoMAE async failed for %s", ws_id)
                continue
            with self._lock:
                self._latest[ws_id] = prediction
                self._last_run[ws_id] = timestamp
