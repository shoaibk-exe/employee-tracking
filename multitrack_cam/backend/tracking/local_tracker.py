"""Per-camera BoT-SORT local tracker.

Local track IDs are camera-session scoped only. Persistent identity is assigned
later by the global association engine.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.types import FramePacket, LocalTrack, TrackLifecycle
from backend.utils.paths import resolve_path

logger = logging.getLogger(__name__)


@dataclass
class _TrackMeta:
    first_seen: float
    last_seen: float
    last_observed: float
    hit_count: int = 0
    age_frames: int = 0


class LocalTracker:
    """Wrap Ultralytics YOLO.track with BoT-SORT for one camera.

    Each camera owns a YOLO wrapper so BoT-SORT persist state never crosses cameras.
    Weights are loaded from the same checkpoint path (Ultralytics caches disk reads).
    """

    def __init__(
        self,
        camera_id: str,
        model_path: str,
        *,
        tracker_config: str = "config/botsort.yaml",
        enable_reid: bool = True,
        confirm_hits: int = 2,
        max_lost_seconds: float = 8.0,
        image_size: int = 640,
        confidence: float = 0.35,
        iou: float = 0.50,
        person_class_id: int = 0,
        max_det: int = 100,
        device: str = "cuda:0",
        half: bool = True,
    ) -> None:
        from ultralytics import YOLO

        self.camera_id = camera_id
        self._model = YOLO(model_path)
        self.tracker_config = str(resolve_path(tracker_config))
        self.enable_reid = enable_reid
        self.confirm_hits = confirm_hits
        self.max_lost_seconds = max_lost_seconds
        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou
        self.person_class_id = person_class_id
        self.max_det = max_det
        self.device = device
        self.half = half

        self.tracker_session_id = uuid.uuid4().hex[:12]
        self._meta: dict[int, _TrackMeta] = {}
        self._active_ids: set[int] = set()

    def reset_session(self) -> None:
        self.tracker_session_id = uuid.uuid4().hex[:12]
        self._meta.clear()
        self._active_ids.clear()
        logger.info("Camera %s tracker session reset -> %s", self.camera_id, self.tracker_session_id)

    def update(self, packet: FramePacket) -> list[LocalTrack]:
        if packet.after_reconnect:
            self.reset_session()

        track_kwargs = dict(
            persist=True,
            tracker=self.tracker_config,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            classes=[self.person_class_id],
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        # Prefer FP16 when supported; Ultralytics 8.4 may warn that half is deprecated.
        if self.half:
            track_kwargs["half"] = True
        results = self._model.track(packet.frame, **track_kwargs)
        result = results[0]
        boxes = getattr(result, "boxes", None)
        observed: dict[int, LocalTrack] = {}

        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            for box, conf, tid in zip(xyxy, confs, ids):
                tid = int(tid)
                meta = self._meta.get(tid)
                if meta is None:
                    meta = _TrackMeta(
                        first_seen=packet.timestamp,
                        last_seen=packet.timestamp,
                        last_observed=packet.timestamp,
                        hit_count=1,
                        age_frames=1,
                    )
                    self._meta[tid] = meta
                else:
                    meta.hit_count += 1
                    meta.age_frames += 1
                    meta.last_seen = packet.timestamp
                    meta.last_observed = packet.timestamp

                lifecycle = (
                    TrackLifecycle.CONFIRMED
                    if meta.hit_count >= self.confirm_hits
                    else TrackLifecycle.TENTATIVE
                )
                x1, y1, x2, y2 = map(float, box.tolist())
                observed[tid] = LocalTrack(
                    camera_id=self.camera_id,
                    tracker_session_id=self.tracker_session_id,
                    local_track_id=tid,
                    frame_id=packet.frame_id,
                    timestamp=packet.timestamp,
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=float(conf),
                    lifecycle=lifecycle,
                    is_predicted=False,
                    hit_count=meta.hit_count,
                    age_frames=meta.age_frames,
                )

        # Mark previously active tracks that disappeared this frame.
        removed: list[LocalTrack] = []
        for tid in list(self._active_ids):
            if tid in observed:
                continue
            meta = self._meta.get(tid)
            if meta is None:
                continue
            gap = packet.timestamp - meta.last_observed
            if gap > self.max_lost_seconds:
                removed.append(
                    LocalTrack(
                        camera_id=self.camera_id,
                        tracker_session_id=self.tracker_session_id,
                        local_track_id=tid,
                        frame_id=packet.frame_id,
                        timestamp=packet.timestamp,
                        bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
                        confidence=0.0,
                        lifecycle=TrackLifecycle.REMOVED,
                        is_predicted=True,
                        hit_count=meta.hit_count,
                        age_frames=meta.age_frames,
                    )
                )
                self._meta.pop(tid, None)
            else:
                # Keep a lost placeholder without inventing a box.
                observed[tid] = LocalTrack(
                    camera_id=self.camera_id,
                    tracker_session_id=self.tracker_session_id,
                    local_track_id=tid,
                    frame_id=packet.frame_id,
                    timestamp=packet.timestamp,
                    bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
                    confidence=0.0,
                    lifecycle=TrackLifecycle.LOST,
                    is_predicted=True,
                    hit_count=meta.hit_count,
                    age_frames=meta.age_frames + 1,
                )
                meta.age_frames += 1

        self._active_ids = {tid for tid, tr in observed.items() if tr.lifecycle != TrackLifecycle.LOST}
        self._active_ids.update(tid for tid, tr in observed.items() if tr.lifecycle == TrackLifecycle.LOST)

        tracks = [tr for tr in observed.values() if tr.lifecycle != TrackLifecycle.LOST or tr.bbox_xyxy != (0.0, 0.0, 0.0, 0.0)]
        # Return confirmed/tentative observed boxes for visualization; include removed for tracklet close.
        visible = [tr for tr in observed.values() if not tr.is_predicted]
        return visible + removed
