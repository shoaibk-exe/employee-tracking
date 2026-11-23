"""Full-frame person detection (YOLO26m) and multi-object tracking (BoT-SORT).

Track IDs from Ultralytics are **camera-session local** only. Permanent SaaS identity
is never equal to a local track ID. Every track exposes a composite key:

    ``{tenant_id}:{camera_id}:{tracking_session_id}:{local_track_id}``

Lifecycle states map to tracker internals where possible:

* ``TENTATIVE`` — newly created, not yet confirmed by consecutive hits
* ``CONFIRMED`` — active with a fresh detection this frame
* ``LOST`` — still in the tracker's buffer, no fresh detection (predicted box only)
* ``REMOVED`` — left the tracker; returned once for callers to close associations

Predicted/lost boxes are marked ``is_predicted=True`` and must not be treated as
fresh visual evidence for seated-time accounting.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class TrackLifecycle(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    REMOVED = "REMOVED"


class IdentityStatus(str, Enum):
    """Person identity relative to a permanent employee record — not a track ID."""

    UNKNOWN = "unknown"
    TENTATIVE = "tentative"
    VERIFIED = "verified"


class TrackerError(RuntimeError):
    """Detector/tracker configuration or runtime failure."""


@dataclass(frozen=True)
class DetectionConfig:
    model: Path
    image_size: int = 1280
    confidence: float = 0.30
    iou: float = 0.70
    device: str = "auto"
    person_class_name: str = "person"
    # When set, track these classes instead of only ``person_class_name``
    # (e.g. fine-tuned ["sitting", "standing"] mapped later to working / not_working).
    track_class_names: tuple[str, ...] | None = None
    max_det: int = 100

    def resolved_track_classes(self) -> tuple[str, ...]:
        if self.track_class_names:
            return tuple(str(n).strip() for n in self.track_class_names if str(n).strip())
        return (self.person_class_name,)


SUPPORTED_TRACKERS = frozenset({
    "tracktrack", "botsort", "deepocsort", "bytetrack", "ocsort", "fasttrack",
})


@dataclass(frozen=True)
class TrackingConfig:
    algorithm: str = "tracktrack"
    enable_reid: bool = True
    tracker_config: Path = Path("config/tracking/tracktrack.yaml")
    confirm_hits: int = 3
    # Soft limit for how long we keep our own metadata after the last observation.
    # Keep aligned with track_buffer/fps so ReID can re-bind before we drop the track.
    max_lost_seconds: float = 60.0


@dataclass
class PersonTrack:
    """One observed person on one camera for one tracking session."""

    local_track_id: int
    composite_track_id: str
    tenant_id: str
    camera_id: str
    tracking_session_id: str
    bbox: tuple[int, int, int, int]
    confidence: float
    lifecycle: TrackLifecycle
    is_predicted: bool
    first_seen_seconds: float
    last_seen_seconds: float
    last_observed_seconds: float
    hit_count: int
    age_frames: int
    time_since_update_frames: int
    lost_age_seconds: float
    identity_status: IdentityStatus = IdentityStatus.UNKNOWN
    # Detector class label for this observation (e.g. "sitting", "standing", "person").
    activity_class: str | None = None
    # Ultralytics internal ID (often 200+); ``local_track_id`` is the dense display ID (1, 2, 3…).
    raw_tracker_id: int | None = None
    # Filled by later milestones; reserved so callers can extend without breaking types.
    workstation_candidate: str | None = None
    association_confidence: float | None = None
    appearance_ref: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_track_id": self.local_track_id,
            "composite_track_id": self.composite_track_id,
            "tenant_id": self.tenant_id,
            "camera_id": self.camera_id,
            "tracking_session_id": self.tracking_session_id,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "lifecycle": self.lifecycle.value,
            "is_predicted": self.is_predicted,
            "activity_class": self.activity_class,
            "raw_tracker_id": self.raw_tracker_id,
            "first_seen_seconds": self.first_seen_seconds,
            "last_seen_seconds": self.last_seen_seconds,
            "last_observed_seconds": self.last_observed_seconds,
            "hit_count": self.hit_count,
            "age_frames": self.age_frames,
            "time_since_update_frames": self.time_since_update_frames,
            "lost_age_seconds": self.lost_age_seconds,
            "identity_status": self.identity_status.value,
            "workstation_candidate": self.workstation_candidate,
            "association_confidence": self.association_confidence,
        }


@dataclass
class TrackerUpdate:
    timestamp_seconds: float
    frame_index: int
    tracks: list[PersonTrack]
    removed: list[PersonTrack] = field(default_factory=list)
    detection_count: int = 0
    detector_latency_ms: float = 0.0
    tracker_latency_ms: float = 0.0


def resolve_device(device: str) -> str | int:
    if device == "auto":
        try:
            import torch

            return 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if device == "cpu":
        return "cpu"
    if device.startswith("cuda"):
        if device == "cuda":
            return 0
        return int(device.split(":")[-1])
    return device


def make_tracking_session_id() -> str:
    return uuid.uuid4().hex[:12]


def composite_id(tenant_id: str, camera_id: str, tracking_session_id: str, local_track_id: int) -> str:
    return f"{tenant_id}:{camera_id}:{tracking_session_id}:{local_track_id}"


def _clip_bbox(xyxy: Sequence[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1 = int(max(0, min(width - 1, round(float(xyxy[0])))))
    y1 = int(max(0, min(height - 1, round(float(xyxy[1])))))
    x2 = int(max(0, min(width, round(float(xyxy[2])))))
    y2 = int(max(0, min(height, round(float(xyxy[3])))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def suppress_overlapping_detections(
    boxes: Sequence[tuple[int, int, int, int]],
    confs: Sequence[float],
    class_ids: Sequence[int],
    track_ids: Sequence[int],
    iou_thresh: float = 0.45,
    prefer_lower_id: bool = False,
) -> list[int]:
    """Class-agnostic keep-indices: one box per person when working/not_working both fire."""
    def sort_key(i: int) -> tuple:
        # Higher confidence first; optionally break ties toward the older/lower tracker ID.
        return (-float(confs[i]), int(track_ids[i]) if prefer_lower_id else 0)

    order = sorted(range(len(boxes)), key=sort_key)
    keep: list[int] = []
    for i in order:
        if any(_box_iou(boxes[i], boxes[j]) >= iou_thresh for j in keep):
            continue
        keep.append(i)
    return keep


class PersonTracker:
    """Per-camera YOLO26m + BoT-SORT tracker. One instance per camera stream."""

    def __init__(
        self,
        tenant_id: str,
        camera_id: str,
        detection: DetectionConfig,
        tracking: TrackingConfig,
        tracking_session_id: str | None = None,
        model: Any | None = None,
    ) -> None:
        if not tenant_id.strip():
            raise TrackerError("tenant_id is required")
        if not camera_id.strip():
            raise TrackerError("camera_id is required")
        algo = tracking.algorithm.lower().strip()
        if algo not in SUPPORTED_TRACKERS:
            raise TrackerError(
                f"Unsupported tracker algorithm '{tracking.algorithm}'. "
                f"Supported: {sorted(SUPPORTED_TRACKERS)}"
            )
        if not detection.model.is_file():
            raise TrackerError(f"Detection model not found: {detection.model}")
        if not tracking.tracker_config.is_file():
            raise TrackerError(f"Tracker config not found: {tracking.tracker_config}")
        self._ensure_tracker_reid_flag(tracking.tracker_config, tracking.enable_reid)

        self.tenant_id = tenant_id
        self.camera_id = camera_id
        self.detection = detection
        self.tracking = tracking
        self.tracking_session_id = tracking_session_id or make_tracking_session_id()
        self.device = resolve_device(detection.device)

        if model is None:
            from ultralytics import YOLO

            if tracking.enable_reid and self.device != "cpu":
                # Tracker ReID is an ONNX model; without this it silently runs on CPU.
                from core.tracking.reid_encoder import install_shared_tracker_reid, preload_onnxruntime_cuda

                preload_onnxruntime_cuda()
                # Same ONNX session the appearance gallery uses. Track crops stay on this path.
                install_shared_tracker_reid()
            logger.info(
                "Loading detector %s on device=%s imgsz=%d (session %s)",
                detection.model, self.device, detection.image_size, self.tracking_session_id,
            )
            model = YOLO(str(detection.model))
        self._model = model
        self._class_ids, self._id_to_name = self._resolve_track_classes(
            model, detection.resolved_track_classes()
        )
        self._meta: dict[int, dict[str, Any]] = {}
        # Ultralytics IDs grow forever (3, 50, 423…). Map them to dense 1, 2, 3 for the UI.
        self._raw_to_display: dict[int, int] = {}
        self._next_display_id = 1
        self._frame_count = 0
        self._last_timestamp: float | None = None
        # Raw Ultralytics IDs suppressed after failed identity (empty-object timeout).
        # Kept blocked while the detector still returns that raw id; cleared after a gap.
        self._blocked_raw: dict[int, int] = {}  # raw_id → consecutive frames absent
        self._block_clear_frames = 60  # ~3 s @ 20 fps without seeing raw id → unblock
        logger.info(
            "PersonTracker ready: tenant=%s camera=%s track_classes=%s ids=%s reid=%s algorithm=%s config=%s",
            tenant_id, camera_id, detection.resolved_track_classes(), self._class_ids,
            tracking.enable_reid, algo, tracking.tracker_config,
        )

    @staticmethod
    def _ensure_tracker_reid_flag(tracker_config: Path, enable_reid: bool) -> None:
        """Keep tracker YAML ``with_reid`` aligned with settings (Ultralytics reads the YAML)."""
        import re

        text = Path(tracker_config).read_text(encoding="utf-8")
        wanted = f"with_reid: {True if enable_reid else False}"
        if "with_reid:" not in text:
            Path(tracker_config).write_text(text.rstrip() + f"\n{wanted}\n", encoding="utf-8")
            return
        updated = re.sub(r"(?m)^with_reid:\s*\S+", wanted, text)
        if updated != text:
            Path(tracker_config).write_text(updated, encoding="utf-8")
            logger.info("Updated %s -> %s", tracker_config, wanted)

    def _display_id(self, raw_tracker_id: int) -> int:
        """Stable dense ID (1, 2, 3…) for one Ultralytics track within this session."""
        display = self._raw_to_display.get(raw_tracker_id)
        if display is None:
            display = self._next_display_id
            self._raw_to_display[raw_tracker_id] = display
            self._next_display_id += 1
        return display

    @staticmethod
    def _resolve_track_classes(model: Any, wanted: Sequence[str]) -> tuple[list[int], dict[int, str]]:
        names = getattr(model, "names", None)
        if names is None:
            raise TrackerError("Detector does not expose class names; cannot validate track classes")
        items = list(names.items() if isinstance(names, Mapping) else enumerate(names))
        by_lower = {str(name).lower().strip(): (int(index), str(name)) for index, name in items}
        ids: list[int] = []
        id_to_name: dict[int, str] = {}
        missing: list[str] = []
        for label in wanted:
            hit = by_lower.get(label.lower().strip())
            if hit is None:
                missing.append(label)
                continue
            class_id, canonical = hit
            if class_id not in id_to_name:
                ids.append(class_id)
                id_to_name[class_id] = canonical
        if missing or not ids:
            available = dict(names) if isinstance(names, Mapping) else list(names)
            raise TrackerError(
                f"Detector class metadata missing {missing or list(wanted)}. Available: {available}"
            )
        return ids, id_to_name

    def reset_session(self, tracking_session_id: str | None = None) -> None:
        """Start a new tracking session (e.g. after a long outage). Local IDs may reuse."""
        self.tracking_session_id = tracking_session_id or make_tracking_session_id()
        self._meta.clear()
        self._raw_to_display.clear()
        self._next_display_id = 1
        self._frame_count = 0
        self._last_timestamp = None
        self._blocked_raw.clear()
        # Ultralytics keeps tracker state on the model; force a fresh tracker next call.
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            self._model.predictor = None
        logger.info("Tracking session reset: %s", self.tracking_session_id)

    def suppress_tracks(self, composite_track_ids: Iterable[str]) -> list[PersonTrack]:
        """Force-remove tracks (e.g. P=calculating timeout on empty objects).

        The Ultralytics raw id is blocked while the detector keeps returning it, so the
        ghost box does not immediately reappear. After ``_block_clear_frames`` without
        that raw id, the block lifts (a real person can be tracked again).
        """
        removed: list[PersonTrack] = []
        want = {str(c) for c in composite_track_ids}
        if not want:
            return removed
        ts = self._last_timestamp if self._last_timestamp is not None else 0.0
        for local_id in list(self._meta):
            cid = composite_id(self.tenant_id, self.camera_id, self.tracking_session_id, local_id)
            if cid not in want:
                continue
            meta = self._meta[local_id]
            raw = meta.get("raw_tracker_id")
            if raw is not None:
                self._blocked_raw[int(raw)] = 0
            removed.append(self._build(local_id, meta, ts, observed=False, removed=True))
            del self._meta[local_id]
            logger.info("Suppressed track T%s (identity timeout / empty object)", local_id)
        return removed

    def update(self, frame: np.ndarray, timestamp_seconds: float, frame_index: int | None = None) -> TrackerUpdate:
        if frame is None or frame.size == 0 or frame.ndim != 3:
            raise TrackerError("PersonTracker.update requires a non-empty BGR frame")
        if self._last_timestamp is not None and timestamp_seconds < self._last_timestamp:
            raise TrackerError(
                f"Non-monotonic timestamp {timestamp_seconds} < {self._last_timestamp}; "
                "reset the tracking session after seeking or reconnecting"
            )

        height, width = frame.shape[:2]
        self._frame_count += 1
        index = self._frame_count - 1 if frame_index is None else frame_index

        t0 = time.perf_counter()
        from core.gpu_gate import gpu_gate

        with gpu_gate():
            results = self._model.track(
                source=frame,
                persist=True,
                tracker=str(self.tracking.tracker_config),
                conf=self.detection.confidence,
                iou=self.detection.iou,
                imgsz=self.detection.image_size,
                classes=list(self._class_ids),
                # Critical for working/not_working: default NMS is per-class, so one person
                # often gets two overlapping boxes (sitting + standing) and two track IDs.
                agnostic_nms=True,
                device=self.device,
                verbose=False,
                max_det=self.detection.max_det,
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = results[0]
        boxes = result.boxes
        active_ids: set[int] = set()
        tracks: list[PersonTrack] = []
        detection_count = 0

        if boxes is not None and getattr(boxes, "xyxy", None) is not None and len(boxes.xyxy) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
            if getattr(boxes, "cls", None) is not None:
                class_ids = boxes.cls.cpu().numpy().astype(int)
            else:
                class_ids = np.full(len(xyxy), self._class_ids[0], dtype=int)
            if ids is None:
                logger.debug("Tracker returned detections without IDs on frame %d", index)
            else:
                clipped_rows: list[tuple[tuple[int, int, int, int], float, int, int]] = []
                for box, conf, local_id, class_id in zip(xyxy, confs, ids, class_ids):
                    clipped = _clip_bbox(box, width, height)
                    if clipped is None:
                        continue
                    clipped_rows.append((clipped, float(conf), int(local_id), int(class_id)))
                if clipped_rows:
                    # Prefer older/lower Ultralytics IDs when confidences are close — reduces
                    # "new ID wins" flicker when working/not_working both fire.
                    keep = suppress_overlapping_detections(
                        [r[0] for r in clipped_rows],
                        [r[1] for r in clipped_rows],
                        [r[3] for r in clipped_rows],
                        [r[2] for r in clipped_rows],
                        iou_thresh=0.45,
                        prefer_lower_id=True,
                    )
                    detection_count = len(keep)
                    seen_raw: set[int] = set()
                    for idx in keep:
                        clipped, conf, raw_id, class_id = clipped_rows[idx]
                        seen_raw.add(raw_id)
                        if raw_id in self._blocked_raw:
                            # Detector still fires on a suppressed empty object — keep blocked.
                            self._blocked_raw[raw_id] = 0
                            continue
                        display_id = self._display_id(raw_id)
                        activity = self._id_to_name.get(class_id, str(class_id))
                        active_ids.add(display_id)
                        tracks.append(self._upsert(
                            local_id=display_id,
                            bbox=clipped,
                            confidence=conf,
                            timestamp=timestamp_seconds,
                            observed=True,
                            activity_class=activity,
                            raw_tracker_id=raw_id,
                        ))
                    for rid in list(self._blocked_raw):
                        if rid in seen_raw:
                            continue
                        self._blocked_raw[rid] += 1
                        if self._blocked_raw[rid] >= self._block_clear_frames:
                            del self._blocked_raw[rid]

        removed: list[PersonTrack] = []
        # Drop ghost duplicates: a LOST box that still overlaps a live person is a false ID.
        observed = [(t.local_track_id, t.bbox) for t in tracks if not t.is_predicted]
        for local_id in list(self._meta):
            if local_id in active_ids:
                continue
            bbox = self._meta[local_id]["bbox"]
            if any(_box_iou(bbox, ob) >= 0.45 for oid, ob in observed):
                removed.append(self._build(local_id, self._meta[local_id], timestamp_seconds,
                                           observed=False, removed=True))
                del self._meta[local_id]

        for local_id in list(self._meta):
            if local_id in active_ids:
                continue
            meta = self._meta[local_id]
            lost_age = timestamp_seconds - meta["last_observed_seconds"]
            if lost_age > self.tracking.max_lost_seconds:
                removed.append(self._build(local_id, meta, timestamp_seconds, observed=False, removed=True))
                del self._meta[local_id]
            else:
                # Keep last known box as predicted/lost — not fresh visual evidence.
                tracks.append(self._build(local_id, meta, timestamp_seconds, observed=False, removed=False))

        self._last_timestamp = timestamp_seconds
        return TrackerUpdate(
            timestamp_seconds=timestamp_seconds,
            frame_index=index,
            tracks=tracks,
            removed=removed,
            detection_count=detection_count,
            detector_latency_ms=latency_ms,
            tracker_latency_ms=latency_ms,
        )

    def _upsert(
        self,
        local_id: int,
        bbox: tuple[int, int, int, int],
        confidence: float,
        timestamp: float,
        observed: bool,
        activity_class: str | None = None,
        raw_tracker_id: int | None = None,
    ) -> PersonTrack:
        meta = self._meta.get(local_id)
        if meta is None:
            meta = {
                "first_seen_seconds": timestamp,
                "last_seen_seconds": timestamp,
                "last_observed_seconds": timestamp,
                "hit_count": 1,
                "age_frames": 1,
                "time_since_update_frames": 0,
                "bbox": bbox,
                "confidence": confidence,
                "activity_class": activity_class,
                "raw_tracker_id": raw_tracker_id,
            }
            self._meta[local_id] = meta
        else:
            meta["last_seen_seconds"] = timestamp
            meta["age_frames"] += 1
            if observed:
                meta["last_observed_seconds"] = timestamp
                meta["hit_count"] += 1
                meta["time_since_update_frames"] = 0
                meta["bbox"] = bbox
                meta["confidence"] = confidence
                if activity_class is not None:
                    meta["activity_class"] = activity_class
                if raw_tracker_id is not None:
                    meta["raw_tracker_id"] = raw_tracker_id
            else:
                meta["time_since_update_frames"] += 1
        return self._build(local_id, meta, timestamp, observed=observed, removed=False)

    def _build(
        self,
        local_id: int,
        meta: Mapping[str, Any],
        timestamp: float,
        observed: bool,
        removed: bool,
    ) -> PersonTrack:
        hits = int(meta["hit_count"])
        if removed:
            lifecycle = TrackLifecycle.REMOVED
        elif not observed:
            lifecycle = TrackLifecycle.LOST
        elif hits < self.tracking.confirm_hits:
            lifecycle = TrackLifecycle.TENTATIVE
        else:
            lifecycle = TrackLifecycle.CONFIRMED
        return PersonTrack(
            local_track_id=local_id,
            composite_track_id=composite_id(
                self.tenant_id, self.camera_id, self.tracking_session_id, local_id
            ),
            tenant_id=self.tenant_id,
            camera_id=self.camera_id,
            tracking_session_id=self.tracking_session_id,
            bbox=tuple(meta["bbox"]),  # type: ignore[arg-type]
            confidence=float(meta["confidence"]),
            lifecycle=lifecycle,
            is_predicted=not observed,
            first_seen_seconds=float(meta["first_seen_seconds"]),
            last_seen_seconds=float(meta["last_seen_seconds"]),
            last_observed_seconds=float(meta["last_observed_seconds"]),
            hit_count=hits,
            age_frames=int(meta["age_frames"]),
            time_since_update_frames=int(meta["time_since_update_frames"]),
            lost_age_seconds=max(0.0, timestamp - float(meta["last_observed_seconds"])),
            identity_status=IdentityStatus.UNKNOWN,
            activity_class=meta.get("activity_class"),
            raw_tracker_id=None if meta.get("raw_tracker_id") is None else int(meta["raw_tracker_id"]),
        )
