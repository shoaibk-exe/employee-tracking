"""Per-camera BoT-SORT driven by externally supplied detections.

Unlike ``LocalTracker`` (which runs its own YOLO per camera), this tracker takes
detections from the shared, batched ``PersonDetector`` so one detector model
serves all cameras. Track lifetimes are expressed in *seconds* and converted to
frames from the measured processing rate, because RTSP delivery rate varies a
lot over slow links.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from backend.types import Detection, FramePacket, LocalTrack, TrackLifecycle

logger = logging.getLogger(__name__)


class _DetBatch:
    """Minimal Boxes-like container accepted by Ultralytics trackers."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data.reshape(-1, 6).astype(np.float32)

    @property
    def xyxy(self) -> np.ndarray:
        return self.data[:, :4]

    @property
    def xywh(self) -> np.ndarray:
        d = self.data
        w = d[:, 2] - d[:, 0]
        h = d[:, 3] - d[:, 1]
        return np.stack([d[:, 0] + w / 2.0, d[:, 1] + h / 2.0, w, h], axis=1)

    @property
    def conf(self) -> np.ndarray:
        return self.data[:, 4]

    @property
    def cls(self) -> np.ndarray:
        return self.data[:, 5]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: Any) -> "_DetBatch":
        return _DetBatch(self.data[idx])


@dataclass
class _Meta:
    first_seen: float
    last_seen: float
    hits: int = 0


class BotSortCameraTracker:
    def __init__(
        self,
        camera_id: str,
        *,
        track_high_thresh: float = 0.35,
        track_low_thresh: float = 0.12,
        new_track_thresh: float = 0.40,
        match_thresh: float = 0.85,
        lost_seconds: float = 6.0,
        min_buffer_frames: int = 4,
        fuse_score: bool = True,
        gmc_method: str | None = None,
        confirm_hits: int = 2,
    ) -> None:
        self.camera_id = camera_id
        self.lost_seconds = float(lost_seconds)
        self.min_buffer_frames = int(min_buffer_frames)
        self.confirm_hits = int(confirm_hits)
        self._args = SimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=float(track_high_thresh),
            track_low_thresh=float(track_low_thresh),
            new_track_thresh=float(new_track_thresh),
            track_buffer=max(self.min_buffer_frames, 30),
            match_thresh=float(match_thresh),
            fuse_score=bool(fuse_score),
            gmc_method=gmc_method if gmc_method not in ("", "none", "None") else None,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            with_reid=False,
            model="auto",
        )
        self._tracker = self._new_tracker()
        self.tracker_session_id = uuid.uuid4().hex[:12]
        self._meta: dict[int, _Meta] = {}
        self._last_ts: float | None = None
        self._fps_ema: float | None = None

    @staticmethod
    def _preserve_counter(fn):
        # Ultralytics keeps one class-level track-ID counter shared by every
        # tracker instance; constructing/resetting a tracker zeroes it, which
        # would make other cameras re-issue live local IDs.
        from ultralytics.trackers.basetrack import BaseTrack

        saved = BaseTrack._count
        out = fn()
        BaseTrack._count = max(saved, BaseTrack._count)
        return out

    def _new_tracker(self):
        from ultralytics.trackers.bot_sort import BOTSORT

        return self._preserve_counter(lambda: BOTSORT(self._args))

    def reset_session(self) -> None:
        self._tracker = self._new_tracker()
        self.tracker_session_id = uuid.uuid4().hex[:12]
        self._meta.clear()
        self._last_ts = None
        self._fps_ema = None
        logger.info("Camera %s BoT-SORT session reset -> %s", self.camera_id, self.tracker_session_id)

    @property
    def measured_fps(self) -> float:
        return float(self._fps_ema or 0.0)

    def _update_rate(self, ts: float) -> None:
        if self._last_ts is not None:
            dt = ts - self._last_ts
            if dt > 1e-3:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema is None else 0.8 * self._fps_ema + 0.2 * inst
        self._last_ts = ts
        fps = self._fps_ema or 2.0
        self._tracker.max_frames_lost = max(self.min_buffer_frames, int(round(self.lost_seconds * fps)))

    def update(self, packet: FramePacket, detections: Sequence[Detection]) -> list[LocalTrack]:
        """Return visible tracks for this frame plus tracks removed this frame."""
        if packet.after_reconnect:
            self.reset_session()
        self._update_rate(packet.timestamp)

        if detections:
            arr = np.array(
                [[*d.bbox_xyxy, d.confidence, 0.0] for d in detections],
                dtype=np.float32,
            )
        else:
            arr = np.zeros((0, 6), dtype=np.float32)
        out = self._tracker.update(_DetBatch(arr), packet.frame)

        visible: list[LocalTrack] = []
        seen: set[int] = set()
        for row in np.asarray(out).reshape(-1, 8) if len(out) else []:
            x1, y1, x2, y2, tid, score = (float(v) for v in row[:6])
            tid = int(tid)
            seen.add(tid)
            meta = self._meta.get(tid)
            if meta is None:
                meta = _Meta(first_seen=packet.timestamp, last_seen=packet.timestamp)
                self._meta[tid] = meta
            # Tracks coasting on the Kalman prediction still appear in
            # tracked_stracks for the frame they are matched; only count real hits.
            meta.hits += 1
            meta.last_seen = packet.timestamp
            visible.append(
                LocalTrack(
                    camera_id=self.camera_id,
                    tracker_session_id=self.tracker_session_id,
                    local_track_id=tid,
                    frame_id=packet.frame_id,
                    timestamp=packet.timestamp,
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=score,
                    lifecycle=TrackLifecycle.CONFIRMED if meta.hits >= self.confirm_hits else TrackLifecycle.TENTATIVE,
                    is_predicted=False,
                    hit_count=meta.hits,
                    age_frames=meta.hits,
                )
            )

        alive = {t.track_id for t in self._tracker.tracked_stracks} | {t.track_id for t in self._tracker.lost_stracks}
        removed: list[LocalTrack] = []
        for tid in list(self._meta):
            if tid in seen or tid in alive:
                continue
            meta = self._meta.pop(tid)
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
                    hit_count=meta.hits,
                    age_frames=meta.hits,
                )
            )
        return visible + removed
