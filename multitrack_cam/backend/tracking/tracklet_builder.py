"""Accumulate per-track observations into tracklets for global association."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

import numpy as np

from backend.types import LocalTrack, TrackLifecycle, Tracklet, TrackletObservation

logger = logging.getLogger(__name__)


@dataclass
class _OpenTracklet:
    tracklet: Tracklet
    last_sample_time: float = -1e9


class TrackletBuilder:
    def __init__(
        self,
        *,
        min_tracklet_seconds: float = 1.0,
        min_tracklet_samples: int = 3,
        sample_interval_seconds: float = 0.5,
    ) -> None:
        self.min_tracklet_seconds = min_tracklet_seconds
        self.min_tracklet_samples = min_tracklet_samples
        self.sample_interval_seconds = sample_interval_seconds
        self._open: dict[str, _OpenTracklet] = {}

    def update(
        self,
        tracks: list[LocalTrack],
        embeddings: dict[str, tuple[np.ndarray, float]] | None = None,
    ) -> list[Tracklet]:
        """Update open tracklets. Return finalized tracklets ready for association."""
        embeddings = embeddings or {}
        finalized: list[Tracklet] = []
        seen_keys: set[str] = set()

        for track in tracks:
            key = track.compound_key
            if track.lifecycle == TrackLifecycle.REMOVED:
                closed = self._close(key)
                if closed is not None:
                    finalized.append(closed)
                continue
            if track.is_predicted:
                continue

            seen_keys.add(key)
            open_item = self._open.get(key)
            if open_item is None:
                tracklet = Tracklet(
                    tracklet_id=f"tl_{uuid.uuid4().hex[:10]}",
                    camera_id=track.camera_id,
                    tracker_session_id=track.tracker_session_id,
                    local_track_id=track.local_track_id,
                    start_time=track.timestamp,
                    end_time=track.timestamp,
                )
                open_item = _OpenTracklet(tracklet=tracklet)
                self._open[key] = open_item

            tracklet = open_item.tracklet
            tracklet.end_time = track.timestamp
            emb_pack = embeddings.get(key)
            if emb_pack is not None and (track.timestamp - open_item.last_sample_time) >= self.sample_interval_seconds:
                emb, quality = emb_pack
                tracklet.observations.append(
                    TrackletObservation(
                        timestamp=track.timestamp,
                        frame_id=track.frame_id,
                        bbox_xyxy=track.bbox_xyxy,
                        confidence=track.confidence,
                        embedding=emb,
                        quality_score=quality,
                    )
                )
                open_item.last_sample_time = track.timestamp
            elif not tracklet.observations or tracklet.observations[-1].frame_id != track.frame_id:
                # Keep geometry even without embedding for duration accounting.
                if (track.timestamp - open_item.last_sample_time) >= self.sample_interval_seconds:
                    tracklet.observations.append(
                        TrackletObservation(
                            timestamp=track.timestamp,
                            frame_id=track.frame_id,
                            bbox_xyxy=track.bbox_xyxy,
                            confidence=track.confidence,
                        )
                    )
                    open_item.last_sample_time = track.timestamp

        return finalized

    def force_close_all(self) -> list[Tracklet]:
        keys = list(self._open.keys())
        out: list[Tracklet] = []
        for key in keys:
            closed = self._close(key)
            if closed is not None:
                out.append(closed)
        return out

    def snapshot_ready(self) -> list[Tracklet]:
        """Return open tracklets that already meet quality thresholds (for live association)."""
        ready: list[Tracklet] = []
        for item in self._open.values():
            tl = item.tracklet
            if self._is_ready(tl):
                ready.append(self._with_aggregated_embedding(tl))
        return ready

    def _close(self, key: str) -> Tracklet | None:
        item = self._open.pop(key, None)
        if item is None:
            return None
        tl = item.tracklet
        if not self._is_ready(tl):
            logger.debug("Dropping short tracklet %s duration=%.2f samples=%d",
                         tl.tracklet_id, tl.duration_seconds, len(tl.observations))
            return None
        return self._with_aggregated_embedding(tl)

    def _is_ready(self, tl: Tracklet) -> bool:
        emb_count = sum(1 for o in tl.observations if o.embedding is not None)
        return (
            tl.duration_seconds >= self.min_tracklet_seconds
            and emb_count >= self.min_tracklet_samples
        )

    @staticmethod
    def _with_aggregated_embedding(tl: Tracklet) -> Tracklet:
        vectors = [o.embedding for o in tl.observations if o.embedding is not None]
        if not vectors:
            return tl
        weights = np.array(
            [max(0.05, o.quality_score) for o in tl.observations if o.embedding is not None],
            dtype=np.float32,
        )
        mat = np.stack(vectors, axis=0).astype(np.float32)
        weights = weights / weights.sum()
        agg = (mat * weights[:, None]).sum(axis=0)
        norm = float(np.linalg.norm(agg)) + 1e-12
        tl.embedding = agg / norm
        return tl
