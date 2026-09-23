"""Track-specific identity. There is no global 'last recognized person'."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.face.recognizer import FaceObservation
from app.tracking.tracker import TrackObservation
from app.utils.geometry import bbox_center, iou, point_in_bbox


@dataclass
class _Vote:
    employee_id: str
    similarity: float
    timestamp: datetime


@dataclass
class _TrackIdentity:
    votes: list[_Vote] = field(default_factory=list)
    employee_id: str | None = None
    confidence: float = 0.0
    last_seen: datetime | None = None


class TrackManager:
    def __init__(self, min_observations: int = 3, window_seconds: float = 2.0) -> None:
        self.min_observations = min_observations
        self.window_seconds = window_seconds
        self._tracks: dict[int, _TrackIdentity] = {}

    def confirmed_employee(self, track_id: int) -> str | None:
        track = self._tracks.get(track_id)
        if track is None:
            return None
        return track.employee_id

    def identity_confidence(self, track_id: int) -> float:
        track = self._tracks.get(track_id)
        if track is None:
            return 0.0
        return track.confidence

    def observe(
        self,
        track_id: int,
        employee_id: str | None,
        similarity: float,
        timestamp: datetime,
    ) -> str | None:
        """Add one reliable recognition. Unknown and rejected faces do not vote."""
        track = self._tracks.setdefault(track_id, _TrackIdentity())
        track.last_seen = timestamp
        if track.employee_id is not None:
            return track.employee_id
        if not employee_id:
            return None
        track.votes.append(_Vote(employee_id, similarity, timestamp))
        window_start = timestamp - timedelta(seconds=self.window_seconds)
        track.votes = [vote for vote in track.votes if vote.timestamp >= window_start]
        counts: dict[str, list[_Vote]] = {}
        for vote in track.votes:
            counts.setdefault(vote.employee_id, []).append(vote)
        best_id = None
        best_votes: list[_Vote] = []
        for candidate, votes in counts.items():
            if len(votes) > len(best_votes):
                best_id = candidate
                best_votes = votes
        if best_id is not None and len(best_votes) >= self.min_observations:
            # Lock the first identity that clears the bar. A later lookalike should not steal the track.
            track.employee_id = best_id
            track.confidence = sum(vote.similarity for vote in best_votes) / len(best_votes)
        return track.employee_id

    def drop_stale(self, active_ids: set[int], now: datetime, stale_seconds: float = 5.0) -> None:
        stale: list[int] = []
        for track_id, track in self._tracks.items():
            if track_id in active_ids:
                continue
            if track.last_seen is None or (now - track.last_seen).total_seconds() > stale_seconds:
                stale.append(track_id)
        for track_id in stale:
            self._tracks.pop(track_id, None)


def associate_faces(
    tracks: list[TrackObservation],
    faces: list[FaceObservation],
) -> dict[int, FaceObservation]:
    """Each face binds to at most one person box. Two people crossing keep two identities."""
    pairs: list[tuple[float, int, int]] = []
    for face_index, face in enumerate(faces):
        if face.quality_reason != "OK":
            continue
        center = bbox_center(face.bbox)
        for track in tracks:
            score = iou(face.bbox, track.bbox)
            if point_in_bbox(track.bbox, center):
                score += 1.0
            if score > 0:
                pairs.append((score, track.track_id, face_index))
    pairs.sort(reverse=True)
    used_tracks: set[int] = set()
    used_faces: set[int] = set()
    assigned: dict[int, FaceObservation] = {}
    for _score, track_id, face_index in pairs:
        if track_id in used_tracks or face_index in used_faces:
            continue
        used_tracks.add(track_id)
        used_faces.add(face_index)
        assigned[track_id] = faces[face_index]
    return assigned
