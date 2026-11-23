"""Person-to-workstation association with posture inference and temporal confirmation.

Separation of concerns (never merged):

* posture        per track: SEATED / STANDING / MOVING / UNKNOWN_POSTURE
                 (sitting/standing detector class + box geometry + motion)
* association    per track: UNASSOCIATED / CANDIDATE / ASSOCIATED / AMBIGUOUS
* occupancy      per workstation: produced later by the fusion policy, not here
* identity       per track: handled by the identity manager, not here

Per frame, observed tracks with seated-compatible posture are scored against every
workstation (gated: context containment, seat overlap, distance to anchor, head height),
then assigned one-to-one with the Hungarian algorithm; each track may stay unassigned
(dummy column). Near-ties are reported as AMBIGUOUS and do not count as evidence.

Temporal rules:
* CANDIDATE -> ASSOCIATED after ``confirm_seconds`` of consistent assignment; the
  association start is backdated to the first supporting frame.
* ASSOCIATED ends after ``leave_confirm_seconds`` of contradicting evidence (standing,
  moving, out of zone, assigned elsewhere) or ``max_gap_seconds`` without support, or when
  the track is removed / the source drops. The end is backdated to the last support.
* A workstation held by one track cannot be confirmed for another until released.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.tracking.person_tracker import PersonTrack
from core.scene.scene_mapper import Workstation, center, containment, diagonal, iou


class Posture(str, Enum):
    SEATED = "seated"
    STANDING = "standing"
    MOVING = "moving"
    UNKNOWN = "unknown_posture"


class AssociationState(str, Enum):
    UNASSOCIATED = "unassociated"
    CANDIDATE = "candidate"
    ASSOCIATED = "associated"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class AssociationConfig:
    min_score: float = 0.35
    ambiguity_margin: float = 0.08
    max_center_distance: float = 0.75
    min_context_containment: float = 0.50
    min_seat_iou: float = 0.10
    head_margin: float = 0.35
    confirm_seconds: float = 3.0
    candidate_max_gap_seconds: float = 1.0
    leave_confirm_seconds: float = 2.0
    max_gap_seconds: float = 5.0
    occupant_fresh_seconds: float = 2.0
    standing_ratio: float = 1.40
    seated_ratio: float = 1.25
    moving_speed: float = 0.50
    stationary_speed: float = 0.25
    motion_window_seconds: float = 1.0
    posture_confirm_seconds: float = 1.0
    near_containment: float = 0.25

    def validate(self) -> None:
        if not 0 < self.min_score < 1:
            raise ValueError("association.min_score must be in (0, 1)")
        if self.seated_ratio >= self.standing_ratio:
            raise ValueError("association.seated_ratio must be < standing_ratio")
        if self.stationary_speed >= self.moving_speed:
            raise ValueError("association.stationary_speed must be < moving_speed")
        for name in ("confirm_seconds", "candidate_max_gap_seconds", "leave_confirm_seconds", "max_gap_seconds",
                     "motion_window_seconds", "posture_confirm_seconds", "occupant_fresh_seconds"):
            if getattr(self, name) < 0:
                raise ValueError(f"association.{name} must be >= 0")


class SeatedHeightModel:
    """Expected seated-person box height as a linear function of image row (ground-plane perspective)."""

    def __init__(self, workstations: Sequence[Workstation]) -> None:
        points = [(float(w.seat_box[3]), float(w.seated_height_ref)) for w in workstations if w.seated_height_ref]
        self.points = points
        self.coef: tuple[float, float] | None = None
        self._bounds = (0.0, float("inf"))
        if len(points) >= 2 and np.ptp([p[0] for p in points]) > 1.0:
            ys, hs = np.array(points).T
            slope, intercept = np.polyfit(ys, hs, 1)
            self.coef = (float(slope), float(intercept))
            self._bounds = (0.5 * float(hs.min()), 2.0 * float(hs.max()))
        elif len(points) == 1:
            self._bounds = (0.5 * points[0][1], 2.0 * points[0][1])

    @property
    def available(self) -> bool:
        return self.coef is not None or len(self.points) == 1

    def expected(self, y_bottom: float) -> float | None:
        if self.coef is not None:
            value = self.coef[0] * y_bottom + self.coef[1]
        elif len(self.points) == 1:
            value = self.points[0][1]
        else:
            return None
        return float(min(max(value, self._bounds[0]), self._bounds[1]))


@dataclass
class PostureEstimate:
    raw: Posture
    confirmed: Posture
    height_ratio: float | None
    speed: float | None
    reason: str


class _PostureTrack:
    def __init__(self) -> None:
        self.history: deque[tuple[float, tuple[int, int, int, int]]] = deque()
        self.confirmed = Posture.UNKNOWN
        self.pending: Posture | None = None
        self.pending_since: float | None = None


class PostureEstimator:
    def __init__(self, model: SeatedHeightModel, config: AssociationConfig) -> None:
        self.model = model
        self.config = config
        self._tracks: dict[str, _PostureTrack] = {}

    def forget(self, track_id: str) -> None:
        self._tracks.pop(track_id, None)

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, track: PersonTrack, timestamp: float, local_reference: float | None = None) -> PostureEstimate:
        """Decision order: motion > sitting/standing detector class > box geometry.

        ``local_reference``: measured seated height at the seat the person overlaps, if any.
        """
        cfg = self.config
        state = self._tracks.setdefault(track.composite_track_id, _PostureTrack())
        if track.is_predicted:
            return PostureEstimate(Posture.UNKNOWN, state.confirmed, None, None, "predicted_box")
        box = track.bbox
        state.history.append((timestamp, box))
        horizon = max(cfg.motion_window_seconds, cfg.posture_confirm_seconds) * 3
        while state.history and timestamp - state.history[0][0] > horizon:
            state.history.popleft()

        height = float(box[3] - box[1])
        speed = None
        older = [(t, b) for t, b in state.history if timestamp - t >= cfg.motion_window_seconds]
        if older and height > 0:
            t0, b0 = older[-1]
            (cx0, cy0), (cx1, cy1) = center(b0), center(box)
            speed = math.hypot(cx1 - cx0, cy1 - cy0) / height / max(timestamp - t0, 1e-6)
        expected = local_reference or self.model.expected(float(box[3]))
        ratio = height / expected if expected else None

        # Fine-tuned detector labels: sitting -> working/seated, standing -> not_working.
        activity = (track.activity_class or "").strip().lower().replace("-", "_").replace(" ", "_")
        yolo_posture = None
        if activity in {"working", "sitting", "seated"}:
            yolo_posture = Posture.SEATED
        elif activity in {"not_working", "notworking", "standing"}:
            yolo_posture = Posture.STANDING

        if speed is not None and speed > cfg.moving_speed:
            raw, reason = Posture.MOVING, f"speed {speed:.2f} h/s"
        elif yolo_posture is not None:
            raw, reason = yolo_posture, f"yolo_activity {track.activity_class}"
        elif ratio is not None and ratio >= cfg.standing_ratio:
            raw, reason = Posture.STANDING, f"height ratio {ratio:.2f}"
        elif ratio is not None and ratio <= cfg.seated_ratio and speed is not None and speed <= cfg.stationary_speed:
            raw, reason = Posture.SEATED, f"height ratio {ratio:.2f}, speed {speed:.2f}"
        else:
            raw = Posture.UNKNOWN
            reason = "insufficient history" if speed is None else (
                "no seated reference" if ratio is None else f"ratio {ratio:.2f}, speed {speed:.2f}")

        if raw == state.confirmed:
            state.pending = state.pending_since = None
        elif raw != state.pending:
            state.pending, state.pending_since = raw, timestamp
        if state.pending is not None and state.pending_since is not None and \
                timestamp - state.pending_since >= cfg.posture_confirm_seconds:
            state.confirmed = state.pending
            state.pending = state.pending_since = None
        return PostureEstimate(raw, state.confirmed, ratio, speed, reason)


@dataclass(frozen=True)
class AssociationEvent:
    kind: str  # "start" | "end"
    composite_track_id: str
    local_track_id: int
    workstation_id: str
    event_seconds: float
    decided_seconds: float
    reason: str
    score: float


@dataclass
class TrackAssociationView:
    composite_track_id: str
    local_track_id: int
    state: AssociationState
    workstation_id: str | None
    candidate_workstation_id: str | None
    posture: Posture
    raw_posture: Posture
    score: float
    reasons: list[str]
    associated_since: float | None


@dataclass
class AssociationFrame:
    timestamp_seconds: float
    tracks: dict[str, TrackAssociationView]
    occupants: dict[str, str]
    candidates: dict[str, list[str]]
    ambiguous_workstations: set[str]
    standing_near: dict[str, list[str]]
    events: list[AssociationEvent]
    persons_in_zone: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class _TrackRecord:
    local_track_id: int
    state: AssociationState = AssociationState.UNASSOCIATED
    workstation_id: str | None = None
    since: float | None = None
    last_confirmed: float | None = None
    candidate_ws: str | None = None
    candidate_since: float | None = None
    candidate_last_support: float | None = None
    contra_since: float | None = None
    contra_reason: str | None = None
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


class AssociationEngine:
    def __init__(self, workstations: Sequence[Workstation], config: AssociationConfig = AssociationConfig()) -> None:
        config.validate()
        self.config = config
        self.workstations = [w for w in workstations if w.active]
        self.height_model = SeatedHeightModel(self.workstations)
        self.posture = PostureEstimator(self.height_model, config)
        self._records: dict[str, _TrackRecord] = {}
        self._owner: dict[str, str] = {}

    # ------------------------------------------------------------------ scoring

    def pair_score(self, track: PersonTrack, ws: Workstation) -> tuple[float, str | None]:
        """Return (score, None) or (0, gate_failure_reason)."""
        cfg = self.config
        box = track.bbox
        contained = containment(box, ws.context_box)
        seat_iou = iou(box, ws.seat_box)
        if contained < cfg.min_context_containment and seat_iou < cfg.min_seat_iou:
            return 0.0, "outside_zone"
        (px, py), (ax, ay) = center(box), ws.anchor
        distance = math.hypot(px - ax, py - ay) / max(diagonal(ws.seat_box), 1.0)
        if distance > cfg.max_center_distance:
            return 0.0, f"far_from_anchor {distance:.2f}"
        reference = ws.seated_height_ref or self.height_model.expected(float(ws.seat_box[3]))
        if reference and box[1] < ws.seat_box[1] - cfg.head_margin * reference:
            return 0.0, "head_above_seated_level"
        score = 0.5 * min(1.0, seat_iou / 0.6) + 0.3 * (1.0 - distance / cfg.max_center_distance) + 0.2 * contained
        return float(score), None

    # ------------------------------------------------------------------ update

    def update(self, timestamp: float, tracks: Sequence[PersonTrack],
               removed: Iterable[PersonTrack] = ()) -> AssociationFrame:
        cfg = self.config
        events: list[AssociationEvent] = []
        for track in removed:
            record = self._records.pop(track.composite_track_id, None)
            self.posture.forget(track.composite_track_id)
            if record is not None and record.state == AssociationState.ASSOCIATED:
                events.append(self._end(track.composite_track_id, record, timestamp, "track_removed"))

        postures = {t.composite_track_id: self.posture.update(t, timestamp, self._local_reference(t))
                    for t in tracks}
        eligible: list[PersonTrack] = []
        ineligible_reason: dict[str, str] = {}
        for t in tracks:
            p = postures[t.composite_track_id]
            if t.is_predicted:
                ineligible_reason[t.composite_track_id] = "predicted_box"
            elif p.confirmed in (Posture.STANDING, Posture.MOVING):
                ineligible_reason[t.composite_track_id] = f"posture_{p.confirmed.value}"
            elif p.raw == Posture.MOVING:
                ineligible_reason[t.composite_track_id] = "moving_now"
            else:
                eligible.append(t)

        assigned, ambiguous, scores, gate_notes = self._assign(eligible)
        views: dict[str, TrackAssociationView] = {}
        for t in tracks:
            tid = t.composite_track_id
            record = self._records.setdefault(tid, _TrackRecord(t.local_track_id))
            record.reasons = []
            ws_id = assigned.get(tid)
            if tid in ineligible_reason:
                record.reasons.append(ineligible_reason[tid])
            elif tid in gate_notes:
                record.reasons.append(gate_notes[tid])

            # Evidence classes: supporting (assigned), neutral (predicted box, ambiguity, lost a
            # one-to-one competition), contradicting (posture/geometry rules out the seat).
            neutral = t.is_predicted or tid in ambiguous or gate_notes.get(tid) == "lost_assignment"
            if ws_id is not None:
                record.score = scores[(tid, ws_id)]
                if record.state == AssociationState.ASSOCIATED and record.workstation_id == ws_id:
                    record.last_confirmed = timestamp
                    record.contra_since = record.contra_reason = None
                elif record.state == AssociationState.ASSOCIATED:
                    self._contradict(record, timestamp, f"assigned_to_{ws_id}")
                else:
                    self._advance_candidate(tid, record, ws_id, timestamp, events)
            elif neutral:
                if tid in ambiguous:
                    record.reasons.append(ambiguous[tid])
                if record.candidate_last_support is not None and \
                        timestamp - record.candidate_last_support > cfg.candidate_max_gap_seconds:
                    self._reset_candidate(record)
                if record.state not in (AssociationState.ASSOCIATED, AssociationState.CANDIDATE):
                    record.state = AssociationState.AMBIGUOUS if tid in ambiguous else AssociationState.UNASSOCIATED
            else:
                self._reset_candidate(record)
                if record.state == AssociationState.ASSOCIATED:
                    self._contradict(record, timestamp, ineligible_reason.get(tid) or gate_notes.get(tid, "no_match"))
                else:
                    record.state = AssociationState.UNASSOCIATED

            if record.state == AssociationState.ASSOCIATED:
                if record.contra_since is not None and timestamp - record.contra_since >= cfg.leave_confirm_seconds:
                    events.append(self._end(tid, record, timestamp, record.contra_reason or "contradicted"))
                elif record.last_confirmed is not None and timestamp - record.last_confirmed > cfg.max_gap_seconds:
                    events.append(self._end(tid, record, timestamp, "evidence_gap"))

            p = postures[tid]
            views[tid] = TrackAssociationView(
                tid, t.local_track_id, record.state, record.workstation_id, record.candidate_ws,
                p.confirmed, p.raw, record.score, list(record.reasons) + [f"posture: {p.reason}"], record.since,
            )

        occupants = {ws: tid for ws, tid in self._owner.items()
                     if (r := self._records.get(tid)) is not None and r.last_confirmed is not None
                     and timestamp - r.last_confirmed <= cfg.occupant_fresh_seconds}
        candidates: dict[str, list[str]] = {}
        for tid, record in self._records.items():
            if record.state == AssociationState.CANDIDATE and record.candidate_ws:
                candidates.setdefault(record.candidate_ws, []).append(tid)
        ambiguous_ws = {ws for tid in ambiguous for (t2, ws) in scores if t2 == tid and scores[(t2, ws)] > 0}
        standing_near: dict[str, list[str]] = {}
        in_zone: dict[str, list[str]] = {}
        for t in tracks:
            if t.is_predicted:
                continue
            upright = postures[t.composite_track_id].confirmed in (Posture.STANDING, Posture.MOVING)
            for ws in self.workstations:
                if containment(t.bbox, ws.context_box) >= cfg.near_containment:
                    in_zone.setdefault(ws.workstation_id, []).append(t.composite_track_id)
                    if upright:
                        standing_near.setdefault(ws.workstation_id, []).append(t.composite_track_id)
        return AssociationFrame(timestamp, views, occupants, candidates, ambiguous_ws, standing_near, events, in_zone)

    def mark_unavailable(self, timestamp: float, reason: str = "source_unavailable") -> list[AssociationEvent]:
        """End every association at its last support (outage / reconnect / session reset)."""
        events = [self._end(tid, r, timestamp, reason) for tid, r in list(self._records.items())
                  if r.state == AssociationState.ASSOCIATED]
        self._records.clear()
        self._owner.clear()
        self.posture.reset()
        return events

    def finalize(self, timestamp: float, reason: str = "end_of_stream") -> list[AssociationEvent]:
        return self.mark_unavailable(timestamp, reason)

    # ------------------------------------------------------------------ internals

    def _assign(self, eligible: list[PersonTrack]):
        cfg = self.config
        scores: dict[tuple[str, str], float] = {}
        gate_notes: dict[str, str] = {}
        assigned: dict[str, str] = {}
        ambiguous: dict[str, str] = {}
        if not eligible or not self.workstations:
            return assigned, ambiguous, scores, gate_notes
        n, m = len(eligible), len(self.workstations)
        matrix = np.zeros((n, m))
        for i, t in enumerate(eligible):
            failures = []
            for j, ws in enumerate(self.workstations):
                value, failure = self.pair_score(t, ws)
                matrix[i, j] = value
                scores[(t.composite_track_id, ws.workstation_id)] = value
                if failure and failure != "outside_zone":
                    failures.append(f"{ws.workstation_id}:{failure}")
            if not matrix[i].any():
                gate_notes[t.composite_track_id] = failures[0] if failures else "outside_all_zones"
        big = 1e6
        cost = np.full((n, m + n), big)
        cost[:, :m] = np.where(matrix >= cfg.min_score, 1.0 - matrix, big)
        for i in range(n):
            cost[i, m + i] = 1.0 - cfg.min_score
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if j >= m or cost[i, j] >= big:
                continue
            tid, ws_id, value = eligible[i].composite_track_id, self.workstations[j].workstation_id, matrix[i, j]
            rival_ws = np.delete(matrix[i], j)
            rival_tracks = np.delete(matrix[:, j], i)
            if rival_ws.size and rival_ws.max() >= value - cfg.ambiguity_margin and rival_ws.max() >= cfg.min_score:
                other = self.workstations[int(np.argmax(np.where(np.arange(m) == j, -1, matrix[i])))].workstation_id
                ambiguous[tid] = f"ambiguous_between {ws_id}/{other}"
            elif rival_tracks.size and rival_tracks.max() >= value - cfg.ambiguity_margin \
                    and rival_tracks.max() >= cfg.min_score:
                ambiguous[tid] = f"ambiguous_competition_for {ws_id}"
            else:
                assigned[tid] = ws_id
        for i, t in enumerate(eligible):
            tid = t.composite_track_id
            if tid not in assigned and tid not in ambiguous and tid not in gate_notes:
                best = float(matrix[i].max())
                gate_notes[tid] = f"best_score {best:.3f} < {cfg.min_score}" if best < cfg.min_score else "lost_assignment"
        return assigned, ambiguous, scores, gate_notes

    def _local_reference(self, track: PersonTrack) -> float | None:
        """Measured seated height (observed seated person during mapping) of the best-overlapping seat."""
        best, ref = 0.3, None
        for ws in self.workstations:
            if not ws.seated_height_ref or not ws.evidence.get("dwell_persistence"):
                continue
            value = iou(track.bbox, ws.seat_box)
            if value > best:
                best, ref = value, float(ws.seated_height_ref)
        return ref

    @staticmethod
    def _reset_candidate(record: _TrackRecord) -> None:
        record.candidate_ws = record.candidate_since = record.candidate_last_support = None
        if record.state == AssociationState.CANDIDATE:
            record.state = AssociationState.UNASSOCIATED

    def _advance_candidate(self, tid: str, record: _TrackRecord, ws_id: str, timestamp: float,
                           events: list[AssociationEvent]) -> None:
        stale = (record.candidate_last_support is not None
                 and timestamp - record.candidate_last_support > self.config.candidate_max_gap_seconds)
        if record.candidate_ws != ws_id or stale:
            record.candidate_ws, record.candidate_since = ws_id, timestamp
        record.candidate_last_support = timestamp
        record.state = AssociationState.CANDIDATE
        owner = self._owner.get(ws_id)
        if owner is not None and owner != tid:
            record.reasons.append(f"workstation_held_by {owner}")
            return
        if record.candidate_since is not None and timestamp - record.candidate_since >= self.config.confirm_seconds:
            record.state = AssociationState.ASSOCIATED
            record.workstation_id = ws_id
            record.since = record.candidate_since
            record.last_confirmed = timestamp
            record.contra_since = record.contra_reason = None
            record.candidate_ws = record.candidate_since = record.candidate_last_support = None
            self._owner[ws_id] = tid
            events.append(AssociationEvent("start", tid, record.local_track_id, ws_id, record.since, timestamp,
                                           "confirmed_seated", record.score))

    @staticmethod
    def _contradict(record: _TrackRecord, timestamp: float, reason: str) -> None:
        if record.contra_since is None:
            record.contra_since, record.contra_reason = timestamp, reason
        record.reasons.append(f"contradiction: {reason}")

    def _end(self, tid: str, record: _TrackRecord, timestamp: float, reason: str) -> AssociationEvent:
        ws_id = record.workstation_id or ""
        end = record.last_confirmed if record.last_confirmed is not None else timestamp
        event = AssociationEvent("end", tid, record.local_track_id, ws_id, end, timestamp, reason, record.score)
        if self._owner.get(ws_id) == tid:
            del self._owner[ws_id]
        record.state = AssociationState.UNASSOCIATED
        record.workstation_id = record.since = record.last_confirmed = None
        record.contra_since = record.contra_reason = None
        return event
