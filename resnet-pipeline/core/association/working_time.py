"""Working-time accounting for the fine-tuned sitting/standing detector.

Per **person track**: accumulate working time only while YOLO class is ``working``
(sitting) **and** the person overlaps a workstation ROI that is ResNet-``occupied``.

Per **workstation** (top-left overlay): accumulate
* ``empty`` time — desk state is empty
* ``working`` time — desk is occupied **and** a sitting (``working``) person is inside it
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from core.tracking.person_tracker import PersonTrack, TrackLifecycle
from core.scene.scene_mapper import containment

WORKING_LABELS = frozenset({"working", "sitting", "seated"})
NOT_WORKING_LABELS = frozenset({"not_working", "notworking", "standing", "not working"})


@dataclass(frozen=True)
class WorkstationOccupancy:
    workstation_id: str
    context_box: tuple[int, int, int, int]
    state: str  # occupied | empty | unknown | ...


@dataclass
class WorkingPersonView:
    local_track_id: int
    composite_track_id: str
    activity_class: str | None
    workstation_id: str | None
    total_working_seconds: float
    active: bool
    reason: str


@dataclass
class DeskTimeView:
    workstation_id: str
    empty_seconds: float
    working_seconds: float
    current_mode: str  # empty | working | other
    occupied: bool


@dataclass
class _TrackState:
    local_track_id: int
    total_seconds: float = 0.0
    open_start: float | None = None
    last_activity: str | None = None
    last_workstation: str | None = None
    last_reason: str = "waiting"
    seen: bool = False


@dataclass
class _DeskState:
    empty_seconds: float = 0.0
    working_seconds: float = 0.0
    mode: str | None = None  # empty | working | None
    open_start: float | None = None


@dataclass
class WorkingTimeTracker:
    """Accumulate working seconds per track and empty/working seconds per desk."""

    min_containment: float = 0.25
    _tracks: dict[str, _TrackState] = field(default_factory=dict)
    _desks: dict[str, _DeskState] = field(default_factory=dict)

    @staticmethod
    def normalize_activity(label: str | None) -> str | None:
        if label is None:
            return None
        key = str(label).strip().lower().replace("-", "_").replace(" ", "_")
        if key in WORKING_LABELS:
            return "working"
        if key in NOT_WORKING_LABELS:
            return "not_working"
        return key

    def _state(self, track: PersonTrack) -> _TrackState:
        st = self._tracks.get(track.composite_track_id)
        if st is None:
            st = _TrackState(local_track_id=track.local_track_id)
            self._tracks[track.composite_track_id] = st
        return st

    def _pause(self, st: _TrackState, ts: float, reason: str) -> None:
        if st.open_start is not None:
            st.total_seconds += max(0.0, ts - st.open_start)
            st.open_start = None
        st.last_reason = reason

    def _desk(self, workstation_id: str) -> _DeskState:
        st = self._desks.get(workstation_id)
        if st is None:
            st = _DeskState()
            self._desks[workstation_id] = st
        return st

    def _set_desk_mode(self, workstation_id: str, mode: str | None, ts: float) -> None:
        st = self._desk(workstation_id)
        if st.mode == mode:
            return
        if st.open_start is not None and st.mode is not None:
            dt = max(0.0, ts - st.open_start)
            if st.mode == "empty":
                st.empty_seconds += dt
            elif st.mode == "working":
                st.working_seconds += dt
        st.mode = mode
        st.open_start = ts if mode is not None else None

    def _roi_hit(
        self, bbox: tuple[int, int, int, int], zones: Sequence[WorkstationOccupancy]
    ) -> WorkstationOccupancy | None:
        best: WorkstationOccupancy | None = None
        best_score = 0.0
        for zone in zones:
            score = containment(bbox, zone.context_box)
            if score >= self.min_containment and score > best_score:
                best = zone
                best_score = score
        return best

    def update(
        self,
        timestamp_seconds: float,
        tracks: Sequence[PersonTrack],
        zones: Sequence[WorkstationOccupancy],
        removed: Sequence[PersonTrack] = (),
        home_desks: Mapping[str, str] | None = None,
    ) -> list[WorkingPersonView]:
        live_ids = {t.composite_track_id for t in tracks}
        working_desks: set[str] = set()

        for track in tracks:
            st = self._state(track)
            st.seen = True
            st.local_track_id = track.local_track_id
            activity = self.normalize_activity(track.activity_class)
            st.last_activity = activity
            if track.lifecycle == TrackLifecycle.REMOVED or track.is_predicted:
                self._pause(st, timestamp_seconds, "lost_or_predicted")
                st.last_workstation = None
                continue
            zone = self._roi_hit(track.bbox, zones)
            if zone is None:
                home = (home_desks or {}).get(track.composite_track_id)
                if not home:
                    self._pause(st, timestamp_seconds, "outside_roi")
                    st.last_workstation = None
                    continue
                # Already identified at a marked desk: keep their own activity after they leave.
                # The desk overlay clock is unchanged; this only continues the person's record.
                st.last_workstation = home
                if activity != "working":
                    self._pause(
                        st,
                        timestamp_seconds,
                        "standing" if activity == "not_working" else f"activity_{activity or 'unknown'}",
                    )
                    continue
                if st.open_start is None:
                    st.open_start = timestamp_seconds
                st.last_reason = f"working_away_from_{home}"
                continue
            st.last_workstation = zone.workstation_id
            occupied = zone.state == "occupied"
            if not occupied:
                self._pause(st, timestamp_seconds, f"{zone.workstation_id}_not_occupied")
                continue
            if activity != "working":
                self._pause(
                    st,
                    timestamp_seconds,
                    "standing" if activity == "not_working" else f"activity_{activity or 'unknown'}",
                )
                continue
            if st.open_start is None:
                st.open_start = timestamp_seconds
            st.last_reason = f"working_at_{zone.workstation_id}"
            working_desks.add(zone.workstation_id)

        for gone in removed:
            st = self._state(gone)
            self._pause(st, timestamp_seconds, "track_removed")
            st.last_workstation = None

        for cid, st in self._tracks.items():
            if cid not in live_ids and st.open_start is not None:
                self._pause(st, timestamp_seconds, "not_visible")

        for zone in zones:
            if zone.workstation_id in working_desks:
                self._set_desk_mode(zone.workstation_id, "working", timestamp_seconds)
            elif zone.state == "empty":
                self._set_desk_mode(zone.workstation_id, "empty", timestamp_seconds)
            else:
                # occupied-but-standing / unknown — pause both empty and working clocks
                self._set_desk_mode(zone.workstation_id, None, timestamp_seconds)

        return self.snapshot(timestamp_seconds)

    def total(self, composite_track_id: str, timestamp_seconds: float) -> float:
        st = self._tracks.get(composite_track_id)
        if st is None:
            return 0.0
        extra = (timestamp_seconds - st.open_start) if st.open_start is not None else 0.0
        return st.total_seconds + max(0.0, extra)

    def snapshot(self, timestamp_seconds: float) -> list[WorkingPersonView]:
        views: list[WorkingPersonView] = []
        for cid, st in self._tracks.items():
            if not st.seen and st.total_seconds <= 0 and st.open_start is None:
                continue
            views.append(WorkingPersonView(
                local_track_id=st.local_track_id,
                composite_track_id=cid,
                activity_class=st.last_activity,
                workstation_id=st.last_workstation,
                total_working_seconds=self.total(cid, timestamp_seconds),
                active=st.open_start is not None,
                reason=st.last_reason,
            ))
        views.sort(key=lambda v: (-v.active, -v.total_working_seconds, v.local_track_id))
        return views

    def desk_snapshot(self, timestamp_seconds: float) -> list[DeskTimeView]:
        views: list[DeskTimeView] = []
        for ws_id, st in sorted(self._desks.items()):
            empty = st.empty_seconds
            working = st.working_seconds
            if st.open_start is not None and st.mode is not None:
                dt = max(0.0, timestamp_seconds - st.open_start)
                if st.mode == "empty":
                    empty += dt
                elif st.mode == "working":
                    working += dt
            views.append(DeskTimeView(
                workstation_id=ws_id,
                empty_seconds=empty,
                working_seconds=working,
                current_mode=st.mode or "other",
                occupied=st.mode == "working",
            ))
        return views

    def finalize(self, timestamp_seconds: float, reason: str = "end") -> list[WorkingPersonView]:
        for st in self._tracks.values():
            if st.open_start is not None:
                self._pause(st, timestamp_seconds, reason)
        for ws_id in list(self._desks):
            self._set_desk_mode(ws_id, None, timestamp_seconds)
        return self.snapshot(timestamp_seconds)
