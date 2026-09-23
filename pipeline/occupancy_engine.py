"""YOLO + ByteTrack chair occupancy and away-from-desk timers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from pipeline.config import settings as default_settings
from pipeline.store import Store

_CHAIR_OCCUPIED = (46, 180, 90)
_CHAIR_EMPTY = (40, 140, 255)
_CHAIR_AWAY = (40, 40, 220)
_PERSON = (210, 180, 70)

_shared_models: dict[str, object] = {}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center(box: np.ndarray) -> tuple[float, float]:
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def _contains(box: np.ndarray, point: tuple[float, float]) -> bool:
    x, y = point
    return float(box[0]) <= x <= float(box[2]) and float(box[1]) <= y <= float(box[3])


def _class_id(names: dict, wanted: str) -> Optional[int]:
    wanted = wanted.lower().strip()
    items = names.items() if isinstance(names, dict) else enumerate(names)
    exact = None
    partial = None
    for idx, name in items:
        label = str(name).lower()
        if label == wanted:
            exact = int(idx)
            break
        if wanted in label or label in wanted:
            partial = int(idx)
    return exact if exact is not None else partial


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _get_model(path: str):
    from ultralytics import YOLO

    if path not in _shared_models:
        _shared_models[path] = YOLO(path)
    return _shared_models[path]


@dataclass
class ChairState:
    track_id: int
    box: np.ndarray
    occupied: bool = False
    away: bool = False
    empty_since: Optional[float] = None
    away_started: Optional[float] = None
    event_id: Optional[int] = None
    last_seen: float = 0.0
    away_seconds_closed: float = 0.0


@dataclass
class ChairSnapshot:
    track_id: int
    occupied: bool
    away: bool
    away_seconds_current: float
    away_seconds_today: float
    label: str


@dataclass
class OccupancyResult:
    frame: np.ndarray
    chairs: list[ChairSnapshot] = field(default_factory=list)
    person_count: int = 0


class OccupancyEngine:
    def __init__(
        self,
        camera_id: str,
        store: Store,
        model_path: Optional[str] = None,
        person_model_path: Optional[str] = None,
    ) -> None:
        self.camera_id = camera_id
        self.store = store
        cfg = default_settings
        self.chair_name = cfg.yolo_chair_class
        self.person_name = cfg.yolo_person_class
        self.conf = cfg.yolo_conf
        self.overlap_iou = cfg.person_chair_iou
        self.away_confirm = cfg.away_confirm_seconds
        self.model = _get_model(model_path or cfg.resolved_yolo_path())
        self.chair_id = _class_id(self.model.names, self.chair_name)
        self.person_id = _class_id(self.model.names, self.person_name)
        self.person_model = None
        self.person_model_person_id: Optional[int] = None
        if self.person_id is None:
            person_path = person_model_path or cfg.resolved_person_model_path()
            self.person_model = _get_model(person_path)
            self.person_model_person_id = _class_id(self.person_model.names, self.person_name)
        if self.chair_id is None:
            # Custom single-class chair model often uses index 0
            self.chair_id = 0
        self.chairs: dict[int, ChairState] = {}
        self.model_path = model_path or cfg.resolved_yolo_path()
        self.using_person_fallback = self.person_model is not None
        self.store.close_open_away_events(self.camera_id)

    def process(self, frame_bgr: np.ndarray) -> OccupancyResult:
        now = time.monotonic()
        try:
            chair_boxes, person_boxes = self._detect(frame_bgr)
        except Exception:
            snapshots = [self._snapshot(state, now) for state in self._live_chairs(now)]
            return OccupancyResult(frame=frame_bgr, chairs=snapshots, person_count=0)

        for track_id, box in chair_boxes:
            occupied = self._chair_occupied(box, person_boxes)
            state = self.chairs.get(track_id)
            if state is None:
                state = ChairState(
                    track_id=track_id,
                    box=box,
                    last_seen=now,
                    away_seconds_closed=self.store.away_seconds_today(
                        self.camera_id, track_id
                    ),
                )
                self.chairs[track_id] = state
            state.box = box
            state.last_seen = now
            state.occupied = occupied
            if occupied:
                state.empty_since = None
                if state.away and state.event_id is not None:
                    closed = self.store.close_away_event(state.event_id)
                    state.away_seconds_closed += closed
                state.away = False
                state.away_started = None
                state.event_id = None
            else:
                if state.empty_since is None:
                    state.empty_since = now
                if (
                    not state.away
                    and (now - state.empty_since) >= self.away_confirm
                ):
                    state.away = True
                    state.away_started = now
                    state.event_id = self.store.open_away_event(
                        self.camera_id, track_id
                    )

        snapshots = [self._snapshot(state, now) for state in self._live_chairs(now)]
        annotated = self._draw(frame_bgr, snapshots, person_boxes)
        return OccupancyResult(
            frame=annotated,
            chairs=snapshots,
            person_count=len(person_boxes),
        )

    def _live_chairs(self, now: float) -> list[ChairState]:
        live = []
        stale = []
        for track_id, state in self.chairs.items():
            if now - state.last_seen > 45.0:
                if state.away and state.event_id is not None:
                    closed = self.store.close_away_event(state.event_id)
                    state.away_seconds_closed += closed
                    state.event_id = None
                    state.away = False
                stale.append(track_id)
            else:
                live.append(state)
        for track_id in stale:
            self.chairs.pop(track_id, None)
        return sorted(live, key=lambda s: s.track_id)

    def _snapshot(self, state: ChairState, now: float) -> ChairSnapshot:
        current = 0.0
        if state.away and state.away_started is not None:
            current = now - state.away_started
        today = state.away_seconds_closed + current
        if state.away:
            label = f"Chair-{state.track_id}  AWAY {_format_duration(current)}"
        elif state.occupied:
            label = f"Chair-{state.track_id}  OCCUPIED"
        else:
            label = f"Chair-{state.track_id}  EMPTY"
        return ChairSnapshot(
            track_id=state.track_id,
            occupied=state.occupied,
            away=state.away,
            away_seconds_current=current,
            away_seconds_today=today,
            label=label,
        )

    def _detect(self, frame_bgr: np.ndarray) -> tuple[list[tuple[int, np.ndarray]], list[np.ndarray]]:
        if self.person_model is None and self.person_id is not None:
            track_classes = [self.chair_id, self.person_id]
        else:
            track_classes = [self.chair_id] if self.chair_id is not None else None
        results = self.model.track(
            frame_bgr,
            persist=True,
            verbose=False,
            conf=self.conf,
            classes=track_classes,
            tracker="bytetrack.yaml",
            imgsz=640,
        )
        chair_boxes: list[tuple[int, np.ndarray]] = []
        person_boxes: list[np.ndarray] = []
        if results:
            result = results[0]
            if result.boxes is not None and len(result.boxes):
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    xyxy = box.xyxy[0].detach().cpu().numpy()
                    if cls_id == self.chair_id:
                        tid = getattr(box, "id", None)
                        if tid is None:
                            cx, cy = _center(xyxy)
                            track_id = int(cx // 40) * 1000 + int(cy // 40)
                        else:
                            try:
                                track_id = int(tid[0])
                            except Exception:
                                cx, cy = _center(xyxy)
                                track_id = int(cx // 40) * 1000 + int(cy // 40)
                        chair_boxes.append((track_id, xyxy))
                    elif self.person_id is not None and cls_id == self.person_id:
                        person_boxes.append(xyxy)

        if self.person_model is not None:
            person_results = self.person_model.predict(
                frame_bgr,
                verbose=False,
                conf=self.conf,
                imgsz=640,
                classes=[self.person_model_person_id]
                if self.person_model_person_id is not None
                else None,
            )
            if person_results and person_results[0].boxes is not None:
                for box in person_results[0].boxes:
                    person_boxes.append(box.xyxy[0].detach().cpu().numpy())

        return chair_boxes, person_boxes

    def _chair_occupied(self, chair: np.ndarray, persons: list[np.ndarray]) -> bool:
        chair_c = _center(chair)
        for person in persons:
            if _iou(chair, person) >= self.overlap_iou:
                return True
            if _contains(chair, _center(person)) or _contains(person, chair_c):
                return True
        return False

    def _draw(
        self,
        frame_bgr: np.ndarray,
        snapshots: list[ChairSnapshot],
        person_boxes: list[np.ndarray],
    ) -> np.ndarray:
        out = frame_bgr.copy()
        by_id = {s.track_id: s for s in snapshots}
        for track_id, state in self.chairs.items():
            snap = by_id.get(track_id)
            if snap is None:
                continue
            if snap.away:
                color = _CHAIR_AWAY
            elif snap.occupied:
                color = _CHAIR_OCCUPIED
            else:
                color = _CHAIR_EMPTY
            x1, y1, x2, y2 = [int(v) for v in state.box]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                snap.label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        for person in person_boxes:
            x1, y1, x2, y2 = [int(v) for v in person]
            cv2.rectangle(out, (x1, y1), (x2, y2), _PERSON, 1)
        return out
