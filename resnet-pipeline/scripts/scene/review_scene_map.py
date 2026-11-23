"""Administrator review of a workstation scene map (required before production use).

    python -m scripts.scene.review_scene_map --map config/scene_map.json

Thick box = seat zone (where a seated occupant appears; used for person association).
Thin box  = context crop (ResNet-18 input). Green = confirmed, orange = needs review,
red = rejected.

Controls
  click inside a box   select workstation           Tab / n   next (needs-review first)
  drag                 redraw the selected box      e         toggle editing seat / context
  a                    accept (confirmed)           r         reject (excluded from runtime)
  w then drag          add a new workstation        u         undo last change
  s                    save as next version         q / Esc   quit (asks if unsaved)
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.scene.scene_mapper import (  # noqa: E402
    SOURCE_ADMIN, STATUS_CONFIRMED, STATUS_NEEDS_REVIEW, STATUS_REJECTED, SceneMap, SceneMapError, Workstation,
    box_area, now_utc,
)
from scripts.scene.select_rois import DisplayMapper, has_display, normalize_box  # noqa: E402

logger = logging.getLogger("review_scene_map")
MIN_SIDE = 16
COLORS = {STATUS_CONFIRMED: (0, 200, 0), STATUS_NEEDS_REVIEW: (0, 165, 255), STATUS_REJECTED: (0, 0, 255)}


@dataclass
class ReviewSession:
    """GUI-independent review state over a copy of the scene map."""

    scene: SceneMap
    selected: str | None = None
    edit_target: str = "seat"
    dirty: bool = False
    _history: list[list[Workstation]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.scene = copy.deepcopy(self.scene)
        order = self.review_order()
        self.selected = order[0] if order else None

    def review_order(self) -> list[str]:
        rank = {STATUS_NEEDS_REVIEW: 0, STATUS_CONFIRMED: 1, STATUS_REJECTED: 2}
        return [w.workstation_id for w in sorted(self.scene.workstations,
                                                 key=lambda w: (rank.get(w.status, 3), w.workstation_id))]

    def _checkpoint(self) -> None:
        self._history.append(copy.deepcopy(self.scene.workstations))
        self.dirty = True

    def _current(self) -> Workstation | None:
        return next((w for w in self.scene.workstations if w.workstation_id == self.selected), None)

    def select_next(self) -> str | None:
        order = self.review_order()
        if not order:
            return None
        index = (order.index(self.selected) + 1) % len(order) if self.selected in order else 0
        self.selected = order[index]
        return self.selected

    def select_at(self, x: int, y: int) -> str | None:
        hits = [w for w in self.scene.workstations
                if w.seat_box[0] <= x <= w.seat_box[2] and w.seat_box[1] <= y <= w.seat_box[3]]
        hits = hits or [w for w in self.scene.workstations
                        if w.context_box[0] <= x <= w.context_box[2] and w.context_box[1] <= y <= w.context_box[3]]
        if hits:
            self.selected = min(hits, key=lambda w: box_area(w.seat_box)).workstation_id
        return self.selected

    def set_status(self, status: str) -> bool:
        ws = self._current()
        if ws is None or ws.status == status:
            return False
        self._checkpoint()
        ws.status = status
        ws.evidence["admin_reviewed_utc"] = now_utc()
        return True

    def set_box(self, box: tuple[int, int, int, int]) -> bool:
        ws = self._current()
        if ws is None or box[2] - box[0] < MIN_SIDE or box[3] - box[1] < MIN_SIDE:
            return False
        self._checkpoint()
        if self.edit_target == "seat":
            ws.seat_box = box
        else:
            ws.context_box = box
        ws.source = SOURCE_ADMIN
        ws.evidence["admin_edited_utc"] = now_utc()
        return True

    def add(self, box: tuple[int, int, int, int]) -> str | None:
        if box[2] - box[0] < MIN_SIDE or box[3] - box[1] < MIN_SIDE:
            return None
        self._checkpoint()
        used = {w.workstation_id for w in self.scene.workstations}
        number = 1
        while f"WS{number:02d}" in used:
            number += 1
        ws_id = f"WS{number:02d}"
        self.scene.workstations.append(Workstation(
            workstation_id=ws_id, seat_box=box, context_box=box, confidence=1.0, status=STATUS_CONFIRMED,
            source=SOURCE_ADMIN, evidence={"admin_added_utc": now_utc()},
        ))
        self.selected = ws_id
        return ws_id

    def undo(self) -> bool:
        if not self._history:
            return False
        self.scene.workstations = self._history.pop()
        self.dirty = bool(self._history)
        return True

    def finalize(self, previous_version: int) -> SceneMap:
        result = copy.deepcopy(self.scene)
        result.version = previous_version + 1
        result.created_utc = now_utc()
        result.notes = [n for n in result.notes if not n.startswith("administrator review required")]
        pending = [w.workstation_id for w in result.workstations if w.status == STATUS_NEEDS_REVIEW]
        result.notes.append(f"admin review saved {result.created_utc}"
                            + (f"; still needs review: {pending}" if pending else ""))
        return result


def render(frame: np.ndarray, session: ReviewSession, mapper: DisplayMapper, drag: tuple | None, adding: bool) -> np.ndarray:
    view = cv2.resize(frame, None, fx=mapper.scale, fy=mapper.scale) if mapper.scale != 1.0 else frame.copy()
    for w in session.scene.workstations:
        color = COLORS.get(w.status, (200, 200, 200))
        selected = w.workstation_id == session.selected
        c1, c2 = mapper.to_display(*w.context_box[:2]), mapper.to_display(*w.context_box[2:])
        s1, s2 = mapper.to_display(*w.seat_box[:2]), mapper.to_display(*w.seat_box[2:])
        cv2.rectangle(view, c1, c2, color, 2 if selected else 1)
        cv2.rectangle(view, s1, s2, (255, 255, 255) if selected else color, 4 if selected else 2)
        cv2.putText(view, w.workstation_id, (s1[0] + 3, s1[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(view, w.workstation_id, (s1[0] + 3, s1[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    if drag is not None:
        cv2.rectangle(view, drag[0], drag[1], (255, 0, 255), 2)
    ws = session._current()
    lines = [
        f"Selected: {ws.workstation_id} [{ws.status}] conf={ws.confidence:.2f} source={ws.source}" if ws else "Selected: -",
        f"Editing: {'NEW workstation' if adding else session.edit_target} box   unsaved={session.dirty}",
    ]
    if ws and ws.evidence.get("review_reason"):
        lines.append(f"Reason: {ws.evidence['review_reason']}")
    lines.append("Tab next | a accept | r reject | drag redraw | e seat/context | w add | u undo | s save | q quit")
    for i, text in enumerate(lines):
        y = 24 + i * 24
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return view


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", type=Path, default=ROOT / "config/scene_map.json")
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--max-height", type=int, default=900)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        scene = SceneMap.load(args.map)
    except SceneMapError as exc:
        logger.error("%s", exc)
        return 2
    reference = args.map.with_name(scene.reference_frame) if scene.reference_frame else None
    frame = cv2.imread(str(reference)) if reference and reference.is_file() else None
    if frame is None:
        logger.error("Reference frame %s not found; re-run discover_scene", reference)
        return 2
    if not has_display():
        logger.error("No display available. Run this on a desktop session (or with X forwarding).")
        return 2

    height, width = frame.shape[:2]
    mapper = DisplayMapper.fit(width, height, args.max_width, args.max_height)
    session = ReviewSession(scene)
    state = {"start": None, "current": None, "adding": False}
    window = "Scene map review"

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"], state["current"] = (x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["start"] is not None:
            state["current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["start"] is not None:
            (xa, ya), (xb, yb) = mapper.to_frame(*state["start"]), mapper.to_frame(x, y)
            state["start"] = state["current"] = None
            if abs(xb - xa) < 4 and abs(yb - ya) < 4:
                session.select_at(xa, ya)
                return
            box = normalize_box(xa, ya, xb, yb, width, height)
            if state["adding"]:
                session.add(box)
                state["adding"] = False
            else:
                session.set_box(box)

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    saved = False
    while True:
        drag = (state["start"], state["current"]) if state["start"] is not None else None
        cv2.imshow(window, render(frame, session, mapper, drag, state["adding"]))
        key = cv2.waitKey(30) & 0xFF
        if key in (9, ord("n")):
            session.select_next()
        elif key == ord("a"):
            session.set_status(STATUS_CONFIRMED)
        elif key == ord("r"):
            session.set_status(STATUS_REJECTED)
        elif key == ord("e"):
            session.edit_target = "context" if session.edit_target == "seat" else "seat"
        elif key == ord("w"):
            state["adding"] = True
        elif key == ord("u"):
            session.undo()
        elif key == ord("s"):
            result = session.finalize(scene.version)
            result.save(args.map)
            scene = result
            session = ReviewSession(result)
            saved = True
            logger.info("Saved scene map v%d to %s", result.version, args.map)
        elif key in (ord("q"), 27):
            if session.dirty:
                logger.warning("Unsaved changes; press q again to discard or s to save")
                session.dirty = False
                continue
            break
    cv2.destroyAllWindows()
    pending = [w.workstation_id for w in scene.workstations if w.status == STATUS_NEEDS_REVIEW]
    if pending:
        logger.warning("Workstations still needing review: %s (they run with reduced trust)", pending)
    return 0 if saved or not session.dirty else 1


if __name__ == "__main__":
    sys.exit(main())
