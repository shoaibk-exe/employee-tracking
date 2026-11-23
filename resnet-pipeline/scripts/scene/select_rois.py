"""Interactive workstation ROI calibration.

Draw one rectangle per workstation on a full CCTV frame with the mouse, inspect each
crop, and save the boxes (in original frame pixels) to ``config/workstations.json``.

    python -m scripts.scene.select_rois --source data/input/office.mp4 --rotation 180

Controls (also shown in the window, toggle with H):
    Left-drag          draw a new box (shown yellow, pending)
    Enter / Space / A  accept the pending box (gets the next free ID: W01, W02, ...)
    X / Backspace      reject the pending box
    Right-click a box  delete that box so it can be redrawn
    U                  undo the last accepted box
    N                  inspect the next box's crop in the "crop" window
    [ / ]              step backward / forward through the video to check boxes on other frames
    O                  rotate the frame 90 degrees (only before any box is drawn)
    S                  save and exit
    Q / Esc            cancel without saving (press twice if there are unsaved boxes)
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from core.occupancy.roi_manager import ALLOWED_ROTATIONS, MIN_ROI_SIDE_PX, ROIConfigError, ROIManager

logger = logging.getLogger("select_rois")

WINDOW = "ROI calibration"
CROP_WINDOW = "crop"
ROTATE_CODES = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
TRAIN_RESIZE, TRAIN_CROP = 256, 224

HELP_LINES = [
    "Left-drag: draw box | Enter/Space/A: accept | X/Backspace: reject",
    "Right-click box: delete | U: undo | N: inspect crop | [ ]: change frame",
    "O: rotate (no boxes yet) | S: save & exit | Q/Esc: cancel | H: hide help",
]


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    return frame if rotation == 0 else cv2.rotate(frame, ROTATE_CODES[rotation])


def next_free_id(existing: list[str], prefix: str = "W") -> str:
    """Smallest unused ``W01``-style ID, so deleted IDs are reused by the redrawn box."""
    used = set(existing)
    number = 1
    while f"{prefix}{number:02d}" in used:
        number += 1
    return f"{prefix}{number:02d}"


def normalize_box(x_a: int, y_a: int, x_b: int, y_b: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Order drag corners and clamp them to the frame (the user is dragging on the frame itself)."""
    x1, x2 = sorted((max(0, min(width, x_a)), max(0, min(width, x_b))))
    y1, y2 = sorted((max(0, min(height, y_a)), max(0, min(height, y_b))))
    return x1, y1, x2, y2


def model_view_box(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Approximate region of the crop the model sees after Resize(256) + CenterCrop(224)."""
    x1, y1, x2, y2 = bbox
    side = min(x2 - x1, y2 - y1) * TRAIN_CROP / TRAIN_RESIZE
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)


@dataclass
class DisplayMapper:
    """Maps between original-frame pixels and the (possibly downscaled) display window."""

    scale: float

    @classmethod
    def fit(cls, width: int, height: int, max_width: int, max_height: int) -> "DisplayMapper":
        return cls(min(1.0, max_width / width, max_height / height))

    def to_frame(self, x: int, y: int) -> tuple[int, int]:
        return int(round(x / self.scale)), int(round(y / self.scale))

    def to_display(self, x: int, y: int) -> tuple[int, int]:
        return int(round(x * self.scale)), int(round(y * self.scale))


@dataclass
class SelectionSession:
    """GUI-independent calibration state."""

    width: int
    height: int
    boxes: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    pending: tuple[int, int, int, int] | None = None
    dirty: bool = False
    history: list[str] = field(default_factory=list)

    def propose(self, x_a: int, y_a: int, x_b: int, y_b: int) -> str:
        box = normalize_box(x_a, y_a, x_b, y_b, self.width, self.height)
        if box[2] - box[0] < MIN_ROI_SIDE_PX or box[3] - box[1] < MIN_ROI_SIDE_PX:
            self.pending = None
            return f"Box too small (min {MIN_ROI_SIDE_PX}px per side); draw again"
        self.pending = box
        return f"Pending box {box}: Enter/A to accept, X to reject"

    def accept(self) -> str | None:
        if self.pending is None:
            return None
        workstation_id = next_free_id(list(self.boxes))
        self.boxes[workstation_id] = self.pending
        self.boxes = dict(sorted(self.boxes.items()))
        self.history.append(workstation_id)
        self.pending = None
        self.dirty = True
        return workstation_id

    def reject(self) -> bool:
        had = self.pending is not None
        self.pending = None
        return had

    def undo(self) -> str | None:
        """Remove the most recently accepted box (or, for loaded boxes, the highest ID)."""
        while self.history and self.history[-1] not in self.boxes:
            self.history.pop()
        if self.history:
            last = self.history.pop()
        elif self.boxes:
            last = max(self.boxes)
        else:
            return None
        del self.boxes[last]
        self.dirty = True
        return last

    def delete_at(self, x: int, y: int) -> str | None:
        """Delete the smallest box containing the point (smallest wins when boxes overlap)."""
        hits = [
            (ws_id, (b[2] - b[0]) * (b[3] - b[1]))
            for ws_id, b in self.boxes.items()
            if b[0] <= x < b[2] and b[1] <= y < b[3]
        ]
        if not hits:
            return None
        ws_id = min(hits, key=lambda item: item[1])[0]
        del self.boxes[ws_id]
        self.dirty = True
        return ws_id


class FrameSource:
    """Grabs representative frames from a video file, RTSP URL, or camera index."""

    def __init__(self, source: str, rotation: int) -> None:
        self.source = source
        self.rotation = rotation
        capture_arg: str | int = int(source) if source.isdigit() else source
        self.is_file = isinstance(capture_arg, str) and "://" not in source
        if self.is_file and not Path(source).is_file():
            raise FileNotFoundError(f"Video file not found: {source}")
        self.capture = cv2.VideoCapture(capture_arg)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        frames = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration = frames / fps if self.is_file and fps > 0 and frames > 0 else None
        self.time_seconds = 0.0
        self.raw: np.ndarray | None = None

    def read(self, time_seconds: float | None = None) -> np.ndarray:
        if self.is_file and time_seconds is not None:
            if self.duration is not None:
                time_seconds = min(max(0.0, time_seconds), max(0.0, self.duration - 0.5))
            self.capture.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)
        ok, frame = self.capture.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError(f"Could not read a frame from {self.source} at t={time_seconds}")
        if self.is_file:
            self.time_seconds = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self.raw = frame
        return self.frame

    @property
    def frame(self) -> np.ndarray:
        assert self.raw is not None
        return rotate_frame(self.raw, self.rotation)

    def release(self) -> None:
        self.capture.release()


def draw_overlay(
    frame: np.ndarray, session: SelectionSession, mapper: DisplayMapper, selected: str | None,
    status: str, show_help: bool, drag: tuple[int, int, int, int] | None,
) -> np.ndarray:
    canvas = frame if mapper.scale == 1.0 else cv2.resize(
        frame, None, fx=mapper.scale, fy=mapper.scale, interpolation=cv2.INTER_AREA
    )
    canvas = canvas.copy()
    thickness = 2

    def rect(box: tuple[int, int, int, int], color: tuple[int, int, int], label: str | None) -> None:
        p1, p2 = mapper.to_display(box[0], box[1]), mapper.to_display(box[2], box[3])
        cv2.rectangle(canvas, p1, p2, color, thickness)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(canvas, (p1[0], p1[1] - th - 8), (p1[0] + tw + 6, p1[1]), color, -1)
            cv2.putText(canvas, label, (p1[0] + 3, p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    for ws_id, box in session.boxes.items():
        rect(box, (255, 255, 0) if ws_id == selected else (0, 220, 0), ws_id)
    if session.pending is not None:
        rect(session.pending, (0, 220, 255), "pending")
    if drag is not None:
        x_a, y_a, x_b, y_b = drag
        cv2.rectangle(canvas, mapper.to_display(x_a, y_a), mapper.to_display(x_b, y_b), (0, 220, 255), 1)

    lines = (HELP_LINES if show_help else []) + [status]
    y = 22
    for line in lines:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 22
    return canvas


def render_crop(frame: np.ndarray, box: tuple[int, int, int, int], label: str) -> np.ndarray:
    """Crop preview, enlarged for visibility, with the model's center-crop region outlined."""
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    zoom = max(1.0, min(3.0, 360 / max(crop.shape[:2])))
    view = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST).copy()
    mx1, my1, mx2, my2 = model_view_box(box)
    cv2.rectangle(
        view, (int((mx1 - x1) * zoom), int((my1 - y1) * zoom)), (int((mx2 - x1) * zoom), int((my2 - y1) * zoom)),
        (255, 0, 255), 1,
    )
    banner = np.zeros((44, max(view.shape[1], 320), 3), dtype=np.uint8)
    cv2.putText(banner, f"{label}  {x2 - x1}x{y2 - y1}px", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(banner, "magenta = model center-crop", (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
    if banner.shape[1] > view.shape[1]:
        view = cv2.copyMakeBorder(view, 0, 0, 0, banner.shape[1] - view.shape[1], cv2.BORDER_CONSTANT)
    return np.vstack([banner, view])


def save_configuration(
    session: SelectionSession, output: Path, rotation: int, source: str, frame: np.ndarray, frame_time: float | None
) -> Path:
    """Back up any existing ROI file, then write ROIs and the reference frame."""
    if output.is_file() and output.stat().st_size > 0:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = output.with_name(f"{output.name}.{stamp}.bak")
        shutil.copy2(output, backup)
        logger.info("Backed up previous ROI file to %s", backup)
    reference_image = output.with_name(f"{output.stem}_reference.jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(reference_image), frame):
        raise RuntimeError(f"Failed to write reference frame {reference_image}")
    manager = ROIManager(session.boxes, (session.width, session.height), rotation)
    manager.to_json(
        output,
        extra={
            "source": source,
            "frame_time_seconds": frame_time,
            "reference_frame": reference_image.name,
            "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    return reference_image


def has_display() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw workstation ROIs on a CCTV frame")
    parser.add_argument("--source", required=True, help="MP4 path, RTSP URL, or camera index")
    parser.add_argument("--output", type=Path, default=Path("config/workstations.json"))
    parser.add_argument("--rotation", type=int, default=0, choices=ALLOWED_ROTATIONS,
                        help="Clockwise frame rotation applied before drawing (use 180 for an upside-down camera)")
    parser.add_argument("--frame-time", type=float, default=0.0, help="Seconds into a video file to start on")
    parser.add_argument("--step", type=float, default=10.0, help="Seconds moved by [ and ]")
    parser.add_argument("--edit", action="store_true", help="Load the existing --output file and edit its boxes")
    parser.add_argument("--max-display-width", type=int, default=1600)
    parser.add_argument("--max-display-height", type=int, default=900)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if not has_display():
        logger.error("No display found (DISPLAY/WAYLAND_DISPLAY unset). Run this on the desktop "
                     "session or over 'ssh -X'.")
        return 2

    source = FrameSource(args.source, args.rotation)
    try:
        frame = source.read(args.frame_time)
    except Exception:
        source.release()
        raise
    height, width = frame.shape[:2]
    session = SelectionSession(width, height)

    if args.edit:
        existing = ROIManager.from_json(args.output)
        if existing.reference_resolution != (width, height):
            raise ROIConfigError(
                f"Existing ROIs are for {existing.reference_resolution}, frame is {width}x{height}"
            )
        existing.check_rotation(args.rotation)
        session.boxes = {ws.workstation_id: ws.bbox for ws in existing.workstations}

    mapper = DisplayMapper.fit(width, height, args.max_display_width, args.max_display_height)
    state: dict = {"drag_start": None, "drag": None, "status": "Draw a box around a workstation", "selected": None}

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        fx, fy = mapper.to_frame(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag_start"] = (fx, fy)
            state["drag"] = (fx, fy, fx, fy)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag_start"] is not None:
            state["drag"] = (*state["drag_start"], fx, fy)
        elif event == cv2.EVENT_LBUTTONUP and state["drag_start"] is not None:
            state["status"] = session.propose(*state["drag_start"], fx, fy)
            state["drag_start"] = state["drag"] = None
            if session.pending is not None:
                cv2.imshow(CROP_WINDOW, render_crop(source.frame, session.pending, "pending"))
        elif event == cv2.EVENT_RBUTTONDOWN:
            removed = session.delete_at(fx, fy)
            state["status"] = f"Deleted {removed}; draw it again" if removed else "No box under cursor"

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    show_help, cancel_armed, inspect_index, saved = True, False, -1, False
    try:
        while True:
            frame = source.frame
            cv2.imshow(WINDOW, draw_overlay(frame, session, mapper, state["selected"], state["status"],
                                            show_help, state["drag"]))
            key = cv2.waitKey(30) & 0xFF
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                logger.warning("Window closed; calibration canceled, nothing saved")
                return 1
            if key == 255:
                continue
            if key not in (ord("q"), 27):
                cancel_armed = False
            if key in (13, 10, ord(" "), ord("a")):
                ws_id = session.accept()
                state["status"] = f"Accepted {ws_id}" if ws_id else "Nothing pending; drag to draw a box"
                if ws_id:
                    state["selected"] = ws_id
                    cv2.imshow(CROP_WINDOW, render_crop(frame, session.boxes[ws_id], ws_id))
            elif key in (ord("x"), 8):
                state["status"] = "Rejected pending box" if session.reject() else "Nothing pending"
            elif key == ord("u"):
                removed = session.undo()
                state["status"] = f"Removed {removed}" if removed else "No boxes to undo"
            elif key == ord("n"):
                if session.boxes:
                    ids = list(session.boxes)
                    inspect_index = (inspect_index + 1) % len(ids)
                    state["selected"] = ids[inspect_index]
                    cv2.imshow(CROP_WINDOW, render_crop(frame, session.boxes[ids[inspect_index]], ids[inspect_index]))
                    state["status"] = f"Inspecting {ids[inspect_index]}"
                else:
                    state["status"] = "No boxes to inspect"
            elif key in (ord("["), ord("]")):
                target = source.time_seconds + (args.step if key == ord("]") else -args.step)
                try:
                    source.read(target if source.is_file else None)
                    state["status"] = f"Frame at {source.time_seconds:.1f}s" if source.is_file else "New live frame"
                except RuntimeError as exc:
                    state["status"] = str(exc)
            elif key == ord("o"):
                if session.boxes or session.pending:
                    state["status"] = "Rotation is locked once boxes exist (U to remove them)"
                else:
                    source.rotation = (source.rotation + 90) % 360
                    session.height, session.width = source.frame.shape[:2]
                    mapper = DisplayMapper.fit(session.width, session.height,
                                               args.max_display_width, args.max_display_height)
                    state["status"] = f"Rotation {source.rotation} deg"
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("s"):
                if not session.boxes:
                    state["status"] = "No boxes to save"
                    continue
                if session.pending is not None:
                    state["status"] = "Accept (Enter) or reject (X) the pending box before saving"
                    continue
                reference = save_configuration(
                    session, args.output, source.rotation, args.source, frame,
                    source.time_seconds if source.is_file else None,
                )
                saved = True
                logger.info("Saved %d workstation(s) to %s (rotation=%d, resolution=%dx%d); reference frame %s",
                            len(session.boxes), args.output, source.rotation, session.width, session.height,
                            reference)
                for ws_id, box in session.boxes.items():
                    logger.info("  %s bbox=%s", ws_id, list(box))
                return 0
            elif key in (ord("q"), 27):
                if session.dirty and not cancel_armed:
                    cancel_armed = True
                    state["status"] = "Unsaved boxes! Press Q/Esc again to discard, S to save"
                else:
                    logger.info("Calibration canceled; %s left unchanged", args.output)
                    return 1
    finally:
        if not saved:
            logger.debug("Exiting without saving")
        source.release()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return run(parse_args(argv))
    except (FileNotFoundError, RuntimeError, ROIConfigError) as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
