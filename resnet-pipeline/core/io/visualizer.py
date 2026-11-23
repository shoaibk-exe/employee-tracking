"""OpenCV overlays, live display, and annotated video output."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from core.tracking.person_tracker import PersonTrack, TrackLifecycle
from core.occupancy.state_manager import UNKNOWN

logger = logging.getLogger(__name__)

DEFAULT_COLORS: dict[str, tuple[int, int, int]] = {
    "occupied": (0, 200, 0),
    "empty": (0, 140, 255),
    "standing_near": (255, 180, 0),
    UNKNOWN: (160, 160, 160),
    "track_confirmed": (0, 220, 255),
    "track_tentative": (200, 200, 80),
    "track_lost": (120, 120, 120),
    # VideoMAE desk activity → ROI box color (OpenCV BGR).
    "computer_interaction": (0, 200, 0),  # green
    "phone_using": (0, 255, 255),         # yellow
    "neither": (128, 128, 128),           # grey
}
FALLBACK_COLOR = (255, 0, 255)
TRACK_COLORS = {
    TrackLifecycle.CONFIRMED: "track_confirmed",
    TrackLifecycle.TENTATIVE: "track_tentative",
    TrackLifecycle.LOST: "track_lost",
    TrackLifecycle.REMOVED: "track_lost",
}


def display_available() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass(frozen=True)
class WorkstationView:
    workstation_id: str
    bbox: tuple[int, int, int, int]
    state: str
    duration_seconds: float
    predicted_class: str | None = None
    confidence: float | None = None


ASSOCIATION_COLORS = {
    "associated": (0, 230, 0),
    "candidate": (0, 230, 230),
    "ambiguous": (230, 0, 230),
    "unassociated": (60, 60, 230),
}


@dataclass(frozen=True)
class FullWorkstationView:
    workstation_id: str
    seat_box: tuple[int, int, int, int]
    anchor: tuple[float, float]
    state: str
    duration_seconds: float
    evidence_source: str
    reason: str
    needs_review: bool = False
    disagreement: str | None = None
    # VideoMAE desk activity (computer_interaction / neither / phone_using).
    activity_class: str | None = None
    activity_confidence: float | None = None
    # YOLO activity of the person at this desk (sitting/standing mapped to working/not_working).
    yolo_activity: str | None = None


@dataclass(frozen=True)
class FullTrackView:
    track: PersonTrack
    association_state: str
    workstation_id: str | None
    candidate_workstation_id: str | None
    posture: str
    employee_id: str
    seated_seconds: float
    reasons: tuple[str, ...] = ()
    activity_class: str | None = None
    working_seconds: float = 0.0
    working_active: bool = False
    person_id: str | None = None  # same-day appearance ID (P01…), not employee_id
    # Stable T## from gallery (survives tracker ID churn); falls back to track.local_track_id.
    display_track_id: int | None = None
    # Desk VideoMAE class for the workstation this person is associated to.
    videomae_class: str | None = None
    videomae_confidence: float | None = None


@dataclass(frozen=True)
class WorkingTimeOverlayRow:
    """Deprecated track-row overlay; kept for backward-compatible call sites."""

    local_track_id: int
    activity_class: str | None
    workstation_id: str | None
    working_seconds: float
    active: bool
    reason: str = ""


@dataclass(frozen=True)
class DeskTimeOverlayRow:
    """Top-left panel: per-workstation empty vs occupied+working time."""

    workstation_id: str
    empty_seconds: float
    working_seconds: float
    current_mode: str


class Visualizer:
    def __init__(
        self,
        camera_id: str,
        colors: Mapping[str, Sequence[int]] | None = None,
        draw_confidence: bool = True,
        draw_duration: bool = True,
        show_window: bool = False,
        window_name: str = "Employee monitoring",
    ) -> None:
        self.camera_id = camera_id
        self.colors = {**DEFAULT_COLORS, **{k: tuple(v) for k, v in (colors or {}).items()}}
        self.draw_confidence = draw_confidence
        self.draw_duration = draw_duration
        self.window_name = window_name
        self.show_window = show_window
        if show_window and not display_available():
            logger.warning("No display available; disabling the live window (video/CSV output still work)")
            self.show_window = False
        self._window_created = False
        self.paused = False

    def color(self, state: str) -> tuple[int, int, int]:
        return self.colors.get(state, FALLBACK_COLOR)  # type: ignore[return-value]

    def draw_tracks(self, frame: np.ndarray, tracks: Sequence[PersonTrack], status_lines: Sequence[str]) -> np.ndarray:
        """Draw person tracks only (Milestone 2). Never labels a track as a permanent employee."""
        canvas = frame.copy()
        height, width = canvas.shape[:2]
        scale = max(0.45, min(1.2, width / 2400))
        thickness = max(1, int(round(scale * 2)))
        # Thin person outlines so they do not visually dominate workstation ROIs.
        person_thickness = 1
        line_h = int(20 * scale) + 4

        for track in tracks:
            if track.lifecycle == TrackLifecycle.REMOVED:
                continue
            if track.is_predicted or track.lifecycle == TrackLifecycle.LOST:
                continue  # no empty ghost boxes when the person is gone
            color = self.color(TRACK_COLORS.get(track.lifecycle, "track_confirmed"))
            x1, y1, x2, y2 = track.bbox
            style = person_thickness
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, style)
            if track.is_predicted:
                # Dashed look for predicted/lost boxes: mark corners only.
                for (ax, ay), (bx, by) in (
                    ((x1, y1), (x1 + 12, y1)), ((x1, y1), (x1, y1 + 12)),
                    ((x2, y1), (x2 - 12, y1)), ((x2, y1), (x2, y1 + 12)),
                    ((x1, y2), (x1 + 12, y2)), ((x1, y2), (x1, y2 - 12)),
                    ((x2, y2), (x2 - 12, y2)), ((x2, y2), (x2, y2 - 12)),
                ):
                    cv2.line(canvas, (ax, ay), (bx, by), color, style)
            label = f"T{track.local_track_id} {track.lifecycle.value}"
            if self.draw_confidence and not track.is_predicted:
                label += f" {track.confidence:.0%}"
            # Explicitly show identity is unknown — track ID is not an employee ID.
            label2 = f"id={track.identity_status.value}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7, thickness)
            by1 = max(0, y1 - th - line_h - 4)
            cv2.rectangle(canvas, (x1, by1), (min(width, x1 + max(tw, 80) + 8), y1), color, -1)
            cv2.putText(canvas, label, (x1 + 3, y1 - line_h + 2), cv2.FONT_HERSHEY_SIMPLEX,
                        scale * 0.7, (0, 0, 0), thickness, cv2.LINE_AA)
            cv2.putText(canvas, label2, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        scale * 0.55, (0, 0, 0), max(1, thickness - 1), cv2.LINE_AA)

        confirmed = sum(1 for t in tracks if t.lifecycle == TrackLifecycle.CONFIRMED and not t.is_predicted)
        tentative = sum(1 for t in tracks if t.lifecycle == TrackLifecycle.TENTATIVE)
        lost = sum(1 for t in tracks if t.lifecycle == TrackLifecycle.LOST or t.is_predicted)
        summary = [
            f"Camera: {self.camera_id}",
            f"Tracks confirmed: {confirmed}",
            f"Tracks tentative: {tentative}",
            f"Tracks lost/predicted: {lost}",
            "Employee ID: not assigned (track ID != employee)",
            *status_lines,
        ]
        panel_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] for t in summary) + 20
        panel_h = line_h * len(summary) + 10
        panel = canvas[0:panel_h, 0:panel_w]
        canvas[0:panel_h, 0:panel_w] = (panel * 0.35).astype(np.uint8)
        for i, text in enumerate(summary):
            cv2.putText(canvas, text, (10, line_h * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
        return canvas

    def draw_full(
        self,
        frame: np.ndarray,
        workstations: Sequence[FullWorkstationView],
        tracks: Sequence[FullTrackView],
        status_lines: Sequence[str],
        debug: bool = False,
        working_rows: Sequence[WorkingTimeOverlayRow] | None = None,
        desk_rows: Sequence[DeskTimeOverlayRow] | None = None,
    ) -> np.ndarray:
        """Fused view: workstation ROIs only (no YOLO person boxes / corner panels)."""
        canvas = frame.copy()
        height, width = canvas.shape[:2]
        scale = max(0.45, min(1.2, width / 2400))
        thickness = max(1, int(round(scale * 2)))
        font_scale = scale * 0.85
        line_h = int(24 * scale) + 6
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Keep unused overlay args for backward-compatible call sites.
        _ = (working_rows, desk_rows, tracks, status_lines)

        def label(lines: Sequence[str], x: int, y: int, color: tuple[int, int, int], above: bool) -> None:
            text_w = max(cv2.getTextSize(t, font, font_scale, thickness)[0][0] for t in lines) + 10
            box_h = line_h * len(lines) + 6
            y1 = max(0, y - box_h) if above else min(height - box_h, y)
            x1 = max(0, min(width - text_w, x))
            region = canvas[y1:y1 + box_h, x1:x1 + text_w]
            if region.size:
                canvas[y1:y1 + box_h, x1:x1 + text_w] = cv2.addWeighted(region, 0.3, np.full_like(region, color), 0.7, 0)
            for i, text in enumerate(lines):
                cv2.putText(canvas, text, (x1 + 5, y1 + line_h * (i + 1) - 2), font, font_scale, (0, 0, 0),
                            thickness, cv2.LINE_AA)

        for w in workstations:
            # Prefer VideoMAE activity color when available (yellow phone, grey neither).
            activity_key = (w.activity_class or "").strip().lower().replace("-", "_").replace(" ", "_")
            color = self.color(activity_key) if activity_key in self.colors else self.color(w.state)
            x1, y1, x2, y2 = w.seat_box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness + 1)
            if w.activity_class:
                conf = "" if w.activity_confidence is None else f" {w.activity_confidence:.0%}"
                activity_line = f"Activity: {w.activity_class}{conf}"
            else:
                activity_line = "Activity:"
            person_line = f"Person: {w.yolo_activity}" if w.yolo_activity else "Person:"
            lines = [
                f"{w.workstation_id}{' *review' if w.needs_review else ''}",
                activity_line,
                person_line,
            ]
            if debug:
                lines.append(w.reason[:40])
                if w.disagreement:
                    lines.append(f"! {w.disagreement[:38]}")
            label(lines, x1, y1, color, above=True)

        return canvas

    def draw(self, frame: np.ndarray, views: Sequence[WorkstationView], status_lines: Sequence[str]) -> np.ndarray:
        canvas = frame.copy()
        height, width = canvas.shape[:2]
        scale = max(0.4, min(1.2, width / 2400))
        thickness = max(1, int(round(scale * 2)))
        line_h = int(22 * scale) + 4

        for view in views:
            x1, y1, x2, y2 = view.bbox
            color = self.color(view.state)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness + 1)
            lines = [view.workstation_id, view.state.upper()]
            if self.draw_confidence and view.confidence is not None:
                if view.predicted_class is not None and view.predicted_class != view.state:
                    lines.append(f"Pred: {view.predicted_class} {view.confidence:.0%}")
                else:
                    lines.append(f"Confidence: {view.confidence:.0%}")
            if self.draw_duration:
                lines.append(f"Duration: {format_duration(view.duration_seconds)}")
            text_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] for t in lines)
            box_h = line_h * len(lines) + 6
            bx1, by1 = x1, max(0, y1)
            bx2, by2 = min(width, x1 + text_w + 10), min(height, by1 + box_h)
            overlay = canvas[by1:by2, bx1:bx2]
            if overlay.size:
                canvas[by1:by2, bx1:bx2] = cv2.addWeighted(overlay, 0.35, np.full_like(overlay, color), 0.65, 0)
            for i, text in enumerate(lines):
                cv2.putText(canvas, text, (bx1 + 5, by1 + line_h * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (0, 0, 0), thickness, cv2.LINE_AA)

        counts: dict[str, int] = {}
        for view in views:
            counts[view.state] = counts.get(view.state, 0) + 1
        summary = [f"Camera: {self.camera_id}"]
        summary += [f"{state.replace('_', ' ').title()}: {counts.get(state, 0)}"
                    for state in sorted(set(counts) | ({"occupied", "empty", UNKNOWN} & set(self.colors)))]
        summary += list(status_lines)
        panel_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] for t in summary) + 20
        panel_h = line_h * len(summary) + 10
        panel = canvas[0:panel_h, 0:panel_w]
        canvas[0:panel_h, 0:panel_w] = (panel * 0.35).astype(np.uint8)
        for i, text in enumerate(summary):
            cv2.putText(canvas, text, (10, line_h * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
        return canvas

    def show(self, frame: np.ndarray) -> str | None:
        """Display a frame. Returns 'quit' when Q/Esc is pressed or the window is closed.

        P toggles pause (offline review); while paused, N steps one processed frame.
        """
        if not self.show_window:
            return None
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, min(frame.shape[1], 1600), min(frame.shape[0], 900))
            self._window_created = True
        cv2.imshow(self.window_name, frame)
        while True:
            key = cv2.waitKey(1 if not self.paused else 50) & 0xFF
            if self._window_created and cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                return "quit"
            if key in (ord("q"), 27):
                return "quit"
            if key == ord("p"):
                self.paused = not self.paused
                logger.info("Playback %s", "paused (P resume, N step)" if self.paused else "resumed")
                if not self.paused:
                    return None
            elif key == ord("n") and self.paused:
                return None
            elif not self.paused:
                return None

    def close(self) -> None:
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False


class AnnotatedVideoWriter:
    """MP4 writer that logs and disables itself instead of crashing on codec errors."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int], fourcc: str = "mp4v") -> None:
        self.path = Path(path)
        self.size = size
        self.frames = 0
        self._writer: cv2.VideoWriter | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*fourcc), float(fps), size)
            if writer.isOpened():
                self._writer = writer
                logger.info("Writing annotated video to %s (%dx%d @ %.2f fps)", self.path, *size, fps)
            else:
                logger.error("Could not open video writer for %s (codec %s); annotated video disabled",
                             self.path, fourcc)
        except Exception as exc:
            logger.error("Video writer initialization failed for %s: %s; annotated video disabled", self.path, exc)

    @property
    def enabled(self) -> bool:
        return self._writer is not None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            return
        if (frame.shape[1], frame.shape[0]) != self.size:
            logger.error("Frame size %s does not match writer size %s; skipping frame",
                         (frame.shape[1], frame.shape[0]), self.size)
            return
        self._writer.write(frame)
        self.frames += 1

    def release(self) -> None:
        """Finalize the MP4. Must run on interrupt — OpenCV leaves files unplayable until this."""
        writer = self._writer
        if writer is None:
            return
        self._writer = None
        try:
            writer.release()
            logger.info(
                "Annotated video finalized: %s (%d frames)", self.path, self.frames,
            )
        except Exception as exc:  # noqa: BLE001 — never block shutdown on codec teardown
            logger.error("Annotated video release failed for %s: %s", self.path, exc)
