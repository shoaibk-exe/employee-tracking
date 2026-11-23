"""Employee presence monitoring entry point.

Modes (``pipeline.mode`` / ``--mode``):

* ``occupancy`` — existing ResNet-18 workstation ROI classifier
* ``tracking``  — YOLO26m + BoT-SORT full-frame person tracking (Milestone 2)
* ``full``      — reserved for the fused person-centric pipeline (later milestones)

    python main.py --mode tracking --source testing/input2.mp4 --show --save-video
    python main.py --mode occupancy --source testing/input.mp4 --show --save-video
"""

from __future__ import annotations

import argparse
import csv
import logging
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2

from core.activity.time_tracker import ACTIVITY_CLASSES, ActivityTimeTracker
from core.activity.videomae_classifier import (
    ActivityError,
    ActivityPrediction,
    VideoMAEActivityClassifier,
    WorkstationActivityMonitor,
)
from core.association.association_engine import AssociationEngine, AssociationEvent, AssociationFrame
from core.occupancy.classifier import CheckpointError, CropClassifier, ResNet18Classifier
from core.config import ConfigError, Settings, load_settings
from core.io.csv_writer import CsvAppender, CsvSchemaError, ReportWriter
from core.io.day_ledger import DayLedger
from core.identity.identity_manager import (
    UNKNOWN_EMPLOYEE, EmployeeDirectory, IdentityError, IdentityEvent, IdentityManager, IdentitySource,
    NullIdentitySource,
)
from core.association.occupancy_fusion import OccupancyFusion, crop_compatibility
from core.tracking.person_tracker import PersonTracker, TrackerError
from core.association.presence_time import PresenceTimeTracker, SeatedInterval
from core.occupancy.roi_manager import ROIConfigError, ROIManager
from core.scene.scene_mapper import (
    STATUS_NEEDS_REVIEW, SceneMap, SceneMapError, SceneMapper,
    scene_map_from_admin_rois,
)
from core.occupancy.state_manager import UNKNOWN, StateTransition, WorkstationStateManager
from core.occupancy.time_tracker import WorkstationTimeTracker
from core.io.video_reader import FramePacket, ReconnectPolicy, SourceOutage, VideoReader, VideoSourceError
from core.io.visualizer import (
    AnnotatedVideoWriter, DeskTimeOverlayRow, FullTrackView, FullWorkstationView, Visualizer,
    WorkstationView,
)
from core.identity.appearance_gallery import AppearanceGallery
from core.io.excel_report import write_progress_workbook
from core.identity.identity_resolver import PersonIdentityResolver
from core.association.working_time import WorkingTimeTracker, WorkstationOccupancy
from core.scalability import OverloadGuard, ensure_runtime_speed_flags, phase_offset_seconds
from storage.repository import JsonlRepository, RepositoryError

logger = logging.getLogger("monitor")

STANDARD_STATES = ("occupied", "empty", "standing_near", UNKNOWN)
TRACK_CSV_FIELDS = [
    "tenant_id", "camera_id", "session_id", "tracking_session_id", "timestamp_seconds",
    "frame_index", "local_track_id", "composite_track_id", "lifecycle", "is_predicted",
    "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "identity_status",
]
FULL_TRACK_CSV_FIELDS = TRACK_CSV_FIELDS + [
    "activity_class", "posture", "raw_posture", "association_state", "workstation_id",
    "candidate_workstation_id", "association_score", "employee_id", "person_id", "reason",
    "working_seconds", "working_active",
]
WORKING_SUMMARY_FIELDS = [
    "tenant_id", "camera_id", "session_id", "local_track_id", "composite_track_id",
    "activity_class", "workstation_id", "working_seconds", "active", "reason",
]
FUSION_CSV_FIELDS = [
    "tenant_id", "camera_id", "session_id", "timestamp_seconds", "workstation_id", "observed_class",
    "confidence", "evidence_source", "reason", "occupant_track_id", "resnet_class", "resnet_confidence",
    "disagreement", "state_after", "activity_class", "activity_confidence", "yolo_activity",
]
SEATED_CSV_FIELDS = [
    "tenant_id", "camera_id", "session_id", "composite_track_id", "local_track_id", "workstation_id",
    "employee_id", "start_timestamp", "end_timestamp", "duration_seconds", "end_reason",
]
ASSOCIATION_EVENT_FIELDS = [
    "tenant_id", "camera_id", "session_id", "kind", "composite_track_id", "local_track_id", "workstation_id",
    "event_timestamp", "decided_timestamp", "reason", "score",
]
EMPLOYEE_SUMMARY_FIELDS = ["tenant_id", "camera_id", "session_id", "employee_id", "workstation_id",
                           "seated_seconds"]


def new_session_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"


def setup_logging(directory: Path, level: str, session_id: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"monitor_{session_id}.log"
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    return log_path


@dataclass
class RunResult:
    session_id: str
    end_reason: str
    frames_seen: int = 0
    inference_frames: int = 0
    tracking_frames: int = 0
    last_timestamp: float | None = None
    annotated_video: Path | None = None
    summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    track_stats: dict[str, Any] = field(default_factory=dict)


class MonitoringApp:
    """Legacy ROI + ResNet-18 occupancy path (pipeline.mode = occupancy)."""

    def __init__(self, settings: Settings, classifier: CropClassifier | None = None,
                 session_id: str | None = None) -> None:
        self.settings = settings
        self.session_id = session_id or new_session_id()
        self._classifier = classifier
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _handle(self, transitions: list[StateTransition], trackers: dict[str, WorkstationTimeTracker],
                reports: ReportWriter) -> None:
        for transition in transitions:
            interval = trackers[transition.workstation_id].apply(transition)
            if interval is not None:
                reports.write_interval(interval)
            reports.write_event(transition)
            if transition.previous_state is not None:
                logger.info("%s: %s -> %s at %.2fs (confirmed %.2fs, %s)", transition.workstation_id,
                            transition.previous_state, transition.new_state, transition.transition_timestamp,
                            transition.confirmation_timestamp, transition.reason)

    def run(self, max_frames: int | None = None) -> RunResult:
        s = self.settings
        if s.roi_config is None or s.checkpoint is None:
            raise ConfigError("occupancy mode requires rois.config and model.checkpoint")
        rois = ROIManager.from_json(s.roi_config)
        rois.check_rotation(s.rotation)
        reader = VideoReader(
            s.source, s.inference_interval_seconds, tracking_interval_seconds=None,
            rotation=s.rotation, yield_all_frames=s.annotate_all_frames,
            recording_start_utc=s.recording_start_utc, reconnect=ReconnectPolicy(**s.stream),
        )
        result = RunResult(self.session_id, "error")
        visualizer: Visualizer | None = None
        video: AnnotatedVideoWriter | None = None
        reports: ReportWriter | None = None
        trackers: dict[str, WorkstationTimeTracker] = {}
        try:
            rois.configure_for_frame(reader.width, reader.height, allow_scaling=s.allow_roi_scaling)
            classifier = self._classifier or ResNet18Classifier(
                s.checkpoint, s.class_names, s.preprocessing_config, s.device, s.batch_size)
            class_names = list(classifier.class_names)
            managers = {ws.workstation_id: WorkstationStateManager(ws.workstation_id, class_names, s.state)
                        for ws in rois.workstations}
            trackers = {ws.workstation_id: WorkstationTimeTracker(
                ws.workstation_id, class_names, reader.origin_utc, s.report_timezone) for ws in rois.workstations}
            reports = ReportWriter(s.output_directory, s.camera_id, self.session_id, s.flush_interval_seconds)
            visualizer = Visualizer(s.camera_id, s.colors, s.draw_confidence, s.draw_duration, s.show_window)
            if s.save_video:
                fps = s.output_fps or (
                    reader.fps if (s.annotate_all_frames or s.inference_interval_seconds <= 0) and reader.fps
                    else (1.0 / s.inference_interval_seconds if s.inference_interval_seconds > 0 else 10.0))
                result.annotated_video = s.output_directory / f"annotated_{self.session_id}.mp4"
                video = AnnotatedVideoWriter(result.annotated_video, fps, (reader.width, reader.height))
            logger.info("Session %s occupancy: %d workstations, classes %s, source %s",
                        self.session_id, len(managers), class_names, s.source)

            latest: dict[str, Any] = {}
            ema_fps: float | None = None
            last_inference_wall: float | None = None
            infer_ms = 0.0
            end_reason = "end_of_stream" if reader.is_live else "end_of_video"
            for item in reader:
                if self._stop_requested:
                    end_reason = "interrupted"
                    break
                if isinstance(item, SourceOutage):
                    logger.warning("Source outage (%s) at %.2fs; last frame %s", item.reason,
                                   item.detected_seconds, item.last_frame_seconds)
                    for ws_id, manager in managers.items():
                        self._handle(manager.mark_unavailable(item.detected_seconds), trackers, reports)
                    continue
                packet: FramePacket = item
                result.frames_seen += 1
                ts = packet.timestamp_seconds
                if packet.run_inference:
                    crops = rois.extract(packet.frame)
                    started = time.perf_counter()
                    predictions = classifier.classify(crops)
                    infer_ms = (time.perf_counter() - started) * 1000.0
                    for ws_id, manager in managers.items():
                        prediction = predictions.get(ws_id)
                        if prediction is None:
                            self._handle(manager.observe_missing(ts), trackers, reports)
                            continue
                        reports.write_prediction(ts, packet.wall_clock_utc, ws_id, prediction)
                        transitions = manager.observe(ts, prediction.predicted_class, prediction.confidence)
                        trackers[ws_id].observe(ts)
                        self._handle(transitions, trackers, reports)
                        latest[ws_id] = prediction
                    result.inference_frames += 1
                    result.last_timestamp = ts
                    now = time.perf_counter()
                    if last_inference_wall is not None and now > last_inference_wall:
                        inst = 1.0 / (now - last_inference_wall)
                        ema_fps = inst if ema_fps is None else 0.9 * ema_fps + 0.1 * inst
                    last_inference_wall = now

                views = []
                for ws in rois.workstations:
                    snap = managers[ws.workstation_id].snapshot()
                    prediction = latest.get(ws.workstation_id)
                    views.append(WorkstationView(
                        ws.workstation_id, ws.bbox, snap.state,
                        trackers[ws.workstation_id].current_duration(ts),
                        prediction.predicted_class if prediction else None,
                        prediction.confidence if prediction else None,
                    ))
                status = [
                    f"Source time: {ts:.1f}s",
                    f"Source FPS: {reader.fps:.1f}" if reader.fps else "Source FPS: n/a",
                    f"Processing FPS: {ema_fps:.1f}" if ema_fps else "Processing FPS: -",
                    f"Batch inference: {infer_ms:.0f} ms",
                ]
                if packet.wall_clock_utc is not None:
                    status.insert(1, f"UTC: {packet.wall_clock_utc:%Y-%m-%d %H:%M:%S}")
                if s.save_video or visualizer.show_window:
                    annotated = visualizer.draw(packet.frame, views, status)
                    if video is not None:
                        video.write(annotated)
                    if visualizer.show(annotated) == "quit":
                        end_reason = "user_stop"
                        break
                if max_frames is not None and result.inference_frames >= max_frames:
                    end_reason = "max_frames"
                    break
            else:
                if reader.source_lost:
                    end_reason = "source_lost"
            result.end_reason = end_reason
        except KeyboardInterrupt:
            result.end_reason = "interrupted"
            logger.warning("Interrupted; finalizing reports")
        finally:
            # Finalize MP4 before slow report work so Ctrl+C still leaves a playable file.
            if video is not None:
                video.release()
            if visualizer is not None:
                visualizer.close()
            if reports is not None:
                self._finalize(result, trackers, reports)
                reports.close()
            reader.close()
        return result

    def _finalize(self, result: RunResult, trackers: dict[str, WorkstationTimeTracker], reports: ReportWriter) -> None:
        reason = result.end_reason
        for tracker in trackers.values():
            interval = tracker.finalize(reason)
            if interval is not None:
                reports.write_interval(interval)
            result.summaries[tracker.workstation_id] = tracker.summary()
        extra = sorted({st for t in trackers.values() for st in t.states} - set(STANDARD_STATES))
        if trackers:
            path = reports.write_summaries(((ws, row) for ws, row in result.summaries.items()), extra)
            logger.info("Wrote summary to %s", path)
        daily = [(ws, day, state, sec) for ws, t in trackers.items()
                 for (day, state), sec in sorted(t.daily.items())]
        if daily:
            reports.write_daily(daily)
        logger.info("Session %s ended (%s): %d frames, %d classified, last t=%s",
                    result.session_id, reason, result.frames_seen, result.inference_frames, result.last_timestamp)


class TrackingApp:
    """Milestone 2: full-frame person detection + BoT-SORT. No ROI dependency. No employee identity."""

    def __init__(self, settings: Settings, session_id: str | None = None,
                 person_tracker: PersonTracker | None = None) -> None:
        self.settings = settings
        self.session_id = session_id or new_session_id()
        self._person_tracker = person_tracker
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self, max_frames: int | None = None) -> RunResult:
        s = self.settings
        reader = VideoReader(
            s.source, inference_interval_seconds=s.inference_interval_seconds,
            tracking_interval_seconds=s.tracking_interval_seconds,
            rotation=s.rotation, yield_all_frames=True,
            recording_start_utc=s.recording_start_utc, reconnect=ReconnectPolicy(**s.stream),
        )
        result = RunResult(self.session_id, "error")
        visualizer: Visualizer | None = None
        video: AnnotatedVideoWriter | None = None
        csv_handle = None
        csv_writer = None
        try:
            tracker = self._person_tracker or PersonTracker(
                s.tenant_id, s.camera_id, s.detection, s.tracking)
            s.output_directory.mkdir(parents=True, exist_ok=True)
            tracks_path = s.output_directory / f"tracks_{self.session_id}.csv"
            csv_handle = tracks_path.open("w", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(csv_handle, fieldnames=TRACK_CSV_FIELDS)
            csv_writer.writeheader()

            visualizer = Visualizer(s.camera_id, s.colors, s.draw_confidence, s.draw_duration, s.show_window,
                                    window_name="Person tracking")
            if s.save_video:
                fps = s.output_fps or (reader.fps if reader.fps else 20.0)
                result.annotated_video = s.output_directory / f"annotated_tracks_{self.session_id}.mp4"
                video = AnnotatedVideoWriter(result.annotated_video, fps, (reader.width, reader.height))

            logger.info(
                "Session %s tracking: tenant=%s camera=%s detector=%s imgsz=%d reid=%s source=%s",
                self.session_id, s.tenant_id, s.camera_id, s.detection.model, s.detection.image_size,
                s.tracking.enable_reid, s.source,
            )

            unique_locals: set[int] = set()
            peak_active = 0
            removed_count = 0
            ema_fps: float | None = None
            last_wall: float | None = None
            latency_ms = 0.0
            end_reason = "end_of_stream" if reader.is_live else "end_of_video"
            latest_tracks = []

            for item in reader:
                if self._stop_requested:
                    end_reason = "interrupted"
                    break
                if isinstance(item, SourceOutage):
                    logger.warning("Source outage (%s); resetting tracking session continuity", item.reason)
                    tracker.reset_session()
                    latest_tracks = []
                    continue
                packet: FramePacket = item
                result.frames_seen += 1
                ts = packet.timestamp_seconds

                if packet.after_reconnect:
                    logger.warning("Frame after reconnect at %.2fs; resetting tracking session", ts)
                    tracker.reset_session()

                if packet.run_tracking:
                    update = tracker.update(packet.frame, ts, packet.frame_index)
                    latest_tracks = update.tracks
                    latency_ms = update.detector_latency_ms
                    result.tracking_frames += 1
                    result.last_timestamp = ts
                    for track in update.tracks:
                        if not track.is_predicted:
                            unique_locals.add(track.local_track_id)
                            csv_writer.writerow({
                                "tenant_id": track.tenant_id,
                                "camera_id": track.camera_id,
                                "session_id": self.session_id,
                                "tracking_session_id": track.tracking_session_id,
                                "timestamp_seconds": f"{ts:.3f}",
                                "frame_index": packet.frame_index,
                                "local_track_id": track.local_track_id,
                                "composite_track_id": track.composite_track_id,
                                "lifecycle": track.lifecycle.value,
                                "is_predicted": int(track.is_predicted),
                                "confidence": f"{track.confidence:.4f}",
                                "bbox_x1": track.bbox[0], "bbox_y1": track.bbox[1],
                                "bbox_x2": track.bbox[2], "bbox_y2": track.bbox[3],
                                "identity_status": track.identity_status.value,
                            })
                    removed_count += len(update.removed)
                    active = sum(1 for t in update.tracks if not t.is_predicted)
                    peak_active = max(peak_active, active)
                    now = time.perf_counter()
                    if last_wall is not None and now > last_wall:
                        inst = 1.0 / (now - last_wall)
                        ema_fps = inst if ema_fps is None else 0.9 * ema_fps + 0.1 * inst
                    last_wall = now
                    if result.tracking_frames % 100 == 0:
                        csv_handle.flush()

                status = [
                    f"Source time: {ts:.1f}s",
                    f"Source FPS: {reader.fps:.1f}" if reader.fps else "Source FPS: n/a",
                    f"Tracking FPS: {ema_fps:.1f}" if ema_fps else "Tracking FPS: -",
                    f"Detector+tracker: {latency_ms:.0f} ms",
                    f"Tracking session: {tracker.tracking_session_id}",
                ]
                if s.save_video or visualizer.show_window:
                    annotated = visualizer.draw_tracks(packet.frame, latest_tracks, status)
                    if video is not None:
                        video.write(annotated)
                    if visualizer.show(annotated) == "quit":
                        end_reason = "user_stop"
                        break
                if max_frames is not None and result.tracking_frames >= max_frames:
                    end_reason = "max_frames"
                    break
            else:
                if reader.source_lost:
                    end_reason = "source_lost"
            result.end_reason = end_reason
            result.track_stats = {
                "unique_local_track_ids": len(unique_locals),
                "peak_active_tracks": peak_active,
                "removed_tracks": removed_count,
                "tracking_session_id": tracker.tracking_session_id,
                "tracks_csv": str(tracks_path),
            }
        except KeyboardInterrupt:
            result.end_reason = "interrupted"
            logger.warning("Interrupted; finalizing tracking outputs")
        finally:
            # Finalize MP4 first — CSV flush can wait; an unreleased writer is unplayable.
            if video is not None:
                video.release()
            if visualizer is not None:
                visualizer.close()
            if csv_handle is not None:
                csv_handle.flush()
                csv_handle.close()
            reader.close()
        logger.info(
            "Session %s ended (%s): %d frames, %d tracked, unique_local_ids=%s peak_active=%s",
            result.session_id, result.end_reason, result.frames_seen, result.tracking_frames,
            result.track_stats.get("unique_local_track_ids"), result.track_stats.get("peak_active_tracks"),
        )
        return result


def _file_sha256(path: Path | None) -> str | None:
    import hashlib

    if path is None or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FullApp:
    """Person-centric presence pipeline (pipeline.mode = full).

    tracking (sitting/standing YOLO + tracker) -> posture (detector class + box geometry) ->
    person-to-workstation association -> occupancy fusion (person first, ResNet-18 support)
    -> workstation state machine -> intervals; identity only from verified sources.
    """

    def __init__(self, settings: Settings, session_id: str | None = None, *,
                 classifier: CropClassifier | None = None, person_tracker: PersonTracker | None = None,
                 identity_source: IdentitySource | None = None,
                 activity_monitor: WorkstationActivityMonitor | None = None,
                 activity_classifier: VideoMAEActivityClassifier | None = None,
                 debug_overlay: bool = False,
                 on_frame: Any | None = None, on_progress: Any | None = None) -> None:
        self.settings = settings
        self.session_id = session_id or new_session_id()
        self._classifier = classifier
        self._tracker = person_tracker
        self._identity_source = identity_source or NullIdentitySource()
        self._activity_monitor = activity_monitor
        # Shared across cameras in the multi-cam hub (one VideoMAE on the GPU).
        self._activity_classifier = activity_classifier
        self.debug_overlay = debug_overlay
        self._on_frame = on_frame
        self._on_progress = on_progress
        self._progress_at = 0.0
        self._stop_requested = False
        self._activity_by_ws: dict[str, ActivityPrediction] = {}
        self._yolo_by_ws: dict[str, str] = {}
        self._activity_trackers: dict[str, ActivityTimeTracker] = {}

    def request_stop(self) -> None:
        self._stop_requested = True

    def _day_ledger(self) -> DayLedger:
        return DayLedger(self.settings.output_directory, self.settings.report_timezone)

    def _local_day(self, ts: float, trackers: dict[str, WorkstationTimeTracker]) -> date:
        origin = next((tracker.origin_utc for tracker in trackers.values()), None)
        if origin is not None:
            return (origin + timedelta(seconds=float(ts))).astimezone(self.settings.report_timezone).date()
        return datetime.now(self.settings.report_timezone).date()

    def _session_slices(
        self, ts: float, workstations: list[Any], managers: dict[str, Any],
        trackers: dict[str, WorkstationTimeTracker],
    ) -> dict[date, dict[str, dict[str, Any]]]:
        """This session's seconds split by calendar day. The current day also carries live state."""
        today = self._local_day(ts, trackers)
        slices: dict[date, dict[str, dict[str, Any]]] = {}
        for ws in workstations:
            ws_id = ws.workstation_id
            tracker = trackers.get(ws_id)
            if tracker is None:
                continue
            by_day = tracker.seconds_by_day(ts)
            if not by_day:
                by_day = {today: tracker.totals_at(ts)}
            activity_tracker = self._activity_trackers.get(ws_id)
            activity_by_day = activity_tracker.seconds_by_day(ts) if activity_tracker is not None else {}
            if activity_tracker is not None and not activity_by_day:
                activity_by_day = {today: activity_tracker.totals_at(ts)}
            snap = managers[ws_id].snapshot() if ws_id in managers else None
            activity = self._activity_by_ws.get(ws_id)
            meta = {
                "state": None if snap is None else snap.state,
                "predicted_class": None if snap is None else snap.last_predicted_class,
                "confidence": None if snap is None or snap.last_confidence is None else round(float(snap.last_confidence), 4),
                "bbox": [int(v) for v in ws.context_box],
                "activity_class": None if activity is None else activity.predicted_class,
                "activity_confidence": None if activity is None else round(float(activity.confidence), 4),
                "yolo_activity": self._yolo_by_ws.get(ws_id),
            }
            for day in set(by_day) | set(activity_by_day) | {today}:
                states = by_day.get(day) or {}
                activity_totals = activity_by_day.get(day) or {}
                row = {
                    "occupied_seconds": round(float(states.get("occupied", 0.0) or 0.0), 3),
                    "empty_seconds": round(float(states.get("empty", 0.0) or 0.0), 3),
                    "unknown_seconds": round(float(states.get(UNKNOWN, 0.0) or 0.0), 3),
                    "standing_near_seconds": round(float(states.get("standing_near", 0.0) or 0.0), 3),
                }
                if activity_tracker is not None:
                    row.update(activity_tracker.as_seconds_fields(activity_totals))
                if day == today:
                    row.update(meta)
                slices.setdefault(day, {})[ws_id] = row
        return slices

    def _publish_day_totals(
        self, ts: float, workstations: list[Any], managers: dict[str, Any],
        trackers: dict[str, WorkstationTimeTracker], *, closed: bool,
    ) -> None:
        """Store this session into the calendar-day ledger and publish the summed day."""
        if not trackers or not workstations:
            return
        ledger = self._day_ledger()
        if not getattr(self, "_summary_imported", False):
            ledger.import_summary(self.settings.output_directory / "summary.csv")
            self._summary_imported = True
        for day, rows in self._session_slices(ts, workstations, managers, trackers).items():
            ledger.record_session(day, self.session_id, rows, closed=closed)
        today = self._local_day(ts, trackers)
        combined = ledger.combined(today)
        if not getattr(self, "_logged_day", False):
            occupied = sum(float(row.get("occupied_seconds") or 0.0) for row in combined["workstations"])
            logger.info(
                "Calendar day %s occupied total %.1fs across %d sessions; this session continues that day",
                today.isoformat(), occupied, int(combined.get("session_count") or 0),
            )
            self._logged_day = True
        if self._on_progress is None:
            return
        s = self.settings
        payload = {
            "camera_id": s.camera_id,
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rotation": int(s.rotation),
            "timestamp_seconds": round(float(ts), 3),
            "day": today.isoformat(),
            "session_id": self.session_id,
            "workstations": combined["workstations"],
        }
        try:
            self._on_progress(payload)
        except Exception:
            logger.warning("Could not publish live progress", exc_info=True)

    def _emit_live_progress(self, ts: float, workstations: list[Any], managers: dict[str, Any],
                            trackers: dict[str, Any]) -> None:
        """Publish the calendar-day empty/occupied totals, not only this session."""
        now = time.monotonic()
        if now - self._progress_at < 2.0:
            return
        self._progress_at = now
        try:
            self._publish_day_totals(ts, workstations, managers, trackers, closed=False)
        except Exception:
            logger.warning("Could not update daily totals", exc_info=True)

    def run(self, max_frames: int | None = None) -> RunResult:
        s = self.settings
        if s.checkpoint is None:
            raise ConfigError("full mode requires model.checkpoint")
        if s.use_admin_rois and s.roi_config is None:
            raise ConfigError("full mode with scene.use_admin_rois requires rois.config")
        scale = s.scalability
        if scale.enabled and scale.runtime_speed_flags:
            ensure_runtime_speed_flags(True)
        infer_phase = (
            phase_offset_seconds(s.camera_id, s.inference_interval_seconds)
            if scale.enabled and scale.phase_offset else 0.0
        )
        reader = VideoReader(
            s.source, inference_interval_seconds=s.inference_interval_seconds,
            tracking_interval_seconds=s.tracking_interval_seconds if s.tracking_interval_seconds is not None else 0.0,
            rotation=s.rotation, yield_all_frames=True,
            recording_start_utc=s.recording_start_utc, reconnect=ReconnectPolicy(**s.stream),
            inference_phase_offset_seconds=infer_phase,
        )
        if infer_phase > 0:
            logger.info(
                "Camera %s inference phase offset %.3fs (stagger ResNet ticks)",
                s.camera_id, infer_phase,
            )
        result = RunResult(self.session_id, "error")
        visualizer: Visualizer | None = None
        video: AnnotatedVideoWriter | None = None
        reports: ReportWriter | None = None
        repo: JsonlRepository | None = None
        csvs: dict[str, CsvAppender] = {}
        trackers: dict[str, WorkstationTimeTracker] = {}
        workstations: list[Any] = []
        managers: dict[str, Any] = {}
        presence = PresenceTimeTracker()
        last_ts: float | None = None
        finalizer = None
        resolver: PersonIdentityResolver | None = None
        write_excel = None
        working_clock: WorkingTimeTracker | None = None
        activity_monitor: WorkstationActivityMonitor | None = None
        overload: OverloadGuard | None = None
        if scale.enabled and scale.overload_shedding:
            overload = OverloadGuard(scale.overload_budget_seconds, scale.overload_consecutive)
        try:
            if s.use_admin_rois:
                # Exact admin-drawn desks — no automatic workstation discovery / seat refine.
                rois = ROIManager.from_json(s.roi_config)
                rois.check_rotation(s.rotation)
                rois.configure_for_frame(reader.width, reader.height, allow_scaling=s.allow_roi_scaling)
                ref_name = rois.metadata.get("reference_frame")
                # Prefer the ROI reference image (same folder as the ROI file) for alignment checks.
                scene = scene_map_from_admin_rois(
                    s.tenant_id, s.camera_id, (reader.width, reader.height), s.rotation,
                    {ws.workstation_id: ws.bbox for ws in rois.workstations},
                    reference_frame=str(ref_name) if ref_name else None,
                    roi_path=s.roi_config,
                )
                logger.info("Full mode workstations from admin ROIs %s (%d desks; seat=context=your ROI, no auto-detect)",
                            s.roi_config, len(scene.workstations))
            else:
                scene = SceneMap.load(s.scene_map_path)
            scene.validate_for(s.tenant_id, s.camera_id, reader.width, reader.height, s.rotation)
            workstations = scene.active_workstations
            review = [w.workstation_id for w in workstations if w.status == STATUS_NEEDS_REVIEW]
            if review:
                logger.warning("Scene map v%d: %s still need administrator review "
                               "(python -m scripts.scene.review_scene_map)", scene.version, review)
            classifier = self._classifier or ResNet18Classifier(
                s.checkpoint, s.class_names, s.preprocessing_config, s.device, s.batch_size)
            class_names = list(classifier.class_names)
            fusion = OccupancyFusion(class_names, s.fusion)
            crop_ok: dict[str, bool] = {}
            for ws in workstations:
                ok, why = crop_compatibility(ws.context_box, s.fusion)
                crop_ok[ws.workstation_id] = ok
                if not ok:
                    logger.warning("%s context crop not classifiable (%s); ResNet evidence disabled for it",
                                   ws.workstation_id, why)
            tracker = self._tracker or PersonTracker(s.tenant_id, s.camera_id, s.detection, s.tracking)
            reference_path = None
            if scene.reference_frame:
                candidates = [s.scene_map_path.with_name(scene.reference_frame)]
                if s.roi_config is not None:
                    candidates.insert(0, s.roi_config.with_name(scene.reference_frame))
                reference_path = next((p for p in candidates if p.is_file()), candidates[0])
            reference_gray = None
            if reference_path is not None and reference_path.is_file():
                reference_gray = cv2.cvtColor(cv2.imread(str(reference_path)), cv2.COLOR_BGR2GRAY)
            mapper = SceneMapper(scene, s.scene_runtime, reference_gray=reference_gray)
            association = AssociationEngine(workstations, s.association)
            working_clock = WorkingTimeTracker()
            latest_working: list = []
            latest_desks: list = []
            person_of: dict[str, str] = {}
            gallery: AppearanceGallery | None = None
            activity_monitor = self._activity_monitor
            if activity_monitor is None and s.activity.enabled:
                try:
                    activity_clf = self._activity_classifier or VideoMAEActivityClassifier(s.activity)
                    activity_phase = (
                        phase_offset_seconds(s.camera_id, s.activity.interval_seconds)
                        if scale.enabled and scale.phase_offset else 0.0
                    )
                    activity_monitor = WorkstationActivityMonitor(
                        activity_clf,
                        [ws.workstation_id for ws in workstations],
                        s.activity,
                        async_infer=bool(scale.enabled and scale.async_videomae),
                        phase_offset_seconds=activity_phase,
                    )
                    logger.info(
                        "VideoMAE activity ON checkpoint=%s classes=%s shared=%s async=%s phase=%.3fs",
                        s.activity.checkpoint,
                        activity_monitor.classifier.class_names,
                        self._activity_classifier is not None,
                        bool(scale.enabled and scale.async_videomae),
                        activity_phase,
                    )
                except ActivityError as exc:
                    logger.error("VideoMAE activity disabled: %s", exc)
                    activity_monitor = None
            if s.appearance_gallery.enabled:
                if s.appearance_gallery.backend == "vlm":
                    gallery = AppearanceGallery(s.appearance_gallery)
                else:
                    resolver = PersonIdentityResolver(s.appearance_gallery)
                    resolver.enroll_requires_desk = True
                logger.info(
                    "Same-day appearance gallery ON backend=%s model=%s — anonymous P## IDs",
                    s.appearance_gallery.backend, s.appearance_gallery.model,
                )
            id_engine: AppearanceGallery | PersonIdentityResolver | None = resolver or gallery
            directory = (EmployeeDirectory.from_json(s.employees_path, s.tenant_id) if s.employees_path
                         else EmployeeDirectory(s.tenant_id))
            identity = IdentityManager(s.tenant_id, directory, s.identity_min_confidence)

            managers = {ws.workstation_id: WorkstationStateManager(ws.workstation_id, class_names, s.state)
                        for ws in workstations}
            trackers = {ws.workstation_id: WorkstationTimeTracker(
                ws.workstation_id, class_names, reader.origin_utc, s.report_timezone) for ws in workstations}
            self._activity_trackers = {}
            if activity_monitor is not None:
                activity_class_names = list(activity_monitor.classifier.class_names)
                self._activity_trackers = {
                    ws.workstation_id: ActivityTimeTracker(
                        ws.workstation_id, activity_class_names, reader.origin_utc, s.report_timezone,
                    )
                    for ws in workstations
                }
            reports = ReportWriter(s.output_directory, s.camera_id, self.session_id, s.flush_interval_seconds)
            out = s.output_directory
            csvs = {
                "tracks": CsvAppender(out / f"tracks_{self.session_id}.csv", FULL_TRACK_CSV_FIELDS,
                                      s.flush_interval_seconds),
                "fusion": CsvAppender(out / f"fusion_{self.session_id}.csv", FUSION_CSV_FIELDS,
                                      s.flush_interval_seconds),
                "assoc": CsvAppender(out / f"association_events_{self.session_id}.csv", ASSOCIATION_EVENT_FIELDS,
                                     s.flush_interval_seconds),
                "seated": CsvAppender(out / f"seated_intervals_{self.session_id}.csv", SEATED_CSV_FIELDS,
                                      s.flush_interval_seconds),
                "working": CsvAppender(out / f"working_summary_{self.session_id}.csv", WORKING_SUMMARY_FIELDS,
                                       s.flush_interval_seconds),
            }
            repo = JsonlRepository(s.storage_directory, s.tenant_id)
            repo.put("sessions", {"session_id": self.session_id}, {
                "session_id": self.session_id, "camera_id": s.camera_id, "mode": "full",
                "source": s.source if "://" not in s.source else s.source.split("://")[0] + "://<redacted>",
                "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "origin_utc": reader.origin_utc.isoformat() if reader.origin_utc else None,
                "scene_map_version": scene.version, "scene_map_fingerprint": scene.fingerprint(),
                "detector_sha256": _file_sha256(s.detection.model),
                "classifier_sha256": _file_sha256(s.checkpoint),
            })
            repo.put("scene_maps", {"camera_id": s.camera_id, "version": scene.version,
                                    "fingerprint": scene.fingerprint()}, scene.to_dict())
            visualizer = Visualizer(s.camera_id, s.colors, s.draw_confidence, s.draw_duration, s.show_window,
                                    window_name="Employee presence monitoring")
            if s.save_video:
                fps = s.output_fps or (reader.fps if reader.fps else 20.0)
                result.annotated_video = out / f"annotated_full_{self.session_id}.mp4"
                video = AnnotatedVideoWriter(result.annotated_video, fps, (reader.width, reader.height))
            logger.info("Session %s full: tenant=%s camera=%s workstations=%d scene_map=v%d identity=%s",
                        self.session_id, s.tenant_id, s.camera_id, len(workstations), scene.version,
                        f"{len(directory)} employees" if len(directory) else "none (UNKNOWN)")

            base = {"tenant_id": s.tenant_id, "camera_id": s.camera_id, "session_id": self.session_id}
            ws_by_id = {w.workstation_id: w for w in workstations}
            latest_frame = AssociationFrame(0.0, {}, {}, {}, set(), {}, [])
            latest_tracks: list = []
            latest_decisions: dict[str, Any] = {}
            disagreements: dict[str, int] = {}
            unique_tracks: set[str] = set()
            assignments: dict[str, Any] = {}
            # P## -> the marked desk where that person was identified.
            person_home: dict[str, str] = {}
            recorded_tracks: set[str] = set()
            end_reason = "end_of_stream" if reader.is_live else "end_of_video"

            def handle_assoc(events: list[AssociationEvent]) -> None:
                for event in events:
                    csvs["assoc"].write({**base, "kind": event.kind, "composite_track_id": event.composite_track_id,
                                         "local_track_id": event.local_track_id,
                                         "workstation_id": event.workstation_id,
                                         "event_timestamp": f"{event.event_seconds:.3f}",
                                         "decided_timestamp": f"{event.decided_seconds:.3f}",
                                         "reason": event.reason, "score": f"{event.score:.3f}"})
                    logger.info("Association %s %s T%d %s at %.2fs (%s)", event.kind, event.workstation_id,
                                event.local_track_id, event.composite_track_id, event.event_seconds, event.reason)
                    for part in presence.apply(event):
                        write_seated(part)

            def write_seated(part: SeatedInterval) -> None:
                row = {**base, "composite_track_id": part.composite_track_id, "local_track_id": part.local_track_id,
                       "workstation_id": part.workstation_id, "employee_id": part.employee_id,
                       "start_timestamp": f"{part.start:.3f}", "end_timestamp": f"{part.end:.3f}",
                       "duration_seconds": f"{part.duration:.3f}", "end_reason": part.end_reason}
                csvs["seated"].write(row)
                key = {"session_id": self.session_id, "track": part.composite_track_id,
                       "ws": part.workstation_id, "start": round(part.start, 3), "employee": part.employee_id}
                repo.put("seated_intervals", key, row)
                if part.employee_id != UNKNOWN_EMPLOYEE:
                    repo.put("employee_intervals", key, row)

            def handle_identity(events: list[IdentityEvent]) -> None:
                for event in events:
                    presence.identity_event(event)
                    repo.put("identity_events", {"session_id": self.session_id, "track": event.composite_track_id,
                                                 "kind": event.kind, "ts": round(event.timestamp_seconds, 3),
                                                 "employee": event.employee_id},
                             {**base, "kind": event.kind, "composite_track_id": event.composite_track_id,
                              "employee_id": event.employee_id, "timestamp_seconds": event.timestamp_seconds,
                              "source": event.source, "reason": event.reason})

            def handle_ws(transitions: list[StateTransition]) -> None:
                for transition in transitions:
                    interval = trackers[transition.workstation_id].apply(transition)
                    if interval is not None:
                        reports.write_interval(interval)
                        repo.put("workstation_intervals",
                                 {"session_id": self.session_id, "ws": interval.workstation_id,
                                  "start": round(interval.start, 3)},
                                 {**base, "workstation_id": interval.workstation_id, "state": interval.state,
                                  "start_timestamp": interval.start, "end_timestamp": interval.end,
                                  "duration_seconds": interval.duration, "end_reason": interval.end_reason})
                    reports.write_event(transition)
                    repo.put("workstation_events",
                             {"session_id": self.session_id, "ws": transition.workstation_id,
                              "ts": round(transition.transition_timestamp, 3), "state": transition.new_state},
                             {**base, "workstation_id": transition.workstation_id,
                              "previous_state": transition.previous_state, "new_state": transition.new_state,
                              "transition_timestamp": transition.transition_timestamp,
                              "confirmation_timestamp": transition.confirmation_timestamp,
                              "reason": transition.reason})
                    if transition.previous_state is not None:
                        logger.info("%s: %s -> %s at %.2fs (%s)", transition.workstation_id,
                                    transition.previous_state, transition.new_state,
                                    transition.transition_timestamp, transition.reason)

            excel_path = out / f"progress_{self.session_id}.xlsx"

            def write_excel_fn(reason: str, *, working_snap=None, desk_snap=None) -> None:
                """Always-safe progress workbook (end, quit, interrupt, pause mid-video)."""
                try:
                    cfg_g = s.appearance_gallery
                    person_rows = []
                    person_desk_rows: list[dict] = []
                    if id_engine is not None:
                        for p in id_engine.people():
                            home = None
                            if hasattr(p, "home_desk"):
                                home = p.home_desk(cfg_g.desk_home_share, cfg_g.desk_home_min_seconds)
                            desk_secs = getattr(p, "desk_seconds", {}) or {}
                            pid = getattr(p, "person_id", "")
                            if not desk_secs and pid not in person_home:
                                continue
                            person_rows.append({
                                "person_id": pid,
                                "home_workstation": home or "",
                                "workstation_seconds": sum(float(v) for v in desk_secs.values()),
                                "seen_seconds": float(getattr(p, "seen_seconds", 0.0) or 0.0),
                                "samples": int(getattr(p, "samples", 0) or 0),
                                "anchor_track_T": (
                                    f"T{p.anchor_local_id}" if getattr(p, "anchor_local_id", None) else ""
                                ),
                            })
                            for ws_id, secs in desk_secs.items():
                                person_desk_rows.append({
                                    "person_id": pid,
                                    "workstation_id": str(ws_id),
                                    "seated_seconds": float(secs or 0.0),
                                })
                    working_rows = []
                    for row in (working_snap or []):
                        if row.composite_track_id not in recorded_tracks:
                            continue
                        working_rows.append({
                            "local_track_id": row.local_track_id,
                            "composite_track_id": row.composite_track_id,
                            "person_id": person_of.get(row.composite_track_id, ""),
                            "employee_id": identity.employee_of(row.composite_track_id),
                            "workstation_id": row.workstation_id or person_home.get(
                                person_of.get(row.composite_track_id, ""), ""),
                            "activity_class": row.activity_class or "",
                            "working_seconds": row.total_working_seconds,
                            "active": row.active,
                        })
                    desk_rows = []
                    for d in (desk_snap or []):
                        desk_rows.append({
                            "workstation_id": d.workstation_id,
                            "empty_seconds": d.empty_seconds,
                            "working_seconds": d.working_seconds,
                        })
                    # ResNet empty/occupied (live totals including open interval).
                    occupancy_rows = []
                    for ws_id, tr in trackers.items():
                        row = tr.summary(last_ts)
                        occupancy_rows.append({
                            "workstation_id": ws_id,
                            "empty_seconds": float(row.get("empty_seconds", 0.0) or 0.0),
                            "occupied_seconds": float(row.get("occupied_seconds", 0.0) or 0.0),
                            "unknown_seconds": float(row.get("unknown_seconds", 0.0) or 0.0),
                            "observed_seconds": float(row.get("observed_seconds", 0.0) or 0.0),
                        })
                    write_progress_workbook(
                        excel_path,
                        session_id=self.session_id,
                        tenant_id=s.tenant_id,
                        camera_id=s.camera_id,
                        end_reason=reason,
                        last_timestamp=last_ts,
                        employee_seated=presence.employee_totals(),
                        workstation_seated=presence.workstation_totals(),
                        person_rows=person_rows,
                        working_rows=working_rows,
                        desk_rows=desk_rows,
                        occupancy_rows=occupancy_rows,
                        person_desk_rows=person_desk_rows,
                    )
                    result.track_stats["progress_excel"] = str(excel_path)
                except Exception as exc:  # noqa: BLE001 — never block shutdown on report I/O
                    logger.warning("Could not write progress Excel: %s", exc)

            write_excel = write_excel_fn

            def finalize_continuity(reason: str) -> None:
                # Snapshot before finalize so mid-run pause still has live totals.
                working_snap = working_clock.snapshot(last_ts) if last_ts is not None else []
                desk_snap = working_clock.desk_snapshot(last_ts) if last_ts is not None else []
                handle_assoc(association.finalize(last_ts, reason))
                handle_identity(identity.reset(last_ts, reason))
                if id_engine is not None:
                    try:
                        id_engine.save()
                        logger.info("Saved same-day appearance gallery (%d people)", len(id_engine.people()))
                    except OSError as exc:
                        logger.warning("Could not save appearance gallery: %s", exc)
                if last_ts is not None and "working" in csvs:
                    for row in working_clock.finalize(last_ts, reason):
                        if row.composite_track_id not in recorded_tracks:
                            continue
                        csvs["working"].write({
                            **base,
                            "local_track_id": row.local_track_id,
                            "composite_track_id": row.composite_track_id,
                            "activity_class": row.activity_class or "",
                            "workstation_id": row.workstation_id or "",
                            "working_seconds": f"{row.total_working_seconds:.3f}",
                            "active": 1 if row.active else 0,
                            "reason": row.reason,
                        })
                    logger.info(
                        "Working time by track: %s",
                        {f"T{r.local_track_id}": round(r.total_working_seconds, 1)
                         for r in working_snap},
                    )
                write_excel(reason, working_snap=working_snap, desk_snap=desk_snap)

            finalizer = finalize_continuity

            def reset_continuity(at: float, reason: str) -> None:
                nonlocal latest_frame, latest_tracks, latest_working, latest_desks
                working_clock.finalize(at, reason)
                tracker.reset_session()
                if resolver is not None:
                    resolver.reset_tracks()
                person_of.clear()
                handle_assoc(association.mark_unavailable(at, reason))
                handle_identity(identity.reset(at, reason))
                latest_frame = AssociationFrame(at, {}, {}, {}, set(), {}, [])
                latest_tracks = []
                latest_working = []
                latest_desks = []

            for item in reader:
                if self._stop_requested:
                    end_reason = "interrupted"
                    break
                if isinstance(item, SourceOutage):
                    logger.warning("Source outage (%s) at %.2fs", item.reason, item.detected_seconds)
                    reset_continuity(item.detected_seconds, "source_outage")
                    for manager in managers.values():
                        handle_ws(manager.mark_unavailable(item.detected_seconds))
                    continue
                packet: FramePacket = item
                loop_started = time.monotonic()
                result.frames_seen += 1
                ts = packet.timestamp_seconds
                last_ts = ts
                if packet.after_reconnect:
                    reset_continuity(ts, "stream_reconnect")

                if packet.run_tracking:
                    update = tracker.update(packet.frame, ts, packet.frame_index)
                    for gone in update.removed:
                        handle_identity(identity.track_removed(gone.composite_track_id, ts))
                        person_of.pop(gone.composite_track_id, None)
                        if gallery is not None:
                            gallery.forget_track(gone.composite_track_id)
                    live = [t.composite_track_id for t in update.tracks]
                    for evidence in self._identity_source.poll(ts):
                        handle_identity(identity.submit(evidence, live))
                    latest_frame = association.update(ts, update.tracks, update.removed)
                    handle_assoc(latest_frame.events)
                    latest_tracks = update.tracks
                    # Same-day P## identities: frame-level resolver (reid) or legacy VLM gallery.
                    assignments = {}
                    if resolver is not None:
                        # Desk prior for returnees: tracks currently seated at a workstation.
                        seated_desks = {
                            cid: v.workstation_id for cid, v in latest_frame.tracks.items()
                            if v.workstation_id and v.state.value == "associated"
                        }
                        assignments = resolver.update(
                            packet.frame, ts, update.tracks,
                            [g.composite_track_id for g in update.removed],
                            desks=seated_desks,
                            frame_index=packet.frame_index,
                        )
                        # Empty-object / stuck P=calculating → remove box + suppress tracker.
                        for gone in tracker.suppress_tracks(resolver.consume_drops()):
                            handle_identity(identity.track_removed(gone.composite_track_id, ts))
                            person_of.pop(gone.composite_track_id, None)
                            update.removed.append(gone)
                        person_of = {cid: a.person_id for cid, a in assignments.items() if a.person_id}
                    # Observed-only occupancy: lost ghosts must not block rematch.
                    if gallery is not None:
                        observed_ids = {
                            t.composite_track_id for t in update.tracks if not t.is_predicted
                        }
                        gallery.set_observed_tracks(observed_ids)
                        fh, fw = packet.frame.shape[:2]
                        for t in update.tracks:
                            if t.is_predicted or t.lifecycle.value == "TENTATIVE":
                                continue
                            x1, y1, x2, y2 = t.bbox
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(fw, x2), min(fh, y2)
                            if x2 - x1 < 8 or y2 - y1 < 8:
                                continue
                            crop = packet.frame[y1:y2, x1:x2]
                            assignment = gallery.observe(
                                t.composite_track_id, t.local_track_id, crop, ts)
                            if assignment.person_id:
                                person_of[t.composite_track_id] = assignment.person_id
                            elif assignment.reason.startswith("enrolling"):
                                logger.debug(
                                    "Gallery pending T%s: %s", t.local_track_id, assignment.reason,
                                )
                    seated_now = {
                        cid: v.workstation_id for cid, v in latest_frame.tracks.items()
                        if v.workstation_id and v.state.value == "associated"
                    }
                    for cid, ws_id in seated_now.items():
                        recorded_tracks.add(cid)
                        pid = person_of.get(cid)
                        if pid:
                            person_home[pid] = ws_id
                    home_desks = {
                        t.composite_track_id: person_home[person_of[t.composite_track_id]]
                        for t in update.tracks
                        if person_of.get(t.composite_track_id) in person_home
                    }
                    recorded_tracks.update(home_desks)
                    zones = [
                        WorkstationOccupancy(
                            ws.workstation_id, ws.context_box,
                            managers[ws.workstation_id].snapshot().state,
                        )
                        for ws in workstations
                    ]
                    latest_working = working_clock.update(
                        ts, update.tracks, zones, update.removed, home_desks=home_desks,
                    )
                    latest_desks = working_clock.desk_snapshot(ts)
                    for ws_id in latest_frame.occupants:
                        mapper.note_person_evidence(ws_id, ts)
                    status = mapper.update(packet.frame, ts)
                    if status.layout_change_suspected and status.last_check_seconds == ts:
                        logger.error("Layout/camera change suspected: %s", "; ".join(status.reasons))
                    result.tracking_frames += 1
                    result.last_timestamp = ts
                    working_by_id = {w.composite_track_id: w for w in latest_working}
                    for t in update.tracks:
                        if t.is_predicted:
                            continue
                        v = latest_frame.tracks[t.composite_track_id]
                        pid = person_of.get(t.composite_track_id, "")
                        home = person_home.get(pid, "")
                        at_desk = v.state.value == "associated" and bool(v.workstation_id)
                        if t.composite_track_id not in recorded_tracks and not at_desk and not home:
                            continue
                        unique_tracks.add(t.composite_track_id)
                        wt = working_by_id.get(t.composite_track_id)
                        csvs["tracks"].write({
                            **base, "tracking_session_id": t.tracking_session_id, "timestamp_seconds": f"{ts:.3f}",
                            "frame_index": packet.frame_index, "local_track_id": t.local_track_id,
                            "composite_track_id": t.composite_track_id, "lifecycle": t.lifecycle.value,
                            "is_predicted": 0, "confidence": f"{t.confidence:.4f}",
                            "bbox_x1": t.bbox[0], "bbox_y1": t.bbox[1], "bbox_x2": t.bbox[2], "bbox_y2": t.bbox[3],
                            "identity_status": t.identity_status.value,
                            "activity_class": t.activity_class or "",
                            "posture": v.posture.value,
                            "raw_posture": v.raw_posture.value, "association_state": v.state.value,
                            "workstation_id": v.workstation_id or home,
                            "candidate_workstation_id": v.candidate_workstation_id or "",
                            "association_score": f"{v.score:.3f}",
                            "employee_id": identity.employee_of(t.composite_track_id),
                            "person_id": pid,
                            "reason": v.reasons[0] if v.reasons else "",
                            "working_seconds": f"{(wt.total_working_seconds if wt else 0.0):.3f}",
                            "working_active": 1 if (wt and wt.active) else 0,
                        })

                if activity_monitor is not None and (packet.run_tracking or packet.run_inference):
                    boxes = {ws.workstation_id: ws.context_box for ws in workstations}
                    # Overload sheds OPTIONAL VideoMAE only; buffers still update.
                    # Never freeze desk labels for minutes: force a heartbeat when stale.
                    run_activity = bool(packet.run_inference)
                    if run_activity and overload is not None and overload.should_shed():
                        if activity_monitor.needs_heartbeat(ts, max_age_seconds=8.0):
                            run_activity = True
                        else:
                            run_activity = False
                    activity_monitor.observe(
                        ts, packet.frame, boxes, run_inference=run_activity,
                    )
                    self._activity_by_ws = activity_monitor.snapshot()
                    for ws_id, activity_tracker in self._activity_trackers.items():
                        pred = self._activity_by_ws.get(ws_id)
                        activity_tracker.observe(ts, None if pred is None else pred.predicted_class)

                yolo_by_ws: dict[str, str] = {}
                for composite_id, view in latest_frame.tracks.items():
                    if view.state.value != "associated" or not view.workstation_id:
                        continue
                    track = next((t for t in latest_tracks if t.composite_track_id == composite_id), None)
                    if track is not None and track.activity_class:
                        yolo_by_ws[view.workstation_id] = track.activity_class
                self._yolo_by_ws = yolo_by_ws

                if packet.run_inference:
                    crops = {ws.workstation_id: packet.frame[ws.context_box[1]:ws.context_box[3],
                                                             ws.context_box[0]:ws.context_box[2]]
                             for ws in workstations if crop_ok[ws.workstation_id]}
                    predictions = classifier.classify(crops) if crops else {}
                    for ws in workstations:
                        ws_id = ws.workstation_id
                        prediction = predictions.get(ws_id)
                        if prediction is not None:
                            reports.write_prediction(ts, packet.wall_clock_utc, ws_id, prediction)
                        decision = fusion.decide(ws, latest_frame, prediction)
                        if decision.observed_class is None:
                            handle_ws(managers[ws_id].observe_missing(ts))
                        else:
                            handle_ws(managers[ws_id].observe(ts, decision.observed_class, decision.confidence))
                            trackers[ws_id].observe(ts)
                        latest_decisions[ws_id] = decision
                        state_after = managers[ws_id].snapshot().state
                        activity = self._activity_by_ws.get(ws_id)
                        row = {**base, "timestamp_seconds": f"{ts:.3f}", "workstation_id": ws_id,
                               "observed_class": decision.observed_class or "",
                               "confidence": f"{decision.confidence:.3f}", "evidence_source": decision.source,
                               "reason": decision.reason, "occupant_track_id": decision.occupant_track_id or "",
                               "resnet_class": decision.resnet_class or "",
                               "resnet_confidence": "" if decision.resnet_confidence is None
                               else f"{decision.resnet_confidence:.3f}",
                               "disagreement": decision.disagreement or "", "state_after": state_after,
                               "activity_class": "" if activity is None else activity.predicted_class,
                               "activity_confidence": "" if activity is None else f"{activity.confidence:.3f}",
                               "yolo_activity": yolo_by_ws.get(ws_id, ""),
                               }
                        csvs["fusion"].write(row)
                        if decision.disagreement:
                            disagreements[decision.disagreement] = disagreements.get(decision.disagreement, 0) + 1
                            repo.put("fusion_disagreements",
                                     {"session_id": self.session_id, "ws": ws_id, "ts": round(ts, 3)}, row)
                    result.inference_frames += 1

                if s.save_video or visualizer.show_window or self._on_frame is not None:
                    ws_views = []
                    for ws in workstations:
                        snap = managers[ws.workstation_id].snapshot()
                        decision = latest_decisions.get(ws.workstation_id)
                        activity = self._activity_by_ws.get(ws.workstation_id)
                        ws_views.append(FullWorkstationView(
                            ws.workstation_id, ws.seat_box, ws.anchor, snap.state,
                            trackers[ws.workstation_id].current_duration(ts),
                            decision.source if decision else "-", decision.reason if decision else "waiting",
                            ws.status == STATUS_NEEDS_REVIEW, decision.disagreement if decision else None,
                            activity_class=None if activity is None else activity.predicted_class,
                            activity_confidence=None if activity is None else activity.confidence,
                            yolo_activity=yolo_by_ws.get(ws.workstation_id),
                        ))
                    working_by_id = {w.composite_track_id: w for w in latest_working}
                    track_views = []
                    for t in latest_tracks:
                        if t.is_predicted:
                            continue  # no empty ghost boxes on the live overlay
                        v = latest_frame.tracks.get(t.composite_track_id)
                        if v is None:
                            continue
                        wt = working_by_id.get(t.composite_track_id)
                        pid = person_of.get(t.composite_track_id)
                        a = assignments.get(t.composite_track_id) if assignments else None
                        if a is not None and a.state == "drop":
                            continue
                        # Overlay: P=<id>, P=calculating (active resolve), or P=- (warm-up).
                        if pid is None:
                            if a is not None and a.state == "warming":
                                pid = "-"
                            else:
                                pid = "calculating"
                        stable_t = (
                            (a.display_track_id if a is not None else None)
                            or (
                                id_engine.display_track_id(pid)
                                if id_engine is not None and pid and not str(pid).startswith("calculating")
                                else None
                            )
                        )
                        desk_activity = self._activity_by_ws.get(v.workstation_id) if v.workstation_id else None
                        track_views.append(FullTrackView(
                            t, v.state.value, v.workstation_id, v.candidate_workstation_id, v.posture.value,
                            identity.employee_of(t.composite_track_id),
                            presence.open_duration(t.composite_track_id, ts), tuple(v.reasons),
                            activity_class=t.activity_class,
                            working_seconds=wt.total_working_seconds if wt else 0.0,
                            working_active=bool(wt and wt.active),
                            person_id=pid,
                            display_track_id=stable_t,
                            videomae_class=None if desk_activity is None else desk_activity.predicted_class,
                            videomae_confidence=None if desk_activity is None else desk_activity.confidence,
                        ))
                    scene_status = mapper.status
                    lines = [f"Source time: {ts:.1f}s   Scene map v{scene.version}"
                             + ("   LAYOUT CHANGE SUSPECTED" if scene_status.layout_change_suspected else "")]
                    if packet.wall_clock_utc is not None:
                        lines.append(f"UTC: {packet.wall_clock_utc:%Y-%m-%d %H:%M:%S}")
                    desk_rows = [
                        DeskTimeOverlayRow(
                            d.workstation_id, d.empty_seconds, d.working_seconds, d.current_mode,
                        )
                        for d in latest_desks
                    ]
                    annotated = visualizer.draw_full(
                        packet.frame, ws_views, track_views, lines, self.debug_overlay,
                        desk_rows=desk_rows,
                    )
                    if self._on_frame is not None:
                        self._on_frame(annotated)
                    if video is not None:
                        video.write(annotated)
                    if visualizer.show_window and visualizer.show(annotated) == "quit":
                        end_reason = "user_stop"
                        break
                self._emit_live_progress(ts, workstations, managers, trackers)
                if overload is not None:
                    overload.record(time.monotonic() - loop_started)
                if max_frames is not None and result.tracking_frames >= max_frames:
                    end_reason = "max_frames"
                    break
            else:
                if reader.source_lost:
                    end_reason = "source_lost"
            result.end_reason = end_reason
            result.track_stats = {"unique_tracks": len(unique_tracks), "disagreements": disagreements,
                                  "scene_status": mapper.status.reasons}
        except KeyboardInterrupt:
            result.end_reason = "interrupted"
            logger.warning("Interrupted; finalizing outputs")
        finally:
            # Release annotated video BEFORE gallery/Excel/CSV finalize. Those steps can take
            # many seconds (or hang on a second Ctrl+C); OpenCV MP4 is unplayable until release().
            if video is not None:
                video.release()
            if visualizer is not None:
                visualizer.close()
            if activity_monitor is not None and activity_monitor is not self._activity_monitor:
                activity_monitor.close()
            try:
                if finalizer is not None and last_ts is not None:
                    finalizer(result.end_reason)
            finally:
                # Guarantee Excel exists even if finalizer partially failed / video paused mid-run.
                if (
                    write_excel is not None
                    and working_clock is not None
                    and "progress_excel" not in result.track_stats
                    and last_ts is not None
                ):
                    try:
                        write_excel(
                            result.end_reason or "interrupted",
                            working_snap=working_clock.snapshot(last_ts),
                            desk_snap=working_clock.desk_snapshot(last_ts),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Excel fallback failed: %s", exc)
            if reports is not None:
                self._finalize(result, trackers, presence, reports, repo, workstations, managers)
                reports.close()
            for appender in csvs.values():
                appender.close()
            if repo is not None:
                repo.close()
            if resolver is not None:
                resolver.close()
            reader.close()
        return result

    def _finalize(self, result: RunResult, trackers: dict[str, WorkstationTimeTracker],
                  presence: PresenceTimeTracker, reports: ReportWriter, repo: JsonlRepository | None,
                  workstations: list[Any], managers: dict[str, Any]) -> None:
        s = self.settings
        for tracker in trackers.values():
            interval = tracker.finalize(result.end_reason)
            if interval is not None:
                reports.write_interval(interval)
                if repo is not None:
                    repo.put("workstation_intervals",
                             {"session_id": self.session_id, "ws": interval.workstation_id,
                              "start": round(interval.start, 3)},
                             {"camera_id": s.camera_id, "session_id": self.session_id,
                              "workstation_id": interval.workstation_id, "state": interval.state,
                              "start_timestamp": interval.start, "end_timestamp": interval.end,
                              "duration_seconds": interval.duration, "end_reason": interval.end_reason})
            result.summaries[tracker.workstation_id] = tracker.summary()
        extra = sorted({st for t in trackers.values() for st in t.states} - set(STANDARD_STATES))
        if trackers:
            reports.write_summaries(((ws, row) for ws, row in result.summaries.items()), extra)
            if result.last_timestamp is not None:
                try:
                    self._publish_day_totals(
                        result.last_timestamp, workstations, managers, trackers, closed=True,
                    )
                except Exception:
                    logger.warning("Could not store the final daily totals", exc_info=True)
        path = s.output_directory / f"employee_summary_{self.session_id}.csv"
        appender = CsvAppender(path, EMPLOYEE_SUMMARY_FIELDS, 0.0)
        try:
            for employee, per_ws in sorted(presence.employee_totals().items()):
                for ws_id, seconds in sorted(per_ws.items()):
                    appender.write({"tenant_id": s.tenant_id, "camera_id": s.camera_id,
                                    "session_id": self.session_id, "employee_id": employee,
                                    "workstation_id": ws_id, "seated_seconds": f"{seconds:.3f}"})
        finally:
            appender.close()
        result.track_stats["seated_seconds_by_workstation"] = {
            k: round(v, 1) for k, v in sorted(presence.workstation_totals().items())}
        logger.info("Session %s ended (%s): %d frames, %d tracked, %d classified, last t=%s",
                    self.session_id, result.end_reason, result.frames_seen, result.tracking_frames,
                    result.inference_frames, result.last_timestamp)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Employee presence monitoring (occupancy / tracking)")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--mode", choices=("occupancy", "tracking", "full"),
                        help="Pipeline mode (overrides pipeline.mode)")
    parser.add_argument("--source", help="MP4 path, RTSP URL, or camera index")
    parser.add_argument("--checkpoint", help="ResNet-18 checkpoint (occupancy/full)")
    parser.add_argument("--detector", help="YOLO weights path (overrides detection.model)")
    parser.add_argument("--rois", help="Workstation ROI JSON (occupancy/full)")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--camera-id")
    parser.add_argument("--tenant-id")
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270))
    parser.add_argument("--interval", type=float, help="Seconds between ResNet classifications")
    parser.add_argument("--device", help="auto | cpu | cuda | cuda:N")
    parser.add_argument("--recording-start-utc", help="ISO-8601 UTC start of an MP4 recording")
    parser.add_argument("--show", dest="show", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-video", dest="save_video", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-frames", type=int, help="Stop after N classified/tracked frames")
    parser.add_argument("--scene-map", help="Workstation scene map JSON (full mode)")
    parser.add_argument("--debug-overlay", action="store_true",
                        help="Full mode: draw association/fusion reasons on the video")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args(argv)


def overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    o: dict[str, dict[str, Any]] = {}

    def put(section: str, key: str, value: Any) -> None:
        if value is not None:
            o.setdefault(section, {})[key] = value

    put("pipeline", "mode", args.mode)
    put("tenant", "tenant_id", args.tenant_id)
    put("camera", "source", args.source)
    put("camera", "camera_id", args.camera_id)
    put("camera", "rotation", args.rotation)
    put("camera", "recording_start_utc", args.recording_start_utc)
    put("model", "checkpoint", args.checkpoint)
    put("model", "device", args.device)
    put("detection", "model", args.detector)
    put("detection", "device", args.device)
    put("rois", "config", args.rois)
    put("scene", "map", getattr(args, "scene_map", None))
    put("inference", "interval_seconds", args.interval)
    put("visualization", "show_window", args.show)
    put("visualization", "save_video", args.save_video)
    put("output", "directory", args.output_dir)
    put("logging", "level", args.log_level)
    return o


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings(args.config, overrides_from_args(args))
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 2
    if args.max_frames is not None and args.max_frames <= 0:
        logger.error("--max-frames must be positive")
        return 2
    session_id = new_session_id()
    log_path = setup_logging(settings.log_directory, settings.log_level, session_id)
    logger.info("Log file: %s", log_path)
    logger.info("Pipeline mode: %s  tenant: %s  camera: %s",
                settings.pipeline_mode, settings.tenant_id, settings.camera_id)

    app: MonitoringApp | TrackingApp | FullApp
    if settings.pipeline_mode == "tracking":
        app = TrackingApp(settings, session_id=session_id)
    elif settings.pipeline_mode == "full":
        app = FullApp(settings, session_id=session_id, debug_overlay=args.debug_overlay)
    else:
        app = MonitoringApp(settings, session_id=session_id)

    def on_stop_signal(signum: int, _frame: Any) -> None:
        # First Ctrl+C / SIGTERM: finish the current frame, release video, then finalize.
        # Restore default SIGINT so a second Ctrl+C can still force-exit if shutdown hangs.
        logger.warning("Received signal %d; stopping after the current frame (annotated video will be saved)",
                       signum)
        app.request_stop()
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGTERM, on_stop_signal)
    signal.signal(signal.SIGINT, on_stop_signal)
    try:
        result = app.run(args.max_frames)
    except (CheckpointError, ROIConfigError, VideoSourceError, CsvSchemaError, ConfigError, TrackerError,
            SceneMapError, IdentityError, RepositoryError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 2

    logger.info("Outputs in %s (session %s)%s", settings.output_directory, result.session_id,
                f"; annotated video {result.annotated_video}" if result.annotated_video else "")
    if result.summaries:
        for ws_id, row in result.summaries.items():
            logger.info("  %s occupied=%.1fs empty=%.1fs unknown=%.1fs transitions=%s", ws_id,
                        row.get("occupied_seconds", 0.0), row.get("empty_seconds", 0.0),
                        row.get("unknown_seconds", 0.0), row.get("transition_count", 0))
    if result.track_stats and settings.pipeline_mode == "tracking":
        logger.info("  tracks csv: %s", result.track_stats.get("tracks_csv"))
        logger.info("  unique local track IDs: %s (not employee IDs)",
                    result.track_stats.get("unique_local_track_ids"))
    elif result.track_stats:
        logger.info("  seated seconds by workstation (anonymous tracks): %s",
                    result.track_stats.get("seated_seconds_by_workstation"))
        logger.info("  ResNet/person disagreements: %s", result.track_stats.get("disagreements"))
    return 0 if result.end_reason not in ("error", "source_lost") else 1


if __name__ == "__main__":
    sys.exit(main())
