"""Capture, inference, and state run on separate threads.

A single loop that reads RTSP, runs a model, and writes SQL will stall the camera
whenever the database or the GPU hiccups.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

from app.analytics import TimeSpan, away_seconds, longest_away_seconds, spans_on_day, total_seconds
from app.config import ROOT_DIR, AppConfig, CameraConfig, load_config
from app.database.repository import Repository
from app.database.session import init_db, make_engine, make_session_factory, session_scope
from app.events.event_manager import EventManager
from app.events.event_types import EventType
from app.events.evidence import EvidenceBuffer
from app.face.gallery import FaceGallery
from app.face.recognizer import InsightFaceRecognizer
from app.health.camera_health import CameraStatus
from app.health.inference_health import InferenceHealthMonitor
from app.pipelines.entrance import EntranceTick, run_entrance
from app.pipelines.workspace import WorkspaceTick, run_workspace
from app.states.attendance_state import AttendanceState, AttendanceStateManager
from app.states.desk_state import DeskStateManager
from app.states.phone_state import PhoneState
from app.states.runtime import EmployeeRuntimeState
from app.streams.rtsp_reader import RTSPReader
from app.tracking.track_manager import TrackManager
from app.tracking.tracker import UltralyticsByteTracker
from app.utils.timestamps import format_clock, utc_now
from app.zones.desk_manager import DeskManager
from app.zones.line_crossing import LineCrossingDetector

logger = logging.getLogger(__name__)


class PipelineService:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.repository = Repository()
        self.gallery = FaceGallery()
        self.attendance = AttendanceStateManager()
        self.desk_states = DeskStateManager(self.config.desk_state)
        self.desk_zones = DeskManager(self.config.desks)
        self.events = EventManager(listener=self._persist_event)
        self.evidence = EvidenceBuffer(
            root=ROOT_DIR / "data" / "evidence",
            enabled=self.config.evidence.enabled,
            pre_seconds=self.config.evidence.pre_seconds,
            post_seconds=self.config.evidence.post_seconds,
            retention_days=self.config.evidence.retention_days,
        )
        self.runtime: dict[str, EmployeeRuntimeState] = {}
        self.camera_monitors: dict[str, InferenceHealthMonitor] = {}
        self._readers: dict[str, RTSPReader] = {}
        self._observations: queue.Queue = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._engine = None
        self._session_factory = None
        self._db_ready = False
        self._last_camera_status: dict[str, CameraStatus] = {}
        self._gallery_loaded_at = 0.0

    def start(self) -> None:
        self._connect_database()
        self._restore_sessions()
        self._reload_gallery()
        if not self.config.run_pipeline:
            logger.info("pipeline threads disabled", extra={"event": "PIPELINE_DISABLED"})
            return
        for camera in self.config.cameras:
            if not camera.enabled:
                continue
            self._start_camera(camera)
        state_thread = threading.Thread(target=self._state_loop, name="state-engine", daemon=True)
        state_thread.start()
        self._threads.append(state_thread)

    def stop(self) -> None:
        self._stop.set()
        for reader in self._readers.values():
            reader.stop()
        for thread in self._threads:
            thread.join(timeout=3)

    def _connect_database(self) -> None:
        try:
            self._engine = make_engine(self.config.database_url)
            init_db(self._engine)
            self._session_factory = make_session_factory(self._engine)
            with session_scope(self._session_factory) as db:
                self.repository.sync_configuration(db, self.config)
            self._db_ready = True
        except Exception:
            self._db_ready = False
            logger.exception("database unavailable; events stay in memory until it returns", extra={"event": "DATABASE_DOWN"})

    def _restore_sessions(self) -> None:
        if not self._db_ready or self._session_factory is None:
            return
        now = utc_now()
        try:
            with session_scope(self._session_factory) as db:
                for record in self.repository.open_attendance_sessions(db):
                    self.attendance.restore(record)
                for record in self.repository.open_desk_sessions(db):
                    self.desk_states.restore(record, now)
            self._refresh_runtime()
        except Exception:
            logger.exception("session restore failed", extra={"event": "RESTORE_FAILED"})

    def _reload_gallery(self) -> None:
        if not self._db_ready or self._session_factory is None:
            return
        try:
            with session_scope(self._session_factory) as db:
                templates, names = self.repository.load_gallery(db)
            self.gallery.replace(templates, names)
            self._gallery_loaded_at = time.monotonic()
        except Exception:
            logger.exception("gallery reload failed", extra={"event": "GALLERY_RELOAD_FAILED"})

    def _start_camera(self, camera: CameraConfig) -> None:
        reader = RTSPReader(
            camera_id=camera.camera_id,
            source=camera.rtsp_url,
            buffer_size=self.config.stream.buffer_size,
            reconnect_base_seconds=self.config.stream.reconnect_base_seconds,
            reconnect_max_seconds=self.config.stream.reconnect_max_seconds,
            offline_after_seconds=self.config.stream.offline_after_seconds,
            open_timeout_seconds=self.config.stream.open_timeout_seconds,
            flush_on_connect_seconds=self.config.stream.flush_on_connect_seconds,
        )
        self._readers[camera.camera_id] = reader
        self.camera_monitors[camera.camera_id] = InferenceHealthMonitor(camera.camera_id)
        reader.start()
        worker = threading.Thread(
            target=self._inference_loop,
            args=(camera,),
            name=f"infer-{camera.camera_id}",
            daemon=True,
        )
        worker.start()
        self._threads.append(worker)

    def _inference_loop(self, camera: CameraConfig) -> None:
        reader = self._readers[camera.camera_id]
        health = self.camera_monitors[camera.camera_id]
        interval = 1.0 / max(camera.process_fps, 0.1)
        last_frame = -1
        tracker: UltralyticsByteTracker | None = None
        recognizer: InsightFaceRecognizer | None = None
        identities = TrackManager(
            self.config.identity.min_observations,
            self.config.identity.window_seconds,
        )
        line = None
        if camera.type == "entrance" and camera.entrance_line is not None:
            line = LineCrossingDetector(
                camera.entrance_line.p1,
                camera.entrance_line.p2,
                camera.inside_direction,
                self.config.crossing.debounce_seconds,
                self.config.crossing.hysteresis_px,
            )
        elif camera.type == "entrance":
            logger.error(
                "entrance camera has no line; crossings disabled",
                extra={"camera": camera.camera_id, "event": "LINE_MISSING"},
            )
        while not self._stop.is_set():
            started = time.monotonic()
            packet = reader.latest()
            if packet is None or packet.frame_number == last_frame:
                self._stop.wait(0.01)
                continue
            last_frame = packet.frame_number
            if self.evidence.enabled:
                self.evidence.push(camera.camera_id, packet.frame, packet.received_timestamp)
            try:
                if tracker is None:
                    tracker = self._build_tracker(camera)
                if camera.type == "entrance":
                    if recognizer is None:
                        recognizer = InsightFaceRecognizer(
                            self.gallery,
                            self.config.face_quality,
                            self.config.match,
                            self.config.models,
                        )
                    tick = run_entrance(packet, tracker, recognizer, identities, line)
                else:
                    tick = run_workspace(packet, tracker, self.desk_zones, self.config.desk_state)
                self._put_observation(tick)
                health.mark_ok()
                reader.health.mark_processed(packet.received_timestamp)
            except FileNotFoundError as exc:
                health.mark_error(str(exc))
                logger.error(
                    "inference weights missing",
                    extra={"camera": camera.camera_id, "event": "MODEL_MISSING", "detail": str(exc)},
                )
                self._stop.wait(5)
            except Exception as exc:
                health.mark_error(str(exc))
                logger.exception(
                    "inference failed",
                    extra={"camera": camera.camera_id, "event": "INFERENCE_ERROR"},
                )
            elapsed = time.monotonic() - started
            if elapsed < interval:
                self._stop.wait(interval - elapsed)

    def _build_tracker(self, camera: CameraConfig) -> UltralyticsByteTracker:
        if camera.type == "entrance":
            model_path = self.config.models.entrance_person_model
            class_filter = {self.config.models.person_class_name.lower()}
        else:
            model_path = camera.detector_model or self.config.models.workspace_model
            if not Path(model_path).is_absolute():
                model_path = str(ROOT_DIR / model_path)
            class_filter = None
        if not Path(model_path).is_absolute() and model_path != "yolov8n.pt":
            model_path = str((ROOT_DIR / model_path).resolve())
        return UltralyticsByteTracker(
            model_path=model_path,
            image_size=self.config.models.yolo_image_size,
            confidence=self.config.desk_state.min_detection_confidence,
            class_filter=class_filter,
            tracker_config=self.config.models.tracker,
        )

    def _put_observation(self, tick: object) -> None:
        try:
            self._observations.put(tick, timeout=0.2)
        except queue.Full:
            try:
                self._observations.get_nowait()
            except queue.Empty:
                pass
            try:
                self._observations.put_nowait(tick)
            except queue.Full:
                logger.warning("observation queue full; tick dropped", extra={"event": "QUEUE_DROP"})

    def _state_loop(self) -> None:
        while not self._stop.is_set():
            if time.monotonic() - self._gallery_loaded_at > 60:
                self._reload_gallery()
            self._poll_camera_health()
            try:
                tick = self._observations.get(timeout=0.2)
            except queue.Empty:
                self.evidence.poll()
                continue
            try:
                if isinstance(tick, EntranceTick):
                    self._apply_entrance(tick)
                elif isinstance(tick, WorkspaceTick):
                    self._apply_workspace(tick)
                self._refresh_runtime()
            except Exception:
                logger.exception("state update failed", extra={"event": "STATE_ERROR"})
            self.evidence.poll()

    def _apply_entrance(self, tick: EntranceTick) -> None:
        for crossing, employee_id, confidence in tick.crossings:
            if employee_id:
                self.attendance.mark_identity_seen(employee_id)
            events = self.attendance.on_crossing(
                employee_id=employee_id,
                direction=crossing.direction,
                timestamp=crossing.timestamp,
                camera_id=tick.camera_id,
                confidence=confidence or None,
                track_id=crossing.track_id,
            )
            for event in events:
                self.events.publish(event)
                self.evidence.on_event(event)
            if employee_id:
                runtime = self.runtime.setdefault(employee_id, EmployeeRuntimeState(employee_id))
                runtime.last_face_time = tick.timestamp
                runtime.last_seen_time = tick.timestamp
                runtime.current_camera_id = tick.camera_id
                runtime.current_track_id = crossing.track_id
                runtime.identity_confidence = confidence

    def _apply_workspace(self, tick: WorkspaceTick) -> None:
        camera_ok = self._camera_is_ok(tick.camera_id)
        for presence in tick.presence:
            events = self.desk_states.observe(
                employee_id=presence.employee_id,
                desk_id=presence.desk_id,
                present=presence.present,
                timestamp=tick.timestamp,
                confidence=presence.confidence,
                camera_id=tick.camera_id,
                camera_ok=camera_ok,
            )
            for event in events:
                self.events.publish(event)
                self.evidence.on_event(event)
            runtime = self.runtime.setdefault(presence.employee_id, EmployeeRuntimeState(presence.employee_id))
            runtime.desk_id = presence.desk_id
            runtime.current_camera_id = tick.camera_id
            if presence.present:
                runtime.last_desk_detection_time = tick.timestamp
                runtime.last_seen_time = tick.timestamp
                runtime.desk_confidence = presence.confidence

    def _camera_is_ok(self, camera_id: str) -> bool:
        reader = self._readers.get(camera_id)
        if reader is None:
            return False
        status = reader.health_snapshot().connection_status
        return status in {CameraStatus.ONLINE, CameraStatus.DEGRADED}

    def _poll_camera_health(self) -> None:
        now = utc_now()
        for camera_id, reader in self._readers.items():
            snapshot = reader.health_snapshot()
            previous = self._last_camera_status.get(camera_id)
            if previous == snapshot.connection_status:
                continue
            self._last_camera_status[camera_id] = snapshot.connection_status
            camera = self.config.camera(camera_id)
            employee_ids = [desk.employee_id for desk in self.config.desks_for(camera_id)]
            became_down = snapshot.connection_status in {CameraStatus.OFFLINE, CameraStatus.RECONNECTING}
            became_up = snapshot.connection_status in {CameraStatus.ONLINE, CameraStatus.DEGRADED}
            was_up = previous in {CameraStatus.ONLINE, CameraStatus.DEGRADED}
            was_down = previous in {CameraStatus.OFFLINE, CameraStatus.RECONNECTING}
            if became_down:
                if camera is not None and camera.type == "workspace":
                    self.desk_states.camera_failed(employee_ids)
                if was_up and camera is not None and camera.type == "entrance":
                    for event in self.attendance.camera_failed(now, camera_id):
                        self.events.publish(event)
                if was_up:
                    self._record_health(snapshot)
            elif became_up and was_down:
                if camera is not None and camera.type == "workspace":
                    self.desk_states.camera_restored(employee_ids, now)
                if camera is not None and camera.type == "entrance":
                    for event in self.attendance.camera_restored(now, camera_id):
                        self.events.publish(event)
                self._record_health(snapshot)

    def _record_health(self, snapshot) -> None:
        if not self._db_ready or self._session_factory is None:
            return
        try:
            with session_scope(self._session_factory) as db:
                self.repository.record_camera_health(
                    db,
                    camera_id=snapshot.camera_id,
                    status=snapshot.connection_status.value,
                    fps=snapshot.fps,
                    last_frame_time=snapshot.last_frame_time,
                    timestamp=utc_now(),
                )
        except Exception:
            self._db_ready = False
            logger.exception("camera health write failed", extra={"event": "DATABASE_DOWN"})

    def _persist_event(self, event) -> None:
        if not self._db_ready or self._session_factory is None:
            self._connect_database()
        if not self._db_ready or self._session_factory is None:
            return
        try:
            with session_scope(self._session_factory) as db:
                self.repository.apply_event(db, event)
        except Exception:
            self._db_ready = False
            logger.exception("event persist failed", extra={"event": "DATABASE_DOWN", "employee": event.employee_id})

    def _refresh_runtime(self) -> None:
        codes = {desk.employee_id for desk in self.config.desks}
        codes.update(self.runtime)
        for employee_id in codes:
            runtime = self.runtime.setdefault(employee_id, EmployeeRuntimeState(employee_id))
            runtime.attendance_state = self.attendance.state_of(employee_id)
            runtime.desk_state = self.desk_states.state_of(employee_id)
            runtime.phone_state = PhoneState.NO_PHONE
            attendance_session = self.attendance.session_of(employee_id)
            if attendance_session is not None and attendance_session.exit_time is None:
                runtime.current_attendance_session_id = attendance_session.session_id
            else:
                runtime.current_attendance_session_id = None
            desk_session = self.desk_states.session_of(employee_id)
            if desk_session is not None and desk_session.end_time is None:
                runtime.current_desk_session_id = desk_session.session_id
                runtime.desk_id = desk_session.desk_id
            else:
                runtime.current_desk_session_id = None
            runtime.desk_confidence = self.desk_states.confidence_of(employee_id)

    def summary_for(self, employee_id: str, day, db) -> dict:
        now = utc_now()
        zone = self.config.local_timezone
        attendance_rows = self.repository.attendance_sessions_for(db, employee_id)
        desk_rows = self.repository.desk_sessions_for(db, employee_id)
        office_spans = spans_on_day(
            [TimeSpan(row.entry_time, row.exit_time) for row in attendance_rows],
            day,
            zone,
            now,
        )
        desk_spans = spans_on_day(
            [TimeSpan(row.start_time, row.end_time) for row in desk_rows],
            day,
            zone,
            now,
        )
        from datetime import datetime, time
        from zoneinfo import ZoneInfo

        from app.utils.timestamps import to_local

        day_start = datetime.combine(day, time.min, ZoneInfo(zone))
        absences = self.repository.count_events(db, employee_id, EventType.DESK_LEFT.value, day_start)
        day_rows = [
            row
            for row in attendance_rows
            if to_local(row.entry_time, zone).date() == day
        ]
        still_inside = any(row.exit_time is None for row in day_rows)
        closed_exits = [row.exit_time for row in day_rows if row.exit_time is not None]
        runtime = self.runtime.get(employee_id)
        return {
            "employee_id": employee_id,
            "date": day.isoformat(),
            "first_entry": format_clock(min(row.entry_time for row in day_rows), zone) if day_rows else None,
            "last_exit": None if still_inside or not closed_exits else format_clock(max(closed_exits), zone),
            "total_office_seconds": total_seconds(office_spans),
            "total_desk_seconds": total_seconds(desk_spans),
            "total_away_seconds": away_seconds(office_spans, desk_spans),
            "number_of_desk_absences": absences,
            "longest_away_seconds": longest_away_seconds(office_spans, desk_spans),
            "desk_status": runtime.desk_state.value if runtime else "UNKNOWN",
            "attendance_status": runtime.attendance_state.value if runtime else AttendanceState.UNKNOWN.value,
        }
