"""Persistence for sessions, templates, and events. No recognition rules live here."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import AppConfig
from app.database.models import (
    AccessAudit,
    AttendanceSession,
    Camera,
    CameraHealthRow,
    Desk,
    DeskSession,
    Employee,
    EventRow,
    FaceTemplate,
)
from app.events.event_types import Event, EventType
from app.states.attendance_state import AttendanceSessionRecord
from app.states.desk_state import DeskSessionRecord
from app.utils.timestamps import utc_now


class Repository:
    def sync_configuration(self, db: Session, config: AppConfig) -> None:
        now = utc_now()
        for camera in config.cameras:
            row = db.get(Camera, camera.camera_id)
            if row is None:
                db.add(
                    Camera(
                        id=camera.camera_id,
                        name=camera.camera_id,
                        type=camera.type,
                        location=camera.location,
                        enabled=camera.enabled,
                        created_at=now,
                    )
                )
            else:
                row.type = camera.type
                row.enabled = camera.enabled
                row.location = camera.location
        for desk in config.desks:
            row = db.get(Desk, desk.desk_id)
            polygon = [list(point) for point in desk.polygon]
            if row is None:
                db.add(
                    Desk(
                        id=desk.desk_id,
                        camera_id=desk.camera_id,
                        employee_id=desk.employee_id,
                        polygon=polygon,
                    )
                )
            else:
                row.camera_id = desk.camera_id
                row.employee_id = desk.employee_id
                row.polygon = polygon
            self.ensure_employee(db, desk.employee_id, desk.employee_id, desk.desk_id)
        db.flush()

    def ensure_employee(
        self,
        db: Session,
        employee_code: str,
        name: str,
        assigned_desk_id: str | None = None,
    ) -> Employee:
        row = db.scalar(select(Employee).where(Employee.employee_code == employee_code))
        now = utc_now()
        if row is None:
            row = Employee(
                employee_code=employee_code,
                name=name,
                active=True,
                assigned_desk_id=assigned_desk_id,
                created_at=now,
                updated_at=now,
            )
            db.add(row)
            db.flush()
            return row
        if name and name != employee_code:
            row.name = name
        if assigned_desk_id:
            row.assigned_desk_id = assigned_desk_id
        row.updated_at = now
        return row

    def add_template(self, db: Session, employee_code: str, embedding: np.ndarray, quality_score: float) -> None:
        employee = db.scalar(select(Employee).where(Employee.employee_code == employee_code))
        if employee is None:
            raise ValueError(f"Unknown employee {employee_code}")
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        db.add(
            FaceTemplate(
                employee_id=employee.id,
                embedding=vector.tobytes(),
                dim=int(vector.size),
                quality_score=float(quality_score),
                created_at=utc_now(),
            )
        )

    def load_gallery(self, db: Session) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        rows = db.execute(
            select(Employee.employee_code, Employee.name, FaceTemplate.embedding, FaceTemplate.dim).join(
                FaceTemplate, FaceTemplate.employee_id == Employee.id
            ).where(Employee.active.is_(True))
        ).all()
        grouped: dict[str, list[np.ndarray]] = {}
        names: dict[str, str] = {}
        for code, name, blob, dim in rows:
            vector = np.frombuffer(blob, dtype=np.float32)
            if vector.size != int(dim):
                continue
            grouped.setdefault(code, []).append(vector.copy())
            names[code] = name
        templates = {code: np.stack(vectors, axis=0) for code, vectors in grouped.items() if vectors}
        return templates, names

    def list_employees(self, db: Session) -> list[Employee]:
        return list(db.scalars(select(Employee).order_by(Employee.employee_code)).all())

    def get_employee(self, db: Session, employee_code: str) -> Employee | None:
        return db.scalar(select(Employee).where(Employee.employee_code == employee_code))

    def apply_event(self, db: Session, event: Event) -> None:
        db.flush()
        existing = db.scalar(select(EventRow).where(EventRow.event_id == event.event_id))
        if existing is not None:
            return
        db.add(
            EventRow(
                event_id=event.event_id,
                event_type=event.event_type.value,
                employee_id=event.employee_id,
                camera_id=event.camera_id,
                timestamp=event.timestamp,
                confidence=event.confidence,
                metadata_json=event.metadata,
            )
        )
        db.flush()
        if event.event_type is EventType.EMPLOYEE_ENTERED and event.employee_id:
            self._open_attendance(db, event)
        elif event.event_type is EventType.EMPLOYEE_EXITED and event.employee_id:
            self._close_attendance(db, event)
        elif event.event_type is EventType.DESK_ENTERED and event.employee_id:
            self._open_desk(db, event)
        elif event.event_type is EventType.DESK_LEFT and event.employee_id:
            self._close_desk(db, event)

    def _open_attendance(self, db: Session, event: Event) -> None:
        open_row = db.scalar(
            select(AttendanceSession).where(
                AttendanceSession.employee_id == event.employee_id,
                AttendanceSession.status == "open",
            )
        )
        if open_row is not None:
            return
        key = str(event.metadata.get("session_id") or event.event_id)
        if db.scalar(select(AttendanceSession).where(AttendanceSession.session_key == key)):
            return
        db.add(
            AttendanceSession(
                employee_id=event.employee_id,
                entry_time=event.timestamp,
                entry_camera=event.camera_id,
                status="open",
                session_key=key,
            )
        )
        db.flush()

    def _close_attendance(self, db: Session, event: Event) -> None:
        key = event.metadata.get("session_id")
        row = None
        if key:
            row = db.scalar(select(AttendanceSession).where(AttendanceSession.session_key == key))
        if row is None and event.employee_id:
            row = db.scalar(
                select(AttendanceSession).where(
                    AttendanceSession.employee_id == event.employee_id,
                    AttendanceSession.status == "open",
                )
            )
        if row is None or row.exit_time is not None:
            return
        row.exit_time = event.timestamp
        row.exit_camera = event.camera_id
        row.status = "closed"
        db.flush()

    def _open_desk(self, db: Session, event: Event) -> None:
        open_row = db.scalar(
            select(DeskSession).where(
                DeskSession.employee_id == event.employee_id,
                DeskSession.status == "open",
            )
        )
        if open_row is not None:
            return
        key = str(event.metadata.get("session_id") or event.event_id)
        if db.scalar(select(DeskSession).where(DeskSession.session_key == key)):
            return
        db.add(
            DeskSession(
                employee_id=event.employee_id or "",
                desk_id=str(event.metadata.get("desk_id") or ""),
                start_time=event.timestamp,
                status="open",
                session_key=key,
            )
        )
        db.flush()

    def _close_desk(self, db: Session, event: Event) -> None:
        key = event.metadata.get("session_id")
        row = None
        if key:
            row = db.scalar(select(DeskSession).where(DeskSession.session_key == key))
        if row is None and event.employee_id:
            row = db.scalar(
                select(DeskSession).where(
                    DeskSession.employee_id == event.employee_id,
                    DeskSession.status == "open",
                )
            )
        if row is None or row.end_time is not None:
            return
        row.end_time = event.timestamp
        row.status = "closed"
        row.duration_seconds = max(0.0, (event.timestamp - row.start_time).total_seconds())
        db.flush()

    def open_attendance_sessions(self, db: Session) -> list[AttendanceSessionRecord]:
        rows = db.scalars(select(AttendanceSession).where(AttendanceSession.status == "open")).all()
        return [
            AttendanceSessionRecord(
                employee_id=row.employee_id,
                entry_time=row.entry_time,
                entry_camera=row.entry_camera,
                session_id=row.session_key,
                status=row.status,
            )
            for row in rows
        ]

    def open_desk_sessions(self, db: Session) -> list[DeskSessionRecord]:
        rows = db.scalars(select(DeskSession).where(DeskSession.status == "open")).all()
        return [
            DeskSessionRecord(
                employee_id=row.employee_id,
                desk_id=row.desk_id,
                start_time=row.start_time,
                session_id=row.session_key,
                status=row.status,
            )
            for row in rows
        ]

    def attendance_sessions_for(self, db: Session, employee_id: str) -> list[AttendanceSession]:
        return list(
            db.scalars(
                select(AttendanceSession)
                .where(AttendanceSession.employee_id == employee_id)
                .order_by(AttendanceSession.entry_time)
            ).all()
        )

    def desk_sessions_for(self, db: Session, employee_id: str) -> list[DeskSession]:
        return list(
            db.scalars(
                select(DeskSession).where(DeskSession.employee_id == employee_id).order_by(DeskSession.start_time)
            ).all()
        )

    def count_events(self, db: Session, employee_id: str, event_type: str, since: datetime) -> int:
        rows = db.scalars(
            select(EventRow).where(
                EventRow.employee_id == employee_id,
                EventRow.event_type == event_type,
                EventRow.timestamp >= since,
            )
        ).all()
        return len(list(rows))

    def record_camera_health(
        self,
        db: Session,
        camera_id: str,
        status: str,
        fps: float,
        last_frame_time: datetime | None,
        timestamp: datetime,
    ) -> None:
        db.add(
            CameraHealthRow(
                camera_id=camera_id,
                timestamp=timestamp,
                status=status,
                fps=fps,
                last_frame_time=last_frame_time,
            )
        )

    def list_cameras(self, db: Session) -> list[Camera]:
        return list(db.scalars(select(Camera).order_by(Camera.id)).all())

    def audit(self, db: Session, method: str, path: str, status_code: int) -> None:
        db.add(AccessAudit(timestamp=utc_now(), method=method, path=path, status_code=status_code))

    def events_since(self, db: Session, since: datetime) -> list[EventRow]:
        return list(db.scalars(select(EventRow).where(EventRow.timestamp >= since)).all())


def day_start(now: datetime, days: int = 0) -> datetime:
    moment = now - timedelta(days=days)
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)
