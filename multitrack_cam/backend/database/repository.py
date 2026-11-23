"""Database repository with concurrency-safe global ID allocation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from backend.database.models import (
    Base,
    CameraRow,
    GlobalIdentityRow,
    IdSequenceRow,
    IdentityAssociationRow,
    IdentityEmbeddingRow,
    IdentityEventRow,
    LocalTrackletRow,
)
from backend.utils.paths import PROJECT_ROOT, resolve_path

logger = logging.getLogger(__name__)


def _normalize_db_url(url: str) -> str:
    if url.startswith("sqlite:///"):
        raw = url[len("sqlite:///") :]
        if raw == ":memory:":
            return url
        path = resolve_path(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path}"
    return url


class DatabaseRepository:
    def __init__(self, url: str, echo: bool = False) -> None:
        self.url = _normalize_db_url(url)
        connect_args = {"check_same_thread": False, "timeout": 30.0} if self.url.startswith("sqlite") else {}
        self.engine = create_engine(self.url, echo=echo, future=True, connect_args=connect_args)
        if self.url.startswith("sqlite") and ":memory:" not in self.url:
            from sqlalchemy import event

            @event.listens_for(self.engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
                cur = dbapi_conn.cursor()
                # WAL: readers never block the writer thread, writers serialize quickly.
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.execute("PRAGMA busy_timeout=30000")
                cur.close()
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        Base.metadata.create_all(self.engine)
        logger.info("Database ready: %s", self._safe_url(self.url))

    @staticmethod
    def _safe_url(url: str) -> str:
        try:
            parsed = urlparse(url)
            if parsed.password:
                netloc = parsed.netloc.replace(f":{parsed.password}", ":***")
                return parsed._replace(netloc=netloc).geturl()
        except Exception:  # noqa: BLE001
            pass
        return url

    def next_global_id(self, prefix: str, width: int) -> str:
        with self.SessionLocal() as session:
            with session.begin():
                if self.url.startswith("sqlite"):
                    session.execute(text("BEGIN IMMEDIATE"))
                row = session.get(IdSequenceRow, "global_id")
                if row is None:
                    row = IdSequenceRow(name="global_id", value=0)
                    session.add(row)
                    session.flush()
                row.value += 1
                value = row.value
            return f"{prefix}{value:0{width}d}"

    def upsert_identity(self, record: dict[str, Any]) -> None:
        with self.SessionLocal() as session:
            with session.begin():
                row = session.get(GlobalIdentityRow, record["global_id"])
                if row is None:
                    row = GlobalIdentityRow(global_id=record["global_id"])
                    session.add(row)
                row.identity_status = record.get("identity_status", "ACTIVE")
                row.verification_status = record.get("verification_status", "UNVERIFIED")
                row.first_seen_ts = record.get("first_seen_ts")
                row.last_seen_ts = record.get("last_seen_ts")
                row.last_camera_id = record.get("last_camera_id")
                row.cameras_seen = record.get("cameras_seen")
                row.display_name = record.get("display_name")
                row.updated_at = datetime.now(timezone.utc)
                if record.get("created_at") and not row.created_at:
                    row.created_at = datetime.now(timezone.utc)

    def add_embedding(self, record: dict[str, Any]) -> None:
        with self.SessionLocal() as session:
            with session.begin():
                session.merge(
                    IdentityEmbeddingRow(
                        embedding_id=record["embedding_id"],
                        global_id=record["global_id"],
                        model_version=record.get("model_version", "reid-v1"),
                        embedding=record["embedding"],
                        camera_id=record.get("camera_id"),
                        timestamp=record.get("timestamp"),
                        quality_score=float(record.get("quality_score", 0.0)),
                        verification_source=record.get("verification_source", "reid"),
                        active=bool(record.get("active", True)),
                    )
                )

    def reassign_embeddings(self, source_gid: str, target_gid: str) -> int:
        """Re-point stored embeddings of a merged duplicate identity to the surviving one."""
        with self.SessionLocal() as session:
            with session.begin():
                rows = session.scalars(
                    select(IdentityEmbeddingRow).where(IdentityEmbeddingRow.global_id == source_gid)
                ).all()
                for r in rows:
                    r.global_id = target_gid
                return len(rows)

    def add_tracklet(self, record: dict[str, Any]) -> None:
        with self.SessionLocal() as session:
            with session.begin():
                session.merge(
                    LocalTrackletRow(
                        tracklet_id=record["tracklet_id"],
                        camera_id=record["camera_id"],
                        tracker_session_id=record["tracker_session_id"],
                        local_track_id=int(record["local_track_id"]),
                        start_time=float(record["start_time"]),
                        end_time=float(record["end_time"]),
                        global_id=record.get("global_id"),
                        assignment_confidence=record.get("assignment_confidence"),
                    )
                )

    def add_event(self, record: dict[str, Any]) -> None:
        with self.SessionLocal() as session:
            with session.begin():
                session.add(
                    IdentityEventRow(
                        event_id=record["event_id"],
                        timestamp=record["timestamp"],
                        event_type=record["event_type"],
                        global_id=record.get("global_id"),
                        camera_id=record.get("camera_id"),
                        metadata_json=record.get("metadata"),
                    )
                )

    def add_association(self, record: dict[str, Any]) -> None:
        with self.SessionLocal() as session:
            with session.begin():
                session.merge(
                    IdentityAssociationRow(
                        association_id=record["association_id"],
                        source_tracklet_id=record["source_tracklet_id"],
                        destination_global_id=record.get("destination_global_id"),
                        evidence=record.get("evidence"),
                        decision=record["decision"],
                        confidence=float(record.get("confidence", 0.0)),
                        model_version=record.get("model_version"),
                        review_status=record.get("review_status", "pending"),
                    )
                )

    def list_identities(self) -> list[dict[str, Any]]:
        with self.SessionLocal() as session:
            rows = session.scalars(select(GlobalIdentityRow)).all()
            return [
                {
                    "global_id": r.global_id,
                    "identity_status": r.identity_status,
                    "verification_status": r.verification_status,
                    "first_seen": r.first_seen.isoformat() if r.first_seen else None,
                    "last_seen": r.last_seen.isoformat() if r.last_seen else None,
                    "first_seen_ts": r.first_seen_ts,
                    "last_seen_ts": r.last_seen_ts,
                    "last_camera_id": r.last_camera_id,
                    "cameras_seen": r.cameras_seen or [],
                    "display_name": r.display_name,
                    "created_at": r.created_at.isoformat() if r.created_at else "",
                    "updated_at": r.updated_at.isoformat() if r.updated_at else "",
                }
                for r in rows
            ]

    def list_embeddings(self, active_only: bool = True) -> list[dict[str, Any]]:
        with self.SessionLocal() as session:
            stmt = select(IdentityEmbeddingRow)
            if active_only:
                stmt = stmt.where(IdentityEmbeddingRow.active.is_(True))
            rows = session.scalars(stmt).all()
            return [
                {
                    "embedding_id": r.embedding_id,
                    "global_id": r.global_id,
                    "model_version": r.model_version,
                    "embedding": r.embedding,
                    "camera_id": r.camera_id,
                    "timestamp": r.timestamp,
                    "quality_score": r.quality_score,
                    "verification_source": r.verification_source,
                    "active": r.active,
                }
                for r in rows
            ]

    def get_identity(self, global_id: str) -> dict[str, Any] | None:
        with self.SessionLocal() as session:
            r = session.get(GlobalIdentityRow, global_id)
            if r is None:
                return None
            return {
                "global_id": r.global_id,
                "identity_status": r.identity_status,
                "verification_status": r.verification_status,
                "first_seen_ts": r.first_seen_ts,
                "last_seen_ts": r.last_seen_ts,
                "last_camera_id": r.last_camera_id,
                "cameras_seen": r.cameras_seen or [],
                "display_name": r.display_name,
            }

    def list_events(self, global_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self.SessionLocal() as session:
            rows = session.scalars(
                select(IdentityEventRow)
                .where(IdentityEventRow.global_id == global_id)
                .order_by(IdentityEventRow.timestamp.desc())
                .limit(limit)
            ).all()
            return [
                {
                    "event_id": r.event_id,
                    "timestamp": r.timestamp,
                    "event_type": r.event_type,
                    "global_id": r.global_id,
                    "camera_id": r.camera_id,
                    "metadata": r.metadata_json,
                }
                for r in rows
            ]

    def upsert_camera(self, camera_id: str, display_name: str, location: str, status: str, stream_ref: str) -> None:
        with self.SessionLocal() as session:
            with session.begin():
                row = session.get(CameraRow, camera_id)
                if row is None:
                    row = CameraRow(camera_id=camera_id, display_name=display_name)
                    session.add(row)
                row.display_name = display_name
                row.location = location
                row.status = status
                row.stream_config_ref = stream_ref


def create_repository(url: str, echo: bool = False) -> DatabaseRepository:
    return DatabaseRepository(url=url, echo=echo)
