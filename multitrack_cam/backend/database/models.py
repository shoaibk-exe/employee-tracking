"""SQLAlchemy models for persistent MTMCT identity storage."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class CameraRow(Base):
    __tablename__ = "cameras"

    camera_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(128))
    location: Mapped[str | None] = mapped_column(String(256), nullable=True)
    stream_config_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="disconnected")
    topology_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class GlobalIdentityRow(Base):
    __tablename__ = "global_identities"

    global_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    identity_status: Mapped[str] = mapped_column(String(32), default="ACTIVE")
    verification_status: Mapped[str] = mapped_column(String(32), default="UNVERIFIED")
    first_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    first_seen_ts: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_seen_ts: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_camera_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    cameras_seen: Mapped[list | None] = mapped_column(JSON, nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class IdentityEmbeddingRow(Base):
    __tablename__ = "identity_embeddings"

    embedding_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    global_id: Mapped[str] = mapped_column(String(32), index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    embedding: Mapped[list] = mapped_column(JSON)
    camera_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    timestamp: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    verification_source: Mapped[str] = mapped_column(String(32), default="reid")
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class LocalTrackletRow(Base):
    __tablename__ = "local_tracklets"

    tracklet_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    tracker_session_id: Mapped[str] = mapped_column(String(64))
    local_track_id: Mapped[int] = mapped_column(Integer)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    global_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    assignment_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)


class IdentityEventRow(Base):
    __tablename__ = "identity_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    timestamp: Mapped[str] = mapped_column(String(64))
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    global_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    camera_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)


class IdentityAssociationRow(Base):
    __tablename__ = "identity_associations"

    association_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_tracklet_id: Mapped[str] = mapped_column(String(64), index=True)
    destination_global_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    evidence: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    decision: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    review_status: Mapped[str] = mapped_column(String(32), default="pending")


class IdSequenceRow(Base):
    __tablename__ = "id_sequences"
    __table_args__ = (UniqueConstraint("name", name="uq_id_sequences_name"),)

    name: Mapped[str] = mapped_column(String(32), primary_key=True)
    value: Mapped[int] = mapped_column(Integer, default=0)
