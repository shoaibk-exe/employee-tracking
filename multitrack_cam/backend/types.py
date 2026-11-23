"""Shared dataclasses and enums for the MTMCT pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np


class CameraStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    LIVE = "live"
    RECONNECTING = "reconnecting"
    FROZEN = "frozen"
    ERROR = "error"
    DISABLED = "disabled"


class TrackLifecycle(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


class IdentityActivityStatus(str, Enum):
    PROVISIONAL = "PROVISIONAL"
    ACTIVE = "ACTIVE"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"
    OUT_OF_VIEW = "OUT_OF_VIEW"
    INACTIVE = "INACTIVE"
    ARCHIVED = "ARCHIVED"


class VerificationStatus(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    PROVISIONAL = "PROVISIONAL"
    VERIFIED = "VERIFIED"
    MANUAL = "MANUAL"


class AssociationDecision(str, Enum):
    CONFIRMED = "CONFIRMED"
    PROVISIONAL = "PROVISIONAL"
    UNKNOWN = "UNKNOWN"
    NEW = "NEW"
    REJECTED = "REJECTED"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class FramePacket:
    camera_id: str
    frame: np.ndarray
    frame_id: int
    timestamp: float
    wall_clock: datetime
    source_fps: float | None = None
    after_reconnect: bool = False


@dataclass
class Detection:
    camera_id: str
    frame_id: int
    timestamp: float
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass
class LocalTrack:
    camera_id: str
    tracker_session_id: str
    local_track_id: int
    frame_id: int
    timestamp: float
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    lifecycle: TrackLifecycle
    is_predicted: bool = False
    hit_count: int = 1
    age_frames: int = 1

    @property
    def compound_key(self) -> str:
        return f"{self.camera_id}:{self.tracker_session_id}:{self.local_track_id}"


@dataclass
class TrackletObservation:
    timestamp: float
    frame_id: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    embedding: np.ndarray | None = None
    quality_score: float = 0.0


@dataclass
class Tracklet:
    tracklet_id: str
    camera_id: str
    tracker_session_id: str
    local_track_id: int
    start_time: float
    end_time: float
    observations: list[TrackletObservation] = field(default_factory=list)
    embedding: np.ndarray | None = None
    global_id: str | None = None
    assignment_confidence: float | None = None
    decision: AssociationDecision | None = None

    @property
    def compound_key(self) -> str:
        return f"{self.camera_id}:{self.tracker_session_id}:{self.local_track_id}"

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class AnnotatedPerson:
    local_track_id: int
    compound_key: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    global_id: str | None
    identity_status: str
    verification_status: str
    color_bgr: tuple[int, int, int]
    is_predicted: bool = False


@dataclass
class CameraRuntimeState:
    camera_id: str
    display_name: str
    status: CameraStatus
    input_fps: float = 0.0
    processed_fps: float = 0.0
    latency_ms: float = 0.0
    last_frame_id: int = 0
    last_timestamp: float | None = None
    reconnect_count: int = 0
    error_message: str | None = None
    people: list[AnnotatedPerson] = field(default_factory=list)


@dataclass
class SystemStats:
    live_cameras: int
    total_cameras: int
    active_global_identities: int
    total_retained_identities: int
    unresolved_tracks: int
    gpu_memory_mb: float | None
    gpu_util_percent: float | None
    detection_latency_ms: float
    reid_latency_ms: float
    association_latency_ms: float
    end_to_end_latency_ms: float
    timestamp: str


@dataclass
class AssociationResult:
    tracklet_id: str
    decision: AssociationDecision
    global_id: str | None
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
