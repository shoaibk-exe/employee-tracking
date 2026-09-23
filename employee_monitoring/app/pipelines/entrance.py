"""Entrance observations: track, then face, then line. Faces alone never open a session."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.face.recognizer import FaceRecognizer
from app.logging_config import log_event
from app.streams.frame_buffer import FramePacket
from app.tracking.track_manager import TrackManager, associate_faces
from app.tracking.tracker import TrackObservation, UltralyticsByteTracker
from app.utils.geometry import bottom_center
from app.zones.line_crossing import CrossingEvent, LineCrossingDetector

logger = logging.getLogger(__name__)


@dataclass
class EntranceTick:
    camera_id: str
    timestamp: object
    crossings: list[tuple[CrossingEvent, str | None, float]]
    active_track_ids: list[int]


def run_entrance(
    packet: FramePacket,
    tracker: UltralyticsByteTracker,
    recognizer: FaceRecognizer,
    identities: TrackManager,
    line: LineCrossingDetector | None,
) -> EntranceTick:
    timestamp = packet.received_timestamp
    tracks: list[TrackObservation] = tracker.track(packet.frame, timestamp)
    faces = recognizer.recognize(packet.frame)
    assigned = associate_faces(tracks, faces)
    active: set[int] = set()
    for track in tracks:
        active.add(track.track_id)
        face = assigned.get(track.track_id)
        if face is None or face.quality_reason != "OK" or not face.employee_id:
            continue
        before = identities.confirmed_employee(track.track_id)
        confirmed = identities.observe(track.track_id, face.employee_id, face.similarity, timestamp)
        if confirmed and before is None:
            log_event(
                logger,
                "identity confirmed",
                event="FACE_MATCH",
                camera=packet.camera_id,
                employee=confirmed,
                similarity=f"{identities.identity_confidence(track.track_id):.3f}",
                track_id=track.track_id,
            )
    identities.drop_stale(active, timestamp)
    crossings: list[tuple[CrossingEvent, str | None, float]] = []
    if line is not None:
        for track in tracks:
            crossing = line.update(track.track_id, bottom_center(track.bbox), timestamp)
            if crossing is None:
                continue
            employee_id = identities.confirmed_employee(track.track_id)
            crossings.append((crossing, employee_id, identities.identity_confidence(track.track_id)))
    return EntranceTick(
        camera_id=packet.camera_id,
        timestamp=timestamp,
        crossings=crossings,
        active_track_ids=sorted(active),
    )
