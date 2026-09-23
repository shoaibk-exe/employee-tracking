"""Workspace observations. Sitting inside an assigned polygon is a fact. Time is decided later."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import DeskStateConfig
from app.streams.frame_buffer import FramePacket
from app.tracking.tracker import UltralyticsByteTracker
from app.utils.geometry import bottom_center
from app.zones.desk_manager import DeskManager


@dataclass
class DeskPresence:
    employee_id: str
    desk_id: str
    present: bool
    confidence: float
    class_name: str


@dataclass
class WorkspaceTick:
    camera_id: str
    timestamp: object
    presence: list[DeskPresence]


def run_workspace(
    packet: FramePacket,
    tracker: UltralyticsByteTracker,
    desks: DeskManager,
    rules: DeskStateConfig,
) -> WorkspaceTick:
    timestamp = packet.received_timestamp
    tracks = tracker.track(packet.frame, timestamp)
    allowed = {"sitting"}
    if rules.count_standing_as_at_desk:
        allowed.add("standing")
    if rules.count_bending_as_at_desk:
        allowed.add("bending")
    presence: dict[str, DeskPresence] = {}
    for zone in desks.zones_for(packet.camera_id):
        presence[zone.employee_id] = DeskPresence(
            employee_id=zone.employee_id,
            desk_id=zone.desk_id,
            present=False,
            confidence=0.0,
            class_name="",
        )
    for track in tracks:
        if track.class_name not in allowed or track.confidence < rules.min_detection_confidence:
            continue
        zone = desks.locate(packet.camera_id, bottom_center(track.bbox))
        if zone is None:
            continue
        slot = presence.get(zone.employee_id)
        if slot is None:
            continue
        slot.present = True
        slot.class_name = track.class_name
        slot.confidence = max(slot.confidence, track.confidence)
    return WorkspaceTick(
        camera_id=packet.camera_id,
        timestamp=timestamp,
        presence=list(presence.values()),
    )
