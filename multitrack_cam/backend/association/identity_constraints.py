"""Hard physical constraints for cross-camera association."""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.association.camera_topology import CameraTopology


@dataclass
class ActiveObservation:
    global_id: str
    camera_id: str
    timestamp: float
    compound_key: str


@dataclass
class ConstraintContext:
    active: list[ActiveObservation] = field(default_factory=list)
    last_seen: dict[str, tuple[str, float]] = field(default_factory=dict)  # gid -> (cam, ts)


class IdentityConstraints:
    def __init__(self, topology: CameraTopology) -> None:
        self.topology = topology

    def can_assign(
        self,
        *,
        global_id: str,
        camera_id: str,
        timestamp: float,
        ctx: ConstraintContext,
        compound_key: str,
    ) -> tuple[bool, str]:
        # Same identity cannot be assigned to two incompatible simultaneous tracklets.
        for obs in ctx.active:
            if obs.global_id != global_id:
                continue
            if obs.compound_key == compound_key:
                continue
            dt = abs(timestamp - obs.timestamp)
            if dt <= 1.5:
                if not self.topology.allows_simultaneous(camera_id, obs.camera_id):
                    return False, "simultaneous_conflict"
        last = ctx.last_seen.get(global_id)
        if last is not None:
            last_cam, last_ts = last
            ok, reason = self.topology.temporal_feasible(last_cam, camera_id, timestamp - last_ts)
            if not ok:
                return False, reason
        return True, "ok"
