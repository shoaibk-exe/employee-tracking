"""Configurable camera topology and travel-time constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Transition:
    src: str
    dst: str
    min_travel_seconds: float
    max_travel_seconds: float
    overlapping: bool = False


class CameraTopology:
    def __init__(self, topology_doc: dict[str, Any]) -> None:
        self.raw = topology_doc
        self.transitions: dict[tuple[str, str], Transition] = {}
        for item in topology_doc.get("transitions", []):
            tr = Transition(
                src=str(item["from"]),
                dst=str(item["to"]),
                min_travel_seconds=float(item.get("min_travel_seconds", 0.0)),
                max_travel_seconds=float(item.get("max_travel_seconds", 1e9)),
                overlapping=bool(item.get("overlapping", False)),
            )
            self.transitions[(tr.src, tr.dst)] = tr

        self.simultaneous_allowed = {
            frozenset((str(a), str(b)))
            for a, b in topology_doc.get("simultaneous_allowed_pairs", [])
        }
        cameras = topology_doc.get("cameras", {})
        for cam_id, meta in cameras.items():
            for other in meta.get("overlapping_with", []):
                self.simultaneous_allowed.add(frozenset((str(cam_id), str(other))))

    def transition(self, src: str, dst: str) -> Transition | None:
        return self.transitions.get((src, dst))

    def allows_simultaneous(self, cam_a: str, cam_b: str) -> bool:
        if cam_a == cam_b:
            return True
        return frozenset((cam_a, cam_b)) in self.simultaneous_allowed

    def temporal_feasible(
        self,
        src_camera: str,
        dst_camera: str,
        delta_seconds: float,
    ) -> tuple[bool, str]:
        """Return whether a transition is physically plausible.

        Unknown paths do NOT hard-reject; they return feasible with reason=unknown_path.
        """
        if src_camera == dst_camera:
            return True, "same_camera"

        if self.allows_simultaneous(src_camera, dst_camera):
            return True, "overlapping"

        tr = self.transition(src_camera, dst_camera)
        if tr is None:
            return True, "unknown_path"

        if tr.overlapping:
            return True, "configured_overlap"

        if delta_seconds < 0:
            return False, "negative_delta"

        if delta_seconds < tr.min_travel_seconds:
            return False, "too_fast"
        if delta_seconds > tr.max_travel_seconds:
            # Soft: still allow with caution flag rather than hard reject for long absences.
            return True, "exceeds_expected_travel"
        return True, "ok"

    def topology_score(
        self,
        src_camera: str,
        dst_camera: str,
        delta_seconds: float,
    ) -> float:
        ok, reason = self.temporal_feasible(src_camera, dst_camera, delta_seconds)
        if not ok:
            return 0.0
        if reason in {"same_camera", "overlapping", "configured_overlap", "ok"}:
            return 1.0
        if reason == "unknown_path":
            return 0.6
        if reason == "exceeds_expected_travel":
            return 0.4
        return 0.5
