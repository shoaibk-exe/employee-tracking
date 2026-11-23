"""Association cost from appearance, temporal, and topology evidence."""

from __future__ import annotations

from dataclasses import dataclass

from backend.association.camera_topology import CameraTopology


@dataclass(frozen=True)
class CostWeights:
    appearance: float = 0.70
    temporal: float = 0.15
    topology: float = 0.15


def temporal_compatibility(delta_seconds: float, long_gap_caution_seconds: float) -> float:
    delta = abs(delta_seconds)
    if delta <= 5.0:
        return 1.0
    if delta <= 60.0:
        return 0.9
    if delta <= long_gap_caution_seconds:
        return 0.75
    if delta <= 24 * 3600:
        return 0.55
    return 0.40


def association_score(
    *,
    appearance_similarity: float,
    src_camera: str,
    dst_camera: str,
    delta_seconds: float,
    topology: CameraTopology,
    weights: CostWeights,
    long_gap_caution_seconds: float = 1800.0,
) -> tuple[float, dict]:
    """Higher is better. Missing topology evidence is treated as missing, not proof."""
    topo = topology.topology_score(src_camera, dst_camera, delta_seconds)
    temporal = temporal_compatibility(delta_seconds, long_gap_caution_seconds)
    # Appearance in [0,1] assumed cosine already roughly that range for ReID.
    app = max(0.0, min(1.0, (appearance_similarity + 1.0) / 2.0)) if appearance_similarity < 0 else min(1.0, appearance_similarity)
    # For YOLO26 ReID cosine is typically already in ~0..1 positive range.
    if appearance_similarity >= 0:
        app = min(1.0, float(appearance_similarity))

    score = (
        weights.appearance * app
        + weights.temporal * temporal
        + weights.topology * topo
    )
    evidence = {
        "appearance": app,
        "temporal": temporal,
        "topology": topo,
        "weighted_score": score,
        "raw_similarity": appearance_similarity,
        "delta_seconds": delta_seconds,
    }
    return float(score), evidence


def similarity_to_cost(score: float) -> float:
    """Convert higher-is-better score into Hungarian cost (lower-is-better)."""
    return 1.0 - max(0.0, min(1.0, score))
