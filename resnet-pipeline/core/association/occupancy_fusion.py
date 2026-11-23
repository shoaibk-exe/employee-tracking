"""Deterministic workstation-occupancy fusion: person association first, ResNet-18 as support.

ResNet-18 was trained on workstation crops ("empty" / "occupied"). It cannot tell a seated
employee from a person standing beside the desk, so it never overrides confirmed person
evidence and is never trusted alone when a standing/moving person overlaps the crop.

Decision table (per workstation, at each classification tick):

==========================================  ==============================  =========================
Evidence                                    Fed to WorkstationStateManager  Recorded reason
==========================================  ==============================  =========================
seated person associated (fresh)            occupied @ person_confidence    person_associated
  ... and ResNet says empty (>= threshold)  (same)                          disagreement logged
seated candidate + ResNet occupied          occupied @ ResNet conf          candidate_corroborated
seated candidate, ResNet not occupied       neutral                         person_candidate
standing/moving person overlaps the crop    neutral                         standing_person_in_crop
  ... and ResNet says occupied              (same)                          disagreement logged
ResNet occupied, no person visible in crop  occupied only if conf >=        resnet_uncorroborated
                                            uncorroborated threshold, else
                                            neutral
ResNet occupied/empty, no person conflict   ResNet class @ ResNet conf      resnet_only
ResNet unavailable / incompatible crop      missing (desk stays empty)      no_evidence
==========================================  ==============================  =========================

"neutral" is an observation with confidence 0: it neither confirms nor cancels a state.
A desk that is not confirmed occupied is empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from core.association.association_engine import AssociationFrame
from core.occupancy.classifier import Prediction
from core.scene.scene_mapper import Workstation

OCCUPIED = "occupied"
EMPTY = "empty"


@dataclass(frozen=True)
class FusionConfig:
    person_confidence: float = 0.95
    resnet_threshold: float = 0.80
    uncorroborated_occupied_threshold: float = 0.95
    min_crop_side_px: int = 64
    min_crop_aspect: float = 0.45
    max_crop_aspect: float = 2.10

    def validate(self) -> None:
        if not 0 < self.person_confidence <= 1 or not 0 < self.resnet_threshold <= 1:
            raise ValueError("fusion confidences must be in (0, 1]")
        if not self.resnet_threshold <= self.uncorroborated_occupied_threshold <= 1:
            raise ValueError("fusion.uncorroborated_occupied_threshold must be in [resnet_threshold, 1]")
        if self.min_crop_aspect >= self.max_crop_aspect:
            raise ValueError("fusion.min_crop_aspect must be < max_crop_aspect")


def crop_compatibility(box: Sequence[int], config: FusionConfig) -> tuple[bool, str]:
    """Context crops outside the training-crop size/aspect distribution are not classified."""
    width, height = box[2] - box[0], box[3] - box[1]
    if min(width, height) < config.min_crop_side_px:
        return False, f"crop {width}x{height} smaller than {config.min_crop_side_px}px"
    aspect = width / height
    if not config.min_crop_aspect <= aspect <= config.max_crop_aspect:
        return False, f"crop aspect {aspect:.2f} outside [{config.min_crop_aspect}, {config.max_crop_aspect}]"
    return True, "ok"


@dataclass(frozen=True)
class FusionDecision:
    workstation_id: str
    observed_class: str | None  # None -> observe_missing
    confidence: float
    source: str                 # person | resnet | neutral | none
    reason: str
    occupant_track_id: str | None
    resnet_class: str | None
    resnet_confidence: float | None
    disagreement: str | None


class OccupancyFusion:
    def __init__(self, class_names: Sequence[str], config: FusionConfig = FusionConfig()) -> None:
        config.validate()
        missing = {OCCUPIED, EMPTY} - set(class_names)
        if missing:
            raise ValueError(f"Fusion requires classifier classes {sorted({OCCUPIED, EMPTY})}; missing {sorted(missing)}")
        self.config = config
        self.class_names = list(class_names)

    def decide(self, ws: Workstation, frame: AssociationFrame, prediction: Prediction | None) -> FusionDecision:
        cfg, ws_id = self.config, ws.workstation_id
        r_class = prediction.predicted_class if prediction else None
        r_conf = prediction.confidence if prediction else None
        r_strong = prediction is not None and prediction.confidence >= cfg.resnet_threshold
        occupant = frame.occupants.get(ws_id)

        if occupant is not None:
            disagreement = "resnet_empty_with_seated_person" if r_strong and r_class == EMPTY else None
            return FusionDecision(ws_id, OCCUPIED, cfg.person_confidence, "person", "person_associated",
                                  occupant, r_class, r_conf, disagreement)
        if frame.candidates.get(ws_id):
            if r_strong and r_class == OCCUPIED:
                return FusionDecision(ws_id, OCCUPIED, float(r_conf), "resnet", "candidate_corroborated", None,
                                      r_class, r_conf, None)
            return FusionDecision(ws_id, OCCUPIED, 0.0, "neutral", "person_candidate", None, r_class, r_conf, None)
        if frame.standing_near.get(ws_id):
            disagreement = "resnet_occupied_but_person_standing" if r_strong and r_class == OCCUPIED else None
            return FusionDecision(ws_id, r_class or OCCUPIED, 0.0, "neutral", "standing_person_in_crop", None,
                                  r_class, r_conf, disagreement)
        if ws_id in frame.ambiguous_workstations:
            return FusionDecision(ws_id, r_class or OCCUPIED, 0.0, "neutral", "ambiguous_association", None,
                                  r_class, r_conf, None)
        if prediction is not None and r_class == OCCUPIED and not frame.persons_in_zone.get(ws_id):
            # The person detector sees nobody overlapping the crop: occupied is uncorroborated.
            if float(r_conf) >= cfg.uncorroborated_occupied_threshold:
                return FusionDecision(ws_id, OCCUPIED, float(r_conf), "resnet", "resnet_uncorroborated", None,
                                      r_class, r_conf, None)
            return FusionDecision(ws_id, OCCUPIED, 0.0, "neutral", "resnet_occupied_no_person_seen", None,
                                  r_class, r_conf, "resnet_occupied_no_person_seen" if r_strong else None)
        if prediction is not None:
            return FusionDecision(ws_id, r_class, float(r_conf), "resnet", "resnet_only", None, r_class, r_conf, None)
        return FusionDecision(ws_id, None, 0.0, "none", "no_evidence", None, None, None, None)
