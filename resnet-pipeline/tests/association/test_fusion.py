"""Occupancy fusion decision table and crop compatibility."""

from __future__ import annotations

import pytest

from core.association.association_engine import AssociationFrame
from core.occupancy.classifier import Prediction
from core.association.occupancy_fusion import FusionConfig, OccupancyFusion, crop_compatibility
from core.scene.scene_mapper import STATUS_CONFIRMED, Workstation

WS = Workstation("WS01", (100, 100, 200, 300), (90, 90, 210, 310), 1.0, STATUS_CONFIRMED, "admin")
TID = "t1:CAM01:s1:1"


def frame(**kw):
    base = dict(timestamp_seconds=1.0, tracks={}, occupants={}, candidates={}, ambiguous_workstations=set(),
                standing_near={}, events=[], persons_in_zone={})
    base.update(kw)
    return AssociationFrame(**base)


def pred(cls, conf):
    return Prediction(cls, conf, {cls: conf, ("empty" if cls == "occupied" else "occupied"): 1 - conf})


@pytest.fixture
def fusion():
    return OccupancyFusion(["empty", "occupied"])


def test_person_association_wins_and_disagreement_is_logged(fusion):
    d = fusion.decide(WS, frame(occupants={"WS01": TID}, persons_in_zone={"WS01": [TID]}), pred("empty", 0.9))
    assert (d.observed_class, d.source, d.confidence) == ("occupied", "person", 0.95)
    assert d.disagreement == "resnet_empty_with_seated_person"


def test_candidate_is_neutral_unless_resnet_corroborates(fusion):
    d = fusion.decide(WS, frame(candidates={"WS01": [TID]}, persons_in_zone={"WS01": [TID]}), pred("empty", 0.9))
    assert d.source == "neutral" and d.confidence == 0.0
    d = fusion.decide(WS, frame(candidates={"WS01": [TID]}, persons_in_zone={"WS01": [TID]}),
                      pred("occupied", 0.9))
    assert (d.source, d.reason, d.confidence) == ("resnet", "candidate_corroborated", 0.9)


def test_standing_person_blocks_resnet_occupied(fusion):
    d = fusion.decide(WS, frame(standing_near={"WS01": [TID]}, persons_in_zone={"WS01": [TID]}),
                      pred("occupied", 0.97))
    assert d.source == "neutral" and d.reason == "standing_person_in_crop"
    assert d.disagreement == "resnet_occupied_but_person_standing"


def test_resnet_occupied_without_any_visible_person_needs_higher_confidence(fusion):
    weak = fusion.decide(WS, frame(), pred("occupied", 0.90))
    assert weak.source == "neutral" and weak.reason == "resnet_occupied_no_person_seen"
    strong = fusion.decide(WS, frame(), pred("occupied", 0.97))
    assert strong.source == "resnet" and strong.reason == "resnet_uncorroborated"


def test_resnet_only_when_person_visible_or_empty(fusion):
    seen = fusion.decide(WS, frame(persons_in_zone={"WS01": [TID]}), pred("occupied", 0.85))
    assert (seen.observed_class, seen.source) == ("occupied", "resnet")
    empty = fusion.decide(WS, frame(), pred("empty", 0.85))
    assert (empty.observed_class, empty.source, empty.confidence) == ("empty", "resnet", 0.85)


def test_no_evidence_is_missing(fusion):
    d = fusion.decide(WS, frame(), None)
    assert d.observed_class is None and d.source == "none"


def test_ambiguous_association_is_neutral(fusion):
    d = fusion.decide(WS, frame(ambiguous_workstations={"WS01"}, persons_in_zone={"WS01": [TID]}),
                      pred("occupied", 0.9))
    assert d.source == "neutral" and d.reason == "ambiguous_association"


def test_fusion_requires_expected_classes():
    with pytest.raises(ValueError, match="missing"):
        OccupancyFusion(["vacant", "occupied"])


def test_crop_compatibility_matches_training_distribution():
    cfg = FusionConfig()
    assert crop_compatibility((0, 0, 150, 180), cfg)[0]
    assert not crop_compatibility((0, 0, 40, 40), cfg)[0]
    assert not crop_compatibility((0, 0, 500, 100), cfg)[0]
