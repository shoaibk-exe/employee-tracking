"""Scene map: geometry, discovery, admin seeding/refinement, persistence, review session."""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.scene.scene_mapper import (
    STATUS_CONFIRMED, STATUS_NEEDS_REVIEW, STATUS_REJECTED, DiscoveryFrame, SceneMap, SceneMapError, SceneMapper,
    SceneRuntimeConfig, Workstation, carry_over_ids, cluster_detections, containment, discover_workstations, iou,
    overlap_min, refine_with_evidence, scene_map_from_admin_rois, workstations_from_admin_rois,
)
from scripts.scene.review_scene_map import ReviewSession

FRAME = (1920, 1080)


def frames_with(persons=(), chairs=(), tvs=(), n=10, every=1):
    out = []
    for i in range(n):
        visible = i % every == 0
        out.append(DiscoveryFrame(
            [(b, 0.8) for b in persons] if visible else [],
            {"chair": [(b, 0.6) for b in chairs], "tv": [(b, 0.7) for b in tvs], "laptop": [], "keyboard": []},
        ))
    return out


def make_map(workstations, version=1) -> SceneMap:
    return SceneMap("t1", "CAM01", version, FRAME, 0, workstations, "2026-01-01T00:00:00+00:00", "test")


def test_scene_map_from_admin_rois_keeps_exact_boxes_confirmed():
    rois = {"W01": (100, 200, 300, 400), "W02": (500, 200, 700, 400)}
    scene = scene_map_from_admin_rois("t1", "CAM01", FRAME, 0, rois, roi_path="config/workstations.json")
    assert scene.source == "admin" and len(scene.workstations) == 2
    for w in scene.workstations:
        assert w.seat_box == w.context_box and w.status == STATUS_CONFIRMED
        assert w.evidence["admin_roi_id"] in rois and w.seat_box == rois[w.evidence["admin_roi_id"]]


def test_geometry_helpers():
    a, b = (0, 0, 10, 10), (5, 0, 15, 10)
    assert iou(a, b) == pytest.approx(50 / 150)
    assert overlap_min((0, 0, 4, 4), (0, 0, 10, 10)) == 1.0
    assert containment((0, 0, 10, 10), (0, 0, 5, 10)) == pytest.approx(0.5)


def test_cluster_detections_one_box_per_frame_per_cluster():
    samples = [[((0, 0, 10, 10), 0.9), ((1, 0, 11, 10), 0.8)], [((0, 0, 10, 10), 0.9)]]
    clusters = cluster_detections(samples, 0.5, "chair")
    assert sorted(len(c.frames) for c in clusters) == [1, 2]


def test_discovery_confirms_persistent_chair_with_surface_and_flags_edge():
    chair = (400, 500, 500, 650)
    edge_chair = (0, 500, 60, 700)
    tv = (420, 400, 520, 480)
    found, notes = discover_workstations(frames_with(chairs=[chair, edge_chair], tvs=[tv]), FRAME)
    by_seat_x = {w.seat_box[0]: w for w in found}
    assert by_seat_x[400].status == STATUS_CONFIRMED
    assert by_seat_x[0].status == STATUS_NEEDS_REVIEW and by_seat_x[0].evidence["at_frame_edge"]
    assert sorted(w.workstation_id for w in found) == ["WS01", "WS02"]
    assert any("review" in n for n in notes)


def test_discovery_merges_seated_person_into_chair_and_suppresses_duplicates():
    chair = (400, 500, 500, 650)
    person = (395, 430, 505, 640)
    found, _ = discover_workstations(frames_with(persons=[person], chairs=[chair], tvs=[(420, 380, 520, 440)]),
                                     FRAME)
    assert len(found) == 1
    ws = found[0]
    assert ws.seat_box == person
    assert ws.seated_height_ref == pytest.approx(210)
    assert ws.evidence["dwell_persistence"] == 1.0


def test_discovery_ignores_transient_people():
    walker_frames = [DiscoveryFrame([((100 + 80 * i, 300, 160 + 80 * i, 500), 0.9)], {}) for i in range(10)]
    found, _ = discover_workstations(walker_frames, FRAME)
    assert found == []


def test_refine_uses_iou_assignment_and_flags_ambiguous_evidence():
    rois = {"W01": (100, 100, 300, 400), "W02": (280, 100, 480, 400), "W03": (1000, 100, 1200, 400)}
    ws = workstations_from_admin_rois(rois)
    seated_w01 = (120, 150, 280, 380)
    straddling = (200, 600, 400, 900)
    ws_amb = workstations_from_admin_rois({"A": (150, 550, 380, 950), "B": (220, 560, 450, 940)})
    refine_with_evidence(ws, frames_with(persons=[seated_w01]))
    assert ws[0].seat_box == seated_w01 and ws[0].evidence["seat_refined"]
    assert ws[1].seat_box == rois["W02"] and not ws[1].evidence["seat_refined"]
    notes = refine_with_evidence(ws_amb, frames_with(persons=[straddling]))
    assert any("needs review" in n for n in notes)
    assert STATUS_NEEDS_REVIEW in {w.status for w in ws_amb}


def test_refine_chair_only_extends_seat_upward_by_measured_ratio():
    rois = {"W01": (100, 100, 300, 500), "W02": (600, 100, 800, 500)}
    ws = workstations_from_admin_rois(rois)
    frames = frames_with(persons=[(110, 150, 290, 450)], chairs=[(120, 250, 280, 450), (620, 300, 780, 450)])
    refine_with_evidence(ws, frames)
    ratio = 300 / 200
    assert ws[1].seat_box == (620, int(round(450 - 150 * ratio)), 780, 450)
    assert ws[1].seated_height_ref == pytest.approx(150 * ratio)


def test_scene_map_roundtrip_versioning_and_tamper_detection(tmp_path):
    path = tmp_path / "scene_map.json"
    w = Workstation("WS01", (1, 2, 30, 40), (0, 0, 50, 60), 0.9, STATUS_CONFIRMED, "auto")
    make_map([w]).save(path)
    loaded = SceneMap.load(path)
    assert loaded.get("WS01").seat_box == (1, 2, 30, 40)
    make_map([w], version=2).save(path)
    assert (tmp_path / "scene_map.v1.json").is_file()
    data = json.loads(path.read_text())
    data["workstations"][0]["seat_box"] = [5, 5, 50, 50]
    path.write_text(json.dumps(data))
    with pytest.raises(SceneMapError, match="fingerprint"):
        SceneMap.load(path)


def test_scene_map_validate_for_rejects_mismatches():
    smap = make_map([Workstation("WS01", (1, 2, 30, 40), (0, 0, 50, 60), 0.9, STATUS_CONFIRMED, "auto")])
    smap.validate_for("t1", "CAM01", *FRAME, 0)
    with pytest.raises(SceneMapError, match="tenant"):
        smap.validate_for("t2", "CAM01", *FRAME, 0)
    with pytest.raises(SceneMapError, match="frame size"):
        smap.validate_for("t1", "CAM01", 1280, 720, 0)
    with pytest.raises(SceneMapError, match="rotation"):
        smap.validate_for("t1", "CAM01", *FRAME, 180)


def test_carry_over_ids_keeps_ids_stable():
    old = make_map([Workstation("WS07", (100, 100, 200, 300), (90, 90, 210, 310), 0.9, STATUS_CONFIRMED, "auto")])
    new = [Workstation("", (102, 98, 205, 301), (90, 90, 210, 310), 0.8, STATUS_CONFIRMED, "auto"),
           Workstation("", (800, 100, 900, 300), (790, 90, 910, 310), 0.8, STATUS_CONFIRMED, "auto")]
    ids = sorted(w.workstation_id for w in carry_over_ids(new, old))
    assert ids == ["WS01", "WS07"]


def test_runtime_flags_stale_workstations_as_layout_change():
    smap = make_map([Workstation(f"WS0{i}", (i * 100, 0, i * 100 + 50, 50), (i * 100, 0, i * 100 + 50, 50), 1.0,
                                 STATUS_CONFIRMED, "admin") for i in range(1, 3)])
    mapper = SceneMapper(smap, SceneRuntimeConfig(check_interval_seconds=10, evidence_stale_seconds=20,
                                                  layout_change_fraction=0.5))
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert not mapper.update(frame, 0.0).layout_change_suspected
    mapper.note_person_evidence("WS01", 25.0)
    status = mapper.update(frame, 30.0)
    assert status.stale_workstations == ["WS02"] and status.layout_change_suspected


def test_review_session_accept_reject_edit_add_undo():
    ws = [Workstation("WS01", (0, 0, 100, 100), (0, 0, 120, 120), 0.5, STATUS_NEEDS_REVIEW, "auto"),
          Workstation("WS02", (300, 0, 400, 100), (300, 0, 420, 120), 0.9, STATUS_CONFIRMED, "auto")]
    session = ReviewSession(make_map(ws))
    assert session.selected == "WS01"
    assert session.set_status(STATUS_CONFIRMED)
    session.select_at(350, 50)
    assert session.selected == "WS02"
    assert session.set_box((310, 10, 390, 95))
    assert session.add((600, 0, 700, 100)) == "WS03"
    session.selected = "WS02"
    assert session.set_status(STATUS_REJECTED)
    result = session.finalize(previous_version=1)
    assert result.version == 2
    assert result.get("WS01").status == STATUS_CONFIRMED
    assert result.get("WS02").seat_box == (310, 10, 390, 95) and result.get("WS02").status == STATUS_REJECTED
    assert len(result.active_workstations) == 2
    assert session.undo() and session.undo()
    assert len(session.scene.workstations) == 2
