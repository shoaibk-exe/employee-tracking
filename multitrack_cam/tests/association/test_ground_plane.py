from __future__ import annotations

import numpy as np

from backend.association.ground_plane import (
    GroundPlaneRegistry,
    OverlapGroup,
    apply_homography,
    fit_homography,
    parse_overlap_groups,
)


def test_fit_homography_recovers_projective_map():
    H_true = np.array([[0.9, 0.1, 0.05], [-0.05, 1.1, 0.02], [0.1, 0.05, 1.0]])
    src = np.random.default_rng(0).uniform(0.05, 0.95, size=(20, 2))
    dst = apply_homography(H_true, src)
    H, err, inliers = fit_homography(src, dst)
    assert H is not None and err < 1e-4 and inliers.all()


def test_manual_calibration_persists_and_reloads(tmp_path):
    groups = [OverlapGroup("hall", "cam_02", ["cam_02", "cam_03"])]
    kw = dict(manual_path=tmp_path / "cal.yaml", auto_path=tmp_path / "auto.json")
    reg = GroundPlaneRegistry(groups, **kw)
    assert not reg.has_geometry("cam_03")
    pts = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9], [0.5, 0.4]]
    reg.set_manual("cam_03", pts, [[1 - x, y] for x, y in pts])
    reg2 = GroundPlaneRegistry(groups, **kw)
    assert reg2.has_geometry("cam_03")
    p = reg2.project("cam_03", (0.2, 0.6))
    assert np.allclose(p, (0.8, 0.6), atol=1e-3)
    assert np.allclose(reg2.project("cam_02", (0.2, 0.6)), (0.2, 0.6))


def test_anchor_calibration_maps_only_near_anchors_and_reloads(tmp_path):
    groups = [OverlapGroup("hall", "cam_02", ["cam_02", "cam_03"])]
    kw = dict(manual_path=tmp_path / "cal.yaml", auto_path=tmp_path / "auto.json")
    reg = GroundPlaneRegistry(groups, **kw)
    reg.set_anchors("cam_03", [[0.37, 0.43], [0.56, 0.45]], [[0.75, 0.37], [0.43, 0.37]], labels=["rust", "beige"])
    reg2 = GroundPlaneRegistry(groups, **kw)
    assert reg2.has_geometry("cam_03")
    assert np.allclose(reg2.project("cam_03", (0.375, 0.435)), (0.75, 0.37))
    assert reg2.project("cam_03", (0.46, 0.44)) is None
    assert reg2.status()[0]["cameras"][1]["mode"] == "anchors"
    reg2.set_manual("cam_03", [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]], [[0.9, 0.1], [0.1, 0.1], [0.1, 0.9], [0.9, 0.9]])
    assert reg2.project("cam_03", (0.46, 0.44)) is not None


def test_overlap_groups_from_topology():
    explicit = parse_overlap_groups({"overlap_groups": [{"name": "h", "reference_camera": "cam_03", "cameras": ["cam_02", "cam_03"]}]})
    assert explicit[0].reference_camera == "cam_03"
    derived = parse_overlap_groups({"cameras": {"cam_02": {"overlapping_with": ["cam_03"]}, "cam_03": {}}})
    assert derived[0].cameras == ["cam_02", "cam_03"]
