from __future__ import annotations

import numpy as np

from backend.association.ground_plane import GroundPlaneRegistry, OverlapGroup
from backend.association.multiview import MultiViewConfig, MultiViewLinker, ViewObservation

W, H = 1000, 1000


def _obs(cam: str, key: str, head_xy: tuple[float, float], t: float, emb=None) -> ViewObservation:
    # Box chosen so that the "head" anchor (y1 + 0.12 h) equals head_xy.
    cx, hy = head_xy[0] * W, head_xy[1] * H
    bh = 300.0
    y1 = hy - 0.12 * bh
    return ViewObservation(key, cam, (cx - 50, y1, cx + 50, y1 + bh), W, H, t, emb)


def _linker(tmp_path) -> MultiViewLinker:
    reg = GroundPlaneRegistry(
        [OverlapGroup("hall", "cam_02", ["cam_02", "cam_03"])],
        manual_path=tmp_path / "cal.yaml",
        auto_path=tmp_path / "auto.json",
        auto_enabled=False,
    )
    reg.set_anchors("cam_03", [[0.37, 0.43], [0.56, 0.45]], [[0.75, 0.37], [0.43, 0.37]])
    return MultiViewLinker(reg, MultiViewConfig())


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_person_at_verified_seat_links_across_views(tmp_path):
    lk = _linker(tmp_path)
    links = {}
    for t in range(3):
        links = lk.update([_obs("cam_03", "c3:1", (0.372, 0.432), t), _obs("cam_02", "c2:7", (0.752, 0.372), t)])
    assert links["c3:1"].partner == "c2:7" and links["c3:1"].geometric


def test_neighbouring_seat_in_reference_view_is_not_linked(tmp_path):
    lk = _linker(tmp_path)
    links = {}
    for t in range(4):
        links = lk.update([_obs("cam_03", "c3:1", (0.372, 0.432), t), _obs("cam_02", "c2:8", (0.83, 0.35), t)])
    assert "c3:1" not in links
    # Seat occupant missed by the detector; the person 0.04 away must not take the link.
    for t in range(4, 8):
        links = lk.update([_obs("cam_03", "c3:1", (0.372, 0.432), t), _obs("cam_02", "c2:9", (0.712, 0.385), t)])
    assert "c3:1" not in links


def test_stationary_pair_links_despite_slow_stream_but_moving_pair_does_not(tmp_path):
    lk = _linker(tmp_path)
    links = {}
    for t in range(3):
        a = _obs("cam_03", "c3:1", (0.372, 0.432), 5.0 * t)
        b = _obs("cam_02", "c2:7", (0.752, 0.372), 5.0 * t + 4.0)
        a.stationary = b.stationary = True
        links = lk.update([a, b])
    assert links["c3:1"].partner == "c2:7"

    lk2 = _linker(tmp_path)
    for t in range(3):
        links = lk2.update([_obs("cam_03", "c3:1", (0.372, 0.432), 5.0 * t),
                            _obs("cam_02", "c2:7", (0.752, 0.372), 5.0 * t + 4.0)])
    assert "c3:1" not in links


def test_off_anchor_needs_strict_appearance(tmp_path):
    lk = _linker(tmp_path)
    a = _unit([1.0, 0.0])
    moderate = _unit([0.5, 0.866])  # cosine 0.5: above min_link_score, below app_only_threshold
    links = {}
    for t in range(4):
        links = lk.update([_obs("cam_03", "c3:1", (0.20, 0.80), t, a), _obs("cam_02", "c2:7", (0.60, 0.30), t, moderate)])
    assert "c3:1" not in links
