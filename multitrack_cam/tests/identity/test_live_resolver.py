from __future__ import annotations

import numpy as np

from backend.association.camera_topology import CameraTopology
from backend.association.global_associator import AssociatorConfig, GlobalAssociator
from backend.association.ground_plane import GroundPlaneRegistry, OverlapGroup
from backend.association.multiview import MultiViewConfig, MultiViewLinker
from backend.identity.global_identity_manager import GlobalIdentityManager, InMemoryRepository
from backend.identity.live_identity_resolver import LiveIdentityResolver, ResolverConfig
from backend.reid.embedding_gallery import EmbeddingGallery
from backend.types import LocalTrack, TrackLifecycle

RNG = np.random.default_rng(7)
PEOPLE = [v / np.linalg.norm(v) for v in RNG.normal(size=(6, 128)).astype(np.float32)]
SHAPE = (720, 1280)


def _view(person: int, noise: float = 0.25) -> np.ndarray:
    v = PEOPLE[person] + noise * RNG.normal(size=128).astype(np.float32) / np.sqrt(128) * 4
    return v / np.linalg.norm(v)


def _track(cam: str, tid: int, ts: float, box, removed: bool = False, hits: int = 5, session: str = "s") -> LocalTrack:
    return LocalTrack(
        camera_id=cam,
        tracker_session_id=session,
        local_track_id=tid,
        frame_id=int(ts * 10),
        timestamp=ts,
        bbox_xyxy=(0.0, 0.0, 0.0, 0.0) if removed else tuple(map(float, box)),
        confidence=0.0 if removed else 0.9,
        lifecycle=TrackLifecycle.REMOVED if removed else TrackLifecycle.CONFIRMED,
        is_predicted=removed,
        hit_count=hits,
    )


def _resolver(linker=None, topology=None, **cfg):
    gallery = EmbeddingGallery()
    topo = topology or CameraTopology({"transitions": [], "cameras": {}})
    mgr = GlobalIdentityManager(GlobalAssociator(gallery, topo, AssociatorConfig()), gallery, repository=InMemoryRepository())
    base = dict(min_age_to_enroll=0.5, overlap_enroll_delay=1.0)
    base.update(cfg)
    return LiveIdentityResolver(mgr, gallery, topo, linker, ResolverConfig(**base)), mgr


def _run(res, frames):
    """frames: list of (ts, {cam: [(tid, person, box)]}, removed: {cam: [tid]})."""
    for ts, by_cam, removed in frames:
        tracks, embs = {}, {}
        for cam, items in by_cam.items():
            tracks.setdefault(cam, [])
            for tid, person, box in items:
                tr = _track(cam, tid, ts, box)
                tracks[cam].append(tr)
                if person is not None:
                    embs[tr.compound_key] = (_view(person), 0.8)
        for cam, tids in (removed or {}).items():
            tracks.setdefault(cam, []).extend(_track(cam, t, ts, None, removed=True) for t in tids)
        res.update(ts, tracks, {c: SHAPE for c in tracks}, embs, wall_now=1_700_000_000 + ts)


def _gid(res, cam, tid, session="s"):
    st = res.tracks.get(f"{cam}:{session}:{tid}")
    return st.gid if st and st.status == "confirmed" else None


def test_new_person_gets_one_sticky_id():
    res, mgr = _resolver()
    box = (100, 100, 200, 400)
    _run(res, [(0.5 * i, {"cam_01": [(1, 0, box)]}, None) for i in range(10)])
    gid = _gid(res, "cam_01", 1)
    assert gid and gid.startswith("G")
    _run(res, [(5 + 0.5 * i, {"cam_01": [(1, 0, box)]}, None) for i in range(6)])
    assert _gid(res, "cam_01", 1) == gid
    assert len(mgr.list_identities()) == 1


def test_two_people_get_distinct_ids_and_reentry_keeps_id():
    res, mgr = _resolver()
    a, b = (100, 100, 200, 400), (700, 100, 800, 400)
    _run(res, [(0.5 * i, {"cam_01": [(1, 0, a), (2, 1, b)]}, None) for i in range(8)])
    g0, g1 = _gid(res, "cam_01", 1), _gid(res, "cam_01", 2)
    assert g0 and g1 and g0 != g1
    # Person 0 leaves; comes back much later elsewhere in the frame with a new local id.
    _run(res, [(10.0, {"cam_01": [(2, 1, b)]}, {"cam_01": [1]})])
    _run(res, [(60 + 0.5 * i, {"cam_01": [(2, 1, b), (7, 0, (400, 120, 500, 420))]}, None) for i in range(8)])
    assert _gid(res, "cam_01", 7) == g0
    assert _gid(res, "cam_01", 2) == g1
    assert len(mgr.list_identities()) == 2


def test_same_camera_exclusivity_prevents_duplicate_live_id():
    res, _ = _resolver()
    _run(res, [(0.5 * i, {"cam_01": [(1, 0, (100, 100, 200, 400))]}, None) for i in range(8)])
    g = _gid(res, "cam_01", 1)
    # A second simultaneous track that looks like the same person must not get the same id.
    _run(res, [(4 + 0.5 * i, {"cam_01": [(1, 0, (100, 100, 200, 400)), (9, 0, (900, 100, 1000, 400))]}, None) for i in range(8)])
    assert _gid(res, "cam_01", 1) == g
    assert _gid(res, "cam_01", 9) != g


def test_tracker_fragmentation_inherits_id_by_continuity():
    res, mgr = _resolver()
    box = (300, 100, 400, 400)
    _run(res, [(0.5 * i, {"cam_01": [(1, 0, box)]}, None) for i in range(6)])
    g = _gid(res, "cam_01", 1)
    # Local tracker switches id (old one lost, new one at the same place).
    shifted = (305, 102, 405, 402)
    _run(res, [(3.5 + 0.5 * i, {"cam_01": [(2, 0, shifted)]}, None) for i in range(3)])
    assert _gid(res, "cam_01", 2) == g
    assert len(mgr.list_identities()) == 1


def test_overlapping_views_share_one_id_via_geometry(tmp_path):
    registry = GroundPlaneRegistry(
        [OverlapGroup("hall", "cam_02", ["cam_02", "cam_03"])],
        manual_path=tmp_path / "cal.yaml",
        auto_path=tmp_path / "auto.json",
    )
    # cam_03 sees the hall mirrored horizontally relative to cam_02.
    pts3 = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9], [0.5, 0.5], [0.3, 0.7]]
    pts2 = [[1 - x, y] for x, y in pts3]
    registry.set_manual("cam_03", pts3, pts2)
    linker = MultiViewLinker(registry, MultiViewConfig())
    res, mgr = _resolver(linker=linker)

    def box_at(cx, top):
        return (cx * 1280 - 50, top * 720, cx * 1280 + 50, top * 720 + 300)

    frames = []
    for i in range(14):
        ts = 0.5 * i
        frames.append(
            (
                ts,
                {
                    # Person 0 at x=0.3 in cam_02 == x=0.7 in cam_03; appearance differs per view.
                    "cam_02": [(1, 0, box_at(0.3, 0.2)), (2, 1, box_at(0.75, 0.2))],
                    "cam_03": [(5, 2, box_at(0.7, 0.2)), (6, 3, box_at(0.25, 0.2))],
                },
                None,
            )
        )
    _run(res, frames)
    g_a2, g_a3 = _gid(res, "cam_02", 1), _gid(res, "cam_03", 5)
    g_b2, g_b3 = _gid(res, "cam_02", 2), _gid(res, "cam_03", 6)
    assert g_a2 and g_a2 == g_a3
    assert g_b2 and g_b2 == g_b3
    assert g_a2 != g_b2
    live = [r for r in mgr.list_identities() if r["status"] != "ARCHIVED"]
    assert len(live) == 2


def test_overlapping_views_do_not_share_id_without_link():
    topo = CameraTopology({"transitions": [], "cameras": {}, "simultaneous_allowed_pairs": [["cam_02", "cam_03"]]})
    res, _ = _resolver(topology=topo)
    _run(res, [(0.5 * i, {"cam_03": [(1, 0, (100, 100, 200, 400))]}, None) for i in range(8)])
    g = _gid(res, "cam_03", 1)
    # A look-alike in the other hall view at the same time, with no geometric link.
    _run(res, [(4 + 0.5 * i, {"cam_03": [(1, 0, (100, 100, 200, 400))], "cam_02": [(4, 0, (600, 100, 700, 400))]}, None) for i in range(30)])
    assert _gid(res, "cam_03", 1) == g
    assert _gid(res, "cam_02", 4) != g


def test_cross_camera_gallery_match_needs_stricter_score():
    res, _ = _resolver(match_threshold=0.5, cross_camera_match_threshold=0.99, cross_camera_margin=0.5, ambiguous_timeout=1000)
    _run(res, [(0.5 * i, {"cam_01": [(1, 0, (100, 100, 200, 400))]}, None) for i in range(6)])
    g = _gid(res, "cam_01", 1)
    _run(res, [(3.0, {}, {"cam_01": [1]})])
    _run(res, [(300 + 0.5 * i, {"cam_04": [(2, 0, (100, 100, 200, 400))]}, None) for i in range(6)])
    st = res.tracks["cam_04:s:2"]
    assert st.gid != g and st.candidate == g


def test_non_overlapping_cameras_cannot_hold_same_id_simultaneously():
    res, _ = _resolver()
    _run(res, [(0.5 * i, {"cam_01": [(1, 0, (100, 100, 200, 400))]}, None) for i in range(8)])
    g = _gid(res, "cam_01", 1)
    # Same appearance appears in cam_04 while still visible in cam_01.
    _run(res, [(4 + 0.5 * i, {"cam_01": [(1, 0, (100, 100, 200, 400))], "cam_04": [(3, 0, (100, 100, 200, 400))]}, None) for i in range(6)])
    assert _gid(res, "cam_01", 1) == g
    assert _gid(res, "cam_04", 3) != g


def _anchor_scenario(tmp_path, **cfg):
    registry = GroundPlaneRegistry(
        [OverlapGroup("hall", "cam_02", ["cam_02", "cam_03"])],
        manual_path=tmp_path / "cal.yaml",
        auto_path=tmp_path / "auto.json",
        auto_enabled=False,
    )
    registry.set_anchors("cam_03", [[0.37, 0.43]], [[0.75, 0.37]])
    res, mgr = _resolver(linker=MultiViewLinker(registry, MultiViewConfig()), merge_window=0.0, **cfg)
    seat3 = (423.6, 273.6, 523.6, 573.6)  # head anchor at (0.37, 0.43)
    seat2 = (910.0, 230.4, 1010.0, 530.4)  # head anchor at (0.75, 0.37)
    # Each view enrolls the seated person on its own (e.g. before calibration existed).
    _run(res, [(0.5 * i, {"cam_03": [(1, 0, seat3)]}, None) for i in range(10)])
    g3 = _gid(res, "cam_03", 1)
    _run(res, [(5.5, {}, {"cam_03": [1]})])
    _run(res, [(20 + 0.5 * i, {"cam_02": [(2, 1, seat2)]}, None) for i in range(10)])
    g2 = _gid(res, "cam_02", 2)
    _run(res, [(25.5, {}, {"cam_02": [2]})])
    _run(res, [(40 + 0.5 * i, {"cam_03": [(3, 0, seat3)], "cam_02": [(4, 1, seat2)]}, None) for i in range(24)])
    # Not merged while the bindings are younger than anchor_merge_min_bound_seconds.
    assert _gid(res, "cam_03", 3) != _gid(res, "cam_02", 4)
    _run(res, [(52 + 0.5 * i, {"cam_03": [(3, 0, seat3)], "cam_02": [(4, 1, seat2)]}, None) for i in range(50)])
    return res, mgr, g3, g2


def test_seat_anchor_link_merges_established_duplicate_into_reference_id(tmp_path):
    res, mgr, g3, g2 = _anchor_scenario(tmp_path)
    assert g3 and g2 and g3 != g2
    assert _gid(res, "cam_02", 4) == g2
    assert _gid(res, "cam_03", 3) == g2
    assert not mgr.is_live_identity(g3)


def test_seat_anchor_only_relabels_when_member_id_has_other_camera_history(tmp_path):
    registry = GroundPlaneRegistry(
        [OverlapGroup("hall", "cam_02", ["cam_02", "cam_03"])],
        manual_path=tmp_path / "cal.yaml",
        auto_path=tmp_path / "auto.json",
        auto_enabled=False,
    )
    registry.set_anchors("cam_03", [[0.37, 0.43]], [[0.75, 0.37]])
    res, mgr = _resolver(linker=MultiViewLinker(registry, MultiViewConfig()), merge_window=0.0)
    seat3 = (423.6, 273.6, 523.6, 573.6)
    seat2 = (910.0, 230.4, 1010.0, 530.4)
    _run(res, [(0.5 * i, {"cam_03": [(1, 0, seat3)]}, None) for i in range(10)])
    g3 = _gid(res, "cam_03", 1)
    mgr.add_live_embedding(g3, _view(0), camera_id="cam_04", timestamp=1.0, quality=0.8, persist=False)
    _run(res, [(5.5, {}, {"cam_03": [1]})])
    _run(res, [(20 + 0.5 * i, {"cam_02": [(2, 1, seat2)]}, None) for i in range(10)])
    g2 = _gid(res, "cam_02", 2)
    _run(res, [(25.5, {}, {"cam_02": [2]})])
    _run(res, [(40 + 0.5 * i, {"cam_03": [(3, 0, seat3)], "cam_02": [(4, 1, seat2)]}, None) for i in range(80)])
    assert _gid(res, "cam_03", 3) == g2 and _gid(res, "cam_02", 4) == g2
    assert mgr.is_live_identity(g3)


def test_established_duplicate_not_merged_when_anchor_merge_disabled(tmp_path):
    res, mgr, g3, g2 = _anchor_scenario(tmp_path, anchor_merge_min_votes=0)
    assert _gid(res, "cam_03", 3) == g3 and _gid(res, "cam_02", 4) == g2
    assert mgr.is_live_identity(g3)


def _transit_scenario(gap: float):
    topo = CameraTopology({
        "cameras": {},
        "transitions": [{"from": "cam_02", "to": "cam_04", "min_travel_seconds": 1.0, "max_travel_seconds": 180.0}],
    })
    res, mgr = _resolver(topology=topo)
    e = np.eye(128, dtype=np.float32)
    a, c = e[0], e[1]
    a_room = 0.58 * e[0] + 0.55 * e[1] + np.sqrt(1 - 0.58**2 - 0.55**2) * e[2]  # cos(a)=0.58, cos(c)=0.55
    box = (100, 100, 200, 400)

    def run(cam, tid, t0, emb, n=10):
        for k in range(n):
            ts = t0 + 0.5 * k
            tr = _track(cam, tid, ts, box)
            res.update(ts, {cam: [tr]}, {cam: SHAPE}, {tr.compound_key: (emb, 0.8)}, wall_now=1_700_000_000 + ts)
        ts = t0 + 0.5 * n
        res.update(ts, {cam: [_track(cam, tid, ts, None, removed=True)]}, {cam: SHAPE}, {}, wall_now=1_700_000_000 + ts)

    run("cam_01", 1, 0.0, c)                      # distractor, seen long ago elsewhere
    g_c = mgr.list_identities()[-1]["global_id"]
    run("cam_02", 2, 300.0, a)                    # person leaves the hall at ~305 s
    g_a = [r["global_id"] for r in mgr.list_identities() if r["global_id"] != g_c][-1]
    t_room = 305.0 + gap
    for k in range(10):
        ts = t_room + 0.5 * k
        tr = _track("cam_04", 3, ts, box)
        res.update(ts, {"cam_04": [tr]}, {"cam_04": SHAPE}, {tr.compound_key: (a_room / np.linalg.norm(a_room), 0.8)},
                   wall_now=1_700_000_000 + ts)
    return _gid(res, "cam_04", 3), g_a, g_c


def test_person_walking_hall_to_room_keeps_id_via_recent_departure():
    g_room, g_a, g_c = _transit_scenario(gap=15.0)
    assert g_room == g_a and g_room != g_c


def test_recent_departure_prior_does_not_apply_outside_travel_window():
    g_room, g_a, _ = _transit_scenario(gap=420.0)
    assert g_room != g_a


def test_slow_overlap_camera_still_blocks_unlinked_id_sharing_between_its_frames():
    topo = CameraTopology({"transitions": [], "cameras": {}, "simultaneous_allowed_pairs": [["cam_02", "cam_03"]]})
    res, _ = _resolver(topology=topo)
    box = (100, 100, 200, 400)
    _run(res, [(4.0 * i, {"cam_03": [(1, 0, box)]}, None) for i in range(6)])
    g = _gid(res, "cam_03", 1)
    assert g
    frames = []
    for k in range(40):  # cam_02 at 2 fps starting 2.5 s into a gap, cam_03 only every 4 s (0.25 fps)
        t = 22.5 + 0.5 * k
        by_cam = {"cam_02": [(3, 0, (700, 100, 800, 400))]}
        if k % 8 == 3:
            by_cam["cam_03"] = [(1, 0, box)]
        frames.append((t, by_cam, None))
    _run(res, frames)
    assert _gid(res, "cam_03", 1) == g
    assert _gid(res, "cam_02", 3) != g


def _handoff_scenario(room_emb: dict[int, np.ndarray], walkers: int = 1, room_moving: bool = True, seated: bool = True):
    """Hall walkers (cam_02) leave the view while moving; newcomers walk into room 3 (cam_04) 3 s later."""
    topo = CameraTopology({
        "cameras": {},
        "transitions": [{"from": "cam_02", "to": "cam_04", "min_travel_seconds": 1.0, "max_travel_seconds": 180.0}],
    })
    res, mgr = _resolver(topology=topo)
    e = np.eye(128, dtype=np.float32)
    seat = (1000, 300, 1100, 600)
    for k in range(20):  # hall: walkers cross the frame; a seated person stays put
        ts = 0.5 * k
        tracks, embs = [], {}
        for w in range(walkers):
            x = 100 + 40 * k
            tr = _track("cam_02", 10 + w, ts, (x, 100 + 250 * w, x + 100, 400 + 250 * w))
            tracks.append(tr)
            embs[tr.compound_key] = (e[w], 0.8)
        if seated:
            tr = _track("cam_02", 30, ts, seat)
            tracks.append(tr)
            embs[tr.compound_key] = (e[5], 0.8)
        res.update(ts, {"cam_02": tracks}, {"cam_02": SHAPE}, embs, wall_now=1_700_000_000 + ts)
    hall = [_gid(res, "cam_02", 10 + w) for w in range(walkers)]
    assert all(hall)
    for k in range(16):  # walkers are gone (lost, not yet removed); room newcomers arrive
        ts = 13.0 + 0.5 * k
        tracks = {"cam_04": []}
        embs = {}
        for w, emb in room_emb.items():
            x = 100 + (30 * k if room_moving else 0)
            tr = _track("cam_04", 50 + w, ts, (x, 100, x + 100, 400))
            tracks["cam_04"].append(tr)
            embs[tr.compound_key] = (emb / np.linalg.norm(emb), 0.8)
        if seated:
            tracks["cam_02"] = [_track("cam_02", 30, ts, seat)]
            embs[tracks["cam_02"][0].compound_key] = (e[5], 0.8)
        res.update(ts, tracks, {c: SHAPE for c in tracks}, embs, wall_now=1_700_000_000 + ts)
    return [_gid(res, "cam_04", 50 + w) for w in room_emb], hall


def _weak(e_idx: int, cos: float) -> np.ndarray:
    e = np.eye(128, dtype=np.float32)
    return cos * e[e_idx] + np.sqrt(1 - cos**2) * e[100]


def test_walker_keeps_id_hall_to_room_despite_weak_cross_view_appearance():
    room, hall = _handoff_scenario({0: _weak(0, 0.35)})
    assert room[0] == hall[0]


def test_walkout_handoff_rejects_contradicting_appearance():
    room, hall = _handoff_scenario({0: _weak(0, 0.05)})
    assert room[0] is not None and room[0] != hall[0]


def test_walkout_handoff_requires_newcomer_to_be_walking():
    room, hall = _handoff_scenario({0: _weak(0, 0.35)}, room_moving=False)
    assert room[0] != hall[0]


def test_walkout_handoff_ambiguous_between_two_walkers_is_not_forced():
    room, hall = _handoff_scenario({0: _weak(0, 0.33) + _weak(1, 0.33)}, walkers=2)
    assert room[0] not in hall


def _dedup_setup(**cfg):
    base = dict(dedup_min_observed_seconds=0.0, dedup_interval=1.0)
    base.update(cfg)
    res, mgr = _resolver(**base)
    gids = [mgr.create_live_identity("cam_02", 1_700_000_000.0, reason="test") for _ in range(4)]
    people = [0, 0, 1, 2]  # gids[0] and gids[1] are one person split in two
    for g, p in zip(gids, people):
        for k in range(10):
            mgr.add_live_embedding(g, _view(p, noise=0.1), camera_id="cam_02",
                                   timestamp=1_700_000_000.0 + k, quality=0.8, persist=False)
    return res, mgr, gids


def _idle(res, t0, n=4):
    for k in range(n):
        res.update(t0 + 2.0 * k, {}, {}, {}, wall_now=1_700_000_000 + t0 + 2.0 * k)


def test_auto_dedup_merges_split_identity_never_seen_together():
    res, mgr, g = _dedup_setup()
    _idle(res, 0.0)
    assert mgr.is_live_identity(g[0]) and not mgr.is_live_identity(g[1])
    assert all(mgr.is_live_identity(x) for x in g[2:])


def test_auto_dedup_never_merges_ids_seen_together():
    res, mgr, g = _dedup_setup()
    a, b = _track("cam_02", 1, 0.0, (100, 100, 200, 400)), _track("cam_02", 2, 0.0, (600, 100, 700, 400))
    res.update(0.0, {"cam_02": [a, b]}, {"cam_02": SHAPE}, {})
    for key, gid in ((a.compound_key, g[0]), (b.compound_key, g[1])):
        res.tracks[key].gid, res.tracks[key].status = gid, "confirmed"
    a2, b2 = _track("cam_02", 1, 0.5, (100, 100, 200, 400)), _track("cam_02", 2, 0.5, (600, 100, 700, 400))
    res.update(0.5, {"cam_02": [a2, b2]}, {"cam_02": SHAPE}, {})
    _idle(res, 2.0)
    assert mgr.is_live_identity(g[0]) and mgr.is_live_identity(g[1])


def test_auto_dedup_waits_for_enough_observation():
    res, mgr, g = _dedup_setup(dedup_min_observed_seconds=60.0)
    _idle(res, 0.0)
    assert mgr.is_live_identity(g[1])


def test_ids_survive_camera_reconnect_tracker_reset():
    res, _ = _resolver()
    seat = (300, 200, 420, 500)
    _run(res, [(0.5 * k, {"cam_02": [(1, 0, seat)]}, None) for k in range(10)])
    g = _gid(res, "cam_02", 1)
    assert g
    # 7 s outage, tracker restarts with a new session; same seated person, appearance not yet sampled
    t0 = 4.5 + 7.0
    for k in range(4):
        ts = t0 + 0.5 * k
        tr = _track("cam_02", 1, ts, seat, session="s2")
        res.update(ts, {"cam_02": [tr]}, {"cam_02": SHAPE}, {}, wall_now=1_700_000_000 + ts)
    assert _gid(res, "cam_02", 1, session="s2") == g


def test_stable_track_teaches_id_its_new_pose_but_not_an_id_switch():
    res, mgr = _resolver(stable_learn_seconds=5.0, gallery_update_interval=1.0)
    e = np.eye(128, dtype=np.float32)
    box = (300, 200, 420, 500)
    for k in range(20):  # seated look
        ts = 0.5 * k
        tr = _track("cam_02", 1, ts, box)
        res.update(ts, {"cam_02": [tr]}, {"cam_02": SHAPE}, {tr.compound_key: (e[0], 0.8)}, wall_now=1_700_000_000 + ts)
    g = _gid(res, "cam_02", 1)
    stand = 0.45 * e[0] + np.sqrt(1 - 0.45**2) * e[1]  # same person standing: gallery sim 0.45
    for k in range(20):
        ts = 10.0 + 0.5 * k
        tr = _track("cam_02", 1, ts, box)
        res.update(ts, {"cam_02": [tr]}, {"cam_02": SHAPE}, {tr.compound_key: (stand, 0.8)}, wall_now=1_700_000_000 + ts)
    assert _gid(res, "cam_02", 1) == g
    assert res.gallery.max_similarity(g, stand) > 0.99
    other = 0.4 * e[0] + np.sqrt(1 - 0.4**2) * e[7]  # tracker jumped to someone else for one sample
    st = res.tracks["cam_02:s:1"]
    st.bank.append((other, 0.9, 30.0))
    assert not res._stable_new_pose(st, other, 0.4, 30.0)


FACES = [v / np.linalg.norm(v) for v in np.random.default_rng(11).normal(size=(4, 512)).astype(np.float32)]


def _face(person: int, noise: float = 0.6) -> np.ndarray:
    v = FACES[person] + noise * np.random.default_rng().normal(size=512).astype(np.float32) / np.sqrt(512)
    return v / np.linalg.norm(v)


def _run_faces(res, cam, tid, t0, n, body, face, box=(300, 200, 420, 500), session="s"):
    for k in range(n):
        ts = t0 + 0.5 * k
        tr = _track(cam, tid, ts, box, session=session)
        embs = {tr.compound_key: (_view(body), 0.8)} if body is not None else {}
        faces = {tr.compound_key: (_face(face), 0.9)} if face is not None else None
        res.update(ts, {cam: [tr]}, {cam: SHAPE}, embs, wall_now=1_700_000_000 + ts, faces=faces)


def _remove(res, cam, tid, ts, session="s"):
    res.update(ts, {cam: [_track(cam, tid, ts, None, removed=True, session=session)]}, {cam: SHAPE}, {},
               wall_now=1_700_000_000 + ts)


def _attr_cfg():
    # Thresholds for synthetic high-separation anchors (same role as CLIP attrs).
    return dict(face_match_threshold=0.55, face_margin=0.08, face_unknown_threshold=0.30,
                face_veto_threshold=0.25, face_merge_threshold=0.55, face_self_consistency=0.30,
                face_match_requires_reentry=False)

def test_face_keeps_id_when_body_appearance_changes_completely():
    res, mgr = _resolver(**_attr_cfg())
    _run_faces(res, "cam_02", 1, 0.0, 16, body=0, face=0)       # seated: body p0, face A
    g = _gid(res, "cam_02", 1)
    assert g and mgr.face_gallery.count(g) > 0
    _remove(res, "cam_02", 1, 8.0)
    _run_faces(res, "cam_02", 2, 60.0, 12, body=3, face=0, box=(800, 100, 900, 500))  # walking: new body look
    assert _gid(res, "cam_02", 2) == g


def test_face_vetoes_body_match_to_wrong_person():
    res, mgr = _resolver(**_attr_cfg())
    _run_faces(res, "cam_02", 1, 0.0, 20, body=0, face=0)
    g = _gid(res, "cam_02", 1)
    _remove(res, "cam_02", 1, 10.0)
    _run_faces(res, "cam_02", 2, 60.0, 20, body=0, face=1)      # same clothes, different face
    g2 = _gid(res, "cam_02", 2)
    assert g2 is not None and g2 != g


def test_face_merges_one_person_split_into_two_ids():
    res, mgr = _resolver(dedup_interval=1.0, **_attr_cfg())
    g1 = mgr.create_live_identity("cam_02", 1_700_000_000.0, reason="t")
    g2 = mgr.create_live_identity("cam_02", 1_700_000_000.0, reason="t")
    g3 = mgr.create_live_identity("cam_02", 1_700_000_000.0, reason="t")
    for g, p in ((g1, 0), (g2, 0), (g3, 1)):
        for k in range(4):
            mgr.add_face_embedding(g, _face(p), camera_id="cam_02", timestamp=1_700_000_000.0 + k, quality=0.9)
    _idle(res, 0.0)
    assert mgr.is_live_identity(g1) and not mgr.is_live_identity(g2) and mgr.is_live_identity(g3)
