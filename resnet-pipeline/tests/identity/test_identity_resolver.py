"""PersonIdentityResolver: P## survives tracker breaks, never merges, never duplicates."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.identity.appearance_gallery import AppearanceGalleryConfig
from core.identity.identity_resolver import PersonIdentityResolver

ROOT = Path(__file__).resolve().parents[2]
DAY = date(2026, 10, 23)

# Crop fill value → identity vector. LOOKALIKE scores 0.47 against A: inside the
# look-alike band (0.43–0.58) that must not be merged into the only known person.
A, B, LOOKALIKE = 50, 100, 150
_c = 0.47
VECTORS = {
    A: np.array([1.0, 0.0, 0.0, 0.0]),
    B: np.array([0.0, 1.0, 0.0, 0.0]),
    LOOKALIKE: np.array([_c, 0.0, np.sqrt(1 - _c * _c), 0.0]),
}


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def embed_many(self, crops):
        self.calls += 1
        out = []
        for crop in crops:
            v = VECTORS[int(round(float(crop.mean())))]
            out.append((v / np.linalg.norm(v)).astype(np.float32))
        return out


def make_resolver(tmp_path: Path, **overrides) -> PersonIdentityResolver:
    kwargs = {"persist_day_gallery": False, "backend": "reid", **overrides}
    cfg = AppearanceGalleryConfig(
        enabled=True, model=tmp_path / "dummy.onnx",
        directory=tmp_path / "gallery", **kwargs,
    )
    return PersonIdentityResolver(cfg, encoder=FakeEncoder())


def box(x: int, y: int = 200) -> tuple[int, int, int, int]:
    return (x, y, x + 70, y + 180)


def step(res, t, people, removed=(), today=DAY):
    """people: [(local_id, bbox, fill)] → assignments keyed by local id."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    tracks = []
    for lid, bb, fill in people:
        frame[bb[1]:bb[3], bb[0]:bb[2]] = fill
        # Stamp one pixel so successive frames have distinct crop hashes while
        # preserving FakeEncoder mean→identity mapping (mean still rounds to fill).
        stamp = int(round(t * 10)) + lid  # 0.0,0.2,0.4 → 0,2,4,…
        frame[bb[1], bb[0], 0] = (int(fill) + 1 + (stamp % 40)) % 256
        tracks.append(SimpleNamespace(
            composite_track_id=f"cam:{lid}", local_track_id=lid, bbox=bb,
            is_predicted=False, lifecycle=SimpleNamespace(value="CONFIRMED"),
        ))
    out = res.update(frame, t, tracks, [f"cam:{r}" for r in removed], today=today)
    return {int(k.split(":")[1]): v for k, v in out.items()}


def run(res, t0, t1, people, dt=0.1, removed=()):
    seen: dict[int, list] = {}
    t, first = t0, True
    while t < t1 - 1e-9:
        out = step(res, t, people, removed if first else ())
        first = False
        for lid, a in out.items():
            seen.setdefault(lid, []).append(a.person_id)
        t += dt
    return seen


def test_new_person_shows_pending_then_enrolls(tmp_path):
    res = make_resolver(tmp_path)
    seen = run(res, 0.0, 3.0, [(1, box(100), A)])
    assert seen[1][0] is None  # P? while collecting samples, no instant enrollment
    assert seen[1][-1] == "P01"
    assert len(res.people()) == 1


def test_redetected_after_miss_keeps_person_and_display_t(tmp_path):
    res = make_resolver(tmp_path)
    run(res, 0.0, 3.0, [(1, box(100), A)])
    run(res, 3.0, 3.5, [], removed=[1])  # detector misses him for 0.5 s
    seen = run(res, 3.5, 5.0, [(2, box(115), A)])  # tracker gives a new T
    assert seen[2][-1] == "P01"
    assert "P02" not in seen[2]
    assert res.display_track_id("P01") == 1  # stable T shown on screen
    assert len(res.people()) == 1


def test_returning_person_rematched_by_appearance_elsewhere(tmp_path):
    res = make_resolver(tmp_path)
    run(res, 0.0, 3.0, [(1, box(100), A)])
    run(res, 3.0, 40.0, [], dt=1.0, removed=[1])
    seen = run(res, 40.0, 43.0, [(7, box(900), A)])
    assert seen[7][-1] == "P01"
    assert len(res.people()) == 1


def test_two_live_people_never_share_a_person(tmp_path):
    res = make_resolver(tmp_path)
    seen = run(res, 0.0, 3.0, [(1, box(100), A), (2, box(600), B)])
    assert seen[1][-1] and seen[2][-1]
    assert seen[1][-1] != seen[2][-1]
    for p1, p2 in zip(seen[1], seen[2]):
        assert p1 is None or p1 != p2
    assert len(res.people()) == 2


def test_lookalike_is_never_merged_into_only_known_person(tmp_path):
    res = make_resolver(tmp_path)
    run(res, 0.0, 3.0, [(1, box(100), A)])
    run(res, 3.0, 10.0, [], dt=1.0, removed=[1])
    seen = run(res, 10.0, 40.0, [(3, box(900), LOOKALIKE)], dt=0.2)
    assert "P01" not in seen[3]
    assert seen[3][-1] == "P02"


def test_lookalike_of_someone_visible_elsewhere_enrolls_without_long_wait(tmp_path):
    res = make_resolver(tmp_path)
    run(res, 0.0, 3.0, [(1, box(100), A)])
    # A stays at his desk; a look-alike sits at a far desk at the same time.
    seen = run(res, 3.0, 15.0, [(1, box(100), A), (2, box(900), LOOKALIKE)])
    assert "P01" not in seen[2]
    assert seen[2][-1] == "P02"
    assert seen[2].index("P02") * 0.1 < 10.0  # decided at the normal timeout, not 3x


def test_tracker_identity_switch_releases_and_rebinds(tmp_path):
    res = make_resolver(tmp_path)
    run(res, 0.0, 3.0, [(1, box(100), A), (2, box(600), B)])
    pa = res.person_of("cam:1")
    pb = res.person_of("cam:2")
    run(res, 3.0, 4.0, [(1, box(100), A)], removed=[2])
    # Tracker swaps: T1's box now contains B (A walked off behind B).
    seen = run(res, 4.0, 12.0, [(1, box(100), B)])
    assert seen[1][-1] == pb
    person_a = next(p for p in res.people() if p.person_id == pa)
    b_vec = VECTORS[B] / np.linalg.norm(VECTORS[B])
    assert max(float(v @ b_vec) for v in person_a.bank) < 0.5  # A's bank not polluted by B


def test_day_rollover_clears_people(tmp_path):
    res = make_resolver(tmp_path)
    run(res, 0.0, 3.0, [(1, box(100), A)])
    assert res.person_of("cam:1") == "P01"
    out = step(res, 3.1, [(1, box(100), A)], today=date(2026, 10, 24))
    assert out[1].person_id is None
    assert res.people() == []


def test_day_file_round_trip(tmp_path):
    res = make_resolver(tmp_path, persist_day_gallery=True)
    res._day = None
    res._ensure_day(DAY)
    run(res, 0.0, 3.0, [(1, box(100), A)])
    res.save()
    again = make_resolver(tmp_path, persist_day_gallery=True)
    again._day = None
    again._ensure_day(DAY)
    assert [p.person_id for p in again.people()] == ["P01"]
    seen = run(again, 0.0, 3.0, [(9, box(900), A)])
    assert seen[9][-1] == "P01"


def test_settings_parse_resolver_keys():
    from core.config import load_settings

    s = load_settings(ROOT / "config" / "settings.yaml", {
        "appearance_gallery": {
            "backend": "hybrid",
            "rival_margin": 0.07,
            "continuity_min_score": 0.4,
            "drift_samples": 4,
        },
    })
    ag = s.appearance_gallery
    assert ag.backend == "hybrid"
    assert ag.rival_margin == 0.07
    assert ag.continuity_min_score == 0.4
    assert ag.drift_samples == 4 and isinstance(ag.drift_samples, int)
    assert ag.match_threshold == 0.50
    assert ag.consensus_samples == 5


def _crop_fill_key(crop) -> int:
    """Map a (possibly padded) solid crop to FakeEncoder fill value.

    VLM crops may include black bbox padding; ignore near-black pixels.
    """
    flat = crop.reshape(-1, crop.shape[-1]).astype(np.float32).mean(axis=1)
    mask = flat > 5.0
    vals = flat[mask] if bool(mask.any()) else flat
    return int(round(float(np.median(vals))))


class FakeFeatures:
    """Deterministic VLM stand-in keyed by crop brightness (same as FakeEncoder)."""

    def __init__(self, by_fill: dict[int, dict[str, str]]):
        self.by_fill = by_fill

    def extract(self, crop, **_ctx):
        return self.extract_detailed(crop, **_ctx).features

    def extract_detailed(self, crop, **ctx):
        from core.vlm.vlm_features import VlmExtractResult, crop_sha1

        fill = _crop_fill_key(crop)
        sha = ctx.get("expect_sha1") or crop_sha1(crop)
        return VlmExtractResult(
            features=dict(self.by_fill[fill]),
            raw_by_key={},
            crop_sha1=sha,
            track_id=ctx.get("track_id"),
            frame_index=ctx.get("frame_index"),
            timestamp_seconds=ctx.get("timestamp_seconds"),
        )


def test_sticky_hold_reuses_p_without_recalculating(tmp_path):
    res = make_resolver(tmp_path, sticky_hold_frames=7, sticky_radius=0.8)
    run(res, 0.0, 3.0, [(1, box(100), A)])
    assert res.person_of("cam:1") == "P01"
    # Tracker drops the person for a few frames (lost track).
    run(res, 3.0, 3.3, [], removed=[1], dt=0.05)
    # Reappears nearby with a new track id — must get P01 instantly (sticky), not a new P.
    out = step(res, 3.35, [(9, box(110), A)])
    assert out[9].person_id == "P01"
    assert "sticky" in out[9].reason
    assert res.display_track_id("P01") == 1


def test_cooldown_blocks_new_enroll_but_allows_gallery_rematch(tmp_path):
    """During cool-down: no new P##, but a known gallery person rematches."""
    from dataclasses import replace

    res = make_resolver(tmp_path, min_stable_seconds=0.0)
    run(res, 0.0, 3.0, [(1, box(100), A)])
    assert res.person_of("cam:1") == "P01"
    run(res, 3.0, 3.3, [], removed=[1], dt=0.05)
    # Turn cool-down on: enroll blocked; rematch still OK.
    res.config = replace(res.config, min_stable_seconds=60.0)
    out = step(res, 3.5, [(2, box(110), A)])
    assert out[2].person_id == "P01"
    # Brand-new look during cool-down must NOT enroll P02 yet.
    out_b = None
    t = 4.0
    while t < 8.0:
        out_b = step(res, t, [(3, box(700), B)])
        t += 0.2
    assert out_b is not None
    assert out_b[3].person_id is None
    assert out_b[3].state == "warming"
    assert len(res.people()) == 1


def test_calculating_timeout_drops_track(tmp_path):
    res = make_resolver(
        tmp_path,
        min_stable_seconds=1.0,
        max_calculating_seconds=2.0,
        # Keep unresolved: never enroll / match quickly.
        max_pending_seconds=60.0,
        ambiguous_enroll_seconds=60.0,
        enroll_samples=50,
        consensus_samples=50,
        min_enroll_span_seconds=60.0,
    )
    run(res, 0.0, 1.5, [(1, box(100), A)], dt=0.2)
    assert res.person_of("cam:1") is None
    # Past warm-up: calculating (may sample).
    out = step(res, 1.6, [(1, box(100), A)])
    assert out[1].state == "pending"
    # Still no P after max_calculating → drop.
    out = step(res, 4.0, [(1, box(100), A)])
    assert out[1].state == "drop"
    assert res.consume_drops() == ["cam:1"]


def test_sticky_rejected_when_different_person_takes_the_spot(tmp_path):
    """Position alone must not hand P01 to someone else walking into the same spot."""
    res = make_resolver(tmp_path, sticky_hold_frames=7, sticky_radius=0.8)
    run(res, 0.0, 3.0, [(1, box(100), A)])
    assert res.person_of("cam:1") == "P01"
    run(res, 3.0, 3.2, [], removed=[1], dt=0.05)
    out = step(res, 3.25, [(9, box(110), B)])
    assert out[9].person_id is None
    assert out[9].reason.startswith("sticky_rejected") or out[9].reason.startswith("calculating")
    # Wrong person must not keep re-sticky looping; rematch uses emb gallery instead.
    out2 = step(res, 3.30, [(9, box(110), B)])
    assert out2[9].person_id is None
    # The real P01 coming back right after still gets P01.
    run(res, 3.3, 3.4, [(9, box(600), B)], dt=0.05)
    out = step(res, 3.45, [(9, box(600), B), (12, box(105), A)])
    assert out[12].person_id == "P01"


def test_low_confidence_boxes_are_not_used_for_identity(tmp_path):
    res = make_resolver(tmp_path, min_identity_confidence=0.35)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    bb = box(100)
    frame[bb[1]:bb[3], bb[0]:bb[2]] = A
    weak = SimpleNamespace(
        composite_track_id="cam:1", local_track_id=1, bbox=bb, confidence=0.12,
        is_predicted=False, lifecycle=SimpleNamespace(value="CONFIRMED"),
    )
    t = 0.0
    while t < 4.0:
        res.update(frame, t, [weak], (), today=DAY)
        t += 0.1
    assert res.encoder.calls == 0
    assert res.people() == []


def test_soft_sticky_handoff_from_predicted_ghost(tmp_path):
    """After a few LOST/predicted frames, a new nearby T gets the same P (no calculating)."""
    res = make_resolver(tmp_path, sticky_hold_frames=7, sticky_radius=0.8)
    run(res, 0.0, 3.0, [(1, box(100), A)])
    assert res.person_of("cam:1") == "P01"

    ghost = SimpleNamespace(
        composite_track_id="cam:1", local_track_id=1, bbox=box(100),
        is_predicted=True, lifecycle=SimpleNamespace(value="LOST"),
    )
    # Soft sticky waits ~2–3 missed frames so overlapping duplicate boxes cannot ping-pong.
    for i in range(3):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        res.update(frame, 3.0 + 0.05 * (i + 1), [ghost], (), today=DAY)

    bb = box(110)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    fresh = SimpleNamespace(
        composite_track_id="cam:9", local_track_id=9, bbox=bb,
        is_predicted=False, lifecycle=SimpleNamespace(value="CONFIRMED"),
    )
    # Fill the full crop region so ReID can verify the provisional sticky bind.
    frame[bb[1]:bb[3], bb[0]:bb[2]] = A
    out = res.update(frame, 3.25, [ghost, fresh], (), today=DAY)
    assert out["cam:9"].person_id == "P01"
    assert "sticky" in out["cam:9"].reason
    assert res.person_of("cam:1") is None  # handed off
    assert res.display_track_id("P01") == 1


def test_covisible_lookalike_neighbour_cannot_hijack_seated_person(tmp_path):
    """Run 0c1ed1 / WS01: seated P (T507) blinks out for ~1.5 s; a same-outfit neighbour
    (T531, ReID 0.63 vs P05) that had been visible *next to him* for a minute grabbed P05
    by continuity, and the real person was re-enrolled as P14. Co-visible tracks are two
    people: the neighbour must never get P01 and the returning holder keeps it."""
    # Production cool-down: the neighbour cannot enroll yet, but (before the fix) could
    # still take an existing P## by continuity while the holder was missing.
    res = make_resolver(tmp_path, hold_seconds=1.5, continuity_seconds=3.0, min_stable_seconds=60.0)
    desks = {"cam:1": "WS01"}
    seated = (1, box(100), A)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def go(t, people, removed=()):
        for lid, bb, fill in people:
            frame[bb[1]:bb[3], bb[0]:bb[2]] = fill
        tracks = [SimpleNamespace(
            composite_track_id=f"cam:{lid}", local_track_id=lid, bbox=bb,
            is_predicted=False, lifecycle=SimpleNamespace(value="CONFIRMED"),
        ) for lid, bb, _ in people]
        out = res.update(frame, t, tracks, [f"cam:{r}" for r in removed], today=DAY, desks=desks)
        return {int(k.split(":")[1]): v for k, v in out.items()}

    from dataclasses import replace

    res.config = replace(res.config, min_stable_seconds=0.0)  # let the seated person enroll
    t = 0.0
    while t < 3.0:
        go(t, [seated]); t += 0.1
    assert res.person_of("cam:1") == "P01"
    res.config = replace(res.config, min_stable_seconds=60.0)
    # Neighbour sits down one box-width away (non-overlapping) and stays visible for 20 s
    # (still in cool-down: unresolved, no P## of his own yet).
    neigh = (2, box(180), LOOKALIKE)
    while t < 23.0:
        go(t, [seated, neigh]); t += 0.1
    assert res.person_of("cam:2") is None
    # Detector misses the seated person for 1.6 s (track kept alive by tracker as LOST).
    ghost = SimpleNamespace(
        composite_track_id="cam:1", local_track_id=1, bbox=box(100),
        is_predicted=True, lifecycle=SimpleNamespace(value="LOST"),
    )
    nt = SimpleNamespace(
        composite_track_id="cam:2", local_track_id=2, bbox=box(180),
        is_predicted=False, lifecycle=SimpleNamespace(value="CONFIRMED"),
    )
    while t < 24.6:
        res.update(frame, t, [ghost, nt], (), today=DAY, desks=desks); t += 0.1
    assert res.person_of("cam:2") != "P01"  # neighbour never inherits P01
    # Seated person re-detected at his desk: still P01, no P## churn.
    out = go(t, [seated, neigh])
    assert out[1].person_id == "P01"
    while t < 30.0:
        out = go(t, [seated, neigh]); t += 0.1
    assert out[1].person_id == "P01"
    assert out[2].person_id != "P01"


def test_lost_holder_reclaims_p_from_positional_handover(tmp_path):
    """A P## handed to a new box by position alone goes back to the original track if
    that track is re-detected at its own spot within continuity_seconds."""
    res = make_resolver(tmp_path, coexist_seconds=0.0, hold_seconds=0.2, sticky_hold_frames=1)
    run(res, 0.0, 3.0, [(1, box(100), A)])
    assert res.person_of("cam:1") == "P01"
    ghost = SimpleNamespace(
        composite_track_id="cam:1", local_track_id=1, bbox=box(100),
        is_predicted=True, lifecycle=SimpleNamespace(value="LOST"),
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    t = 3.0
    while t < 3.6:
        res.update(frame, t, [ghost], (), today=DAY); t += 0.1
    # A new box appears next to the lost spot and inherits P01 by continuity.
    bb = box(150)
    frame[bb[1]:bb[3], bb[0]:bb[2]] = LOOKALIKE
    newbie = SimpleNamespace(
        composite_track_id="cam:9", local_track_id=9, bbox=bb,
        is_predicted=False, lifecycle=SimpleNamespace(value="CONFIRMED"),
    )
    got = None
    while t < 6.0 and got != "P01":
        out = res.update(frame, t, [ghost, newbie], (), today=DAY); t += 0.1
        got = out["cam:9"].person_id
    assert got == "P01"  # positional (continuity) hand-over happened
    assert res._tracks["cam:9"].reason.startswith(("continuity", "sticky"))
    # Original track re-detected exactly where it vanished → reclaims P01.
    bb1 = box(100)
    frame[bb1[1]:bb1[3], bb1[0]:bb1[2]] = A
    back = SimpleNamespace(
        composite_track_id="cam:1", local_track_id=1, bbox=bb1,
        is_predicted=False, lifecycle=SimpleNamespace(value="CONFIRMED"),
    )
    out = res.update(frame, t, [back, newbie], (), today=DAY)
    assert out["cam:1"].person_id == "P01"
    assert out["cam:1"].reason.startswith("reclaimed")
    assert out["cam:9"].person_id is None


def test_hybrid_vlm_veto_when_enabled(tmp_path):
    """With vlm_identity_veto=True, emb look-alike + conflicting VLM must not rematch."""
    feats_a = {
        "gender": "male", "glasses": "no", "top_color": "orange",
        "bottom_color": "blue", "hair": "short", "hat": "yes",
    }
    feats_b = {
        "gender": "male", "glasses": "yes", "top_color": "blue",
        "bottom_color": "black", "hair": "short", "hat": "no",
    }
    cfg = AppearanceGalleryConfig(
        enabled=True, backend="hybrid", model=tmp_path / "dummy.onnx",
        directory=tmp_path / "gallery", persist_day_gallery=False,
        # One VLM sample is enough for the veto gate in this unit test.
        consensus_samples=1, enroll_samples=5, min_enroll_span_seconds=0.5,
        pending_sample_interval_seconds=0.0, update_interval_seconds=0.0,
        min_crop_side=16, vlm_min_comparable=2, vlm_majority_min=1,
        vlm_identity_veto=True,
        vlm_gallery_color_min_votes=1, vlm_gallery_color_min_share=0.5,
        max_pending_seconds=4.0, ambiguous_enroll_seconds=6.0,
        timeout_new_threshold=0.55,  # LOOKALIKE@0.47 is "new", not a timeout match
        match_threshold=0.50, match_margin=0.05,
    )
    res = PersonIdentityResolver(
        cfg, encoder=FakeEncoder(),
        feature_extractor=FakeFeatures({A: feats_a, LOOKALIKE: feats_b, B: feats_b}),
    )
    seen = run(res, 0.0, 3.0, [(1, box(100), A)], dt=0.2)
    assert seen[1][-1] == "P01"
    # Allow async VLM backfill onto the enrolled person.
    for t in np.linspace(3.0, 4.0, 6):
        step(res, float(t), [(1, box(100), A)])
    assert res.people()[0].features["top_color"] == "orange"
    run(res, 4.0, 10.0, [], dt=1.0, removed=[1])
    seen2 = run(res, 10.0, 18.0, [(2, box(900), LOOKALIKE)], dt=0.2)
    assert "P01" not in seen2[2]
    assert seen2[2][-1] == "P02"
    seen3 = run(res, 20.0, 28.0, [(3, box(200), A)], dt=0.2)
    assert seen3[3][-1] == "P01"


def test_hybrid_vlm_veto_disabled_by_default(tmp_path):
    """Default: VLM mismatch must not block embedding rematch (veto off)."""
    feats_a = {
        "gender": "male", "glasses": "no", "top_color": "orange",
        "bottom_color": "blue", "hair": "short", "hat": "yes",
    }
    feats_b = {
        "gender": "male", "glasses": "yes", "top_color": "blue",
        "bottom_color": "black", "hair": "short", "hat": "no",
    }
    cfg = AppearanceGalleryConfig(
        enabled=True, backend="hybrid", model=tmp_path / "dummy.onnx",
        directory=tmp_path / "gallery", persist_day_gallery=False,
        consensus_samples=5, enroll_samples=5, min_enroll_span_seconds=0.5,
        pending_sample_interval_seconds=0.0, update_interval_seconds=0.0,
        min_crop_side=16, vlm_min_comparable=2, vlm_majority_min=2,
        # vlm_identity_veto defaults False
        max_pending_seconds=2.0, ambiguous_enroll_seconds=3.0,
        match_threshold=0.50, match_margin=0.05,
        strong_match_threshold=0.45, strong_match_samples=3,
    )
    res = PersonIdentityResolver(
        cfg, encoder=FakeEncoder(),
        feature_extractor=FakeFeatures({A: feats_a, LOOKALIKE: feats_b, B: feats_b}),
    )
    run(res, 0.0, 3.0, [(1, box(100), A)], dt=0.2)
    run(res, 3.0, 10.0, [], dt=1.0, removed=[1])
    # LOOKALIKE ≈ A on embedding; VLM differs but veto is off → rematch P01
    seen2 = run(res, 10.0, 16.0, [(2, box(900), LOOKALIKE)], dt=0.2)
    assert seen2[2][-1] == "P01"


def test_vlm_consensus_rejects_duplicate_crop_hash(tmp_path):
    """Identical crops must not inflate consensus sample count."""
    from core.vlm.vlm_features import crop_sha1, empty_features

    feats = {
        "gender": "male", "glasses": "yes", "top_color": "gray",
        "bottom_color": "black", "hair": "short", "hat": "no",
    }
    cfg = AppearanceGalleryConfig(
        enabled=True, backend="hybrid", model=tmp_path / "dummy.onnx",
        directory=tmp_path / "gallery", persist_day_gallery=False,
        consensus_samples=3, pending_sample_interval_seconds=0.0,
        min_crop_side=16, vlm_identity_veto=False,
    )
    res = PersonIdentityResolver(
        cfg, encoder=FakeEncoder(),
        feature_extractor=FakeFeatures({A: feats}),
    )
    # Seed a track and inject the same feature dict with the same hash thrice.
    step(res, 0.0, [(1, box(100), A)])
    tr = res._tracks["cam:1"]
    crop = np.full((180, 70, 3), A, dtype=np.uint8)
    sha = crop_sha1(crop)
    assert res._append_vlm_sample(tr, feats, sha1=sha, frame_index=1, timestamp_seconds=0.0)
    assert not res._append_vlm_sample(tr, feats, sha1=sha, frame_index=2, timestamp_seconds=0.4)
    assert not res._append_vlm_sample(tr, feats, sha1=sha, frame_index=3, timestamp_seconds=0.8)
    assert len(tr.pending_features) == 1
    assert tr.pending_feature_hashes == [sha]
    # Distinct hash counts as a new consensus sample.
    other = empty_features()
    other.update(feats)
    assert res._append_vlm_sample(
        tr, other, sha1="b" * 40, frame_index=10, timestamp_seconds=1.0,
    )
    assert len(tr.pending_features) == 2
    assert len({m["crop_sha1"] for m in tr.pending_feature_meta}) == 2


def test_vlm_static_person_commits_after_dup_skips(tmp_path):
    """Identical seated crops must not block feature commit forever."""
    from core.vlm.vlm_features import crop_sha1

    feats = {
        "gender": "male", "glasses": "no", "top_color": "gray",
        "bottom_color": "black", "hair": "short", "hat": "no",
    }
    cfg = AppearanceGalleryConfig(
        enabled=True, backend="hybrid", model=tmp_path / "dummy.onnx",
        directory=tmp_path / "gallery", persist_day_gallery=False,
        consensus_samples=5, pending_sample_interval_seconds=0.0,
        min_crop_side=16, vlm_dup_skip_limit=3, vlm_identity_veto=False,
    )
    res = PersonIdentityResolver(
        cfg, encoder=FakeEncoder(), feature_extractor=FakeFeatures({A: feats}),
    )
    step(res, 0.0, [(1, box(100), A)])
    tr = res._tracks["cam:1"]
    sha = crop_sha1(np.full((180, 70, 3), A, dtype=np.uint8))
    assert res._append_vlm_sample(tr, feats, sha1=sha, frame_index=1, timestamp_seconds=0.0)
    assert not res._vlm_samples_sufficient(tr)
    for _ in range(3):
        tr.vlm_dup_skips += 1
    assert res._vlm_unique_exhausted(tr)
    assert res._vlm_samples_sufficient(tr)


def test_vlm_async_hash_mismatch_discarded(tmp_path):
    """Finished future whose crop hash ≠ submitted hash must not enter consensus."""
    from concurrent.futures import Future

    from core.vlm.vlm_features import VlmExtractResult, empty_features

    cfg = AppearanceGalleryConfig(
        enabled=True, backend="hybrid", model=tmp_path / "dummy.onnx",
        directory=tmp_path / "gallery", persist_day_gallery=False,
        consensus_samples=3, min_crop_side=16,
    )
    res = PersonIdentityResolver(cfg, encoder=FakeEncoder(), feature_extractor=FakeFeatures({A: empty_features()}))
    step(res, 0.0, [(1, box(100), A)])
    tr = res._tracks["cam:1"]
    tr.vlm_inflight_sha1 = "a" * 40
    fut: Future = Future()
    fut.set_result(VlmExtractResult(
        features={"gender": "male", "glasses": "yes", "top_color": "gray",
                  "bottom_color": "black", "hair": "short", "hat": "no"},
        raw_by_key={},
        crop_sha1="c" * 40,  # mismatch
        track_id=1, frame_index=5, timestamp_seconds=1.0,
    ))
    tr.vlm_future = fut
    res._collect_vlm_future(tr)
    assert tr.pending_features == []
    assert tr.vlm_future is None
    assert tr.vlm_inflight_sha1 is None


def _hybrid_veto_resolver(tmp_path, feats_by_fill, **overrides):
    kwargs = dict(
        enabled=True, backend="hybrid", model=tmp_path / "dummy.onnx",
        directory=tmp_path / "gallery", persist_day_gallery=False,
        consensus_samples=5, vlm_majority_min=2, vlm_min_comparable=2,
        pending_sample_interval_seconds=0.0, min_crop_side=16,
        vlm_identity_veto=True,
        vlm_gallery_color_min_votes=2, vlm_gallery_color_min_share=0.5,
    )
    kwargs.update(overrides)
    return PersonIdentityResolver(
        AppearanceGalleryConfig(**kwargs), encoder=FakeEncoder(),
        feature_extractor=FakeFeatures(feats_by_fill),
    )


def test_gallery_features_corrected_by_later_votes(tmp_path):
    """An early occluded crop (brown) must not freeze the gallery top colour."""
    from core.identity.identity_resolver import _Person

    res = _hybrid_veto_resolver(tmp_path, {A: {}})
    person = _Person("P06", [np.array([1.0, 0, 0, 0], dtype=np.float32)], 1,
                     box(100), 0.0, 1, "")
    base = {"gender": "male", "bottom_color": "black", "hair": "short"}
    res._add_person_votes(person, [{**base, "top_color": "brown"}])
    assert person.features["top_color"] == "unknown"
    res._add_person_votes(person, [
        {**base, "top_color": "red"},
        {**base, "top_color": "red"},
        {**base, "top_color": "red"},
    ])
    assert person.features["top_color"] == "red"
    assert person.features["bottom_color"] == "black"


def test_positional_bind_released_on_vlm_conflict(tmp_path):
    """Continuity hand-over to someone in clearly different clothes is undone."""
    feats_a = {"gender": "male", "glasses": "no", "top_color": "red",
               "bottom_color": "black", "hair": "short", "hat": "no"}
    feats_b = {"gender": "male", "glasses": "no", "top_color": "teal",
               "bottom_color": "blue", "hair": "short", "hat": "no"}
    res = _hybrid_veto_resolver(tmp_path, {A: feats_a, B: feats_b})
    step(res, 0.0, [(1, box(100), A)])
    t1 = res._tracks["cam:1"]
    t1.pending_features = [dict(feats_a)] * 3
    pid = res._enroll(t1, 0.0, "new")
    assert res._people[pid].features["top_color"] == "red"
    step(res, 0.2, [(2, box(110), B)], removed=[1])
    t2 = res._tracks["cam:2"]
    res._bind(t2, pid, 0.2, "continuity (0.60)", 0.6)
    assert t2.vlm_check_pending
    t2.pending_features = [dict(feats_b)] * 3
    res._finish_positional_check(t2, res._people[pid], 0.4)
    assert t2.person is None
    assert pid in t2.vlm_rejected
    assert res._people[pid].bound_track is None
    assert res._people[pid].features["top_color"] == "red"  # impostor did not vote


def test_positional_bind_kept_when_vlm_agrees(tmp_path):
    feats_a = {"gender": "male", "glasses": "no", "top_color": "red",
               "bottom_color": "black", "hair": "short", "hat": "no"}
    res = _hybrid_veto_resolver(tmp_path, {A: feats_a})
    step(res, 0.0, [(1, box(100), A)])
    t1 = res._tracks["cam:1"]
    t1.pending_features = [dict(feats_a)] * 3
    pid = res._enroll(t1, 0.0, "new")
    step(res, 0.2, [(2, box(110), A)], removed=[1])
    t2 = res._tracks["cam:2"]
    res._bind(t2, pid, 0.2, "continuity (0.60)", 0.6)
    t2.pending_features = [dict(feats_a)] * 3
    res._finish_positional_check(t2, res._people[pid], 0.4)
    assert t2.person == pid
    assert not t2.vlm_check_pending
    assert res._people[pid].feature_samples == 6  # 3 enroll + 3 continuity accept


def test_veto_tolerates_single_conflict_with_strong_agreement(tmp_path):
    from core.identity.identity_resolver import _Person

    res = _hybrid_veto_resolver(tmp_path, {A: {}})
    person = _Person("P01", [np.array([1.0, 0, 0, 0], dtype=np.float32)], 1,
                     box(100), 0.0, 1, "")
    person.features = {"gender": "male", "glasses": "no", "top_color": "red",
                       "bottom_color": "black", "hair": "short", "hat": "no"}
    step(res, 0.0, [(1, box(100), A)])
    tr = res._tracks["cam:1"]
    tr.pending_features = [dict(person.features, top_color="brown")] * 3
    assert res._vlm_verdict(tr, person) == "ok"  # 1 noisy colour vs 5 agreeing keys
    tr.pending_features = [dict(person.features, top_color="teal", bottom_color="blue")] * 3
    assert res._vlm_verdict(tr, person) == "mismatch"
    tr.pending_features = [dict(person.features)]
    assert res._vlm_verdict(tr, person) == "wait"


def test_vlm_sample_skipped_when_crop_overlapped_by_other_person(tmp_path):
    res = _hybrid_veto_resolver(tmp_path, {A: {}})
    step(res, 0.0, [(1, box(100), A)])
    tr = res._tracks["cam:1"]
    if tr.vlm_future is not None:
        tr.vlm_future.result(timeout=5)
    tr.vlm_future = None
    tr.last_vlm_sample = -1e9
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[200:380, 100:170] = A
    res._maybe_queue_vlm(tr, frame, 1.0, [box(120)])
    assert tr.vlm_future is None  # neighbour covers the crop
    res._maybe_queue_vlm(tr, frame, 1.0, [box(900)])
    assert tr.vlm_future is not None
