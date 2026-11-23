"""Same-day appearance gallery: enroll P01/P02, rematch after leave, no double live ID."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np

from core.identity.appearance_gallery import AppearanceGallery, AppearanceGalleryConfig


class FakeEncoder:
    dim = 4

    def __init__(self, vectors: dict[str, np.ndarray]):
        self.vectors = {k: v / np.linalg.norm(v) for k, v in vectors.items()}

    def embed(self, crop: np.ndarray) -> np.ndarray:
        key = "blue" if float(crop.mean()) < 80 else "red"
        return self.vectors[key].astype(np.float32)


class FakeDescriber:
    """Deterministic VLM stand-in: crop brightness → clothing label."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _label(self, crop: np.ndarray) -> str:
        m = float(crop.mean())
        if m < 80:
            return "person wearing a dark blue shirt and black pants"
        if m < 150:
            return "person wearing a green hoodie and gray jeans"
        return "person wearing a bright red jacket and blue jeans"

    def describe(self, bgr_crop: np.ndarray) -> str:
        self.calls.append("describe")
        return self._label(bgr_crop)

    def match_id(
        self,
        bgr_crop: np.ndarray,
        candidates: dict[str, str],
        *,
        query_description: str | None = None,
    ) -> str | None:
        self.calls.append("match")
        label = query_description or self._label(bgr_crop)
        # Use a distinctive phrase unique to each fake clothing class.
        keys = ("dark blue shirt", "bright red jacket", "green hoodie")
        key = next((k for k in keys if k in label), None)
        if key is None:
            return None
        for pid, desc in candidates.items():
            if key in desc:
                return pid
        return None


def make_reid_gallery(tmp_path: Path, encoder: FakeEncoder) -> AppearanceGallery:
    cfg = AppearanceGalleryConfig(
        enabled=True,
        backend="reid",
        model=tmp_path / "dummy.onnx",
        enroll_samples=2,
        match_threshold=0.85,
        sole_match_threshold=0.85,
        match_margin=0.05,
        update_interval_seconds=0.0,
        min_crop_side=16,
        directory=tmp_path / "gallery",
    )
    return AppearanceGallery(cfg, encoder=encoder)


def make_vlm_gallery(tmp_path: Path, describer: FakeDescriber) -> AppearanceGallery:
    cfg = AppearanceGalleryConfig(
        enabled=True,
        backend="vlm",
        model="fake-vlm",
        enroll_samples=2,
        update_interval_seconds=0.0,
        min_crop_side=16,
        directory=tmp_path / "gallery_vlm",
    )
    return AppearanceGallery(cfg, describer=describer)


def test_second_person_gets_p02_not_p01(tmp_path):
    blue = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    red = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    g = make_reid_gallery(tmp_path, FakeEncoder({"blue": blue, "red": red}))
    blue_crop = np.full((80, 60, 3), 40, dtype=np.uint8)
    red_crop = np.full((80, 60, 3), 200, dtype=np.uint8)
    today = date(2026, 10, 23)

    g.observe("a", 1, blue_crop, 0.0, today=today)
    assert g.observe("a", 1, blue_crop, 0.1, today=today).person_id == "P01"
    g.observe("b", 2, red_crop, 1.0, today=today)
    assert g.observe("b", 2, red_crop, 1.1, today=today).person_id == "P02"
    assert g.person_of("a") == "P01" and g.person_of("b") == "P02"


def test_leave_and_return_reuses_same_p_id(tmp_path):
    blue = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    red = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    g = make_reid_gallery(tmp_path, FakeEncoder({"blue": blue, "red": red}))
    blue_crop = np.full((80, 60, 3), 40, dtype=np.uint8)
    red_crop = np.full((80, 60, 3), 200, dtype=np.uint8)
    today = date(2026, 10, 23)

    g.observe("t1", 1, blue_crop, 0.0, today=today)
    g.observe("t1", 1, blue_crop, 0.1, today=today)
    g.observe("t2", 2, red_crop, 0.2, today=today)
    g.observe("t2", 2, red_crop, 0.3, today=today)
    assert {p.person_id for p in g.people()} == {"P01", "P02"}

    g.forget_track("t1")
    g.forget_track("t2")
    assert g.person_of("t1") is None and len(g.people()) == 2

    back = g.observe("t99", 99, blue_crop, 40.0, today=today)
    assert back.person_id == "P01" and back.reason == "matched_gallery"

    g.observe("t99", 99, blue_crop, 40.1, today=today)
    green = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    class ThreeEncoder(FakeEncoder):
        def embed(self, crop: np.ndarray) -> np.ndarray:
            m = float(crop.mean())
            if m < 80:
                return self.vectors["blue"]
            if m < 150:
                return green / np.linalg.norm(green)
            return self.vectors["red"]

    g.encoder = ThreeEncoder({"blue": blue, "red": red})  # type: ignore[assignment]
    mid = np.full((80, 60, 3), 120, dtype=np.uint8)
    g.observe("t3", 3, mid, 41.0, today=today)
    enrolled = g.observe("t3", 3, mid, 41.1, today=today)
    assert enrolled.person_id == "P03" and enrolled.enrolled


def test_day_rollover_discards_gallery(tmp_path):
    blue = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    g = make_reid_gallery(tmp_path, FakeEncoder({"blue": blue, "red": blue}))
    crop = np.full((80, 60, 3), 40, dtype=np.uint8)
    d0 = date(2026, 10, 23)
    g.observe("a", 1, crop, 0.0, today=d0)
    g.observe("a", 1, crop, 0.1, today=d0)
    assert len(g.people()) == 1
    first = g.observe("b", 2, crop, 0.0, today=d0 + timedelta(days=1))
    assert first.person_id is None and len(g.people()) == 0
    second = g.observe("b", 2, crop, 0.1, today=d0 + timedelta(days=1))
    assert second.person_id == "P01" and second.enrolled


def test_vlm_second_person_gets_p02_not_p01(tmp_path):
    g = make_vlm_gallery(tmp_path, FakeDescriber())
    blue_crop = np.full((80, 60, 3), 40, dtype=np.uint8)
    red_crop = np.full((80, 60, 3), 200, dtype=np.uint8)
    today = date(2026, 10, 23)

    g.observe("a", 1, blue_crop, 0.0, today=today)
    assert g.observe("a", 1, blue_crop, 0.1, today=today).person_id == "P01"
    g.observe("b", 2, red_crop, 1.0, today=today)
    assert g.observe("b", 2, red_crop, 1.1, today=today).person_id == "P02"
    assert g.person_of("a") == "P01" and g.person_of("b") == "P02"
    assert "blue" in g.people()[0].description or "red" in g.people()[0].description


def test_ghost_lost_track_does_not_block_gallery_rematch(tmp_path):
    """Tracker miss leaves a LOST ghost holding P01 — new detection must reclaim P01+T."""
    blue = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    g = make_reid_gallery(tmp_path, FakeEncoder({"blue": blue, "red": blue}))
    crop = np.full((80, 60, 3), 40, dtype=np.uint8)
    today = date(2026, 10, 23)

    g.set_observed_tracks({"old"})
    g.observe("old", 1, crop, 0.0, today=today)
    assert g.observe("old", 1, crop, 0.1, today=today).person_id == "P01"
    assert g.display_track_id("P01") == 1

    # Person briefly missed: ghost track still mapped, but NOT observed.
    g.set_observed_tracks(set())
    # New Ultralytics ID → new dense T5 — must rematch P01 and keep display T1.
    g.set_observed_tracks({"new"})
    back = g.observe("new", 5, crop, 1.0, today=today)
    assert back.person_id == "P01"
    assert back.reason == "matched_gallery"
    assert back.display_track_id == 1
    assert g.person_of("new") == "P01"
    assert g.person_of("old") is None  # stolen from ghost


def test_vlm_leave_and_return_reuses_description(tmp_path):
    g = make_vlm_gallery(tmp_path, FakeDescriber())
    blue_crop = np.full((80, 60, 3), 40, dtype=np.uint8)
    red_crop = np.full((80, 60, 3), 200, dtype=np.uint8)
    today = date(2026, 10, 23)

    g.observe("t1", 1, blue_crop, 0.0, today=today)
    g.observe("t1", 1, blue_crop, 0.1, today=today)
    g.observe("t2", 2, red_crop, 0.2, today=today)
    g.observe("t2", 2, red_crop, 0.3, today=today)
    g.forget_track("t1")
    g.forget_track("t2")

    back = g.observe("t99", 99, blue_crop, 40.0, today=today)
    assert back.person_id == "P01" and back.reason == "matched_gallery"

    # New look while P01 live → P03
    g.observe("t99", 99, blue_crop, 40.1, today=today)
    mid = np.full((80, 60, 3), 120, dtype=np.uint8)
    g.observe("t3", 3, mid, 41.0, today=today)
    enrolled = g.observe("t3", 3, mid, 41.1, today=today)
    assert enrolled.person_id == "P03" and enrolled.enrolled
