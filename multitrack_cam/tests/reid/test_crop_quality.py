from __future__ import annotations

import numpy as np

from backend.reid.crop_quality import is_crop_acceptable, score_crop


def test_rejects_tiny_or_empty_crops():
    assert score_crop(np.zeros((5, 5, 3), dtype=np.uint8)) == 0.0
    assert not is_crop_acceptable(0.0)


def test_accepts_reasonable_crop():
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 255, size=(160, 70, 3), dtype=np.uint8)
    q = score_crop(crop, min_side=40, min_blur_laplacian=1.0)
    assert q > 0
    assert is_crop_acceptable(q)
