"""Small image helpers. Models never see these; quality checks do."""

from __future__ import annotations

import numpy as np


def blur_score(gray_or_bgr: np.ndarray) -> float:
    """Higher means blurrier.

    Laplacian variance rises as an image gets sharper, which is the opposite of a
    "maximum blur" threshold. Invert it so YAML can reject large scores.
    The scale must be calibrated on the real entrance camera.
    """
    import cv2

    image = gray_or_bgr
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(image, cv2.CV_64F).var())
    return 1000.0 / (variance + 1e-6)


def mean_brightness(bgr: np.ndarray) -> float:
    if bgr.size == 0:
        return 0.0
    return float(bgr.mean())


def clip_bbox(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    ix1 = max(0, min(width - 1, int(x1)))
    iy1 = max(0, min(height - 1, int(y1)))
    ix2 = max(ix1 + 1, min(width, int(x2)))
    iy2 = max(iy1 + 1, min(height, int(y2)))
    return ix1, iy1, ix2, iy2
