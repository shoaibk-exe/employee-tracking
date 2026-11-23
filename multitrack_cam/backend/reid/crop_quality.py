"""Quality gates for person crops before ReID embedding."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from backend.utils.geometry import iou_xyxy


def score_crop(
    crop_bgr: np.ndarray,
    *,
    min_side: int = 40,
    min_blur_laplacian: float = 25.0,
) -> float:
    """Return quality in [0, 1]. Zero means reject."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    h, w = crop_bgr.shape[:2]
    if min(h, w) < min_side:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur < min_blur_laplacian:
        return 0.0
    aspect = h / max(1.0, float(w))
    # Prefer upright person-like aspect ratios.
    aspect_score = 1.0 if 1.2 <= aspect <= 4.5 else 0.4
    size_score = min(1.0, min(h, w) / 120.0)
    blur_score = min(1.0, blur / 150.0)
    return float(0.45 * blur_score + 0.35 * size_score + 0.20 * aspect_score)


def is_crop_acceptable(quality: float, min_quality: float = 0.25) -> bool:
    return quality >= min_quality


def occlusion_ratio(
    box: Sequence[float],
    others: Sequence[Sequence[float]],
) -> float:
    """Approximate occlusion as max IoU with other person boxes."""
    if not others:
        return 0.0
    return max(iou_xyxy(box, other) for other in others)


def extract_person_crop(
    frame: np.ndarray,
    bbox_xyxy: Sequence[float],
    pad_x: float = 0.08,
    pad_top: float = 0.05,
    pad_bottom: float = 0.02,
) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    nx1 = int(max(0, x1 - bw * pad_x))
    ny1 = int(max(0, y1 - bh * pad_top))
    nx2 = int(min(w - 1, x2 + bw * pad_x))
    ny2 = int(min(h - 1, y2 + bh * pad_bottom))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return frame[ny1:ny2, nx1:nx2].copy()
