"""Face quality gate. A poor face is rejected instead of forced into an identity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.config import FaceQualityConfig
from app.utils.image import blur_score, mean_brightness


class FaceQualityReason(str, Enum):
    OK = "OK"
    FACE_TOO_SMALL = "FACE_TOO_SMALL"
    FACE_BLURRY = "FACE_BLURRY"
    FACE_BAD_ANGLE = "FACE_BAD_ANGLE"
    FACE_LOW_CONFIDENCE = "FACE_LOW_CONFIDENCE"
    FACE_BAD_EXPOSURE = "FACE_BAD_EXPOSURE"
    FACE_OCCLUDED = "FACE_OCCLUDED"


@dataclass
class FaceSample:
    bbox: tuple[float, float, float, float]
    det_score: float
    yaw: float | None
    pitch: float | None
    landmark_count: int
    crop: np.ndarray


def assess(sample: FaceSample, rules: FaceQualityConfig) -> FaceQualityReason:
    if sample.det_score < rules.minimum_detection_confidence:
        return FaceQualityReason.FACE_LOW_CONFIDENCE
    x1, y1, x2, y2 = sample.bbox
    if (x2 - x1) < rules.minimum_width or (y2 - y1) < rules.minimum_height:
        return FaceQualityReason.FACE_TOO_SMALL
    if sample.crop.size == 0:
        return FaceQualityReason.FACE_TOO_SMALL
    if blur_score(sample.crop) > rules.maximum_blur_score:
        return FaceQualityReason.FACE_BLURRY
    if sample.yaw is None or sample.pitch is None:
        return FaceQualityReason.FACE_BAD_ANGLE
    if abs(sample.yaw) > rules.maximum_yaw or abs(sample.pitch) > rules.maximum_pitch:
        return FaceQualityReason.FACE_BAD_ANGLE
    brightness = mean_brightness(sample.crop)
    if brightness < rules.minimum_exposure or brightness > rules.maximum_exposure:
        return FaceQualityReason.FACE_BAD_EXPOSURE
    # Five-point landmarks are the alignment input. Missing them usually means occlusion.
    if sample.landmark_count < 5:
        return FaceQualityReason.FACE_OCCLUDED
    return FaceQualityReason.OK
