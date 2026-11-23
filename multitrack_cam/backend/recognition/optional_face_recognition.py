"""Optional face recognition module (disabled unless explicitly configured).

Designed for consent-based enrolled deployments. This stub keeps the interface
stable without requiring InsightFace at install time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceMatch:
    enrolled_id: str
    score: float
    quality: float


class FaceRecognitionModule:
    def __init__(self, enabled: bool = False, **kwargs: Any) -> None:
        self.enabled = bool(enabled)
        self.kwargs = kwargs
        self._enrolled: dict[str, list[np.ndarray]] = {}
        if self.enabled:
            logger.warning(
                "Face recognition enabled in config, but runtime backend is a safe stub. "
                "Install/configure InsightFace separately before production biometric use."
            )

    def enroll(self, enrolled_id: str, face_images: list[np.ndarray]) -> None:
        if not self.enabled:
            raise RuntimeError("Face recognition is disabled")
        self._enrolled[enrolled_id] = face_images

    def match(self, face_crop: np.ndarray) -> FaceMatch | None:
        if not self.enabled:
            return None
        # Never invent a match when face is not usable / backend not loaded.
        return None

    def delete_enrollment(self, enrolled_id: str) -> None:
        self._enrolled.pop(enrolled_id, None)
