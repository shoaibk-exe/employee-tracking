"""Phone detector boundary for a later phase.

V1 does not run this. A phone lying on a desk is not phone use, and the crop-then-detect
path only matters after attendance and desk sessions are already trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.detection.employee_detector import Detection


@dataclass
class PhoneDetection(Detection):
    pass


class PhoneDetector:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def detect_crop(self, person_crop: np.ndarray) -> list[PhoneDetection]:
        raise RuntimeError(
            "Phone detection is disabled. Finish attendance and desk monitoring before enabling it."
        )
