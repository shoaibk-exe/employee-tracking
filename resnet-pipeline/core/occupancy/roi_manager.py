"""Workstation region-of-interest (ROI) loading, validation, and crop extraction.

ROIs are axis-aligned rectangles ``[x1, y1, x2, y2]`` in pixel coordinates of the
(already rotated) frame used during calibration. Crops are taken as
``frame[y1:y2, x1:x2]`` and returned as views, without resizing.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

logger = logging.getLogger(__name__)

MIN_ROI_SIDE_PX = 8
ALLOWED_ROTATIONS = (0, 90, 180, 270)


class ROIConfigError(ValueError):
    """Raised when workstation ROI configuration is missing, malformed, or incompatible."""


@dataclass(frozen=True)
class Workstation:
    workstation_id: str
    bbox: tuple[int, int, int, int]

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


def validate_bbox(
    workstation_id: str, bbox: Iterable[Any], frame_width: int, frame_height: int
) -> tuple[int, int, int, int]:
    """Return ``bbox`` as an int tuple, or raise if it is malformed or not fully inside the frame."""
    values = list(bbox)
    if len(values) != 4:
        raise ROIConfigError(f"{workstation_id}: bbox must have 4 values [x1, y1, x2, y2], got {values}")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ROIConfigError(f"{workstation_id}: bbox values must be numbers, got {values}")
        if isinstance(value, float) and not value.is_integer():
            raise ROIConfigError(f"{workstation_id}: bbox values must be whole pixels, got {values}")
    x1, y1, x2, y2 = (int(v) for v in values)
    if x2 <= x1 or y2 <= y1:
        raise ROIConfigError(f"{workstation_id}: bbox requires x1 < x2 and y1 < y2, got {values}")
    if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
        raise ROIConfigError(
            f"{workstation_id}: bbox {values} lies partly outside the {frame_width}x{frame_height} frame. "
            "Recalibrate with scripts/select_rois.py instead of relying on clipping."
        )
    if (x2 - x1) < MIN_ROI_SIDE_PX or (y2 - y1) < MIN_ROI_SIDE_PX:
        raise ROIConfigError(
            f"{workstation_id}: bbox {values} is smaller than {MIN_ROI_SIDE_PX}px on a side"
        )
    return x1, y1, x2, y2


class ROIManager:
    """Holds workstation ROIs and extracts per-workstation crops from frames."""

    def __init__(
        self,
        workstations: Mapping[str, Iterable[Any]],
        reference_resolution: tuple[int, int],
        rotation: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not workstations:
            raise ROIConfigError(
                "No workstations configured. Run: python -m scripts.scene.select_rois --source <video>"
            )
        ref_w, ref_h = (int(v) for v in reference_resolution)
        if ref_w <= 0 or ref_h <= 0:
            raise ROIConfigError(f"reference_resolution must be positive, got {reference_resolution}")
        if rotation is not None and rotation not in ALLOWED_ROTATIONS:
            raise ROIConfigError(f"rotation must be one of {ALLOWED_ROTATIONS}, got {rotation}")
        self.reference_resolution = (ref_w, ref_h)
        self.rotation = rotation
        self.metadata = dict(metadata or {})
        self._reference: dict[str, Workstation] = {}
        for workstation_id, bbox in workstations.items():
            if not isinstance(workstation_id, str) or not workstation_id.strip():
                raise ROIConfigError(f"Workstation IDs must be non-empty strings, got {workstation_id!r}")
            self._reference[workstation_id] = Workstation(
                workstation_id, validate_bbox(workstation_id, bbox, ref_w, ref_h)
            )
        self._active: dict[str, Workstation] = dict(self._reference)
        self._frame_size: tuple[int, int] | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "ROIManager":
        path = Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            raise ROIConfigError(
                f"ROI file {path} is missing or empty. "
                "Run: python -m scripts.scene.select_rois --source <video>"
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ROIConfigError(f"ROI file {path} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ROIConfigError(f"ROI file {path} must contain a JSON object")
        resolution = data.get("reference_resolution")
        if not isinstance(resolution, list) or len(resolution) != 2:
            raise ROIConfigError(f"ROI file {path} needs reference_resolution: [width, height]")
        raw = data.get("workstations")
        if not isinstance(raw, dict):
            raise ROIConfigError(f"ROI file {path} needs a 'workstations' object")
        workstations: dict[str, Any] = {}
        for workstation_id, entry in raw.items():
            if not isinstance(entry, dict) or "bbox" not in entry:
                raise ROIConfigError(f"{workstation_id}: expected an object with a 'bbox' field")
            workstations[workstation_id] = entry["bbox"]
        metadata = {
            k: v for k, v in data.items() if k not in ("reference_resolution", "workstations", "rotation")
        }
        return cls(workstations, tuple(resolution), data.get("rotation"), metadata)

    def to_json(self, path: str | Path, extra: Mapping[str, Any] | None = None) -> None:
        """Write the reference ROIs atomically so an interrupted save never corrupts the file."""
        path = Path(path)
        payload: dict[str, Any] = {
            "reference_resolution": list(self.reference_resolution),
            "rotation": self.rotation,
            **self.metadata,
            **(extra or {}),
            "workstations": {
                ws.workstation_id: {"bbox": list(ws.bbox)} for ws in self._reference.values()
            },
        }
        payload.setdefault("saved_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    @property
    def workstation_ids(self) -> list[str]:
        return list(self._reference)

    @property
    def workstations(self) -> list[Workstation]:
        """ROIs in the coordinates of the frame size passed to ``configure_for_frame``."""
        return list(self._active.values())

    def check_rotation(self, rotation: int) -> None:
        if self.rotation is None:
            logger.warning(
                "ROI file does not record the frame rotation used during calibration; "
                "make sure it matches the runtime rotation (%s).", rotation,
            )
        elif self.rotation != rotation:
            raise ROIConfigError(
                f"ROIs were calibrated with rotation={self.rotation} but runtime rotation is {rotation}. "
                "Use the same rotation or recalibrate."
            )

    def configure_for_frame(
        self, frame_width: int, frame_height: int, allow_scaling: bool = True, aspect_tolerance: float = 0.01
    ) -> None:
        """Bind ROIs to a runtime frame size.

        Identical size uses ROIs as-is. A different size with the same aspect ratio is
        scaled proportionally when ``allow_scaling`` is set; anything else is rejected.
        """
        ref_w, ref_h = self.reference_resolution
        if (frame_width, frame_height) == (ref_w, ref_h):
            self._active = dict(self._reference)
            self._frame_size = (frame_width, frame_height)
            return
        ref_aspect = ref_w / ref_h
        aspect = frame_width / frame_height
        if not allow_scaling or abs(aspect - ref_aspect) / ref_aspect > aspect_tolerance:
            raise ROIConfigError(
                f"Frame size {frame_width}x{frame_height} is incompatible with ROI reference "
                f"resolution {ref_w}x{ref_h}. Recalibrate ROIs on this stream."
            )
        sx, sy = frame_width / ref_w, frame_height / ref_h
        logger.warning(
            "Scaling ROIs from %dx%d to %dx%d (same geometry assumed). Verify the camera has not moved.",
            ref_w, ref_h, frame_width, frame_height,
        )
        scaled: dict[str, Workstation] = {}
        for ws in self._reference.values():
            x1, y1, x2, y2 = ws.bbox
            box = (
                math.floor(x1 * sx), math.floor(y1 * sy),
                min(math.ceil(x2 * sx), frame_width), min(math.ceil(y2 * sy), frame_height),
            )
            scaled[ws.workstation_id] = Workstation(
                ws.workstation_id, validate_bbox(ws.workstation_id, box, frame_width, frame_height)
            )
        self._active = scaled
        self._frame_size = (frame_width, frame_height)

    def extract(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """Return ``{workstation_id: crop}`` views into ``frame`` (no copy, no resize)."""
        if frame is None or frame.ndim < 2 or frame.size == 0:
            raise ValueError("Cannot extract ROIs from an empty frame")
        height, width = frame.shape[:2]
        if self._frame_size is None:
            self.configure_for_frame(width, height)
        elif self._frame_size != (width, height):
            raise ROIConfigError(
                f"Frame size changed from {self._frame_size[0]}x{self._frame_size[1]} to {width}x{height}"
            )
        crops: dict[str, np.ndarray] = {}
        for ws in self._active.values():
            x1, y1, x2, y2 = ws.bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                raise ROIConfigError(f"{ws.workstation_id}: produced an empty crop for bbox {ws.bbox}")
            crops[ws.workstation_id] = crop
        return crops
