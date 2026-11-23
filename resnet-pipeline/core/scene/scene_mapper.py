"""Physical workstation scene map: admin ROI persistence and runtime upkeep.

A workstation (``WS01``...) is a physical work position, never an employee and never the
current position of a rolling chair. Its geometry is fixed once mapped:

* ``seat_box``    where a seated occupant's visible body appears (association zone core)
* ``context_box`` the workstation-context crop used for the ResNet-18 classifier
* ``anchor``      centre of the seat box

Production maps come from administrator-drawn ROIs. Offline helpers can still cluster
synthetic chair/person evidence for tests and tooling, but the runtime pipeline does not
load a COCO object detector for layout checks.

Runtime upkeep (``SceneMapper.update``) runs at a low cadence and flags probable camera
movement via feature alignment against the map reference frame. Chair movement inside a
workstation never changes the workstation's identity or geometry.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Box = tuple[int, int, int, int]

STATUS_CONFIRMED = "confirmed"
STATUS_NEEDS_REVIEW = "needs_review"
STATUS_REJECTED = "rejected"
SOURCE_AUTO = "auto"
SOURCE_ADMIN = "admin"
SCENE_OBJECT_CLASSES = ("chair", "tv", "laptop", "keyboard")
SURFACE_CLASSES = ("tv", "laptop", "keyboard")
MAP_FORMAT_VERSION = 1


class SceneMapError(ValueError):
    """Missing, malformed, or incompatible scene map."""


# --------------------------------------------------------------------------- geometry


def box_area(b: Sequence[float]) -> float:
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def intersection(a: Sequence[float], b: Sequence[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, float(w)) * max(0.0, float(h))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = intersection(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def overlap_min(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over the smaller box (1.0 when one box contains the other)."""
    denom = min(box_area(a), box_area(b))
    return intersection(a, b) / denom if denom > 0 else 0.0


def containment(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Fraction of ``inner`` lying inside ``outer``."""
    area = box_area(inner)
    return intersection(inner, outer) / area if area > 0 else 0.0


def center(b: Sequence[float]) -> tuple[float, float]:
    return (float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0


def diagonal(b: Sequence[float]) -> float:
    return math.hypot(float(b[2]) - float(b[0]), float(b[3]) - float(b[1]))


def union_box(boxes: Iterable[Sequence[float]]) -> Box:
    boxes = list(boxes)
    return (int(min(b[0] for b in boxes)), int(min(b[1] for b in boxes)),
            int(max(b[2] for b in boxes)), int(max(b[3] for b in boxes)))


def clip_box(b: Sequence[float], width: int, height: int) -> Box | None:
    x1 = int(max(0, min(width, round(b[0]))))
    y1 = int(max(0, min(height, round(b[1]))))
    x2 = int(max(0, min(width, round(b[2]))))
    y2 = int(max(0, min(height, round(b[3]))))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def expand_box(b: Sequence[float], scale_x: float, scale_y: float, shift_y: float = 0.0) -> tuple[float, ...]:
    """Scale a box around its centre; ``shift_y`` moves the centre by a fraction of its height."""
    cx, cy = center(b)
    w, h = float(b[2]) - float(b[0]), float(b[3]) - float(b[1])
    cy += shift_y * h
    return cx - w * scale_x / 2, cy - h * scale_y / 2, cx + w * scale_x / 2, cy + h * scale_y / 2


# --------------------------------------------------------------------------- data model


@dataclass
class Workstation:
    workstation_id: str
    seat_box: Box
    context_box: Box
    confidence: float
    status: str
    source: str
    seated_height_ref: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def anchor(self) -> tuple[float, float]:
        return center(self.seat_box)

    @property
    def active(self) -> bool:
        return self.status != STATUS_REJECTED

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["seat_box"] = list(self.seat_box)
        data["context_box"] = list(self.context_box)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Workstation":
        try:
            return cls(
                workstation_id=str(data["workstation_id"]),
                seat_box=tuple(int(v) for v in data["seat_box"]),  # type: ignore[arg-type]
                context_box=tuple(int(v) for v in data["context_box"]),  # type: ignore[arg-type]
                confidence=float(data["confidence"]),
                status=str(data["status"]),
                source=str(data["source"]),
                seated_height_ref=None if data.get("seated_height_ref") is None else float(data["seated_height_ref"]),
                evidence=dict(data.get("evidence") or {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SceneMapError(f"Malformed workstation entry {data!r}: {exc}") from exc


@dataclass
class SceneMap:
    tenant_id: str
    camera_id: str
    version: int
    frame_size: tuple[int, int]
    rotation: int
    workstations: list[Workstation]
    created_utc: str
    source: str
    reference_frame: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def active_workstations(self) -> list[Workstation]:
        return [w for w in self.workstations if w.active]

    def get(self, workstation_id: str) -> Workstation:
        for w in self.workstations:
            if w.workstation_id == workstation_id:
                return w
        raise KeyError(workstation_id)

    def fingerprint(self) -> str:
        """Content hash used to tie reports to the exact map geometry."""
        payload = json.dumps(
            [(w.workstation_id, w.seat_box, w.context_box, w.status) for w in self.workstations],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": MAP_FORMAT_VERSION,
            "tenant_id": self.tenant_id,
            "camera_id": self.camera_id,
            "version": self.version,
            "frame_size": list(self.frame_size),
            "rotation": self.rotation,
            "created_utc": self.created_utc,
            "source": self.source,
            "reference_frame": self.reference_frame,
            "fingerprint": self.fingerprint(),
            "notes": list(self.notes),
            "workstations": [w.to_dict() for w in self.workstations],
        }

    def save(self, path: str | Path) -> Path:
        """Write atomically; previous versions are kept as ``<name>.v<N>.json``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size > 0:
            try:
                old_version = int(json.loads(path.read_text()).get("version", 0))
            except (ValueError, json.JSONDecodeError):
                old_version = 0
            archive = path.with_name(f"{path.stem}.v{old_version}{path.suffix}")
            if not archive.exists():
                archive.write_bytes(path.read_bytes())
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp, 0o644)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SceneMap":
        path = Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            raise SceneMapError(f"Scene map {path} is missing or empty. Run: python -m scripts.scene.discover_scene")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SceneMapError(f"Scene map {path} is not valid JSON: {exc}") from exc
        if data.get("format_version") != MAP_FORMAT_VERSION:
            raise SceneMapError(f"Unsupported scene map format_version {data.get('format_version')}")
        workstations = [Workstation.from_dict(w) for w in data.get("workstations", [])]
        ids = [w.workstation_id for w in workstations]
        if len(ids) != len(set(ids)):
            raise SceneMapError(f"Duplicate workstation IDs in {path}")
        smap = cls(
            tenant_id=str(data["tenant_id"]), camera_id=str(data["camera_id"]), version=int(data["version"]),
            frame_size=tuple(int(v) for v in data["frame_size"]),  # type: ignore[arg-type]
            rotation=int(data.get("rotation", 0)), workstations=workstations,
            created_utc=str(data["created_utc"]), source=str(data.get("source", "")),
            reference_frame=data.get("reference_frame"), notes=list(data.get("notes") or []),
        )
        stored = data.get("fingerprint")
        if stored and stored != smap.fingerprint():
            raise SceneMapError(f"Scene map {path} fingerprint mismatch (edited without re-saving?)")
        return smap

    def validate_for(self, tenant_id: str, camera_id: str, width: int, height: int, rotation: int) -> None:
        if self.tenant_id != tenant_id or self.camera_id != camera_id:
            raise SceneMapError(
                f"Scene map belongs to tenant={self.tenant_id} camera={self.camera_id}, "
                f"not tenant={tenant_id} camera={camera_id}"
            )
        if self.frame_size != (width, height):
            raise SceneMapError(f"Scene map frame size {self.frame_size} != stream {width}x{height}; re-map the scene")
        if self.rotation != rotation:
            raise SceneMapError(f"Scene map rotation {self.rotation} != runtime rotation {rotation}")
        if not self.active_workstations:
            raise SceneMapError("Scene map has no active workstations")


# --------------------------------------------------------------------------- discovery


@dataclass(frozen=True)
class DiscoveryConfig:
    chair_iou: float = 0.40
    person_iou: float = 0.50
    min_chair_persistence: float = 0.30
    min_dwell_persistence: float = 0.40
    merge_overlap: float = 0.35
    surface_distance_factor: float = 1.6
    duplicate_overlap: float = 0.55
    confirm_confidence: float = 0.70
    review_confidence: float = 0.40
    edge_margin_px: int = 4
    # Context crop = union(seat, seated person) scaled like admin-validated ROIs.
    context_scale_x: float = 1.10
    context_scale_y: float = 1.10
    context_shift_y: float = 0.0


@dataclass
class _Cluster:
    label: str
    boxes: list[Box] = field(default_factory=list)
    confs: list[float] = field(default_factory=list)
    frames: set[int] = field(default_factory=set)
    _median: Box | None = None

    def add(self, box: Box, conf: float, frame: int) -> None:
        self.boxes.append(box)
        self.confs.append(conf)
        self.frames.add(frame)
        self._median = None

    @property
    def median_box(self) -> Box:
        if self._median is None:
            arr = np.array(self.boxes, dtype=float)
            self._median = tuple(int(round(v)) for v in np.median(arr, axis=0))  # type: ignore[assignment]
        return self._median  # type: ignore[return-value]

    def persistence(self, total_frames: int) -> float:
        return len(self.frames) / total_frames if total_frames else 0.0


def cluster_detections(samples: Sequence[Sequence[tuple[Box, float]]], iou_threshold: float, label: str) -> list[_Cluster]:
    """Greedy temporal clustering: each frame contributes at most one box per cluster."""
    clusters: list[_Cluster] = []
    for frame_index, detections in enumerate(samples):
        used: set[int] = set()
        for box, conf in sorted(detections, key=lambda d: -d[1]):
            best_iou, best = 0.0, None
            for ci, cluster in enumerate(clusters):
                if ci in used:
                    continue
                value = iou(box, cluster.median_box)
                if value > best_iou:
                    best_iou, best = value, ci
            if best is not None and best_iou >= iou_threshold:
                clusters[best].add(box, conf, frame_index)
                used.add(best)
            else:
                cluster = _Cluster(label)
                cluster.add(box, conf, frame_index)
                clusters.append(cluster)
                used.add(len(clusters) - 1)
    return clusters


@dataclass
class DiscoveryFrame:
    """Detections from one sampled discovery frame."""

    persons: list[tuple[Box, float]]
    objects: dict[str, list[tuple[Box, float]]]


def discover_workstations(
    frames: Sequence[DiscoveryFrame], frame_size: tuple[int, int], config: DiscoveryConfig = DiscoveryConfig()
) -> tuple[list[Workstation], list[str]]:
    """Deterministic workstation proposals from sampled detections. Returns (workstations, notes)."""
    if not frames:
        raise SceneMapError("Discovery needs at least one sampled frame")
    width, height = frame_size
    total = len(frames)
    notes: list[str] = []

    chairs = [c for c in cluster_detections([f.objects.get("chair", []) for f in frames], config.chair_iou, "chair")
              if c.persistence(total) >= config.min_chair_persistence]
    dwells = [c for c in cluster_detections([f.persons for f in frames], config.person_iou, "person")
              if c.persistence(total) >= config.min_dwell_persistence]
    surfaces: list[_Cluster] = []
    for cls in SURFACE_CLASSES:
        surfaces += [c for c in cluster_detections([f.objects.get(cls, []) for f in frames], 0.4, cls)
                     if c.persistence(total) >= config.min_chair_persistence]

    # Merge each seated-person cluster into the best overlapping chair cluster.
    merged: dict[int, list[_Cluster]] = {i: [] for i in range(len(chairs))}
    unmatched_dwells: list[_Cluster] = []
    for dwell in dwells:
        scores = [(overlap_min(dwell.median_box, ch.median_box), i) for i, ch in enumerate(chairs)]
        best = max(scores, default=(0.0, None))
        if best[1] is not None and best[0] >= config.merge_overlap:
            merged[best[1]].append(dwell)
        else:
            unmatched_dwells.append(dwell)

    candidates: list[dict[str, Any]] = []
    for i, chair in enumerate(chairs):
        persons = merged[i]
        person = max(persons, key=lambda c: len(c.frames)) if persons else None
        candidates.append({
            "seat": chair.median_box,
            "person": person.median_box if person else None,
            "chair_persistence": chair.persistence(total),
            "dwell_persistence": person.persistence(total) if person else 0.0,
            "chair_height": chair.median_box[3] - chair.median_box[1],
            "person_height": (person.median_box[3] - person.median_box[1]) if person else None,
        })
    for dwell in unmatched_dwells:
        candidates.append({
            "seat": dwell.median_box, "person": dwell.median_box, "chair_persistence": 0.0,
            "dwell_persistence": dwell.persistence(total), "chair_height": None,
            "person_height": dwell.median_box[3] - dwell.median_box[1],
        })

    # Seated height per pixel of chair height, from workstations where both were seen.
    ratios = [c["person_height"] / c["chair_height"] for c in candidates
              if c["person_height"] and c["chair_height"]]
    height_ratio = float(np.median(ratios)) if ratios else None

    workstations: list[Workstation] = []
    for c in candidates:
        seat = c["seat"]
        if c["person"] is not None:
            seat = c["person"]
        elif height_ratio and c["chair_height"]:
            seat = (seat[0], int(max(0, round(seat[3] - c["chair_height"] * height_ratio))), seat[2], seat[3])
        reach = diagonal(seat) * config.surface_distance_factor
        sx, sy = center(seat)
        surface_hits = sorted({s.label for s in surfaces if math.hypot(center(s.median_box)[0] - sx,
                                                                       center(s.median_box)[1] - sy) <= reach})
        persistence = max(c["chair_persistence"], c["dwell_persistence"])
        confidence = (0.5 * min(1.0, persistence / 0.6) + 0.3 * (1.0 if surface_hits else 0.0)
                      + 0.2 * (1.0 if c["dwell_persistence"] > 0 else 0.0))
        at_edge = (seat[0] <= config.edge_margin_px or seat[1] <= config.edge_margin_px
                   or seat[2] >= width - config.edge_margin_px or seat[3] >= height - config.edge_margin_px)
        if confidence < config.review_confidence:
            continue
        status = STATUS_CONFIRMED if confidence >= config.confirm_confidence and not at_edge else STATUS_NEEDS_REVIEW
        base = union_box([seat] + ([c["person"]] if c["person"] else []))
        context = clip_box(expand_box(base, config.context_scale_x, config.context_scale_y, config.context_shift_y),
                           width, height)
        if context is None:
            continue
        if c["person_height"]:
            seated_ref = float(c["person_height"])
        elif height_ratio and c["chair_height"]:
            seated_ref = float(c["chair_height"] * height_ratio)
        else:
            seated_ref = None
        workstations.append(Workstation(
            workstation_id="", seat_box=seat, context_box=context, confidence=round(confidence, 3),
            status=status, source=SOURCE_AUTO, seated_height_ref=seated_ref,
            evidence={
                "frames_sampled": total,
                "chair_persistence": round(c["chair_persistence"], 3),
                "dwell_persistence": round(c["dwell_persistence"], 3),
                "surfaces": surface_hits,
                "at_frame_edge": at_edge,
            },
        ))

    # Non-maximum suppression of duplicate proposals for the same physical position.
    workstations.sort(key=lambda w: -w.confidence)
    kept: list[Workstation] = []
    for w in workstations:
        if any(overlap_min(w.seat_box, k.seat_box) >= config.duplicate_overlap for k in kept):
            notes.append(f"suppressed duplicate candidate at {list(w.seat_box)}")
            continue
        kept.append(w)

    # Stable reading order: rows (by anchor y, banded by median seat height), then x.
    band = float(np.median([w.seat_box[3] - w.seat_box[1] for w in kept])) / 2 if kept else 1.0
    kept.sort(key=lambda w: (round(w.anchor[1] / max(band, 1.0)), w.anchor[0]))
    for n, w in enumerate(kept, start=1):
        w.workstation_id = f"WS{n:02d}"
    review = [w.workstation_id for w in kept if w.status == STATUS_NEEDS_REVIEW]
    if review:
        notes.append(f"administrator review required for {review}")
    if height_ratio is None:
        notes.append("no seated reference observed; posture inference will report UNKNOWN where unreferenced")
    return kept, notes


def carry_over_ids(new: list[Workstation], previous: SceneMap | None, min_iou: float = 0.3) -> list[Workstation]:
    """Keep workstation IDs stable across map versions by matching seat geometry."""
    if previous is None:
        return new
    taken: set[str] = set()
    for w in sorted(new, key=lambda x: -x.confidence):
        best = max(((iou(w.seat_box, p.seat_box), p.workstation_id) for p in previous.workstations
                    if p.workstation_id not in taken), default=(0.0, None))
        if best[1] is not None and best[0] >= min_iou:
            w.workstation_id = best[1]
            taken.add(best[1])
        else:
            w.workstation_id = ""
    used = {w.workstation_id for w in new if w.workstation_id} | {p.workstation_id for p in previous.workstations}
    number = 1
    for w in new:
        if not w.workstation_id:
            while f"WS{number:02d}" in used:
                number += 1
            w.workstation_id = f"WS{number:02d}"
            used.add(w.workstation_id)
    return sorted(new, key=lambda w: w.workstation_id)


def workstations_from_admin_rois(rois: Mapping[str, Box], prefix: str = "WS") -> list[Workstation]:
    """Admin-validated layout from manually drawn workstation boxes (one-time camera calibration).

    seat_box and context_box are both set to the admin ROI (no automatic seat refinement).
    """
    result = []
    for n, (roi_id, box) in enumerate(sorted(rois.items()), start=1):
        box = tuple(int(v) for v in box)  # type: ignore[assignment]
        result.append(Workstation(
            workstation_id=f"{prefix}{n:02d}", seat_box=box, context_box=box, confidence=1.0,
            status=STATUS_CONFIRMED, source=SOURCE_ADMIN, seated_height_ref=None,
            evidence={"admin_roi_id": roi_id},
        ))
    return result


def scene_map_from_admin_rois(
    tenant_id: str,
    camera_id: str,
    frame_size: tuple[int, int],
    rotation: int,
    rois: Mapping[str, Box],
    *,
    version: int = 1,
    reference_frame: str | None = None,
    roi_path: str | Path | None = None,
) -> SceneMap:
    """Build a confirmed scene map whose geometry is exactly the admin-drawn ROIs.

    Automatic chair/person discovery and seat refinement are not used. This is the
    recommended source of workstation geometry for production when an administrator
    has already drawn the desks.
    """
    workstations = workstations_from_admin_rois(rois)
    note = "admin ROIs only (no automatic workstation discovery or seat refinement)"
    if roi_path is not None:
        note = f"{note}; source={roi_path}"
    return SceneMap(
        tenant_id=tenant_id,
        camera_id=camera_id,
        version=int(version),
        frame_size=(int(frame_size[0]), int(frame_size[1])),
        rotation=int(rotation),
        workstations=workstations,
        created_utc=now_utc(),
        source=SOURCE_ADMIN,
        reference_frame=reference_frame,
        notes=[note],
    )


def refine_with_evidence(
    workstations: list[Workstation], frames: Sequence[DiscoveryFrame], config: DiscoveryConfig = DiscoveryConfig(),
    min_containment: float = 0.6, ambiguity_ratio: float = 0.8,
) -> list[str]:
    """Tighten seat geometry of (admin) workstations using chair / seated-person evidence.

    The context box (ResNet crop) is never changed. Clusters are assigned one-to-one by
    maximum IoU with the context box (admin boxes are drawn around seat + occupant, so IoU
    disambiguates overlapping boxes far better than centre distance). The seat box becomes
    the expected seated-occupant box: the observed seated person if any, else the chair box
    extended upward by the measured seated/chair height ratio. Returns notes.
    """
    from scipy.optimize import linear_sum_assignment

    if not frames:
        return ["no frames for evidence refinement"]
    total = len(frames)
    chairs = [c for c in cluster_detections([f.objects.get("chair", []) for f in frames], config.chair_iou, "chair")
              if c.persistence(total) >= config.min_chair_persistence]
    dwells = [c for c in cluster_detections([f.persons for f in frames], config.person_iou, "person")
              if c.persistence(total) >= config.min_dwell_persistence]

    ambiguous: dict[str, str] = {}

    def claim(clusters: list[_Cluster]) -> dict[str, _Cluster]:
        if not clusters or not workstations:
            return {}
        raw = np.array([[iou(c.median_box, w.context_box) for w in workstations] for c in clusters])
        score = raw.copy()
        for ci, cluster in enumerate(clusters):
            for wi, w in enumerate(workstations):
                if containment(cluster.median_box, w.context_box) < min_containment:
                    score[ci, wi] = 0.0
        rows, cols = linear_sum_assignment(-score)
        result = {}
        for ci, wi in zip(rows, cols):
            if score[ci, wi] <= 0:
                continue
            ws_id = workstations[wi].workstation_id
            result[ws_id] = clusters[ci]
            others = np.delete(raw[ci], wi)
            if others.size and others.max() >= ambiguity_ratio * raw[ci, wi]:
                rival = workstations[int(np.argmax(np.where(np.arange(raw.shape[1]) == wi, -1, raw[ci])))]
                ambiguous[ws_id] = (f"{clusters[ci].label} evidence {list(clusters[ci].median_box)} overlaps "
                                    f"{ws_id} ({raw[ci, wi]:.2f}) and {rival.workstation_id} ({others.max():.2f})")
        return result

    chair_of, dwell_of = claim(chairs), claim(dwells)
    ratios = [(dwell_of[k].median_box[3] - dwell_of[k].median_box[1]) /
              (chair_of[k].median_box[3] - chair_of[k].median_box[1])
              for k in chair_of if k in dwell_of and chair_of[k].median_box[3] > chair_of[k].median_box[1]]
    height_ratio = float(np.median(ratios)) if ratios else None
    notes = []
    for w in workstations:
        chair, dwell = chair_of.get(w.workstation_id), dwell_of.get(w.workstation_id)
        if dwell is not None:
            w.seat_box = dwell.median_box
            w.seated_height_ref = float(dwell.median_box[3] - dwell.median_box[1])
        elif chair is not None:
            x1, y1, x2, y2 = chair.median_box
            expected_h = (y2 - y1) * (height_ratio or 1.0)
            w.seat_box = (x1, int(max(w.context_box[1], round(y2 - expected_h))), x2, y2)
            w.seated_height_ref = float(expected_h) if height_ratio else None
        w.evidence.update({
            "frames_sampled": total,
            "chair_persistence": round(chair.persistence(total), 3) if chair else 0.0,
            "dwell_persistence": round(dwell.persistence(total), 3) if dwell else 0.0,
            "seat_refined": bool(chair or dwell),
        })
        if not (chair or dwell):
            notes.append(f"{w.workstation_id}: no chair/seated evidence inside context; seat = context box")
        if w.workstation_id in ambiguous:
            w.status = STATUS_NEEDS_REVIEW
            w.evidence["review_reason"] = ambiguous[w.workstation_id]
            notes.append(f"{w.workstation_id}: needs review - {ambiguous[w.workstation_id]}")
    if height_ratio is not None:
        notes.append(f"seated/chair height ratio {height_ratio:.2f} from {len(ratios)} workstations")
    return notes


# --------------------------------------------------------------------------- camera alignment


def alignment_offset(reference_gray: np.ndarray, frame_gray: np.ndarray, max_features: int = 2000) -> tuple[float, int]:
    """Translation magnitude (px) of a similarity transform reference->frame, and inlier count."""
    orb = cv2.ORB_create(max_features)
    ka, da = orb.detectAndCompute(reference_gray, None)
    kb, db = orb.detectAndCompute(frame_gray, None)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return float("inf"), 0
    matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db), key=lambda m: m.distance)[:400]
    if len(matches) < 10:
        return float("inf"), 0
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(pa, pb, ransacReprojThreshold=3.0)
    if matrix is None or inliers is None:
        return float("inf"), 0
    cx, cy = reference_gray.shape[1] / 2, reference_gray.shape[0] / 2
    moved = matrix @ np.array([cx, cy, 1.0])
    return float(math.hypot(moved[0] - cx, moved[1] - cy)), int(inliers.sum())


# --------------------------------------------------------------------------- runtime


@dataclass(frozen=True)
class SceneRuntimeConfig:
    check_interval_seconds: float = 30.0
    max_alignment_shift_px: float = 8.0
    min_alignment_inliers: int = 40
    evidence_stale_seconds: float = 600.0
    layout_change_fraction: float = 0.5


@dataclass
class SceneStatus:
    layout_change_suspected: bool = False
    camera_moved: bool = False
    alignment_shift_px: float | None = None
    stale_workstations: list[str] = field(default_factory=list)
    last_check_seconds: float | None = None
    reasons: list[str] = field(default_factory=list)


class SceneMapper:
    """Holds the persistent map and performs low-cadence camera-alignment checks."""

    def __init__(
        self,
        scene_map: SceneMap,
        runtime: SceneRuntimeConfig = SceneRuntimeConfig(),
        reference_gray: np.ndarray | None = None,
    ) -> None:
        self.map = scene_map
        self.runtime = runtime
        self._reference_gray = reference_gray
        self.status = SceneStatus()
        self._last_evidence: dict[str, float] = {}

    @property
    def workstations(self) -> list[Workstation]:
        return self.map.active_workstations

    def note_person_evidence(self, workstation_id: str, timestamp: float) -> None:
        self._last_evidence[workstation_id] = max(timestamp, self._last_evidence.get(workstation_id, timestamp))

    def update(self, frame: np.ndarray, timestamp: float) -> SceneStatus:
        """Run checks only when the cadence has elapsed; otherwise return the current status."""
        last = self.status.last_check_seconds
        if last is not None and timestamp - last < self.runtime.check_interval_seconds:
            return self.status
        self.status.last_check_seconds = timestamp
        reasons: list[str] = []

        if self._reference_gray is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            shift, inliers = alignment_offset(self._reference_gray, gray)
            self.status.alignment_shift_px = None if math.isinf(shift) else round(shift, 2)
            moved = inliers < self.runtime.min_alignment_inliers or shift > self.runtime.max_alignment_shift_px
            if moved and not self.status.camera_moved:
                logger.error("Camera alignment check failed (shift=%.1fpx, inliers=%d); recalibrate the scene map",
                             shift, inliers)
            self.status.camera_moved = moved
            if moved:
                reasons.append(f"camera_alignment shift={shift:.1f}px inliers={inliers}")

        stale = [w.workstation_id for w in self.workstations
                 if timestamp - self._last_evidence.get(w.workstation_id, timestamp) > self.runtime.evidence_stale_seconds]
        for w in self.workstations:
            self._last_evidence.setdefault(w.workstation_id, timestamp)
        self.status.stale_workstations = stale
        if self.workstations and len(stale) / len(self.workstations) >= self.runtime.layout_change_fraction:
            reasons.append(f"{len(stale)}/{len(self.workstations)} workstations without person evidence")
        self.status.layout_change_suspected = bool(reasons)
        self.status.reasons = reasons
        return self.status


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
