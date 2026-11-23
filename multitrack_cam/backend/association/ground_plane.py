"""Ground-plane homographies between overlapping cameras.

Each overlapping camera group has a reference camera. Every other camera in the
group gets a homography H that maps its *normalized* image floor points
(x/W, y/H in the rotated processing frame) into the reference camera's
normalized image plane. A person's foot point (bottom-center of the box) seen
by two overlapping cameras then lands at (approximately) the same reference
point, which is strong, appearance-independent evidence that two tracks are the
same physical person.

Sources of H, in priority order:
1. Manual calibration (operator clicks >= 4 matching floor points) — trusted.
2. Automatic self-calibration from confident cross-view correspondences, fitted
   with RANSAC — used only after it passes inlier and spread checks.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml

from backend.utils.paths import resolve_path

logger = logging.getLogger(__name__)


@dataclass
class OverlapGroup:
    name: str
    reference_camera: str
    cameras: list[str]


@dataclass
class Homography:
    camera_id: str
    reference_camera: str
    matrix: np.ndarray
    source: str  # manual | auto
    reprojection_error: float
    num_points: int
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "reference_camera": self.reference_camera,
            "matrix": self.matrix.tolist(),
            "source": self.source,
            "reprojection_error": float(self.reprojection_error),
            "num_points": int(self.num_points),
            "updated_at": float(self.updated_at),
        }


def parse_overlap_groups(topology_doc: dict[str, Any]) -> list[OverlapGroup]:
    """Read explicit overlap groups, or derive them from overlapping_with links."""
    groups: list[OverlapGroup] = []
    for item in topology_doc.get("overlap_groups", []) or []:
        cams = [str(c) for c in item.get("cameras", [])]
        if len(cams) < 2:
            continue
        ref = str(item.get("reference_camera") or cams[0])
        groups.append(OverlapGroup(name=str(item.get("name", "_".join(cams))), reference_camera=ref, cameras=cams))
    if groups:
        return groups

    # Derive connected components from overlapping_with / simultaneous pairs.
    adj: dict[str, set[str]] = {}
    for cam_id, meta in (topology_doc.get("cameras") or {}).items():
        for other in (meta or {}).get("overlapping_with", []) or []:
            adj.setdefault(str(cam_id), set()).add(str(other))
            adj.setdefault(str(other), set()).add(str(cam_id))
    for a, b in topology_doc.get("simultaneous_allowed_pairs", []) or []:
        adj.setdefault(str(a), set()).add(str(b))
        adj.setdefault(str(b), set()).add(str(a))
    seen: set[str] = set()
    for start in sorted(adj):
        if start in seen:
            continue
        comp, stack = [], [start]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            stack.extend(adj.get(cur, ()))
        if len(comp) >= 2:
            comp.sort()
            groups.append(OverlapGroup(name="_".join(comp), reference_camera=comp[0], cameras=comp))
    return groups


def fit_homography(
    src: Sequence[Sequence[float]],
    dst: Sequence[Sequence[float]],
    *,
    ransac_threshold: float = 0.03,
) -> tuple[np.ndarray | None, float, np.ndarray]:
    """Fit src->dst homography on normalized points. Returns (H, mean_error, inlier_mask)."""
    s = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    d = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(s) < 4 or len(s) != len(d):
        return None, float("inf"), np.zeros(len(s), dtype=bool)
    method = cv2.RANSAC if len(s) > 4 else 0
    H, mask = cv2.findHomography(s, d, method, ransac_threshold)
    if H is None:
        return None, float("inf"), np.zeros(len(s), dtype=bool)
    inliers = mask.reshape(-1).astype(bool) if mask is not None else np.ones(len(s), dtype=bool)
    proj = apply_homography(H, s)
    err = np.linalg.norm(proj - d, axis=1)
    mean_err = float(err[inliers].mean()) if inliers.any() else float("inf")
    return H, mean_err, inliers


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([p, np.ones((len(p), 1))]) @ H.T
    w = hom[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    return hom[:, :2] / w


@dataclass
class AnchorMap:
    """Sparse verified correspondences (e.g. fixed seats) instead of a global homography.

    Used when a single plane model does not hold well (opposite viewpoints,
    people at different heights). A point maps only if it lies within an
    anchor's radius; there is no extrapolation between anchors.
    """

    camera_id: str
    reference_camera: str
    points_camera: np.ndarray
    points_reference: np.ndarray
    radii: np.ndarray
    labels: list[str]
    # Max distance in the reference view between a projected anchor and the partner track.
    reference_radius: float = 0.025
    updated_at: float = field(default_factory=time.time)

    def project(self, point: np.ndarray) -> np.ndarray | None:
        d = np.linalg.norm(self.points_camera - point.reshape(1, 2), axis=1)
        i = int(np.argmin(d))
        if d[i] > self.radii[i]:
            return None
        return self.points_reference[i].copy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "reference_camera": self.reference_camera,
            "mode": "anchors",
            "num_points": int(len(self.points_camera)),
            "labels": list(self.labels),
            "radii": [round(float(r), 4) for r in self.radii],
            "reference_radius": float(self.reference_radius),
            "reprojection_error": 0.0,
            "source": "manual",
        }


def anchor_radii(points: np.ndarray, *, max_radius: float = 0.03, fraction: float = 0.45) -> np.ndarray:
    """Radius per anchor: never more than ``fraction`` of the distance to the nearest other anchor."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return np.full(len(pts), max_radius)
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    np.fill_diagonal(d, np.inf)
    return np.minimum(max_radius, fraction * d.min(axis=1))


@dataclass
class _AutoSamples:
    src: list[tuple[float, float]] = field(default_factory=list)
    dst: list[tuple[float, float]] = field(default_factory=list)
    last_fit_count: int = 0


class GroundPlaneRegistry:
    """Holds per-camera homographies into each overlap group's reference plane."""

    def __init__(
        self,
        groups: list[OverlapGroup],
        *,
        manual_path: str | Path = "config/calibration.yaml",
        auto_path: str | Path = "data/store/calibration_auto.json",
        auto_enabled: bool = True,
        auto_min_points: int = 12,
        auto_min_inlier_ratio: float = 0.6,
        auto_max_error: float = 0.04,
        auto_min_spread: float = 0.25,
        auto_max_points: int = 200,
    ) -> None:
        self.groups = groups
        self.manual_path = resolve_path(manual_path)
        self.auto_path = resolve_path(auto_path)
        self.auto_enabled = auto_enabled
        self.auto_min_points = auto_min_points
        self.auto_min_inlier_ratio = auto_min_inlier_ratio
        self.auto_max_error = auto_max_error
        self.auto_min_spread = auto_min_spread
        self.auto_max_points = auto_max_points
        self._lock = threading.RLock()
        self._h: dict[str, Homography] = {}
        self._anchors: dict[str, AnchorMap] = {}
        self._auto: dict[str, _AutoSamples] = {}
        self._cam_group: dict[str, OverlapGroup] = {}
        for g in groups:
            for cam in g.cameras:
                self._cam_group[cam] = g
        self._load()

    # ------------------------------------------------------------------ queries
    def group_of(self, camera_id: str) -> OverlapGroup | None:
        return self._cam_group.get(camera_id)

    def overlapping(self, cam_a: str, cam_b: str) -> bool:
        ga, gb = self._cam_group.get(cam_a), self._cam_group.get(cam_b)
        return ga is not None and ga is gb and cam_a != cam_b

    def has_geometry(self, camera_id: str) -> bool:
        g = self._cam_group.get(camera_id)
        if g is None:
            return False
        if camera_id == g.reference_camera:
            return True
        with self._lock:
            return camera_id in self._h or camera_id in self._anchors

    def project(self, camera_id: str, point_norm: Sequence[float]) -> np.ndarray | None:
        """Map a normalized image point to the group's reference plane.

        Returns None when the point cannot be mapped (no calibration, or an
        anchor-mode camera and the point is not at a verified anchor).
        """
        g = self._cam_group.get(camera_id)
        if g is None:
            return None
        p = np.asarray(point_norm, dtype=np.float64).reshape(1, 2)
        if camera_id == g.reference_camera:
            return p[0]
        with self._lock:
            h = self._h.get(camera_id)
            anchors = self._anchors.get(camera_id)
        if h is not None:
            return apply_homography(h.matrix, p)[0]
        if anchors is not None:
            return anchors.project(p[0])
        return None

    def match_radius(self, camera_id: str) -> float | None:
        """Tight reference-plane gate for anchor-mode cameras; None for homography / uncalibrated."""
        with self._lock:
            anchors = self._anchors.get(camera_id)
            if anchors is None or camera_id in self._h:
                return None
            return anchors.reference_radius

    def set_anchors(
        self,
        camera_id: str,
        points_cam: Sequence[Sequence[float]],
        points_ref: Sequence[Sequence[float]],
        *,
        labels: Sequence[str] | None = None,
        max_radius: float = 0.03,
        reference_radius: float = 0.025,
    ) -> AnchorMap:
        """Register verified point correspondences (anchor mode); replaces any homography."""
        g = self._cam_group.get(camera_id)
        if g is None:
            raise ValueError(f"{camera_id} is not in any overlap group")
        if camera_id == g.reference_camera:
            raise ValueError(f"{camera_id} is the reference camera of group {g.name}")
        pc = np.asarray(points_cam, dtype=np.float64).reshape(-1, 2)
        pr = np.asarray(points_ref, dtype=np.float64).reshape(-1, 2)
        if len(pc) == 0 or len(pc) != len(pr):
            raise ValueError("Anchor mode needs >= 1 point pair with equal counts")
        if ((pc < 0) | (pc > 1) | (pr < 0) | (pr > 1)).any():
            raise ValueError("Points must be normalized to [0, 1]")
        names = list(labels) if labels else [f"anchor_{i + 1}" for i in range(len(pc))]
        amap = AnchorMap(camera_id, g.reference_camera, pc, pr, anchor_radii(pc, max_radius=max_radius), names,
                         reference_radius=float(reference_radius))
        with self._lock:
            self._anchors[camera_id] = amap
            self._h.pop(camera_id, None)
            doc = self._read_manual_doc()
            doc["homographies"][camera_id] = {
                "reference_camera": g.reference_camera,
                "mode": "anchors",
                "max_radius": float(max_radius),
                "reference_radius": float(reference_radius),
                "labels": names,
                "points_camera": pc.tolist(),
                "points_reference": pr.tolist(),
            }
            self._write_manual_doc(doc)
        logger.info("Anchor calibration %s->%s with %d anchors %s", camera_id, g.reference_camera, len(pc), names)
        return amap

    def status(self) -> list[dict[str, Any]]:
        out = []
        with self._lock:
            for g in self.groups:
                cams = []
                for cam in g.cameras:
                    if cam == g.reference_camera:
                        cams.append({"camera_id": cam, "role": "reference", "source": "reference"})
                        continue
                    h = self._h.get(cam)
                    auto = self._auto.get(cam)
                    anchors = self._anchors.get(cam)
                    if anchors is not None and h is None:
                        cams.append({**anchors.to_dict(), "role": "member",
                                     "auto_samples": len(auto.src) if auto else 0})
                        continue
                    cams.append(
                        {
                            "camera_id": cam,
                            "role": "member",
                            "source": h.source if h else "none",
                            "reprojection_error": h.reprojection_error if h else None,
                            "num_points": h.num_points if h else 0,
                            "auto_samples": len(auto.src) if auto else 0,
                        }
                    )
                out.append({"group": g.name, "reference_camera": g.reference_camera, "cameras": cams})
        return out

    # ------------------------------------------------------------ manual calib
    def set_manual(
        self,
        camera_id: str,
        points_cam: Sequence[Sequence[float]],
        points_ref: Sequence[Sequence[float]],
    ) -> Homography:
        g = self._cam_group.get(camera_id)
        if g is None:
            raise ValueError(f"{camera_id} is not in any overlap group")
        if camera_id == g.reference_camera:
            raise ValueError(f"{camera_id} is the reference camera of group {g.name}")
        H, err, inliers = fit_homography(points_cam, points_ref, ransac_threshold=0.05)
        if H is None:
            raise ValueError("Homography fit failed; use >= 4 well-spread, non-collinear floor points")
        h = Homography(
            camera_id=camera_id,
            reference_camera=g.reference_camera,
            matrix=H,
            source="manual",
            reprojection_error=err,
            num_points=int(inliers.sum()),
        )
        with self._lock:
            self._h[camera_id] = h
            self._anchors.pop(camera_id, None)
            self._save_manual(camera_id, points_cam, points_ref)
        logger.info("Manual homography %s->%s err=%.4f n=%d", camera_id, g.reference_camera, err, h.num_points)
        return h

    def clear(self, camera_id: str) -> None:
        with self._lock:
            self._h.pop(camera_id, None)
            self._anchors.pop(camera_id, None)
            self._auto.pop(camera_id, None)
            doc = self._read_manual_doc()
            doc.get("homographies", {}).pop(camera_id, None)
            self._write_manual_doc(doc)
            self._save_auto()

    # -------------------------------------------------------------- auto calib
    def add_auto_correspondence(self, camera_id: str, point_cam: Sequence[float], point_ref: Sequence[float]) -> None:
        """Feed a confident cross-view foot-point pair (normalized coords)."""
        if not self.auto_enabled:
            return
        g = self._cam_group.get(camera_id)
        if g is None or camera_id == g.reference_camera:
            return
        with self._lock:
            current = self._h.get(camera_id)
            if (current is not None and current.source == "manual") or camera_id in self._anchors:
                return
            samples = self._auto.setdefault(camera_id, _AutoSamples())
            p = (float(point_cam[0]), float(point_cam[1]))
            # Keep spatially diverse samples only.
            if samples.src:
                d = np.linalg.norm(np.asarray(samples.src) - np.asarray(p), axis=1)
                if float(d.min()) < 0.03:
                    return
            samples.src.append(p)
            samples.dst.append((float(point_ref[0]), float(point_ref[1])))
            if len(samples.src) > self.auto_max_points:
                samples.src.pop(0)
                samples.dst.pop(0)
            if len(samples.src) >= self.auto_min_points and len(samples.src) - samples.last_fit_count >= 3:
                samples.last_fit_count = len(samples.src)
                self._try_fit_auto(camera_id, g, samples)

    def _try_fit_auto(self, camera_id: str, g: OverlapGroup, samples: _AutoSamples) -> None:
        src = np.asarray(samples.src)
        spread = float(min(np.ptp(src[:, 0]), np.ptp(src[:, 1])))
        if spread < self.auto_min_spread:
            return
        H, err, inliers = fit_homography(samples.src, samples.dst, ransac_threshold=0.03)
        if H is None:
            return
        ratio = float(inliers.mean()) if len(inliers) else 0.0
        if ratio < self.auto_min_inlier_ratio or err > self.auto_max_error or int(inliers.sum()) < self.auto_min_points:
            logger.debug("Auto homography %s rejected ratio=%.2f err=%.4f", camera_id, ratio, err)
            return
        self._h[camera_id] = Homography(
            camera_id=camera_id,
            reference_camera=g.reference_camera,
            matrix=H,
            source="auto",
            reprojection_error=err,
            num_points=int(inliers.sum()),
        )
        self._save_auto()
        logger.info("Auto homography %s->%s accepted err=%.4f inliers=%d/%d",
                    camera_id, g.reference_camera, err, int(inliers.sum()), len(inliers))

    # ------------------------------------------------------------- persistence
    def _read_manual_doc(self) -> dict[str, Any]:
        if not self.manual_path.is_file():
            return {"homographies": {}}
        with self.manual_path.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        doc.setdefault("homographies", {})
        return doc

    def _write_manual_doc(self, doc: dict[str, Any]) -> None:
        self.manual_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manual_path.open("w", encoding="utf-8") as fh:
            fh.write("# Ground-plane calibration (normalized image coords, rotated frames).\n")
            fh.write("# Written by the /calibrate page. Points: camera floor point -> same point in reference camera.\n")
            yaml.safe_dump(doc, fh, sort_keys=True)

    def _save_manual(self, camera_id: str, points_cam, points_ref) -> None:
        doc = self._read_manual_doc()
        g = self._cam_group[camera_id]
        doc["homographies"][camera_id] = {
            "reference_camera": g.reference_camera,
            "points_camera": [[float(x), float(y)] for x, y in points_cam],
            "points_reference": [[float(x), float(y)] for x, y in points_ref],
        }
        self._write_manual_doc(doc)

    def _save_auto(self) -> None:
        try:
            self.auto_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "homographies": {k: h.to_dict() for k, h in self._h.items() if h.source == "auto"},
                "samples": {k: {"src": s.src, "dst": s.dst} for k, s in self._auto.items()},
            }
            self.auto_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist auto calibration: %s", exc)

    def _load(self) -> None:
        try:
            doc = self._read_manual_doc()
            for cam, item in (doc.get("homographies") or {}).items():
                if cam not in self._cam_group:
                    continue
                if item.get("mode") == "anchors":
                    pc = np.asarray(item["points_camera"], dtype=np.float64)
                    self._anchors[cam] = AnchorMap(
                        cam,
                        self._cam_group[cam].reference_camera,
                        pc,
                        np.asarray(item["points_reference"], dtype=np.float64),
                        anchor_radii(pc, max_radius=float(item.get("max_radius", 0.03))),
                        list(item.get("labels") or [f"anchor_{i + 1}" for i in range(len(pc))]),
                        reference_radius=float(item.get("reference_radius", 0.025)),
                    )
                    continue
                H, err, inliers = fit_homography(item["points_camera"], item["points_reference"], ransac_threshold=0.05)
                if H is not None:
                    self._h[cam] = Homography(cam, self._cam_group[cam].reference_camera, H, "manual", err, int(inliers.sum()))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Manual calibration load failed: %s", exc)
        try:
            if self.auto_path.is_file():
                payload = json.loads(self.auto_path.read_text(encoding="utf-8"))
                for cam, item in (payload.get("homographies") or {}).items():
                    if cam in self._h or cam in self._anchors or cam not in self._cam_group:
                        continue
                    self._h[cam] = Homography(
                        cam,
                        str(item["reference_camera"]),
                        np.asarray(item["matrix"], dtype=np.float64),
                        "auto",
                        float(item.get("reprojection_error", 0.0)),
                        int(item.get("num_points", 0)),
                    )
                for cam, s in (payload.get("samples") or {}).items():
                    if cam in self._cam_group:
                        self._auto[cam] = _AutoSamples(
                            src=[tuple(p) for p in s.get("src", [])],
                            dst=[tuple(p) for p in s.get("dst", [])],
                        )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auto calibration load failed: %s", exc)
        if self._h:
            logger.info("Ground-plane homographies loaded: %s",
                        {k: f"{v.source}(err={v.reprojection_error:.3f})" for k, v in self._h.items()})
        if self._anchors:
            logger.info("Anchor calibrations loaded: %s", {k: v.labels for k, v in self._anchors.items()})
