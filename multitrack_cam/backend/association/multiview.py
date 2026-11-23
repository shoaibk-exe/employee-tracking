"""Simultaneous cross-view linking for overlapping cameras.

When two cameras look at the same space (cam_02 / cam_03 hall), the same person
appears in both views at once. Appearance similarity across opposite viewpoints
is weak for this ReID model (measured on site: true cross-view pairs score
~0.3-0.45, while different people in one view can reach ~0.47), so geometry is
the primary cue:

    score = w_geo * exp(-d^2 / 2 sigma^2) + w_app * appearance

where d is the distance between the two tracks' anchor points projected into
the group's reference plane (see ``ground_plane``). Without calibration only a
very strong appearance match can create a link.

Links use vote hysteresis so a single noisy frame neither creates nor breaks a
link.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from backend.association.ground_plane import GroundPlaneRegistry

logger = logging.getLogger(__name__)


@dataclass
class ViewObservation:
    key: str
    camera_id: str
    bbox_xyxy: tuple[float, float, float, float]
    frame_width: int
    frame_height: int
    timestamp: float
    embedding: np.ndarray | None = None
    stationary: bool = False


@dataclass
class Link:
    partner: str
    score: float
    votes: int
    geometric: bool
    distance: float | None = None


@dataclass
class _PairState:
    votes: int = 0
    misses: int = 0
    score: float = 0.0
    geometric: bool = False
    distance: float | None = None


def anchor_point(obs: ViewObservation, mode: str = "head") -> tuple[float, float]:
    """Normalized anchor point of a person box."""
    x1, y1, x2, y2 = obs.bbox_xyxy
    w = max(1.0, float(obs.frame_width))
    h = max(1.0, float(obs.frame_height))
    cx = (x1 + x2) / 2.0
    bh = max(1.0, y2 - y1)
    if mode == "foot":
        y = y2
    elif mode == "center":
        y = (y1 + y2) / 2.0
    else:  # head: robust to desks/chairs hiding the lower body
        y = y1 + 0.12 * bh
    return cx / w, y / h


@dataclass
class MultiViewConfig:
    anchor: str = "head"
    geo_sigma: float = 0.05
    geo_gate_sigmas: float = 3.0
    geo_weight: float = 0.8
    app_weight: float = 0.2
    min_link_score: float = 0.45
    link_margin: float = 0.10
    app_only_threshold: float = 0.62
    app_only_margin: float = 0.10
    votes_to_link: int = 2
    misses_to_unlink: int = 3
    max_time_skew: float = 2.0
    # Two stationary (seated) tracks can be compared across a larger skew: slow streams deliver ~0.4 fps.
    stationary_time_skew: float = 8.0
    # Calibrated overlap + at least one track seated: tolerate relay jitter (cam_02/cam_03).
    # Fully moving pairs keep max_time_skew so walkers don't false-link across 4+ seconds.
    geometry_time_skew: float = 6.0
    autocal_app_threshold: float = 0.62
    autocal_min_votes: int = 3


@dataclass
class MultiViewLinker:
    registry: GroundPlaneRegistry
    config: MultiViewConfig = field(default_factory=MultiViewConfig)

    def __post_init__(self) -> None:
        self._pairs: dict[tuple[str, str], _PairState] = {}

    @staticmethod
    def _pair_key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def _project(self, obs: ViewObservation) -> np.ndarray | None:
        return self.registry.project(obs.camera_id, anchor_point(obs, self.config.anchor))

    def _score_matrix(
        self, left: list[ViewObservation], right: list[ViewObservation]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        cfg = self.config
        n, m = len(left), len(right)
        score = np.full((n, m), -1.0)
        dist = np.full((n, m), np.nan)
        app = np.full((n, m), np.nan)
        geo_ok = bool(left) and bool(right) and self.registry.has_geometry(left[0].camera_id) and self.registry.has_geometry(right[0].camera_id)
        pl = [self._project(o) for o in left] if geo_ok else [None] * n
        pr = [self._project(o) for o in right] if geo_ok else [None] * m
        gate = cfg.geo_gate_sigmas * cfg.geo_sigma
        if geo_ok:
            for cam in (left[0].camera_id, right[0].camera_id):
                r = self.registry.match_radius(cam)
                if r is not None:
                    gate = min(gate, r)
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                if abs(a.timestamp - b.timestamp) > self._skew_limit(a, b):
                    continue
                if a.embedding is not None and b.embedding is not None:
                    app[i, j] = float(np.dot(a.embedding, b.embedding))
                if geo_ok and pl[i] is not None and pr[j] is not None:
                    d = float(np.linalg.norm(pl[i] - pr[j]))
                    dist[i, j] = d
                    if d > gate:
                        continue
                    g = float(np.exp(-0.5 * (d / cfg.geo_sigma) ** 2))
                    if np.isnan(app[i, j]):
                        score[i, j] = g
                    else:
                        score[i, j] = cfg.geo_weight * g + cfg.app_weight * max(0.0, app[i, j])
                elif not np.isnan(app[i, j]):
                    # Without a projection, appearance alone must clear the strict threshold.
                    if not geo_ok or app[i, j] >= cfg.app_only_threshold:
                        score[i, j] = app[i, j]
        return score, dist, app, geo_ok

    @staticmethod
    def _second_best(mat: np.ndarray, i: int, j: int) -> float:
        row = np.delete(mat[i, :], j)
        col = np.delete(mat[:, j], i)
        vals = np.concatenate([row, col]) if row.size or col.size else np.array([-1.0])
        return float(vals.max()) if vals.size else -1.0

    def update(self, observations: Iterable[ViewObservation]) -> dict[str, Link]:
        cfg = self.config
        by_cam: dict[str, list[ViewObservation]] = {}
        for o in observations:
            by_cam.setdefault(o.camera_id, []).append(o)

        matched_now: set[tuple[str, str]] = set()
        present: set[str] = {o.key for obs in by_cam.values() for o in obs}

        for group in self.registry.groups:
            cams = [c for c in group.cameras if c in by_cam]
            for cam_a, cam_b in combinations(cams, 2):
                left, right = by_cam[cam_a], by_cam[cam_b]
                score, dist, app, geo_ok = self._score_matrix(left, right)
                if score.size == 0:
                    continue
                threshold = cfg.min_link_score if geo_ok else cfg.app_only_threshold
                margin = cfg.link_margin if geo_ok else cfg.app_only_margin
                cost = 1.0 - np.where(score >= threshold, score, -1.0)
                rows, cols = linear_sum_assignment(cost)
                for i, j in zip(rows, cols):
                    s = float(score[i, j])
                    if s < threshold:
                        continue
                    if s - self._second_best(score, i, j) < margin and self._second_best(score, i, j) >= threshold:
                        continue
                    pk = self._pair_key(left[i].key, right[j].key)
                    st = self._pairs.setdefault(pk, _PairState())
                    st.votes += 1
                    st.misses = 0
                    st.score = 0.7 * st.score + 0.3 * s if st.votes > 1 else s
                    st.geometric = geo_ok and not np.isnan(dist[i, j])
                    st.distance = None if np.isnan(dist[i, j]) else float(dist[i, j])
                    matched_now.add(pk)
                    self._maybe_autocalibrate(group.reference_camera, left[i], right[j], app[i, j], st, geo_ok)

        for pk in list(self._pairs):
            if pk in matched_now:
                continue
            st = self._pairs[pk]
            if pk[0] in present and pk[1] in present and not self._compared(pk, by_cam):
                continue  # observations too far apart in time to judge this cycle
            st.misses += 1
            if st.misses >= cfg.misses_to_unlink:
                self._pairs.pop(pk, None)

        return self.links()

    def _compared(self, pk: tuple[str, str], by_cam: dict[str, list[ViewObservation]]) -> bool:
        obs = {o.key: o for lst in by_cam.values() for o in lst}
        a, b = obs.get(pk[0]), obs.get(pk[1])
        return a is not None and b is not None and abs(a.timestamp - b.timestamp) <= self._skew_limit(a, b)

    def _skew_limit(self, a: ViewObservation, b: ViewObservation) -> float:
        if a.stationary and b.stationary:
            return max(self.config.max_time_skew, self.config.stationary_time_skew)
        if (
            self.config.geometry_time_skew > 0
            and (a.stationary or b.stationary)
            and self.registry.has_geometry(a.camera_id)
            and self.registry.has_geometry(b.camera_id)
        ):
            return max(self.config.max_time_skew, self.config.geometry_time_skew)
        return self.config.max_time_skew

    def links(self) -> dict[str, Link]:
        out: dict[str, Link] = {}
        ranked = sorted(self._pairs.items(), key=lambda kv: (kv[1].votes, kv[1].score), reverse=True)
        for (a, b), st in ranked:
            if st.votes < self.config.votes_to_link:
                continue
            if a in out or b in out:
                continue  # one partner per camera pair per track
            out[a] = Link(partner=b, score=st.score, votes=st.votes, geometric=st.geometric, distance=st.distance)
            out[b] = Link(partner=a, score=st.score, votes=st.votes, geometric=st.geometric, distance=st.distance)
        return out

    def forget(self, key: str) -> None:
        for pk in [pk for pk in self._pairs if key in pk]:
            self._pairs.pop(pk, None)

    def add_identity_correspondence(self, ref_obs: ViewObservation, member_obs: ViewObservation) -> None:
        """Two stationary tracks bound to the same identity by independent evidence."""
        g = self.registry.group_of(member_obs.camera_id)
        if g is None or g.reference_camera != ref_obs.camera_id:
            return
        if not (ref_obs.stationary and member_obs.stationary):
            return
        self.registry.add_auto_correspondence(
            member_obs.camera_id,
            anchor_point(member_obs, self.config.anchor),
            anchor_point(ref_obs, self.config.anchor),
        )

    def _maybe_autocalibrate(
        self,
        reference_camera: str,
        a: ViewObservation,
        b: ViewObservation,
        app: float,
        st: _PairState,
        geo_ok: bool,
    ) -> None:
        cfg = self.config
        if geo_ok or np.isnan(app) or app < cfg.autocal_app_threshold or st.votes < cfg.autocal_min_votes:
            return
        if a.camera_id == reference_camera:
            self.add_identity_correspondence(a, b)
        elif b.camera_id == reference_camera:
            self.add_identity_correspondence(b, a)
