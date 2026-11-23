"""Online, per-live-track global identity resolution.

Design (why it is shaped like this):

* **Sticky bindings.** Once a live local track has a global ID it keeps it for
  the whole life of the track. A binding is only released after *repeated*
  appearance contradiction (a BoT-SORT ID switch), never because of one bad crop.
* **Continuity before re-identification.** A new local track that appears where
  a just-ended track of the same camera was (IoU / center distance, a few
  seconds) inherits that ID unless appearance clearly contradicts it. This
  absorbs local tracker fragmentation, the biggest source of ID churn.
* **Multi-view inheritance.** Tracks linked across overlapping cameras (see
  ``MultiViewLinker``) share one ID: an unbound track adopts its partner's ID.
* **Gallery re-identification.** Remaining tracks are matched against the
  identity gallery with a robust bank-vs-identity score, Hungarian assignment,
  absolute threshold + margin, exclusivity (one live track per camera per ID;
  simultaneous presence across cameras only when they overlap) and camera
  topology travel-time feasibility.
* **Delayed, conservative enrollment.** A new ID is created only after the
  track has enough appearance evidence and clearly matches nobody. On
  overlapping cameras enrollment waits longer so the cross-view link can form
  first and both views receive the same ID.
* **Audited duplicate repair.** If two linked views still ended up with two
  different IDs and one of them was created moments ago, the young ID is merged
  into the older one and the merge is written to the audit log.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from backend.association.camera_topology import CameraTopology
from backend.association.multiview import Link, MultiViewLinker, ViewObservation
from backend.identity.global_identity_manager import GlobalIdentityManager
from backend.reid.embedding_gallery import EmbeddingGallery
from backend.types import LocalTrack, TrackLifecycle, VerificationStatus
from backend.utils.geometry import iou_xyxy

logger = logging.getLogger(__name__)


@dataclass
class ResolverConfig:
    match_threshold: float = 0.55
    strong_match_threshold: float = 0.65
    match_margin: float = 0.08
    # Identity never seen by the query camera: viewpoint/lighting change makes
    # scores lower and look-alikes closer (site: wrong cross-cam match at 0.566).
    cross_camera_match_threshold: float = 0.62
    cross_camera_margin: float = 0.10
    new_identity_threshold: float = 0.38
    min_samples_to_match: int = 2
    min_samples_to_enroll: int = 3
    min_age_to_enroll: float = 1.5
    overlap_enroll_delay: float = 4.0
    ambiguous_timeout: float = 12.0
    no_appearance_timeout: float = 20.0
    contradiction_threshold: float = 0.25
    contradiction_votes: int = 3
    continuity_window: float = 4.0
    continuity_iou: float = 0.25
    continuity_center_ratio: float = 0.6
    continuity_min_appearance: float = 0.30
    # After a stream reconnect the tracker restarts; the same spot may be re-detected only after the outage.
    reconnect_continuity_window: float = 20.0
    # A track followed continuously by the tracker may teach its ID new poses (standing/walking),
    # as long as each sample is consistent with the track itself.
    stable_learn_seconds: float = 20.0
    stable_learn_min_similarity: float = 0.35
    stable_learn_self_consistency: float = 0.50
    simultaneity_horizon: float = 2.0
    gallery_update_interval: float = 4.0
    gallery_update_min_similarity: float = 0.50
    gallery_diversity_max_similarity: float = 0.93
    merge_window: float = 90.0
    merge_min_votes: int = 4
    # Established (older than merge_window) IDs merge only via a seat-anchor link held this long; 0 disables.
    anchor_merge_min_votes: int = 8
    anchor_merge_min_bound_seconds: float = 30.0
    # Recently departed IDs (walked to another room / stepped out briefly) are preferred candidates.
    recent_departure_bonus: float = 0.08   # 0 disables the prior
    recent_match_threshold: float = 0.52
    recent_match_margin: float = 0.06
    recent_reentry_window: float = 600.0   # same camera / same hall re-entry
    link_inherit_min_votes: int = 2
    bank_size: int = 12
    persist_interval: float = 3.0
    tracklet_min_hits: int = 3
    stale_track_seconds: float = 30.0
    stationary_max_shift: float = 0.02
    observation_horizon: float = 2.0
    # Slow streams (e.g. 0.4 fps) stay "visible" for this many frame intervals, up to the cap.
    horizon_frame_factor: float = 2.5
    max_observation_horizon: float = 8.0
    # Walk-out handoff: a person who walked out of one camera and walks into a
    # reachable non-overlapping camera keeps the ID even when the views look
    # different (e.g. low-res room camera). Appearance only has to not contradict.
    walkout_handoff: bool = True
    walkout_min_motion: float = 0.5        # displacement in body heights within the window
    walkout_motion_window: float = 3.0
    handoff_min_appearance: float = 0.25   # single candidate: reject only on clear contradiction
    handoff_multi_min_appearance: float = 0.30
    handoff_margin: float = 0.08
    handoff_event_window: float = 4.0      # same exit seen by overlapping views = one event
    handoff_time_slack: float = 1.0
    handoff_hold_seconds: float = 5.0      # delay enrolling a moving newcomer while candidates exist
    # Automatic duplicate consolidation: one person split into two IDs (e.g.
    # after a tracker drop) is merged when the two IDs are statistically the
    # same appearance AND were never visible together. Two different people who
    # look alike are present at the same time at some point, which blocks it.
    dedup_enabled: bool = True
    dedup_interval: float = 20.0
    dedup_min_samples: int = 6             # per identity on the shared camera
    dedup_min_similarity: float = 0.55     # mean cross-identity cosine on that camera
    dedup_self_ratio: float = 0.95         # cross >= ratio * min(self-similarity of each)
    dedup_margin: float = 0.08             # over each identity's next-closest identity
    dedup_min_observed_seconds: float = 60.0
    dedup_confirmations: int = 2
    dedup_covisible_time: float = 0.5
    dedup_covisible_max_iou: float = 0.3   # two boxes on one person are not two people
    # Face anchor (ArcFace): same person ~0.5-1.0, different people < 0.2 on this site.
    # A face match overrides body appearance; a contradicting face vetoes a body match.
    face_enabled: bool = True
    face_min_samples: int = 2
    face_match_threshold: float = 0.45
    face_margin: float = 0.12
    face_unknown_threshold: float = 0.25   # below this for every ID: a face nobody has yet
    face_veto_threshold: float = 0.20
    face_veto_min_gallery: int = 3
    face_self_consistency: float = 0.30    # a track's own faces must agree (no tracker swap)
    face_enroll_min_bound: float = 3.0
    face_learn_interval: float = 2.0
    face_merge_threshold: float = 0.50     # two IDs with the same face are one person
    # Clothing/caption VLMs must not merge two seated people in similar clothes.
    # A caption match may bind only on re-entry / handoff, or when body ReID also agrees.
    face_match_requires_reentry: bool = True
    face_match_body_floor: float = 0.42
    # Overlapping hall views: matching clothing captions can share an ID when seat
    # geometry is unavailable (walker between seats). Does NOT override co-visibility.
    overlap_caption_link: bool = True
    overlap_caption_threshold: float = 0.92
    overlap_caption_margin: float = 0.04


@dataclass
class LiveTrackState:
    key: str
    camera_id: str
    session_id: str
    local_id: int
    first_seen: float
    last_seen: float
    first_seen_wall: float
    bbox: tuple[float, float, float, float]
    frame_w: int
    frame_h: int
    hits: int = 0
    bank: deque = field(default_factory=lambda: deque(maxlen=12))
    gid: str | None = None
    status: str = "pending"  # pending | provisional | confirmed
    source: str = ""
    confidence: float = 0.0
    candidate: str | None = None
    bound_at: float = 0.0
    contradictions: int = 0
    last_gallery_update: float = 0.0
    last_persist: float = 0.0
    last_embed_ts: float = 0.0
    yielded_at: float | None = None
    recent_centers: deque = field(default_factory=lambda: deque(maxlen=6))
    trail: deque = field(default_factory=lambda: deque(maxlen=90))  # (ts, cx_px, cy_px, h_px)
    face_bank: deque = field(default_factory=lambda: deque(maxlen=8))  # (unit emb, quality, ts)
    face_seen: int = 0
    face_evaluated: int = 0
    face_verified: bool = False
    last_face_learn: float = 0.0
    face_vetoed: set = field(default_factory=set)  # IDs this track's face ruled out

    def dormant(self) -> bool:
        """Gave its ID to a successor track and has not been observed since."""
        return self.yielded_at is not None and self.last_seen <= self.yielded_at

    def proto(self) -> np.ndarray | None:
        if not self.bank:
            return None
        embs = np.stack([b[0] for b in self.bank])
        w = np.asarray([max(0.05, b[1]) for b in self.bank])
        v = (embs * w[:, None]).sum(axis=0)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None

    def bank_matrix(self) -> np.ndarray | None:
        if not self.bank:
            return None
        return np.stack([b[0] for b in self.bank]).astype(np.float32)

    def is_stationary(self, max_shift: float) -> bool:
        if len(self.recent_centers) < 3:
            return False
        pts = np.asarray(self.recent_centers)
        return float(np.ptp(pts[:, 0])) < max_shift and float(np.ptp(pts[:, 1])) < max_shift


@dataclass
class _Ended:
    gid: str
    camera_id: str
    bbox: tuple[float, float, float, float]
    ended_at: float
    proto: np.ndarray | None
    walked_out: bool = False   # tracker dropped it while the person was moving (left the view)
    handed_off: bool = False   # already claimed by a track on another camera


class LiveIdentityResolver:
    def __init__(
        self,
        identity: GlobalIdentityManager,
        gallery: EmbeddingGallery,
        topology: CameraTopology,
        linker: MultiViewLinker | None,
        config: ResolverConfig | None = None,
    ) -> None:
        self.identity = identity
        self.gallery = gallery
        self.topology = topology
        self.linker = linker
        self.cfg = config or ResolverConfig()
        self.tracks: dict[str, LiveTrackState] = {}
        self.links: dict[str, Link] = {}
        self._ended: deque[_Ended] = deque(maxlen=256)
        self._frame_obs: list[tuple[str, float, str, tuple[float, float, float, float]]] = []
        self._gid_obs: dict[str, tuple[str, float]] = {}
        self._co_visible: set[frozenset[str]] = set()
        self._observed_seconds: dict[str, float] = {}
        self._dup_streak: dict[tuple[str, str], int] = {}
        self._last_dedup = 0.0
        self._gid_seen: dict[str, tuple[str, float]] = {}
        self._cam_last_update: dict[str, float] = {}
        self._cam_interval: dict[str, float] = {}
        self._created_at: dict[str, float] = {}
        self._deferred_embeddings: dict[str, list[dict[str, Any]]] = {}
        self._gid_last_persist: dict[str, float] = {}
        self.last_latency_ms = 0.0

    # ------------------------------------------------------------ public API
    def needs_embedding(self, key: str, now: float, *, bound_interval: float, unbound_interval: float) -> bool:
        st = self.tracks.get(key)
        if st is None:
            return True
        interval = bound_interval if st.status == "confirmed" else unbound_interval
        return (now - st.last_embed_ts) >= interval

    def priority(self, key: str) -> int:
        st = self.tracks.get(key)
        if st is None or st.status != "confirmed":
            return 0
        return 1

    def binding_for(self, key: str) -> LiveTrackState | None:
        return self.tracks.get(key)

    def update(
        self,
        now: float,
        tracks_by_camera: dict[str, list[LocalTrack]],
        frame_shapes: dict[str, tuple[int, int]],
        embeddings: dict[str, tuple[np.ndarray, float]],
        wall_now: float | None = None,
        faces: dict[str, tuple[np.ndarray, float]] | None = None,
    ) -> None:
        t0 = time.perf_counter()
        wall_now = time.time() if wall_now is None else wall_now
        self._ingest(now, wall_now, tracks_by_camera, frame_shapes, embeddings)
        self._ingest_faces(now, faces)
        self._expire_stale(now, wall_now)
        self._apply_manual_overrides()

        if self.linker is not None:
            self.links = self.linker.update(self._observations(now))
        else:
            self.links = {}

        self._verify_bound(now)
        self._resolve_faces(now, wall_now)
        self._inherit_from_links(now)
        self._inherit_from_overlap_captions(now)
        self._inherit_continuity(now)
        self._match_handoff(now)
        self._match_gallery(now)
        self._enroll(now, wall_now)
        self._inherit_from_links(now)
        self._inherit_from_overlap_captions(now)
        self._repair_linked_duplicates(now)
        self._feed_identity_correspondences(now)
        self._update_gallery_and_bindings(now, wall_now)
        self._record_co_visibility()
        self._consolidate_duplicates(now)
        self._flush_deferred(now)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

    def close_all(self, now: float, wall_now: float | None = None) -> None:
        wall_now = time.time() if wall_now is None else wall_now
        for key in list(self.tracks):
            self._finalize(key, now, wall_now)
        for gid in list(self._deferred_embeddings):
            self._persist_deferred(gid)

    # ------------------------------------------------------------- internals
    def _horizon(self, camera_id: str, base: float) -> float:
        interval = self._cam_interval.get(camera_id)
        if interval is None:
            return base
        return max(base, min(self.cfg.max_observation_horizon, self.cfg.horizon_frame_factor * interval))

    def _ingest(self, now, wall_now, tracks_by_camera, frame_shapes, embeddings) -> None:
        self._frame_obs = [
            (cam, tr.timestamp, tr.compound_key, tr.bbox_xyxy)
            for cam, tracks in tracks_by_camera.items()
            for tr in tracks
            if not tr.is_predicted and tr.lifecycle != TrackLifecycle.REMOVED
        ]
        for cam in tracks_by_camera:
            last = self._cam_last_update.get(cam)
            if last is not None and now > last:
                gap = min(now - last, 30.0)
                prev = self._cam_interval.get(cam)
                self._cam_interval[cam] = gap if prev is None else 0.7 * prev + 0.3 * gap
            self._cam_last_update[cam] = now
        for cam, tracks in tracks_by_camera.items():
            h, w = frame_shapes.get(cam, (720, 1280))
            for tr in tracks:
                key = tr.compound_key
                if tr.lifecycle == TrackLifecycle.REMOVED:
                    if key in self.tracks:
                        self._finalize(key, now, wall_now, removed_by_tracker=True)
                    continue
                st = self.tracks.get(key)
                if st is None:
                    st = LiveTrackState(
                        key=key,
                        camera_id=cam,
                        session_id=tr.tracker_session_id,
                        local_id=tr.local_track_id,
                        first_seen=tr.timestamp,
                        last_seen=tr.timestamp,
                        first_seen_wall=wall_now,
                        bbox=tr.bbox_xyxy,
                        frame_w=w,
                        frame_h=h,
                    )
                    st.bank = deque(maxlen=self.cfg.bank_size)
                    self.tracks[key] = st
                st.last_seen = tr.timestamp
                st.bbox = tr.bbox_xyxy
                st.frame_w, st.frame_h = w, h
                st.hits = tr.hit_count
                x1, y1, x2, y2 = tr.bbox_xyxy
                st.recent_centers.append(((x1 + x2) / 2.0 / max(1, w), (y1 + y2) / 2.0 / max(1, h)))
                st.trail.append((tr.timestamp, (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, y2 - y1)))
                emb = embeddings.get(key)
                if emb is not None:
                    vec, q = emb
                    st.bank.append((np.asarray(vec, dtype=np.float32), float(q), tr.timestamp))
                    st.last_embed_ts = tr.timestamp

    def _expire_stale(self, now: float, wall_now: float) -> None:
        for key, st in list(self.tracks.items()):
            if now - st.last_seen > self.cfg.stale_track_seconds:
                self._finalize(key, now, wall_now)

    def _apply_manual_overrides(self) -> None:
        for key, st in self.tracks.items():
            b = self.identity.get_binding(key)
            if b is None or not b.global_id or b.confidence < 1.0 or b.status != "confirmed":
                continue
            if b.global_id != st.gid and st.source != "manual":
                logger.info("Manual correction applied %s -> %s", key, b.global_id)
                st.gid, st.status, st.source, st.confidence = b.global_id, "confirmed", "manual", 1.0

    def _observations(self, now: float) -> list[ViewObservation]:
        # Slow hall cameras (cam_03 ~0.4–1 fps) lag several seconds behind cam_02. Using each
        # track's packet timestamp as the link clock made every cross-view pair look "skewed"
        # and dropped cam_03 from the linker whenever global now raced ahead. Treat currently
        # live occupancy as simultaneous; gate staleness per-camera instead.
        latest_by_cam: dict[str, float] = {}
        for st in self.tracks.values():
            prev = latest_by_cam.get(st.camera_id)
            if prev is None or st.last_seen > prev:
                latest_by_cam[st.camera_id] = st.last_seen
        out = []
        for st in self.tracks.values():
            cam_latest = latest_by_cam.get(st.camera_id, now)
            if cam_latest - st.last_seen > self._horizon(st.camera_id, self.cfg.observation_horizon):
                continue
            if now - cam_latest > self.cfg.max_observation_horizon:
                continue
            out.append(
                ViewObservation(
                    key=st.key,
                    camera_id=st.camera_id,
                    bbox_xyxy=st.bbox,
                    frame_width=st.frame_w,
                    frame_height=st.frame_h,
                    timestamp=now,
                    embedding=st.proto(),
                    stationary=st.is_stationary(self.cfg.stationary_max_shift),
                )
            )
        return out

    def _overlapping(self, cam_a: str, cam_b: str) -> bool:
        if cam_a == cam_b:
            return False
        if self.linker is not None and self.linker.registry.overlapping(cam_a, cam_b):
            return True
        return self.topology.allows_simultaneous(cam_a, cam_b)

    def _is_overlap_camera(self, cam: str) -> bool:
        return self.linker is not None and self.linker.registry.group_of(cam) is not None

    def _available(self, st: LiveTrackState, gid: str, now: float) -> bool:
        """Can ``gid`` be given to ``st`` without violating physical constraints?"""
        if not self.identity.is_live_identity(gid) or gid in st.face_vetoed:
            return False
        partner_key = self.links.get(st.key).partner if st.key in self.links else None
        for other in self.tracks.values():
            if other.key == st.key or other.gid != gid or other.status != "confirmed":
                continue
            if now - other.last_seen > self._horizon(other.camera_id, self.cfg.simultaneity_horizon):
                continue
            if other.camera_id == st.camera_id:
                return False  # one person cannot be two live tracks in one view
            if not self._overlapping(other.camera_id, st.camera_id):
                return False  # simultaneously in two non-overlapping rooms
            if partner_key != other.key:
                # Overlapping views may share a live ID only when the two tracks
                # are linked as the same physical person (geometry/strong
                # appearance). Similar clothing alone is not enough.
                return False
        seen = self._gid_seen.get(gid)
        if seen is not None:
            cam, ts = seen
            if cam != st.camera_id and not self._overlapping(cam, st.camera_id):
                ok, _ = self.topology.temporal_feasible(cam, st.camera_id, st.first_seen - ts)
                if not ok:
                    return False
        return True

    def _recently_departed(self, st: LiveTrackState, gid: str) -> bool:
        """``gid`` left view shortly before ``st`` appeared, on a camera ``st`` is reachable from."""
        seen = self._gid_seen.get(gid)
        if seen is None:
            return False
        cam, ts = seen
        gap = st.first_seen - ts
        if gap < 0:
            return False
        if cam == st.camera_id or self._overlapping(cam, st.camera_id):
            return gap <= self.cfg.recent_reentry_window
        tr = self.topology.transition(cam, st.camera_id)
        if tr is None or tr.overlapping:
            return False
        return tr.min_travel_seconds <= gap <= tr.max_travel_seconds

    def _bind(self, st: LiveTrackState, gid: str, now: float, *, source: str, confidence: float, status: str = "confirmed") -> None:
        prev = st.gid
        st.gid = gid
        st.status = status
        st.source = source
        st.confidence = float(confidence)
        st.candidate = None
        st.bound_at = now
        st.contradictions = 0
        if prev != gid or source == "face":
            st.face_verified = source == "face"
        if status == "confirmed" and prev != gid:
            self.identity.record_event(
                "TRACK_BOUND",
                gid,
                st.camera_id,
                {"compound_key": st.key, "source": source, "confidence": round(float(confidence), 4), "previous": prev},
            )

    def _verify_bound(self, now: float) -> None:
        cfg = self.cfg
        for st in self.tracks.values():
            if st.status != "confirmed" or st.gid is None or st.source == "manual":
                continue
            if st.face_verified:
                continue  # the face confirmed this ID; body appearance changes with pose
            link = self.links.get(st.key)
            if st.source == "link" and link is not None:
                continue  # cross-view appearance is not reliable evidence against a geometric link
            if not st.bank or st.bank[-1][2] < st.bound_at:
                continue
            latest = st.bank[-1][0]
            sim = self.gallery.max_similarity(st.gid, latest)
            if sim < 0:  # identity has no gallery yet
                continue
            if sim < cfg.contradiction_threshold:
                st.contradictions += 1
            else:
                st.contradictions = max(0, st.contradictions - 1)
            if st.contradictions < cfg.contradiction_votes:
                continue
            recent = np.stack([b[0] for b in list(st.bank)[-cfg.contradiction_votes:]])
            scores = self.gallery.score_bank(recent)
            own = scores.get(st.gid, -1.0)
            best_gid, best = max(scores.items(), key=lambda kv: kv[1], default=(None, -1.0))
            old = st.gid
            if best_gid and best_gid != old and best >= cfg.match_threshold + cfg.match_margin and self._available(st, best_gid, now):
                self._bind(st, best_gid, now, source="gallery_switch", confidence=best)
                logger.info("Track %s switched %s -> %s (contradiction, own=%.2f best=%.2f)", st.key, old, best_gid, own, best)
            elif own < cfg.contradiction_threshold:
                st.gid, st.status, st.source, st.confidence = None, "pending", "", 0.0
                st.bank.clear()
                logger.info("Track %s released %s after repeated contradiction (own=%.2f)", st.key, old, own)
                self.identity.record_event("TRACK_RELEASED", old, st.camera_id, {"compound_key": st.key, "own_score": own})
            st.contradictions = 0

    def _inherit_from_links(self, now: float) -> None:
        for key, link in self.links.items():
            st = self.tracks.get(key)
            partner = self.tracks.get(link.partner)
            if st is None or partner is None:
                continue
            if st.status == "confirmed" or partner.status != "confirmed" or partner.gid is None:
                continue
            if link.votes < self.cfg.link_inherit_min_votes:
                continue
            if self._available(st, partner.gid, now):
                self._bind(st, partner.gid, now, source="link", confidence=link.score)

    def _inherit_from_overlap_captions(self, now: float) -> None:
        """Share IDs across overlapping views when clothing captions strongly agree.

        Seat anchors only fire at calibrated seats; walkers between seats get no geometry.
        A mutual best caption match fills that gap without forcing same-clothes strangers together
        (requires margin + _available simultaneity checks).
        """
        cfg = self.cfg
        if not cfg.overlap_caption_link or not cfg.face_enabled:
            return
        pending = [
            st for st in self.tracks.values()
            if st.status != "confirmed" and st.face_bank
            and now - st.last_seen <= self._horizon(st.camera_id, cfg.observation_horizon)
            and not st.dormant() and self._is_overlap_camera(st.camera_id)
        ]
        confirmed = [
            st for st in self.tracks.values()
            if st.status == "confirmed" and st.gid and st.face_bank
            and now - st.last_seen <= self._horizon(st.camera_id, cfg.simultaneity_horizon)
            and not st.dormant() and self._is_overlap_camera(st.camera_id)
        ]

        def proto(st: LiveTrackState) -> np.ndarray:
            q = np.stack([f[0] for f in st.face_bank])
            v = q.mean(axis=0)
            return v / max(float(np.linalg.norm(v)), 1e-12)

        def mutual_best(st: LiveTrackState, pool: list[LiveTrackState]) -> tuple[float, LiveTrackState] | None:
            mine = proto(st)
            scored: list[tuple[float, LiveTrackState]] = []
            for other in pool:
                if other.camera_id == st.camera_id or not self._overlapping(st.camera_id, other.camera_id):
                    continue
                if other.gid and not self._available(st, other.gid, now):
                    continue
                scored.append((float(mine @ proto(other)), other))
            if not scored:
                return None
            scored.sort(reverse=True, key=lambda x: x[0])
            best, partner = scored[0]
            second = scored[1][0] if len(scored) > 1 else -1.0
            if best < cfg.overlap_caption_threshold or best - second < cfg.overlap_caption_margin:
                return None
            rivals = [
                float(proto(partner) @ proto(p))
                for p in pool
                if p.key != st.key and p.camera_id == st.camera_id
            ]
            if rivals and best - max(rivals) < cfg.overlap_caption_margin:
                return None
            return best, partner

        for st in pending:
            hit = mutual_best(st, confirmed)
            if hit is None:
                continue
            best, partner = hit
            self._bind(st, partner.gid, now, source="caption_link", confidence=best)
            logger.info(
                "Track %s caption-linked to %s as %s (sim=%.2f)",
                st.key, partner.key, partner.gid, best,
            )

        # Both sides already enrolled separately (common when streams are skewed): merge young→old.
        done: set[str] = set()
        for st in confirmed:
            if st.key in done or not st.gid:
                continue
            hit = mutual_best(st, confirmed)
            if hit is None:
                continue
            best, partner = hit
            if not partner.gid or partner.gid == st.gid or partner.key in done:
                continue
            if frozenset((st.gid, partner.gid)) in self._co_visible:
                continue
            ca, cb = self._created_at.get(st.gid), self._created_at.get(partner.gid)
            if ca is None or cb is None:
                continue
            if ca <= cb:
                keep, drop, drop_track = st.gid, partner.gid, partner
            else:
                keep, drop, drop_track = partner.gid, st.gid, st
            if any(t.gid == drop and t.key != drop_track.key for t in self.tracks.values()):
                continue
            other_cams = self.gallery.identity_cameras().get(drop, set()) - {drop_track.camera_id}
            if other_cams:
                if self._available(drop_track, keep, now):
                    self._bind(drop_track, keep, now, source="caption_link", confidence=best)
                    logger.info(
                        "Caption relabel %s: %s -> %s (sim=%.2f; %s has other-camera history)",
                        drop_track.key, drop, keep, best, drop,
                    )
                    done.update({st.key, partner.key})
                continue
            moved = self.identity.merge_identity_into(
                drop, keep, reason=f"overlap_caption_merge sim={best:.2f} {drop_track.key}<->{st.key if drop_track is partner else partner.key}"
            )
            for rec in self._deferred_embeddings.pop(drop, []):
                rec["global_id"] = keep
                self._deferred_embeddings.setdefault(keep, []).append(rec)
            drop_track.gid, drop_track.source = keep, "caption_merge"
            self._created_at.pop(drop, None)
            done.update({st.key, partner.key})
            logger.info(
                "Merged duplicate %s into %s via overlap caption (sim=%.2f, %d embeddings)",
                drop, keep, best, moved,
            )

    def _inherit_continuity(self, now: float) -> None:
        """Hand a just-lost track's ID to the new local track that replaced it."""
        cfg = self.cfg
        for st in list(self.tracks.values()):
            if st.status == "confirmed" or now - st.first_seen > max(cfg.continuity_window, cfg.reconnect_continuity_window):
                continue
            candidates: list[tuple[_Ended, LiveTrackState | None]] = [
                (e, None) for e in self._ended if e.camera_id == st.camera_id
            ]
            for other in self.tracks.values():
                if other.key != st.key and other.camera_id == st.camera_id and other.status == "confirmed" and other.gid:
                    candidates.append((_Ended(other.gid, other.camera_id, other.bbox, other.last_seen, other.proto()), other))
            best: tuple[float, _Ended, LiveTrackState | None] | None = None
            for e, holder in candidates:
                reset = holder is not None and holder.session_id != st.session_id
                if not reset and now - st.first_seen > cfg.continuity_window:
                    continue
                gap = st.first_seen - e.ended_at
                if gap < 0 or gap > (cfg.reconnect_continuity_window if reset else cfg.continuity_window):
                    continue  # concurrent with the new track, or too old
                iou = iou_xyxy(st.bbox, e.bbox)
                cx1, cy1 = (st.bbox[0] + st.bbox[2]) / 2, (st.bbox[1] + st.bbox[3]) / 2
                cx2, cy2 = (e.bbox[0] + e.bbox[2]) / 2, (e.bbox[1] + e.bbox[3]) / 2
                scale = max(1.0, (e.bbox[3] - e.bbox[1]))
                center_ratio = float(np.hypot(cx1 - cx2, cy1 - cy2) / scale)
                if iou < cfg.continuity_iou and center_ratio > cfg.continuity_center_ratio:
                    continue
                proto = st.proto()
                if proto is not None and e.proto is not None and float(proto @ e.proto) < cfg.continuity_min_appearance:
                    continue
                score = iou + (1.0 - min(1.0, center_ratio))
                if best is None or score > best[0]:
                    best = (score, e, holder)
            if best is None:
                continue
            score, ended, holder = best
            if holder is not None:
                # The lost track yields its ID; if it ever reappears it re-resolves.
                holder.gid, holder.status, holder.source, holder.confidence = None, "pending", "", 0.0
                holder.yielded_at = holder.last_seen
            if self._available(st, ended.gid, now):
                self._bind(st, ended.gid, now, source="continuity", confidence=min(1.0, score / 2.0))
                if holder is None and ended in self._ended:
                    self._ended.remove(ended)
            elif holder is not None:
                holder.gid, holder.status, holder.source = ended.gid, "confirmed", "continuity_restore"
                holder.yielded_at = None

    def _ingest_faces(self, now: float, faces: dict[str, tuple[np.ndarray, float]] | None) -> None:
        if not faces or not self.cfg.face_enabled:
            return
        for key, (emb, q) in faces.items():
            st = self.tracks.get(key)
            if st is None:
                continue
            v = np.asarray(emb, dtype=np.float32)
            st.face_bank.append((v / max(float(np.linalg.norm(v)), 1e-12), float(q), now))
            st.face_seen += 1

    def _face_query(self, st: LiveTrackState) -> np.ndarray | None:
        """The track's face samples, or None if too few or mutually inconsistent (tracker swap)."""
        if len(st.face_bank) < self.cfg.face_min_samples:
            return None
        q = np.stack([f[0] for f in st.face_bank])
        if len(q) > 1:
            s = q @ q.T
            if (s.sum() - len(q)) / (len(q) * len(q) - len(q)) < self.cfg.face_self_consistency:
                st.face_bank.clear()
                return None
        return q

    def _release(self, st: LiveTrackState, reason: str) -> None:
        old = st.gid
        st.gid, st.status, st.source, st.confidence = None, "pending", "", 0.0
        st.face_verified = False
        st.bank.clear()
        logger.info("Track %s released %s (%s)", st.key, old, reason)
        self.identity.record_event("TRACK_RELEASED", old, st.camera_id, {"compound_key": st.key, "reason": reason})

    def _resolve_faces(self, now: float, wall_now: float) -> None:
        cfg = self.cfg
        if not cfg.face_enabled:
            return
        fg = self.identity.face_gallery
        for st in list(self.tracks.values()):
            if st.face_seen == st.face_evaluated or st.dormant():
                continue
            st.face_evaluated = st.face_seen
            q = self._face_query(st)
            if q is None:
                continue
            scores = {g: s for g, s in fg.score(q).items() if self.identity.is_live_identity(g)}
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            best_gid, best = ranked[0] if ranked else (None, -1.0)
            second = ranked[1][1] if len(ranked) > 1 else -1.0

            if best_gid and best >= cfg.face_match_threshold and best - second >= cfg.face_margin:
                body_ok = False
                if st.bank and st.proto() is not None:
                    body_ok = self.gallery.max_similarity(best_gid, st.proto()) >= cfg.face_match_body_floor
                reentry = self._recently_departed(st, best_gid) or any(
                    e.gid == best_gid and e.walked_out and not e.handed_off for e in self._ended
                )
                if cfg.face_match_requires_reentry and not (reentry or body_ok):
                    continue  # clothing caption alone is not enough while many people are seated
                if st.gid != best_gid or st.status != "confirmed":
                    if not self._claim_for_face(st, best_gid, now):
                        continue
                    old = st.gid
                    self._bind(st, best_gid, now, source="face", confidence=min(0.999, best))
                    logger.info(
                        "Track %s caption/face-matched %s (sim=%.2f, reentry=%s body=%s, was %s)",
                        st.key, best_gid, best, reentry, body_ok, old,
                    )
                    if old and old != best_gid and old in self._created_at and not any(
                        t.gid == old for t in self.tracks.values()
                    ):
                        self._merge_consolidated(old, best_gid, f"face match {best:.2f} on {st.key}")
                st.face_verified = True
                self._learn_face(st, now, wall_now)
                continue

            if st.status != "confirmed" or not st.gid:
                continue
            own = scores.get(st.gid)
            if own is not None and fg.count(st.gid) >= cfg.face_veto_min_gallery and own < cfg.face_veto_threshold:
                st.face_vetoed.add(st.gid)
                self._release(st, f"face contradicts ID (face={own:.2f})")
                continue
            if best < cfg.face_unknown_threshold and fg.count(st.gid) == 0 and self._reliable_for_face_enroll(st, now):
                self._learn_face(st, now, wall_now, force=True)
                st.face_verified = True
                logger.info("Track %s enrolled first face for %s", st.key, st.gid)

    def _claim_for_face(self, st: LiveTrackState, gid: str, now: float) -> bool:
        """Free ``gid`` for a face-matched track: body-only holders elsewhere were wrong matches."""
        partner = self.links.get(st.key).partner if st.key in self.links else None
        for other in list(self.tracks.values()):
            if other.key == st.key or other.gid != gid or other.status != "confirmed":
                continue
            if now - other.last_seen > self._horizon(other.camera_id, self.cfg.simultaneity_horizon):
                continue
            if other.key == partner or (self._overlapping(other.camera_id, st.camera_id) and other.camera_id != st.camera_id):
                continue  # the other view of the same person may share the ID
            if other.face_verified:
                return False  # two face-verified tracks: leave it to the operator / later evidence
            self._release(other, f"face of {gid} seen on {st.key}")
        return True

    def _reliable_for_face_enroll(self, st: LiveTrackState, now: float) -> bool:
        if st.source in {"new", "continuity", "link", "manual", "dedup_merge", "anchor_merge"}:
            pass
        elif st.source == "gallery" and st.confidence >= self.cfg.strong_match_threshold:
            pass
        else:
            return False
        return st.contradictions == 0 and now - st.bound_at >= self.cfg.face_enroll_min_bound

    def _learn_face(self, st: LiveTrackState, now: float, wall_now: float, force: bool = False) -> None:
        if not st.gid or not st.face_bank:
            return
        if not force and now - st.last_face_learn < self.cfg.face_learn_interval:
            return
        st.last_face_learn = now
        for emb, q, _ in sorted(st.face_bank, key=lambda f: f[1], reverse=True)[:3 if force else 1]:
            self.identity.add_face_embedding(st.gid, emb, camera_id=st.camera_id, timestamp=wall_now, quality=q)

    def _is_moving(self, st: LiveTrackState, window: float | None = -1.0) -> bool:
        """Displacement within the last ``window`` seconds (default config window; None = whole track)."""
        if len(st.trail) < 2:
            return False
        if window is not None and window < 0:
            window = self.cfg.walkout_motion_window
        t_end = st.trail[-1][0]
        pts = [p for p in st.trail if window is None or t_end - p[0] <= window]
        if len(pts) < 2:
            return False
        arr = np.asarray(pts, dtype=np.float64)
        disp = float(np.hypot(np.ptp(arr[:, 1]), np.ptp(arr[:, 2])))
        return disp / float(np.median(arr[:, 3])) >= self.cfg.walkout_min_motion

    def _handoff_candidates(self, st: LiveTrackState, now: float) -> dict[str, tuple[str, float, np.ndarray | None, Any]]:
        """IDs that walked out of a camera ``st``'s camera is reachable from: gid -> (camera, exit_ts, proto, source)."""
        cfg = self.cfg
        out: dict[str, tuple[str, float, np.ndarray | None, Any]] = {}

        def consider(gid: str, cam: str, ts: float, proto, src) -> None:
            if cam == st.camera_id or self._overlapping(cam, st.camera_id):
                return
            tr = self.topology.transition(cam, st.camera_id)
            if tr is None or tr.overlapping:
                return
            gap = st.first_seen - ts
            if not (tr.min_travel_seconds - cfg.handoff_time_slack <= gap <= tr.max_travel_seconds):
                return
            if gid in out and out[gid][1] >= ts:
                return
            out[gid] = (cam, ts, proto, src)

        for e in self._ended:
            if e.walked_out and not e.handed_off:
                consider(e.gid, e.camera_id, e.ended_at, e.proto, e)
        for other in self.tracks.values():
            if other.key == st.key or other.status != "confirmed" or not other.gid or other.dormant():
                continue
            if now - other.last_seen <= self._horizon(other.camera_id, cfg.simultaneity_horizon):
                continue  # still visible
            if self._is_moving(other):
                consider(other.gid, other.camera_id, other.last_seen, other.proto(), other)
        return {g: v for g, v in out.items() if self._available(st, g, now)}

    def _handoff_score(self, st: LiveTrackState, gid: str, proto: np.ndarray | None, scores: dict[str, float]) -> float:
        s = scores.get(gid, -1.0)
        mine = st.proto()
        if mine is not None and proto is not None:
            s = max(s, float(mine @ proto))
        return s

    def _match_handoff(self, now: float) -> None:
        cfg = self.cfg
        if not cfg.walkout_handoff:
            return
        for st in list(self.tracks.values()):
            if st.status == "confirmed" or st.dormant() or len(st.bank) < cfg.min_samples_to_match:
                continue
            if now - st.last_seen > self._horizon(st.camera_id, cfg.observation_horizon) or not self._is_moving(st, None):
                continue  # must have walked in; a person already sitting there is not a transit
            cands = self._handoff_candidates(st, now)
            if not cands:
                continue
            scores = self.gallery.score_bank(st.bank_matrix())
            ranked = sorted(
                ((self._handoff_score(st, g, v[2], scores), g) for g, v in cands.items()), reverse=True
            )
            best, gid = ranked[0]
            # A clear gallery match to someone else wins; the normal matcher handles it.
            other_best = max((s for g, s in scores.items() if g not in cands), default=-1.0)
            if other_best >= cfg.cross_camera_match_threshold and other_best > best:
                continue
            if len(ranked) == 1:
                ok = best >= cfg.handoff_min_appearance
            else:
                second = ranked[1][0]
                ok = best >= cfg.handoff_multi_min_appearance and best - second >= cfg.handoff_margin
                if not ok and best >= cfg.handoff_min_appearance:
                    cams = [cands[g][0] for _, g in ranked]
                    times = [cands[g][1] for _, g in ranked]
                    group = self.linker.registry.group_of(cams[0]) if self.linker is not None else None
                    one_event = (
                        len(set(cams)) == len(cams)
                        and group is not None
                        and all(self.linker.registry.group_of(c) == group for c in cams)
                        and max(times) - min(times) <= cfg.handoff_event_window
                    )
                    ok = one_event  # one person leaving the hall, seen by both hall views
            if not ok:
                continue
            cam, ts, _, src = cands[gid]
            if isinstance(src, _Ended):
                src.handed_off = True
            elif isinstance(src, LiveTrackState):
                src.gid, src.status, src.source, src.confidence = None, "pending", "", 0.0
                src.yielded_at = src.last_seen
            self._bind(st, gid, now, source="handoff", confidence=max(0.0, best))
            logger.info(
                "Track %s handoff %s from %s (gap=%.1fs, appearance=%.2f, candidates=%d)",
                st.key, gid, cam, st.first_seen - ts, best, len(ranked),
            )

    def _match_gallery(self, now: float) -> None:
        cfg = self.cfg
        pending = [
            st for st in self.tracks.values()
            if st.status != "confirmed" and len(st.bank) >= cfg.min_samples_to_match
            and now - st.last_seen <= self._horizon(st.camera_id, cfg.observation_horizon) and not st.dormant()
        ]
        if not pending:
            return
        score_rows: list[dict[str, float]] = []
        for st in pending:
            scores = self.gallery.score_bank(st.bank_matrix())
            score_rows.append({g: s for g, s in scores.items() if self._available(st, g, now)})
        gids = sorted({g for row in score_rows for g in row})
        if not gids:
            for st in pending:
                st.candidate, st.confidence = None, 0.0
            return
        mat = np.full((len(pending), len(gids)), -1.0)
        for i, row in enumerate(score_rows):
            for j, g in enumerate(gids):
                if g in row:
                    mat[i, j] = row[g]
        # Spatio-temporal prior: IDs that just left a camera this one is reachable from.
        recent = np.zeros(mat.shape, dtype=bool)
        if cfg.recent_departure_bonus > 0:
            for i, st in enumerate(pending):
                for j, g in enumerate(gids):
                    if mat[i, j] > -1.0 and self._recently_departed(st, g):
                        recent[i, j] = True
        adj = mat + cfg.recent_departure_bonus * recent
        rows, cols = linear_sum_assignment(1.0 - adj)
        assigned = dict(zip(rows, cols))
        gid_cameras = self.gallery.identity_cameras()
        for i, st in enumerate(pending):
            st.status = "pending"
            row = mat[i]
            best_j = int(np.argmax(row))
            st.confidence = float(row[best_j])
            st.candidate = gids[best_j] if row[best_j] > 0 else None
            if i not in assigned:
                continue
            j = assigned[i]
            s = float(mat[i, j])
            if recent[i, j] and s >= cfg.recent_match_threshold:
                others = np.concatenate([np.delete(adj[i], j), np.delete(adj[:, j], i)])
                if float(adj[i, j]) - float(others.max(initial=-1.0)) >= cfg.recent_match_margin:
                    logger.info("Track %s matched recently departed %s (score=%.2f)", st.key, gids[j], s)
                    self._bind(st, gids[j], now, source="gallery", confidence=s)
                    continue
            same_camera = st.camera_id in gid_cameras.get(gids[j], set())
            threshold = cfg.match_threshold if same_camera else cfg.cross_camera_match_threshold
            margin = cfg.match_margin if same_camera else cfg.cross_camera_margin
            strong = cfg.strong_match_threshold if same_camera else max(cfg.strong_match_threshold, threshold + margin)
            if s < threshold:
                if s >= cfg.new_identity_threshold:
                    st.status, st.candidate = "provisional", gids[j]
                continue
            others_row = np.delete(row, j)
            others_col = np.delete(mat[:, j], i)
            second = float(max(others_row.max(initial=-1.0), others_col.max(initial=-1.0)))
            if s >= strong or s - second >= margin:
                self._bind(st, gids[j], now, source="gallery", confidence=s)
            else:
                st.status, st.candidate = "provisional", gids[j]

    def _enroll(self, now: float, wall_now: float) -> None:
        cfg = self.cfg
        for st in list(self.tracks.values()):
            if st.status == "confirmed" or now - st.last_seen > self._horizon(st.camera_id, cfg.observation_horizon) or st.dormant():
                continue
            age = now - st.first_seen
            overlap_cam = self._is_overlap_camera(st.camera_id)
            min_age = cfg.overlap_enroll_delay if overlap_cam else cfg.min_age_to_enroll
            if age < min_age:
                continue
            if (
                cfg.walkout_handoff and age < cfg.handoff_hold_seconds and self._handoff_candidates(st, now)
            ):
                continue  # a newcomer may be someone who just walked out of a neighbouring camera
            link = self.links.get(st.key)
            if link is not None:
                partner = self.tracks.get(link.partner)
                if partner is not None and partner.status != "confirmed":
                    # Enroll only one side of a linked pair; the other inherits.
                    if (len(partner.bank), partner.hits, partner.key) > (len(st.bank), st.hits, st.key):
                        continue
            best = st.confidence if st.candidate else -1.0
            reason = None
            verification = VerificationStatus.UNVERIFIED
            if len(st.bank) >= cfg.min_samples_to_enroll and best < cfg.new_identity_threshold:
                reason = "no_gallery_match"
            elif len(st.bank) >= cfg.min_samples_to_enroll and age >= cfg.ambiguous_timeout:
                reason = "ambiguous_timeout"
                verification = VerificationStatus.PROVISIONAL
            elif not st.bank and age >= cfg.no_appearance_timeout:
                reason = "no_appearance"
                verification = VerificationStatus.PROVISIONAL
            if reason is None:
                continue
            if st.candidate:
                reason = f"{reason} best={st.candidate}:{best:.3f} samples={len(st.bank)}"
            gid = self.identity.create_live_identity(st.camera_id, wall_now, verification=verification, reason=reason)
            self._created_at[gid] = now
            self._bind(st, gid, now, source="new", confidence=max(0.0, 1.0 - max(0.0, best)))
            for vec, q, ts in sorted(st.bank, key=lambda b: b[1], reverse=True)[:3]:
                rec = self.identity.add_live_embedding(gid, vec, camera_id=st.camera_id, timestamp=wall_now, quality=q, persist=False)
                self._deferred_embeddings.setdefault(gid, []).append(rec)
            st.last_gallery_update = now

    def _repair_linked_duplicates(self, now: float) -> None:
        cfg = self.cfg
        done: set[str] = set()
        for key, link in self.links.items():
            if key in done or link.votes < cfg.merge_min_votes or not link.geometric:
                continue
            a, b = self.tracks.get(key), self.tracks.get(link.partner)
            done.update({key, link.partner})
            if a is None or b is None or a.status != "confirmed" or b.status != "confirmed":
                continue
            if not a.gid or not b.gid or a.gid == b.gid:
                continue
            ca, cb = self._created_at.get(a.gid), self._created_at.get(b.gid)
            young, old, young_track = None, None, None
            if cb is not None and now - cb <= cfg.merge_window and (ca is None or cb > ca):
                young, old, young_track = b.gid, a.gid, b
            elif ca is not None and now - ca <= cfg.merge_window:
                young, old, young_track = a.gid, b.gid, a
            if young is None:
                self._merge_anchor_verified(a, b, link, now)
                continue
            holders = [t for t in self.tracks.values() if t.gid == young and t.key != young_track.key]
            if holders:
                continue
            moved = self.gallery.reassign_identity(young, old)
            for rec in self._deferred_embeddings.pop(young, []):
                rec["global_id"] = old
                self._deferred_embeddings.setdefault(old, []).append(rec)
            young_track.gid, young_track.source = old, "link_merge"
            self._created_at.pop(young, None)
            self.identity.archive_identity(young, merged_into=old, reason=f"linked_views link_score={link.score:.2f}")
            logger.info("Merged duplicate %s into %s (linked views, %d embeddings moved)", young, old, moved)
            if old not in self._created_at:
                self._persist_deferred(old)

    def _merge_anchor_verified(self, a: LiveTrackState, b: LiveTrackState, link: Link, now: float) -> None:
        """Merge two established IDs held by a long-standing seat-anchor link (both people seated)."""
        cfg = self.cfg
        if self.linker is None or cfg.anchor_merge_min_votes <= 0 or link.votes < cfg.anchor_merge_min_votes:
            return
        # Both bindings must be settled: a freshly matched (possibly wrong) ID must never absorb another.
        for t in (a, b):
            if t.contradictions > 0 or now - t.bound_at < cfg.anchor_merge_min_bound_seconds:
                return
        registry = self.linker.registry
        group = registry.group_of(a.camera_id)
        if group is None or group is not registry.group_of(b.camera_id):
            return
        ref, member = (a, b) if a.camera_id == group.reference_camera else (b, a)
        if ref.camera_id != group.reference_camera or registry.match_radius(member.camera_id) is None:
            return  # only verified seat anchors justify merging established identities
        if not (ref.is_stationary(cfg.stationary_max_shift) and member.is_stationary(cfg.stationary_max_shift)):
            return
        src, tgt = member.gid, ref.gid
        if any(t.gid == src and t.key != member.key for t in self.tracks.values()):
            return
        if self.gallery.identity_cameras().get(src, set()) - {member.camera_id}:
            # The member's ID has history on other cameras, so it may be a wrong gallery match of a
            # different person: relabel this track only, never merge that identity's data.
            if self._available(member, tgt, now):
                logger.info("Seat-anchor relabel %s: %s -> %s (no merge: %s seen on other cameras)",
                            member.key, src, tgt, src)
                self._bind(member, tgt, now, source="link", confidence=min(0.999, link.score))
            return
        moved = self.identity.merge_identity_into(
            src, tgt, reason=f"seat_anchor_link votes={link.votes} score={link.score:.2f} {member.key}<->{ref.key}"
        )
        for rec in self._deferred_embeddings.pop(src, []):
            rec["global_id"] = tgt
            self._deferred_embeddings.setdefault(tgt, []).append(rec)
        member.gid, member.source = tgt, "anchor_merge"
        self._created_at.pop(src, None)
        logger.info("Merged established duplicate %s into %s (seat-anchor link, %d votes, %d embeddings moved)",
                    src, tgt, link.votes, moved)

    def _record_co_visibility(self) -> None:
        """Remember ID pairs proven to be different people (visible at the same moment)."""
        cfg = self.cfg
        current: list[tuple[str, str, float, tuple[float, float, float, float]]] = []
        for cam, ts, key, bbox in self._frame_obs:
            st = self.tracks.get(key)
            if st is None or st.status != "confirmed" or not st.gid:
                continue
            current.append((st.gid, cam, ts, bbox))
            prev = self._gid_obs.get(st.gid)
            if prev is not None and ts > prev[1]:
                self._observed_seconds[st.gid] = self._observed_seconds.get(st.gid, 0.0) + min(ts - prev[1], 1.0)
            self._gid_obs[st.gid] = (cam, ts)
        for i in range(len(current)):
            ga, ca, _, ba = current[i]
            for gb, cb, _, bb in current[i + 1:]:
                if ga != gb and ca == cb and iou_xyxy(ba, bb) <= cfg.dedup_covisible_max_iou:
                    self._co_visible.add(frozenset((ga, gb)))
        for g, cam, ts, _ in current:
            for og, (ocam, ots) in self._gid_obs.items():
                if og == g or ocam == cam or self._overlapping(ocam, cam):
                    continue  # overlapping views may show one person under two unlinked IDs
                if abs(ts - ots) <= cfg.dedup_covisible_time:
                    self._co_visible.add(frozenset((g, og)))

    def _consolidate_duplicates(self, now: float) -> None:
        cfg = self.cfg
        if not cfg.dedup_enabled or now - self._last_dedup < cfg.dedup_interval:
            return
        self._last_dedup = now
        found: dict[tuple[str, str], str] = {}
        for cam, by_gid in self.gallery.camera_matrices().items():
            elig = [g for g, m in by_gid.items() if len(m) >= cfg.dedup_min_samples and self.identity.is_live_identity(g)]
            if len(elig) < 2:
                continue
            # Mean pairwise cosine == dot product of the mean unit vectors, so all
            # pairs come from one small matrix product (keeps the live loop fast).
            n = len(elig)
            means = np.zeros((n, by_gid[elig[0]].shape[1]), dtype=np.float64)
            self_sim = np.zeros(n)
            for i, g in enumerate(elig):
                m = by_gid[g].astype(np.float64)
                m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
                k = len(m)
                s = m.sum(axis=0)
                means[i] = s / k
                self_sim[i] = (float(s @ s) - k) / (k * k - k)
            cross = means @ means.T
            np.fill_diagonal(cross, -1.0)
            for i in range(n):
                j = int(np.argmax(cross[i]))
                if j <= i or int(np.argmax(cross[j])) != i:
                    continue  # each mutual-nearest pair once
                c = float(cross[i, j])
                if c < cfg.dedup_min_similarity or c < cfg.dedup_self_ratio * min(self_sim[i], self_sim[j]):
                    continue
                second_i = float(np.delete(cross[i], j).max(initial=-1.0))
                second_j = float(np.delete(cross[j], i).max(initial=-1.0))
                if c - second_i < cfg.dedup_margin or c - second_j < cfg.dedup_margin:
                    continue
                a, b = sorted((elig[i], elig[j]))
                if frozenset((a, b)) in self._co_visible:
                    continue
                if min(self._observed_seconds.get(a, 0.0), self._observed_seconds.get(b, 0.0)) < cfg.dedup_min_observed_seconds:
                    continue
                found[(a, b)] = (
                    f"{cam} cross={c:.2f} self={self_sim[i]:.2f}/{self_sim[j]:.2f} "
                    f"next={max(second_i, second_j):.2f}"
                )
        if cfg.face_enabled:
            found.update(self._face_duplicates())
        self._dup_streak = {p: self._dup_streak.get(p, 0) + 1 for p in found}
        merged: set[str] = set()
        for (a, b), evidence in found.items():
            if self._dup_streak[(a, b)] < cfg.dedup_confirmations or a in merged or b in merged:
                continue
            self._merge_consolidated(b, a, evidence)  # IDs are sequential: keep the older one
            merged.update({a, b})

    def _face_duplicates(self) -> dict[tuple[str, str], str]:
        """ID pairs carrying the same face (one person split into two IDs). Faces override co-visibility
        artifacts such as two tracker boxes on one person."""
        cfg = self.cfg
        mats = self.identity.face_gallery.matrices()
        elig = [g for g, m in mats.items() if len(m) >= cfg.face_min_samples and self.identity.is_live_identity(g)]
        if len(elig) < 2:
            return {}
        means = np.stack([mats[g].mean(axis=0) for g in elig])
        cross = means @ means.T
        np.fill_diagonal(cross, -1.0)
        out: dict[tuple[str, str], str] = {}
        for i in range(len(elig)):
            j = int(np.argmax(cross[i]))
            if j <= i or int(np.argmax(cross[j])) != i:
                continue
            c = float(cross[i, j])
            second = max(float(np.delete(cross[i], j).max(initial=-1.0)), float(np.delete(cross[j], i).max(initial=-1.0)))
            if c >= cfg.face_merge_threshold and c - second >= cfg.face_margin:
                a, b = sorted((elig[i], elig[j]))
                out[(a, b)] = f"face cross={c:.2f} next={second:.2f}"
        return out

    def _merge_consolidated(self, src: str, tgt: str, evidence: str) -> None:
        if not (self.identity.is_live_identity(src) and self.identity.is_live_identity(tgt)):
            return
        moved = self.identity.merge_identity_into(
            src, tgt, reason=f"auto_dedup never_co_visible {evidence}"
        )
        for rec in self._deferred_embeddings.pop(src, []):
            rec["global_id"] = tgt
            self._deferred_embeddings.setdefault(tgt, []).append(rec)
        self._created_at.pop(src, None)
        for st in self.tracks.values():
            if st.gid == src:
                st.gid, st.source = tgt, "dedup_merge"
            if st.candidate == src:
                st.candidate = tgt
        for e in self._ended:
            if e.gid == src:
                e.gid = tgt
        seen = [s for s in (self._gid_seen.pop(src, None), self._gid_seen.get(tgt)) if s is not None]
        if seen:
            self._gid_seen[tgt] = max(seen, key=lambda s: s[1])
        self._observed_seconds[tgt] = self._observed_seconds.get(tgt, 0.0) + self._observed_seconds.pop(src, 0.0)
        self._gid_obs.pop(src, None)
        self._co_visible = {
            frozenset(tgt if g == src else g for g in pair) for pair in self._co_visible
        }
        self._co_visible = {p for p in self._co_visible if len(p) == 2}
        self._dup_streak = {p: v for p, v in self._dup_streak.items() if src not in p}
        logger.info("Auto-merged duplicate identity %s into %s (%s; %d embeddings moved)", src, tgt, evidence, moved)

    def _feed_identity_correspondences(self, now: float) -> None:
        if self.linker is None:
            return
        by_gid: dict[str, list[LiveTrackState]] = {}
        for st in self.tracks.values():
            if st.status == "confirmed" and st.gid and st.source in {"gallery", "manual"} and now - st.last_seen <= 1.0:
                by_gid.setdefault(st.gid, []).append(st)
        for sts in by_gid.values():
            if len(sts) < 2:
                continue
            obs = [o for o in self._observations(now) if o.key in {s.key for s in sts}]
            for i in range(len(obs)):
                for j in range(len(obs)):
                    if i != j:
                        self.linker.add_identity_correspondence(obs[i], obs[j])

    def _update_gallery_and_bindings(self, now: float, wall_now: float) -> None:
        cfg = self.cfg
        unresolved = 0
        for st in self.tracks.values():
            visible = now - st.last_seen <= self._horizon(st.camera_id, cfg.observation_horizon) and not st.dormant()
            if st.status != "confirmed" or not st.gid:
                if visible:
                    unresolved += 1
                status = "provisional" if st.status == "provisional" else "calculating"
                self.identity.bind_live_track(
                    st.key, st.candidate if status == "provisional" else None, status, min(st.confidence, 0.999),
                    timestamp=st.last_seen, camera_id=st.camera_id,
                )
                continue
            self._gid_seen[st.gid] = (st.camera_id, st.last_seen)
            # Confidence 1.0 is reserved for manual operator corrections.
            conf = st.confidence if st.source == "manual" else min(st.confidence, 0.999)
            self.identity.bind_live_track(st.key, st.gid, "confirmed", conf, timestamp=st.last_seen, camera_id=st.camera_id)
            last_persist = self._gid_last_persist.get(st.gid, 0.0)
            persist = now - last_persist >= cfg.persist_interval
            self.identity.touch_live_identity(st.gid, st.camera_id, wall_now, persist=persist)
            if persist:
                self._gid_last_persist[st.gid] = now
            self._maybe_update_gallery(st, now, wall_now)
        self.identity.set_unresolved_count(unresolved)
        self.identity.refresh_activity(wall_now)

    def _maybe_update_gallery(self, st: LiveTrackState, now: float, wall_now: float) -> None:
        cfg = self.cfg
        if now - st.last_gallery_update < cfg.gallery_update_interval or not st.bank:
            return
        vec, q, ts = max(list(st.bank)[-4:], key=lambda b: b[1])
        if ts < st.bound_at:
            return
        link = self.links.get(st.key)
        sim = self.gallery.max_similarity(st.gid, vec)
        if st.source == "link":
            # New viewpoint from a geometric link: appearance can legitimately differ.
            if link is None or not link.geometric or link.votes < cfg.merge_min_votes:
                return
        elif (
            sim >= 0 and sim < cfg.gallery_update_min_similarity
            and not (st.face_verified and sim >= cfg.contradiction_threshold)
            and not self._stable_new_pose(st, vec, sim, now)
        ):
            return
        if sim >= cfg.gallery_diversity_max_similarity:
            st.last_gallery_update = now
            return  # redundant sample
        st.last_gallery_update = now
        young = st.gid in self._created_at
        rec = self.identity.add_live_embedding(st.gid, vec, camera_id=st.camera_id, timestamp=wall_now, quality=q, persist=not young)
        if young:
            self._deferred_embeddings.setdefault(st.gid, []).append(rec)

    def _stable_new_pose(self, st: LiveTrackState, vec: np.ndarray, sim: float, now: float) -> bool:
        """Sample shows a new pose of a long, uncontested track (not a tracker ID switch)."""
        cfg = self.cfg
        if st.source in {"handoff", "link", "gallery_switch"} or st.contradictions > 0:
            return False
        if now - st.bound_at < cfg.stable_learn_seconds or sim < cfg.stable_learn_min_similarity:
            return False
        bank = [b[0] for b in st.bank]
        idx = next((i for i, b in enumerate(bank) if b is vec), None)
        others = bank[:idx] + bank[idx + 1:] if idx is not None else bank
        if len(others) < 3:
            return False
        p = np.mean(np.stack(others), axis=0)
        p /= max(float(np.linalg.norm(p)), 1e-12)
        v = vec / max(float(np.linalg.norm(vec)), 1e-12)
        return float(p @ v) >= cfg.stable_learn_self_consistency

    def _flush_deferred(self, now: float) -> None:
        for gid, created in list(self._created_at.items()):
            if now - created > self.cfg.merge_window:
                self._created_at.pop(gid, None)
                self._persist_deferred(gid)

    def _persist_deferred(self, gid: str) -> None:
        for rec in self._deferred_embeddings.pop(gid, []):
            self.identity.persist_embedding_record(rec)

    def _finalize(self, key: str, now: float, wall_now: float, removed_by_tracker: bool = False) -> None:
        st = self.tracks.pop(key, None)
        if st is None:
            return
        if self.linker is not None:
            self.linker.forget(key)
        self.identity.release_track(key)
        if st.gid and st.status == "confirmed":
            walked = removed_by_tracker and self._is_moving(st)
            self._ended.append(_Ended(st.gid, st.camera_id, st.bbox, st.last_seen, st.proto(), walked_out=walked))
            if walked:
                logger.info("Track %s (%s) walked out of %s", key, st.gid, st.camera_id)
        if st.hits >= self.cfg.tracklet_min_hits:
            tracklet_id = f"tl_{st.camera_id}_{st.session_id}_{st.local_id}"
            duration = max(0.0, st.last_seen - st.first_seen)
            gid = st.gid if st.status == "confirmed" else None
            self.identity.record_tracklet(
                {
                    "tracklet_id": tracklet_id,
                    "camera_id": st.camera_id,
                    "tracker_session_id": st.session_id,
                    "local_track_id": st.local_id,
                    "start_time": st.first_seen_wall,
                    "end_time": st.first_seen_wall + duration,
                    "global_id": gid,
                    "assignment_confidence": st.confidence,
                },
                {
                    "association_id": f"as_{tracklet_id}",
                    "source_tracklet_id": tracklet_id,
                    "destination_global_id": gid,
                    "evidence": {"source": st.source, "hits": st.hits, "samples": len(st.bank)},
                    "decision": "CONFIRMED" if gid else "UNKNOWN",
                    "confidence": st.confidence,
                    "model_version": self.gallery.model_version,
                    "review_status": "auto" if gid else "pending",
                },
            )

    # --------------------------------------------------------------- stats
    def stats(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for st in self.tracks.values():
            statuses[st.status] = statuses.get(st.status, 0) + 1
        return {
            "live_tracks": len(self.tracks),
            "by_status": statuses,
            "cross_view_links": len(self.links) // 2,
            "latency_ms": round(self.last_latency_ms, 2),
        }
