"""Same-day person appearance gallery (stable ``P01`` IDs across track switches).

Rules:
* First distinct person → ``P01``, second → ``P02``, etc. Never give two
  *observed* people the same ``P##``.
* Lost/predicted ghost tracks do **not** keep a P## locked — otherwise a brief
  tracker miss creates a new detection that cannot rematch and gets P02+Tnew.
* Descriptions (or embeddings) of people who left stay in the gallery for the
  day. A returnee is compared to free gallery entries; match reuses that ID
  (and its stable display T), otherwise a new ``P##`` is enrolled.
* Gallery is wiped on calendar-day rollover (clothes change).

Backends:
* ``vlm`` (default) — Moondream2 clothing/appearance captions + VLM match.
* ``reid`` — YOLO26x crop embeddings (legacy / tests).

This is **not** verified employee identity. Permanent names still come only from
``IdentityManager`` verified sources.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class _Embedder(Protocol):
    def embed(self, crop: np.ndarray) -> np.ndarray: ...


class _Describer(Protocol):
    def describe(self, bgr_crop: np.ndarray) -> str: ...

    def match_id(self, bgr_crop: np.ndarray, candidates: dict[str, str]) -> str | None: ...


@dataclass(frozen=True)
class AppearanceGalleryConfig:
    enabled: bool = True
    # ``reid`` = YOLO26x person-ReID only.
    # ``hybrid`` = ReID embeddings + structured VLM features (both must match to reuse a P##).
    # ``vlm`` = legacy Moondream clothing captions.
    backend: str = "hybrid"
    # reid/hybrid: ONNX ReID path. vlm: HuggingFace model id.
    model: Path | str = Path("models/yolo26x-reid.onnx")
    # HuggingFace id for structured VLM (hybrid). Qwen2.5-VL preferred; Moondream still works.
    vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    # Empty = HF default branch. Moondream needs e.g. "2025-06-21".
    vlm_revision: str = ""
    image_size: int = 448
    device: str = "auto"
    # Hybrid: collect this many ReID + VLM samples before deciding (majority / consensus).
    consensus_samples: int = 5
    # VLM feature majority: min votes for a concrete value; else that feature = unknown.
    vlm_majority_min: int = 2
    # Need this many non-unknown features agreeing to count as a VLM match.
    vlm_min_comparable: int = 2
    # Parallel VLM workers (background; keeps the tracking/stream loop responsive).
    vlm_workers: int = 4
    # When set (path string), every Moondream extract saves crop + raw reply for audit.
    vlm_debug_directory: str | None = None
    # When True, gallery VLM features can block embedding rematch (identity veto).
    # Keep False until labeled CCTV evaluation shows acceptable Moondream reliability.
    vlm_identity_veto: bool = False
    # Expand person bbox when forming the VLM crop (from the clean source frame).
    vlm_crop_pad_x: float = 0.15
    vlm_crop_pad_top: float = 0.08
    # After this many duplicate-hash skips with ≥1 unique sample, commit features
    # (static employees never produce 5 distinct crops).
    vlm_dup_skip_limit: int = 3
    # Gallery features are vote counts per person: a value is committed only with this
    # many agreeing observations (capped at consensus_samples) and this share of all
    # concrete votes. Later majorities correct early wrong values.
    vlm_gallery_min_votes: int = 2
    vlm_gallery_min_share: float = 0.6
    # Colour keys: need more agreeing votes / share (accuracy over speed).
    vlm_gallery_color_min_share: float = 0.55
    # 2 is enough for short doorway tracks; warm-vs-cool guard blocks red-tee pollution.
    vlm_gallery_color_min_votes: int = 2
    # Keep sampling trusted bound tracks until a person has this many VLM samples.
    vlm_gallery_max_votes: int = 15
    # Veto may decide after this many unique track samples (capped at consensus_samples).
    vlm_veto_min_samples: int = 3
    # One conflicting key is tolerated when at least this many keys agree.
    vlm_veto_agree_override: int = 3
    # Hybrid: new P## waits for this many unique VLM samples (capped at consensus_samples)…
    vlm_enroll_min_samples: int = 2
    # …or this long after calculation started (VLM stuck / no usable crops).
    vlm_max_wait_seconds: float = 20.0
    # VLM-check positional (continuity / sticky) binds; release when ≥ this many keys
    # conflict with the person's stable gallery features.
    vlm_positional_check: bool = True
    vlm_release_min_conflicts: int = 2
    # Skip a VLM sample when another person's box covers more than this fraction of
    # the (padded) VLM crop — their clothes would be described instead.
    vlm_max_overlap: float = 0.08
    # Skip VLM when another box covers this fraction of the torso band (stricter).
    vlm_torso_max_overlap: float = 0.05
    # Brief miss buffer: if a person/track disappears for up to this many frames and a
    # new box appears near the same spot, reuse the same P## and display T## immediately
    # (no "calculating" restart). ~7 frames @ 20 fps ≈ 0.35 s.
    sticky_hold_frames: int = 7
    sticky_radius: float = 0.6  # box-heights
    # Cosine thresholds on the YOLO26x-reid scale (same person ≈ 0.45–0.65,
    # different person ≈ 0.15–0.25; calibrated on hand-labelled site footage).
    # Look-alike outfits reach 0.43–0.58, so a match also needs a clear margin.
    match_threshold: float = 0.50
    sole_match_threshold: float = 0.50
    match_margin: float = 0.08
    enroll_samples: int = 5
    ema_alpha: float = 0.12
    min_crop_side: int = 40
    # Bound tracks: how often to refresh the person's sample bank.
    update_interval_seconds: float = 1.0
    directory: Path = Path("data/gallery")
    # ---- PersonIdentityResolver (backend=reid) ----
    flip_tta: bool = False
    persist_day_gallery: bool = True
    pending_sample_interval_seconds: float = 0.4
    # Plain appearance match needs this many quality samples over this span. Scores
    # from 2–3 samples overlap between look-alikes; whole-track evidence does not.
    min_match_samples: int = 4
    min_match_span_seconds: float = 1.5
    # A match this strong is accepted with fewer samples.
    strong_match_threshold: float = 0.55
    strong_match_samples: int = 3
    # Unresolved samples kept per track (whole-track evidence; also the initial bank).
    pending_keep: int = 24
    # A stale, weakly-known P## (seen under absorb_weak_seconds, not seen since) whose
    # samples match a richer live P## this well is the same person enrolled twice from
    # a poor first look: it is absorbed so it cannot capture that person's next return.
    absorb_threshold: float = 0.52
    absorb_weak_seconds: float = 20.0
    absorb_interval_seconds: float = 5.0
    # A returnee seated at their own learned desk is accepted from this score.
    desk_match_threshold: float = 0.45
    # Enroll a new P## only when every known person scores below this.
    new_person_threshold: float = 0.30
    min_enroll_span_seconds: float = 2.0
    # Unresolved this long → decide anyway (weaker match or enroll).
    max_pending_seconds: float = 8.0
    timeout_match_threshold: float = 0.50
    # Below this at timeout → clearly nobody known → enroll. Between this and the
    # timeout match score the track stays P? until ambiguous_enroll_seconds, then a
    # NEW person is enrolled: a split is recoverable, a merge is not.
    timeout_new_threshold: float = 0.42
    ambiguous_enroll_seconds: float = 16.0
    # A re-identified track feeds the person's sample bank only after this long and
    # only with samples this close to the anchors (anti-pollution). The enrolling track
    # itself is trusted and grows the anchor set up to max_anchors.
    probation_seconds: float = 5.0
    probation_anchor_threshold: float = 0.45
    probation_bank_threshold: float = 0.50
    max_anchors: int = 8
    # Desk prior: a person's home desk is the one holding this share of their seated time.
    desk_home_share: float = 0.6
    desk_home_min_seconds: float = 20.0
    desk_bonus: float = 0.10
    # Boxes touching the left/right frame edge are partial bodies: not sampled, and a
    # person lost there has left the scene (no continuity hand-over at the door).
    edge_margin_px: int = 4
    # Re-appearance near where a person was just lost inherits that person.
    continuity_seconds: float = 3.0
    continuity_radius: float = 0.8
    continuity_bonus: float = 0.15
    continuity_min_score: float = 0.42
    # A P## stays locked to a track seen this recently (stops ping-pong on 1-frame misses).
    hold_seconds: float = 1.5
    # A holder *seated at a desk* keeps its P## through longer detection gaps (a seated
    # person does not move; a 1–2 s detector miss must not free their ID for a neighbour).
    seated_hold_seconds: float = 5.0
    # Two tracks observed simultaneously as separate, non-overlapping boxes for this long
    # are two different people: such a track can never inherit the other's P## by
    # continuity / sticky / appearance while that holder is remembered. This is what stops
    # a same-outfit neighbour (ReID 0.5–0.65 look-alike) from grabbing a seated person's
    # P## during a brief detector miss. Overlapping duplicate boxes do not count.
    coexist_seconds: float = 2.0
    # Box heights per second a person can move (rules out impossible jumps).
    max_speed_heights_per_second: float = 1.5
    bank_size: int = 32
    bank_diversity: float = 0.92
    # New samples join a person's bank only if they agree with the bank AND the
    # permanent enrollment anchors (prevents a tracker ID switch polluting it).
    bank_add_threshold: float = 0.45
    anchor_add_threshold: float = 0.40
    # Consecutive samples below drift_threshold → tracker swapped people; re-resolve.
    drift_threshold: float = 0.28
    drift_samples: int = 3
    # A sample joins a person's bank only if it beats every other person by this much;
    # a bound track whose samples another person beats by rival_switch_margin (and
    # scores at least strong_match_threshold) is a tracker switch.
    rival_margin: float = 0.08
    rival_switch_margin: float = 0.10
    # Crops overlapping another person this much are not used (mixed appearance).
    occlusion_iou: float = 0.25
    # Unresolved box overlapping a bound person's box this much is a duplicate box.
    duplicate_iou: float = 0.5
    # Sticky (position-only) binds are provisional: the first usable ReID sample must
    # score at least this against the person, else the P## is taken back immediately.
    # Keep near match_threshold so nearby look-alikes do not briefly steal a P##.
    sticky_verify_threshold: float = 0.45
    # Detector boxes below this confidence are still tracked but never sampled for
    # identity (no enroll / match / bank from weak, partial or junk boxes).
    min_identity_confidence: float = 0.0
    # Cool-down after a track first appears before *enrolling a new* P##.
    # During cool-down, ReID (+ VLM) still runs against the existing gallery and can
    # attach a matched P## immediately. 0 = enroll immediately.
    # Production default in settings.yaml is ~60 s (1 min).
    min_stable_seconds: float = 0.0
    # If P## is still unresolved this long after calculation started, drop the track
    # (box + tracker suppressed). A real person will be re-detected; empty ghosts die.
    # 0 = never drop. Production default in settings.yaml is ~300 s (5 min).
    max_calculating_seconds: float = 0.0


@dataclass
class GalleryPerson:
    person_id: str
    embedding: np.ndarray | None = None
    description: str = ""
    samples: int = 1
    last_seen_seconds: float = 0.0
    last_track_id: str | None = None
    # First dense T## shown for this person — reused on rematch so overlay T stays stable.
    anchor_local_id: int | None = None
    created_utc: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "description": self.description,
            "samples": self.samples,
            "last_seen_seconds": self.last_seen_seconds,
            "last_track_id": self.last_track_id,
            "anchor_local_id": self.anchor_local_id,
            "created_utc": self.created_utc,
        }


@dataclass
class AppearanceAssignment:
    person_id: str | None
    score: float
    reason: str
    enrolled: bool = False
    display_track_id: int | None = None


@dataclass
class AppearanceGallery:
    """In-memory (+ optional day file) gallery keyed by anonymous ``P##`` IDs."""

    config: AppearanceGalleryConfig = field(default_factory=AppearanceGalleryConfig)
    encoder: _Embedder | None = None
    describer: _Describer | None = None
    _day: date | None = field(default=None, init=False)
    _people: dict[str, GalleryPerson] = field(default_factory=dict, init=False)
    # Live track → person (released when the track is removed so the P## can return later).
    _track_person: dict[str, str] = field(default_factory=dict, init=False)
    _track_pending: dict[str, list[Any]] = field(default_factory=dict, init=False)
    _track_last_update: dict[str, float] = field(default_factory=dict, init=False)
    # Composite IDs with a fresh detection this frame. None = tests / unknown (all locks count).
    _observed_tracks: set[str] | None = field(default=None, init=False)
    _next_index: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if not self.config.enabled:
            return
        backend = (self.config.backend or "vlm").lower().strip()
        if backend == "vlm":
            if self.describer is None:
                from core.vlm.vlm_describer import MoondreamDescriber, VlmDescriberConfig

                model_id = str(self.config.model)
                self.describer = MoondreamDescriber(VlmDescriberConfig(
                    model_id=model_id,
                    revision=self.config.vlm_revision,
                    device=self.config.device,
                ))
        elif backend == "reid":
            if self.encoder is None:
                from core.tracking.reid_encoder import ReidEncoderConfig, Yolo26ReidEncoder

                self.encoder = Yolo26ReidEncoder(ReidEncoderConfig(
                    model=Path(self.config.model),
                    image_size=self.config.image_size,
                    device=self.config.device,
                ))
        elif backend == "hybrid":
            raise ValueError(
                "appearance_gallery.backend='hybrid' uses PersonIdentityResolver, not AppearanceGallery")
        else:
            raise ValueError(
                f"appearance_gallery.backend must be 'vlm', 'reid', or 'hybrid', got {backend!r}")
        self._ensure_day(date.today())

    @property
    def backend(self) -> str:
        return (self.config.backend or "vlm").lower().strip()

    def _ensure_day(self, today: date) -> None:
        if self._day == today:
            return
        if self._day is not None:
            logger.info("Appearance gallery day rollover %s -> %s; discarding %d people",
                        self._day, today, len(self._people))
        self._day = today
        self._people.clear()
        self._track_person.clear()
        self._track_pending.clear()
        self._track_last_update.clear()
        self._observed_tracks = None
        self._next_index = 1
        self._load_day_file()

    def _day_path(self) -> Path:
        assert self._day is not None
        return Path(self.config.directory) / f"gallery_{self._day.isoformat()}.json"

    def _load_day_file(self) -> None:
        path = self._day_path()
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load appearance gallery %s: %s", path, exc)
            return
        for entry in data.get("people", []):
            pid = str(entry["person_id"])
            emb = None
            if entry.get("embedding") is not None:
                emb = np.asarray(entry["embedding"], dtype=np.float32)
                norm = float(np.linalg.norm(emb))
                if norm > 1e-8:
                    emb = emb / norm
            anchor = entry.get("anchor_local_id")
            self._people[pid] = GalleryPerson(
                person_id=pid,
                embedding=emb,
                description=str(entry.get("description") or ""),
                samples=int(entry.get("samples", 1)),
                last_seen_seconds=float(entry.get("last_seen_seconds", 0.0)),
                last_track_id=entry.get("last_track_id"),
                anchor_local_id=int(anchor) if anchor is not None else None,
                created_utc=str(entry.get("created_utc") or ""),
            )
            try:
                n = int(pid.lstrip("P"))
                self._next_index = max(self._next_index, n + 1)
            except ValueError:
                pass
        logger.info("Loaded %d same-day appearance identities from %s", len(self._people), path)

    def save(self) -> None:
        if not self.config.enabled or self._day is None:
            return
        path = self._day_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        people_rows = []
        for p in self._people.values():
            row = {**p.to_public_dict()}
            if p.embedding is not None:
                row["embedding"] = p.embedding.astype(float).tolist()
            people_rows.append(row)
        payload = {
            "day": self._day.isoformat(),
            "backend": self.backend,
            "model": str(self.config.model),
            "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "people": people_rows,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _new_person_id(self) -> str:
        pid = f"P{self._next_index:02d}"
        self._next_index += 1
        return pid

    def set_observed_tracks(self, track_ids: set[str] | list[str]) -> None:
        """Mark which composite tracks have a fresh detection this frame.

        Lost/predicted ghosts stay in ``_track_person`` until removed, but must
        **not** block rematching — call this every tracking frame before ``observe``.
        """
        self._observed_tracks = set(track_ids)

    def _track_blocks_person(self, composite_track_id: str) -> bool:
        """True if this track currently occupies its P## against other tracks."""
        if self._observed_tracks is None:
            return True
        return composite_track_id in self._observed_tracks

    def _live_person_ids(self, except_track: str | None = None) -> set[str]:
        """Person IDs held by another *observed* track (cannot be double-assigned)."""
        live: set[str] = set()
        for tid, pid in self._track_person.items():
            if except_track is not None and tid == except_track:
                continue
            if not self._track_blocks_person(tid):
                continue
            live.add(pid)
        return live

    def _free_candidates(self, except_track: str) -> dict[str, str]:
        """Gallery descriptions available for rematch (includes people held only by ghosts)."""
        occupied = self._live_person_ids(except_track=except_track)
        out: dict[str, str] = {}
        for pid, person in self._people.items():
            if pid in occupied:
                continue
            desc = (person.description or "").strip()
            if desc:
                out[pid] = desc
            elif person.embedding is not None:
                # reid backend may have no text; still mark as free via empty desc skip —
                # reid path uses embeddings directly, not this map.
                pass
        return out

    def _claim(
        self,
        composite_track_id: str,
        person_id: str,
        local_track_id: int,
        timestamp_seconds: float,
    ) -> int:
        """Bind track→person, steal from ghost holders, keep stable display T."""
        for tid, pid in list(self._track_person.items()):
            if pid == person_id and tid != composite_track_id:
                self._track_person.pop(tid, None)
                self._track_last_update.pop(tid, None)
                self._track_pending.pop(tid, None)
                logger.info(
                    "Reclaimed %s from ghost/stale track %s → T%s",
                    person_id, tid, local_track_id,
                )
        self._track_person[composite_track_id] = person_id
        self._track_last_update[composite_track_id] = timestamp_seconds
        self._track_pending.pop(composite_track_id, None)
        person = self._people[person_id]
        if person.anchor_local_id is None:
            person.anchor_local_id = int(local_track_id)
        person.last_track_id = composite_track_id
        person.last_seen_seconds = timestamp_seconds
        return int(person.anchor_local_id)

    def display_track_id(self, person_id: str | None) -> int | None:
        if not person_id:
            return None
        person = self._people.get(person_id)
        return person.anchor_local_id if person else None

    def _rank_matches(
        self, emb: np.ndarray, *, except_track: str
    ) -> list[tuple[str, float]]:
        """All gallery people scored high→low (including people who left the office)."""
        scored = []
        for pid, person in self._people.items():
            if person.embedding is None:
                continue
            scored.append((pid, self.cosine(emb, person.embedding)))
        scored.sort(key=lambda x: -x[1])
        return scored

    def _ema(self, old: np.ndarray, new: np.ndarray) -> np.ndarray:
        alpha = float(self.config.ema_alpha)
        merged = (1.0 - alpha) * old + alpha * new
        norm = float(np.linalg.norm(merged))
        return merged / norm if norm > 1e-8 else old

    def person_of(self, composite_track_id: str) -> str | None:
        return self._track_person.get(composite_track_id)

    def _accept_match(
        self, best_id: str, best: float, second: float, occupied: set[str]
    ) -> bool:
        if best_id in occupied:
            return False
        need = float(self.config.match_threshold)
        if len(self._people) <= 1 or second < 0.0:
            need = max(need, float(self.config.sole_match_threshold))
        if best < need:
            return False
        if second >= 0.0 and (best - second) < float(self.config.match_margin):
            return False
        return True

    def observe(
        self,
        composite_track_id: str,
        local_track_id: int,
        bgr_crop: np.ndarray,
        timestamp_seconds: float,
        *,
        today: date | None = None,
    ) -> AppearanceAssignment:
        """Describe/embed a person crop and assign/reuse a same-day ``P##`` ID."""
        if not self.config.enabled:
            return AppearanceAssignment(None, 0.0, "gallery_disabled")
        self._ensure_day(today or date.today())

        h, w = bgr_crop.shape[:2]
        if min(h, w) < self.config.min_crop_side:
            return AppearanceAssignment(
                self._track_person.get(composite_track_id), 0.0, "crop_too_small"
            )

        existing = self._track_person.get(composite_track_id)
        last = self._track_last_update.get(composite_track_id)
        if existing is not None and last is not None and \
                timestamp_seconds - last < self.config.update_interval_seconds:
            return AppearanceAssignment(existing, 1.0, "cached_track_assignment")

        if self.backend == "vlm":
            return self._observe_vlm(
                composite_track_id, local_track_id, bgr_crop, timestamp_seconds, existing
            )
        return self._observe_reid(
            composite_track_id, local_track_id, bgr_crop, timestamp_seconds, existing
        )

    def _observe_vlm(
        self,
        composite_track_id: str,
        local_track_id: int,
        bgr_crop: np.ndarray,
        timestamp_seconds: float,
        existing: str | None,
    ) -> AppearanceAssignment:
        from core.vlm.vlm_describer import VlmDescriberError

        if self.describer is None:
            return AppearanceAssignment(None, 0.0, "vlm_unavailable")

        # Same live track already has a P## — keep it; lightly refresh description.
        if existing is not None and existing in self._people:
            person = self._people[existing]
            try:
                if person.samples < 4:
                    desc = self.describer.describe(bgr_crop)
                    if desc and (not person.description or len(desc) >= len(person.description) * 0.6):
                        person.description = desc
                        person.samples += 1
            except VlmDescriberError as exc:
                logger.debug("VLM refresh skipped for %s: %s", existing, exc)
            display = self._claim(
                composite_track_id, existing, local_track_id, timestamp_seconds,
            )
            return AppearanceAssignment(
                existing, 1.0, "updated_existing", display_track_id=display,
            )

        try:
            desc = self.describer.describe(bgr_crop)
        except VlmDescriberError as exc:
            return AppearanceAssignment(
                self._track_person.get(composite_track_id), 0.0, f"vlm_describe_failed:{exc}"
            )

        free = self._free_candidates(composite_track_id)
        logger.debug(
            "VLM gallery compare T%s free=%s gallery=%d desc=%r",
            local_track_id, list(free.keys()), len(self._people), desc[:80],
        )
        if free:
            try:
                matched = self.describer.match_id(
                    bgr_crop, free, query_description=desc,
                )
            except VlmDescriberError as exc:
                return AppearanceAssignment(
                    self._track_person.get(composite_track_id), 0.0, f"vlm_match_failed:{exc}"
                )
            if matched is not None and matched in free and matched in self._people:
                person = self._people[matched]
                if desc:
                    person.description = desc
                person.samples += 1
                display = self._claim(
                    composite_track_id, matched, local_track_id, timestamp_seconds,
                )
                logger.info(
                    "VLM matched T%s → %s (display T%s, free=%d gallery=%d) desc=%r",
                    local_track_id, matched, display, len(free), len(self._people), desc[:80],
                )
                try:
                    self.save()
                except OSError:
                    pass
                return AppearanceAssignment(
                    matched, 1.0, "matched_gallery", enrolled=False, display_track_id=display,
                )

        # No match → accumulate descriptions then enroll a NEW P##.
        pending = self._track_pending.setdefault(composite_track_id, [])
        pending.append(desc)
        if len(pending) < self.config.enroll_samples:
            return AppearanceAssignment(
                None, 0.0,
                f"enrolling {len(pending)}/{self.config.enroll_samples}"
                + (f" free={list(free.keys())}" if free else " free=[]"),
            )

        # Prefer the longest / most recent description as canonical.
        best_desc = max(pending, key=len)
        pid = self._new_person_id()
        self._people[pid] = GalleryPerson(
            person_id=pid,
            description=best_desc,
            samples=len(pending),
            last_seen_seconds=timestamp_seconds,
            last_track_id=composite_track_id,
            anchor_local_id=int(local_track_id),
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        display = self._claim(composite_track_id, pid, local_track_id, timestamp_seconds)
        logger.info(
            "VLM enrolled %s from T%s (display T%s) — %r (gallery now %d)",
            pid, local_track_id, display, best_desc[:120], len(self._people),
        )
        try:
            self.save()
        except OSError:
            pass
        return AppearanceAssignment(
            pid, 1.0, "enrolled_new", enrolled=True, display_track_id=display,
        )

    def _observe_reid(
        self,
        composite_track_id: str,
        local_track_id: int,
        bgr_crop: np.ndarray,
        timestamp_seconds: float,
        existing: str | None,
    ) -> AppearanceAssignment:
        from core.tracking.reid_encoder import ReidEncoderError

        if self.encoder is None:
            return AppearanceAssignment(None, 0.0, "reid_unavailable")

        try:
            emb = self.encoder.embed(bgr_crop)
        except ReidEncoderError as exc:
            return AppearanceAssignment(
                self._track_person.get(composite_track_id), 0.0, f"embed_failed:{exc}"
            )

        if existing is not None and existing in self._people:
            person = self._people[existing]
            sim = 0.0
            if person.embedding is not None:
                sim = self.cosine(emb, person.embedding)
                if sim >= self.config.match_threshold * 0.95:
                    person.embedding = self._ema(person.embedding, emb)
                    person.samples += 1
            display = self._claim(
                composite_track_id, existing, local_track_id, timestamp_seconds,
            )
            return AppearanceAssignment(
                existing, sim, "updated_existing", display_track_id=display,
            )

        occupied = self._live_person_ids(except_track=composite_track_id)
        ranked = self._rank_matches(emb, except_track=composite_track_id)
        best_id, best, second = None, -1.0, -1.0
        free = [(pid, s) for pid, s in ranked if pid not in occupied]
        if len(free) >= 2:
            best_id, best = free[0]
            second = free[1][1]
        elif len(free) == 1:
            best_id, best = free[0]
            second = -1.0

        if best_id is not None and self._accept_match(best_id, best, second, occupied):
            person = self._people[best_id]
            if person.embedding is not None:
                person.embedding = self._ema(person.embedding, emb)
            else:
                person.embedding = emb
            person.samples += 1
            display = self._claim(
                composite_track_id, best_id, local_track_id, timestamp_seconds,
            )
            logger.info(
                "Matched T%s → %s (display T%s, score=%.3f, gallery=%d)",
                local_track_id, best_id, display, best, len(self._people),
            )
            return AppearanceAssignment(
                best_id, best, "matched_gallery", enrolled=False, display_track_id=display,
            )

        pending = self._track_pending.setdefault(composite_track_id, [])
        pending.append(emb)
        if len(pending) < self.config.enroll_samples:
            return AppearanceAssignment(
                None,
                best if best > 0 else 0.0,
                f"enrolling {len(pending)}/{self.config.enroll_samples}"
                + (f" best={best_id}:{best:.2f}" if best_id else ""),
            )

        mean = np.mean(np.stack(pending, axis=0), axis=0)
        norm = float(np.linalg.norm(mean))
        mean = mean / norm if norm > 1e-8 else pending[-1]
        pid = self._new_person_id()
        self._people[pid] = GalleryPerson(
            person_id=pid,
            embedding=mean,
            samples=len(pending),
            last_seen_seconds=timestamp_seconds,
            last_track_id=composite_track_id,
            anchor_local_id=int(local_track_id),
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        display = self._claim(composite_track_id, pid, local_track_id, timestamp_seconds)
        logger.info(
            "Enrolled new identity %s from T%s (display T%s, gallery now %d)",
            pid, local_track_id, display, len(self._people),
        )
        return AppearanceAssignment(
            pid, 1.0, "enrolled_new", enrolled=True, display_track_id=display,
        )

    def forget_track(self, composite_track_id: str) -> None:
        """Track left: release live P## lock but **keep** the day gallery entry."""
        self._track_pending.pop(composite_track_id, None)
        self._track_last_update.pop(composite_track_id, None)
        released = self._track_person.pop(composite_track_id, None)
        if released:
            logger.debug("Released live lock on %s (day gallery entry kept)", released)

    def people(self) -> list[GalleryPerson]:
        return list(self._people.values())
