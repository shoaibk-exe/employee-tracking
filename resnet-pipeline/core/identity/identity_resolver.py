"""Same-day anonymous person identities (``P01``, ``P02``…) that survive tracker breaks.

The tracker's own IDs (T##) break whenever a person is occluded (monitors, other
people) or missed by the detector for a moment. This resolver sits on top of the
tracker and decides, per frame and globally, which same-day person each track is.

Decision rules (calibrated on this site's footage, YOLO26x-reid cosine scale where
same-person ≈ 0.45–0.65 and different-person ≈ 0.15–0.25):

* **Sticky binding** – a track that has a ``P##`` keeps it. It is released only
  after several consecutive quality samples clearly contradict it (a real tracker
  identity switch), never because of a one-frame miss.
* **Exclusive** – one ``P##`` is shown on at most one *observed* track. A ``P##``
  held only by a lost/predicted ghost track can be reclaimed by a new track.
* **Continuity** – a new track appearing within a few seconds next to where a
  person was just lost inherits that person immediately, unless appearance
  clearly contradicts it.
* **Physical feasibility** – a person seen moments ago cannot re-appear far away.
* **Co-visibility** – two tracks seen simultaneously as separate boxes are two people:
  such a track can never inherit the other's ``P##`` (continuity, sticky or appearance),
  and a lost holder re-detected at its own spot reclaims a ``P##`` handed over by
  position alone while it was missing.
* **No instant enrollment** – an unresolved track shows ``P=calculating`` while it
  collects quality crops. It matches when the best gallery score is high *and*
  clearly ahead of the runner-up; a new ``P##`` is enrolled only when nobody scores close.
* **Global assignment** – all unresolved tracks are matched in one Hungarian
  assignment so two tracks can never grab the same person.
* Per-person **sample bank** (multiple poses) instead of one averaged vector.

This is not verified employee identity; names still come only from
``IdentityManager``. The gallery is wiped on calendar-day rollover.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

from core.identity.appearance_gallery import AppearanceGalleryConfig

logger = logging.getLogger(__name__)


class _Embedder(Protocol):
    def embed_many(self, crops: Sequence[np.ndarray]) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class IdentityAssignment:
    person_id: str | None
    display_track_id: int | None
    state: str  # bound | pending | ghost | warming | drop
    score: float = 0.0
    reason: str = ""


@dataclass
class _Person:
    person_id: str
    bank: list[np.ndarray]
    anchor_count: int
    last_bbox: tuple[int, int, int, int]
    last_seen: float
    anchor_local_id: int | None
    created_utc: str
    bound_track: str | None = None
    samples: int = 0
    # Track this person was enrolled from: the only track trusted to grow the anchors.
    enroll_track: str | None = None
    # Seated seconds per workstation (desk prior for returnees).
    desk_seconds: dict[str, float] = field(default_factory=dict)
    # Total observed seconds while bound to a track (how well we know this person).
    seen_seconds: float = 0.0
    # Fixed VLM appearance features (gender, glasses, …); values or "unknown".
    features: dict[str, str] = field(default_factory=dict)
    # Per-key VLM vote counts behind ``features`` (from trusted tracks only).
    feature_votes: dict[str, Counter] = field(default_factory=dict)
    # Number of VLM samples that contributed to ``feature_votes``.
    feature_samples: int = 0
    # Last track this person was bound to (kept after unbinding): a claimant that was
    # co-visible with it as a separate box is a different physical person.
    last_holder_cid: str | None = None

    def matrix(self) -> np.ndarray:
        return np.stack(self.bank, axis=0)

    def home_desk(self, min_share: float, min_seconds: float) -> str | None:
        total = sum(self.desk_seconds.values())
        if total < min_seconds:
            return None
        desk, secs = max(self.desk_seconds.items(), key=lambda kv: kv[1])
        return desk if secs / total >= min_share else None


@dataclass
class _Track:
    cid: str
    local_id: int
    first_seen: float
    last_bbox: tuple[int, int, int, int]
    last_observed: float
    person: str | None = None
    pending: list[np.ndarray] = field(default_factory=list)
    pending_since: float | None = None
    pending_first_sample: float | None = None
    last_sample: float = -1e9
    contradictions: int = 0
    reason: str = "new"
    score: float = 0.0
    # Released from a P## after contradicting samples (switch/occlusion): such crops are
    # unreliable, so this track may only enroll a new person when clearly unlike everyone.
    released: bool = False
    bound_since: float = -1e9
    desk: str | None = None  # workstation this track is currently seated at (hint)
    last_desk_update: float = -1e9
    # Structured VLM feature samples collected while pending (hybrid backend).
    pending_features: list[dict[str, str]] = field(default_factory=list)
    # Crop SHA1s already accepted into pending_features (reject duplicate consensus samples).
    pending_feature_hashes: list[str] = field(default_factory=list)
    # Provenance parallel to pending_features: frame_index, timestamp, sha1.
    pending_feature_meta: list[dict[str, Any]] = field(default_factory=list)
    last_vlm_sample: float = -1e9
    vlm_future: Any = None  # Future | None — in-flight background VLM call
    # SHA1 of the crop currently in flight (reject if result hash mismatches).
    vlm_inflight_sha1: str | None = None
    # Consecutive duplicate-hash skips after ≥1 unique sample (static-person commit).
    vlm_dup_skips: int = 0
    # Bound tracks: how many pending_features were already pushed into person votes.
    vlm_votes_pushed: int = 0
    # Positional (continuity / sticky) bind awaiting a VLM clothing check.
    vlm_check_pending: bool = False
    # P## this track was released from on VLM conflict (never re-taken by this track).
    vlm_rejected: set[str] = field(default_factory=set)
    last_observed_frame: int = 0
    # Provisional sticky bind awaiting its first ReID check, and what to restore on reject.
    sticky_verify: bool = False
    sticky_from: str | None = None
    sticky_prev: tuple[tuple[int, int, int, int], float] | None = None
    # When P## calculation actually started (after min_stable_seconds warm-up).
    calc_started: float | None = None
    discarded: bool = False
    # Seconds this track was observed simultaneously with another track as separate,
    # non-overlapping boxes (cid → seconds). Two such tracks are two different people.
    covis: dict[str, float] = field(default_factory=dict)
    # P## taken from this (then lost) track by a positional bind on another track, and
    # when. If this track is re-detected at its own spot right after, it reclaims the P##.
    superseded_pid: str | None = None
    superseded_at: float = -1e9


@dataclass
class _LostPending:
    """An unresolved track that just disappeared; a nearby new track inherits its evidence."""
    local_id: int
    last_bbox: tuple[int, int, int, int]
    last_observed: float
    pending: list[np.ndarray]
    pending_since: float
    pending_first_sample: float | None
    pending_features: list[dict[str, str]] = field(default_factory=list)
    pending_feature_hashes: list[str] = field(default_factory=list)
    pending_feature_meta: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _StickySighting:
    """Recent bound person sighting — brief miss buffer (sticky_hold_frames)."""
    person_id: str
    local_id: int
    bbox: tuple[int, int, int, int]
    frame_index: int
    timestamp: float


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center(b: Sequence[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _height(b: Sequence[float]) -> float:
    return max(1.0, float(b[3] - b[1]))


class PersonIdentityResolver:
    """Frame-level same-day ``P##`` assignment on top of tracker IDs."""

    def __init__(
        self,
        config: AppearanceGalleryConfig,
        encoder: _Embedder | None = None,
        feature_extractor: Any | None = None,
    ) -> None:
        self.config = config
        if encoder is None and config.enabled:
            from core.tracking.reid_encoder import ReidEncoderConfig, Yolo26ReidEncoder

            encoder = Yolo26ReidEncoder(ReidEncoderConfig(
                model=Path(config.model),
                image_size=config.image_size,
                device=config.device,
                flip_tta=config.flip_tta,
            ))
        self.encoder = encoder
        self.feature_extractor = feature_extractor
        if (
            feature_extractor is None
            and config.enabled
            and (config.backend or "").lower().strip() == "hybrid"
        ):
            from core.vlm.vlm_features import VlmFeatureConfig, create_feature_extractor

            dbg = getattr(config, "vlm_debug_directory", None)
            self.feature_extractor = create_feature_extractor(VlmFeatureConfig(
                model_id=str(getattr(config, "vlm_model", None) or "Qwen/Qwen2.5-VL-3B-Instruct"),
                revision=str(getattr(config, "vlm_revision", "") or ""),
                device=config.device,
                majority_min=config.vlm_majority_min,
                min_comparable=config.vlm_min_comparable,
                debug_directory=Path(dbg) if dbg else None,
            ))
        self._day: date | None = None
        self._people: dict[str, _Person] = {}
        self._tracks: dict[str, _Track] = {}
        self._lost_pending: list[_LostPending] = []
        self._sticky: list[_StickySighting] = []
        self._frame_wh: tuple[int, int] = (0, 0)
        self._frame_index: int = 0
        self._last_update_time: float | None = None
        self._last_absorb = -1e9
        # Retired duplicate IDs → the P## that absorbed them (same person, poor first look).
        self.aliases: dict[str, str] = {}
        self._next_index = 1
        # Composite track ids the caller should force-remove from the person tracker.
        self._drop_cids: list[str] = []
        workers = max(1, int(getattr(config, "vlm_workers", 2) or 2))
        self._vlm_pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vlm-feat")
            if self.feature_extractor is not None else None
        )
        self._ensure_day(date.today())

    def consume_drops(self) -> list[str]:
        """Track ids that timed out on P=calculating — remove box + suppress tracker."""
        out = self._drop_cids
        self._drop_cids = []
        return out

    def _is_stable(self, tr: _Track, now: float) -> bool:
        need = float(self.config.min_stable_seconds or 0.0)
        if need <= 0:
            return True
        return (now - tr.first_seen) >= need

    @property
    def _hybrid(self) -> bool:
        return (self.config.backend or "").lower().strip() == "hybrid"

    def close(self) -> None:
        if self._vlm_pool is not None:
            self._vlm_pool.shutdown(wait=False, cancel_futures=True)
            self._vlm_pool = None

    # ------------------------------------------------------------------ day / persistence
    def _ensure_day(self, today: date) -> None:
        if self._day == today:
            return
        if self._day is not None:
            logger.info("Identity gallery day rollover %s -> %s; discarding %d people",
                        self._day, today, len(self._people))
        self._day = today
        self._people.clear()
        self._lost_pending.clear()
        self._sticky.clear()
        for tr in self._tracks.values():
            tr.person = None
            tr.pending.clear()
            tr.pending_since = None
            tr.pending_first_sample = None
        self._next_index = 1
        if self.config.persist_day_gallery:
            self._load_day_file()

    def _day_path(self) -> Path:
        assert self._day is not None
        return Path(self.config.directory) / f"identities_{self._day.isoformat()}.json"

    def _load_day_file(self) -> None:
        path = self._day_path()
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load identity gallery %s: %s", path, exc)
            return
        if data.get("model") != str(self.config.model) or data.get("format") != 2:
            logger.info("Ignoring %s (different model/format)", path)
            return
        for entry in data.get("people", []):
            bank = [np.asarray(v, dtype=np.float32) for v in entry.get("bank", [])]
            if not bank:
                continue
            pid = str(entry["person_id"])
            self._people[pid] = _Person(
                person_id=pid,
                bank=bank,
                anchor_count=int(entry.get("anchor_count", min(3, len(bank)))),
                last_bbox=tuple(entry.get("last_bbox", (0, 0, 1, 1))),  # type: ignore[arg-type]
                # Positions/times from an earlier process are not comparable to this stream.
                last_seen=-1e9,
                anchor_local_id=entry.get("anchor_local_id"),
                created_utc=str(entry.get("created_utc") or ""),
                samples=int(entry.get("samples", len(bank))),
                desk_seconds={str(k): float(v) for k, v in (entry.get("desk_seconds") or {}).items()},
                seen_seconds=float(entry.get("seen_seconds", 0.0)),
                features={str(k): str(v) for k, v in (entry.get("features") or {}).items()},
                feature_votes={
                    str(k): Counter({str(v): int(n) for v, n in (c or {}).items()})
                    for k, c in (entry.get("feature_votes") or {}).items()
                },
                feature_samples=int(entry.get("feature_samples", 0)),
            )
            try:
                self._next_index = max(self._next_index, int(pid.lstrip("P")) + 1)
            except ValueError:
                pass
        logger.info("Loaded %d same-day identities from %s", len(self._people), path)

    def save(self) -> None:
        if not self.config.enabled or self._day is None:
            return
        path = self._day_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": 2,
            "day": self._day.isoformat(),
            "backend": "reid",
            "model": str(self.config.model),
            "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "people": [
                {
                    "person_id": p.person_id,
                    "samples": p.samples,
                    "anchor_count": p.anchor_count,
                    "anchor_local_id": p.anchor_local_id,
                    "last_bbox": list(p.last_bbox),
                    "created_utc": p.created_utc,
                    "desk_seconds": {k: round(v, 1) for k, v in p.desk_seconds.items()},
                    "seen_seconds": round(p.seen_seconds, 1),
                    "features": dict(p.features),
                    "feature_votes": {k: dict(c) for k, c in p.feature_votes.items()},
                    "feature_samples": p.feature_samples,
                    "bank": [v.astype(float).round(5).tolist() for v in p.bank],
                }
                for p in self._people.values()
            ],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)

    # ------------------------------------------------------------------ public queries
    def person_of(self, composite_track_id: str) -> str | None:
        tr = self._tracks.get(composite_track_id)
        return tr.person if tr else None

    def display_track_id(self, person_id: str | None) -> int | None:
        if not person_id:
            return None
        person = self._people.get(person_id)
        return person.anchor_local_id if person else None

    def people(self) -> list[_Person]:
        return list(self._people.values())

    def forget_track(self, composite_track_id: str) -> None:
        tr = self._tracks.pop(composite_track_id, None)
        if tr is None:
            return
        if tr.vlm_future is not None and not tr.vlm_future.done():
            tr.vlm_future.cancel()
        # covis entries pointing at this track are kept while any person remembers it as
        # their last holder (sticky/continuity veto); pruned in _prune_covisibility.
        if tr.person and tr.person in self._people:
            person = self._people[tr.person]
            if person.bound_track == composite_track_id:
                person.bound_track = None
            # Brief-miss buffer: a new box near here within sticky_hold_frames reuses P/T.
            self._sticky.append(_StickySighting(
                tr.person, tr.local_id, tr.last_bbox, self._frame_index, tr.last_observed,
            ))
            hold = max(1, int(self.config.sticky_hold_frames))
            self._sticky = [s for s in self._sticky if self._frame_index - s.frame_index <= hold]
        elif tr.pending and tr.pending_since is not None and not tr.released:
            # Keep an unresolved track's evidence briefly so a re-detection of the same
            # walking person continues collecting instead of restarting from zero.
            self._lost_pending.append(_LostPending(
                tr.local_id, tr.last_bbox, tr.last_observed, list(tr.pending),
                tr.pending_since, tr.pending_first_sample,
                pending_features=list(tr.pending_features),
                pending_feature_hashes=list(tr.pending_feature_hashes),
                pending_feature_meta=list(tr.pending_feature_meta),
            ))

    def _coexisted(self, person: _Person, tr: _Track) -> bool:
        """Was ``tr`` seen alongside this person's (last) holder as a separate box?

        Two tracks observed simultaneously as non-overlapping boxes for
        ``coexist_seconds`` are two different people, so ``tr`` can never be the
        continuation of that holder — whatever ReID says (same-outfit neighbours score
        0.5–0.65). Duplicate/overlapping boxes do not accumulate co-visibility.
        """
        need = float(self.config.coexist_seconds or 0.0)
        if need <= 0:
            return False
        holder_cid = person.bound_track or person.last_holder_cid
        if not holder_cid or holder_cid == tr.cid:
            return False
        return tr.covis.get(holder_cid, 0.0) >= need

    def _update_covisibility(self, observed_tracks: list[_Track], dt: float) -> None:
        """Accumulate co-visible seconds for every pair of separate observed boxes."""
        if dt <= 0 or len(observed_tracks) < 2:
            return
        dup = float(self.config.duplicate_iou)
        for i, a in enumerate(observed_tracks):
            for b in observed_tracks[i + 1:]:
                if _iou(a.last_bbox, b.last_bbox) >= dup:
                    continue  # same person split into two boxes: not two people
                a.covis[b.cid] = a.covis.get(b.cid, 0.0) + dt
                b.covis[a.cid] = b.covis.get(a.cid, 0.0) + dt

    def _prune_covisibility(self) -> None:
        """Drop co-visibility entries for tracks nobody can refer to any more."""
        keep = set(self._tracks)
        for p in self._people.values():
            if p.last_holder_cid:
                keep.add(p.last_holder_cid)
            if p.bound_track:
                keep.add(p.bound_track)
        for tr in self._tracks.values():
            for cid in [c for c in tr.covis if c not in keep]:
                del tr.covis[cid]

    def _try_reclaim(self, tr: _Track, prev_bbox: tuple[int, int, int, int], now: float) -> bool:
        """A lost holder re-detected at its own spot takes its P## back from a
        *positional* (continuity / sticky) bind made while it was missing.

        The other track only had position evidence; the original track re-appearing
        exactly where it vanished within ``continuity_seconds`` is stronger evidence.
        The displaced track returns to calculating with its samples intact.
        """
        pid = tr.superseded_pid
        if not pid:
            return False
        if now - tr.superseded_at > self.config.continuity_seconds:
            tr.superseded_pid = None
            return False
        person = self._people.get(pid)
        if person is None or not person.bound_track or person.bound_track not in self._tracks:
            tr.superseded_pid = None
            return False
        holder = self._tracks[person.bound_track]
        positional = holder.reason.startswith(("continuity", "sticky"))
        if (holder.cid == tr.cid or holder.person != pid or not positional
                or holder.bound_since < tr.superseded_at - 1e-6
                or _iou(prev_bbox, tr.last_bbox) < float(self.config.duplicate_iou)):
            return False
        logger.info("T%s re-detected at its spot — reclaiming %s from T%s (%s)",
                    tr.local_id, pid, holder.local_id, holder.reason)
        holder.person = None
        holder.reason = f"returned_to_T{tr.local_id}"
        holder.pending_since = now
        holder.pending_first_sample = None
        holder.sticky_verify = False
        holder.sticky_from = None
        holder.sticky_prev = None
        person.bound_track = None
        tr.superseded_pid = None
        self._bind(tr, pid, now, f"reclaimed from T{holder.local_id}", tr.score)
        return True

    def _try_sticky_bind(self, tr: _Track, now: float) -> bool:
        """Reuse P/T from a recent nearby sighting (no calculating restart).

        Covers two cases within ``sticky_hold_frames``:
        1. Track fully removed → sticky buffer (forget_track).
        2. Tracker still has a LOST/predicted ghost → soft hand-off from that holder
           when a new confirmed box appears near the same spot.

        Guards against crowded false steals: require a uniquely nearest candidate,
        reject overlapping duplicate boxes, and require the holder to have been
        missing for a few frames (stops T↔T ping-pong).
        """
        hold = max(1, int(self.config.sticky_hold_frames))
        radius = float(self.config.sticky_radius)
        # Soft-sticky only after the holder has been unobserved for a few frames.
        min_miss_frames = max(2, min(hold // 2, 3))
        candidates: list[tuple[float, str, int, tuple[int, int, int, int], _Track | None]] = []

        def _consider(pid: str, local_id: int, bbox: tuple[int, int, int, int],
                      frame_idx: int, holder: _Track | None) -> None:
            miss = self._frame_index - frame_idx
            if miss > hold or miss < 0:
                return
            if pid not in self._people or pid in tr.vlm_rejected:
                return
            person = self._people[pid]
            # Seen alongside this person's holder as a separate box → someone else.
            if self._coexisted(person, tr):
                return
            if holder is not None:
                if holder.cid == tr.cid:
                    return
                # Soft sticky: still seen this frame, or only missing briefly → do not steal.
                # (Overlapping boxes are OK — tracker often opens a new T on the same person.)
                if holder.last_observed >= now or miss < min_miss_frames:
                    return
            elif person.bound_track and person.bound_track in self._tracks:
                live = self._tracks[person.bound_track]
                if live.person == pid and live.last_observed >= now:
                    return
                # Bound holder still exists: same miss gate as soft sticky.
                if miss < min_miss_frames:
                    return
            if not self._near(bbox, tr.last_bbox, radius):
                return
            h = max(_height(bbox), _height(tr.last_bbox))
            (sx, sy), (tx, ty) = _center(bbox), _center(tr.last_bbox)
            dist = float(np.hypot(sx - tx, sy - ty)) / h
            candidates.append((dist, pid, local_id, bbox, holder))

        for s in self._sticky:
            holder = None
            person = self._people.get(s.person_id)
            if person is not None and person.bound_track and person.bound_track in self._tracks:
                holder = self._tracks[person.bound_track]
            _consider(s.person_id, s.local_id, s.bbox, s.frame_index, holder)

        # Soft sticky: holder still in memory but missing for several frames.
        for person in self._people.values():
            if not person.bound_track or person.bound_track not in self._tracks:
                continue
            holder = self._tracks[person.bound_track]
            if holder.person != person.person_id or holder.cid == tr.cid:
                continue
            _consider(
                person.person_id, holder.local_id, holder.last_bbox,
                holder.last_observed_frame, holder,
            )

        if not candidates:
            return False
        candidates.sort(key=lambda c: c[0])
        best_dist, best_pid, best_local, _bb, _holder = candidates[0]
        # Ambiguous: another lost person almost as close → do not guess.
        if len(candidates) > 1 and candidates[1][0] - best_dist < 0.20:
            if candidates[1][1] != best_pid:
                return False
        # If we already have appearance samples, they must agree with the sticky person.
        # Use sticky_verify_threshold (not the weaker continuity floor) so a nearby
        # look-alike cannot grab P## from position alone.
        if tr.pending and best_pid in self._people:
            s = self._score(tr.pending, self._people[best_pid])
            if s >= 0.0 and s < self.config.sticky_verify_threshold:
                return False

        person = self._people[best_pid]
        prev = (person.last_bbox, person.last_seen)
        sticky_from: str | None = None
        # Release stale holder so exclusivity holds.
        if person.bound_track and person.bound_track in self._tracks:
            old = self._tracks[person.bound_track]
            if old.cid != tr.cid and old.person == best_pid:
                old.person = None
                old.reason = "sticky_handoff"
                sticky_from = old.cid
        if person.anchor_local_id is None:
            person.anchor_local_id = best_local
        self._bind(tr, best_pid, now, f"sticky T{best_local} ({best_dist:.2f})", 1.0)
        tr.sticky_verify = True
        tr.sticky_from = sticky_from
        tr.sticky_prev = prev
        self._sticky = [s for s in self._sticky if s.person_id != best_pid]
        return True

    def _reject_sticky(self, tr: _Track, emb: np.ndarray, score: float, now: float) -> None:
        """Provisional sticky bind failed its ReID check: give the P## back, recalculate.

        Do **not** re-queue a sticky sighting — that caused bind→reject→bind flicker
        loops when the wrong person stood near a lost P##. Rematch goes through
        emb (+ VLM) gallery comparison instead.
        """
        pid = tr.person
        person = self._people.get(pid) if pid else None
        logger.info("T%s sticky %s rejected by appearance (%.2f) — recalculating", tr.local_id, pid, score)
        if person is not None:
            if person.bound_track == tr.cid:
                person.bound_track = None
            if tr.sticky_prev is not None:
                person.last_bbox, person.last_seen = tr.sticky_prev
            old = self._tracks.get(tr.sticky_from) if tr.sticky_from else None
            if old is not None and old.person is None and old.reason == "sticky_handoff":
                old.person = pid
                old.reason = "sticky_restored"
                person.bound_track = old.cid
            # else: leave unbound — gallery rematch (emb+VLM) will reattach if correct
        tr.person = None
        tr.sticky_verify = False
        tr.sticky_from = None
        tr.sticky_prev = None
        tr.contradictions = 0
        tr.released = True  # block immediate re-sticky; rematch via _resolve_pending
        tr.pending = [emb]
        tr.pending_since = now
        tr.pending_first_sample = now
        tr.score = score
        tr.reason = f"sticky_rejected ({score:.2f})"

    def reset_tracks(self) -> None:
        """Tracker session reset (outage/reconnect): all tracks gone, people kept."""
        for cid in list(self._tracks):
            self.forget_track(cid)
        self._lost_pending.clear()
        self._sticky.clear()

    # ------------------------------------------------------------------ scoring helpers
    def _score(self, queries: Sequence[np.ndarray], person: _Person) -> float:
        """Mean over recent query samples of the mean top-k bank similarity."""
        if not queries or not person.bank:
            return -1.0
        q = np.stack(queries[-12:], axis=0)
        sims = q @ person.matrix().T
        k = min(3, sims.shape[1])
        topk = np.sort(sims, axis=1)[:, -k:].mean(axis=1)
        return float(topk.mean())

    def _gate_score(self, emb: np.ndarray, person: _Person) -> tuple[float, float]:
        """(score vs permanent enrollment anchors, score vs whole bank)."""
        anchors = _Person(person.person_id, person.bank[: max(1, person.anchor_count)], 0,
                          person.last_bbox, person.last_seen, None, "")
        return self._score([emb], anchors), self._score([emb], person)

    def _best_rival(self, emb: np.ndarray, person: _Person) -> tuple[str | None, float]:
        """Best-scoring *other* person for one sample."""
        best_pid, best = None, -1.0
        for other in self._people.values():
            if other.person_id == person.person_id:
                continue
            s = self._score([emb], other)
            if s > best:
                best_pid, best = other.person_id, s
        return best_pid, best

    def _maybe_add(self, person: _Person, emb: np.ndarray, *, trusted: bool = False) -> bool:
        """Add a sample only if it agrees with the anchors AND the bank (anti-pollution)
        and no other person explains it about as well (tracker-switch / look-alike guard).

        ``trusted`` = sample comes from the track the person was enrolled from (same
        tracker ID, no re-identification involved): normal gates, and it may become an
        additional anchor. Re-identified tracks use the stricter probation gates.
        """
        cfg = self.config
        anchor_s, bank_s = self._gate_score(emb, person)
        anchor_th = cfg.anchor_add_threshold if trusted else cfg.probation_anchor_threshold
        bank_th = cfg.bank_add_threshold if trusted else cfg.probation_bank_threshold
        if anchor_s >= anchor_th and bank_s >= bank_th:
            _, rival = self._best_rival(emb, person)
            if bank_s - rival < cfg.rival_margin:
                return False
            self._add_to_bank(person, emb, promote=trusted)
            return True
        return False

    def _add_to_bank(self, person: _Person, emb: np.ndarray, *, promote: bool = False) -> None:
        person.samples += 1
        if person.bank:
            sims = person.matrix() @ emb
            if float(sims.max()) >= self.config.bank_diversity:
                return
        if promote and person.anchor_count < self.config.max_anchors:
            # Anchors are stored first; a trusted, diverse sample extends them so the
            # permanent reference covers walking AND seated views, not just the doorway.
            person.bank.insert(person.anchor_count, emb)
            person.anchor_count += 1
        else:
            person.bank.append(emb)
        cap = max(self.config.bank_size, person.anchor_count + 1)
        if len(person.bank) > cap:
            m = person.matrix()
            s = m @ m.T
            np.fill_diagonal(s, -1.0)
            redundancy = s.max(axis=1)
            redundancy[: person.anchor_count] = -2.0  # enrollment anchors are permanent
            person.bank.pop(int(np.argmax(redundancy)))

    def _at_edge(self, bbox: Sequence[float]) -> bool:
        fw = self._frame_wh[0]
        if fw <= 0:
            return False
        m = self.config.edge_margin_px
        return bbox[0] <= m or bbox[2] >= fw - m

    def _quality_ok(self, bbox: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> bool:
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if min(w, h) < self.config.min_crop_side or h < 2 * self.config.min_crop_side:
            return False
        if self._at_edge(bbox):
            return False
        return all(_iou(bbox, o) < self.config.occlusion_iou for o in others)

    def _feasible(self, person: _Person, bbox: tuple[int, int, int, int], now: float) -> tuple[bool, bool]:
        """(physically feasible, spatial-temporal continuation)."""
        gap = now - person.last_seen
        if gap < 0:
            gap = 0.0
        h = max(_height(bbox), _height(person.last_bbox))
        cx, cy = _center(bbox)
        px, py = _center(person.last_bbox)
        dist = float(np.hypot(cx - px, cy - py)) / h
        feasible = dist <= 1.0 + self.config.max_speed_heights_per_second * gap
        scale = _height(bbox) / _height(person.last_bbox)
        continuation = (
            gap <= self.config.continuity_seconds
            and dist <= self.config.continuity_radius
            and 0.6 <= scale <= 1.6
            # Lost at the frame edge = walked out; whoever appears there next is not
            # automatically the same person.
            and not self._at_edge(person.last_bbox)
        )
        return feasible, continuation

    def _near(self, a: Sequence[float], b: Sequence[float], radius: float) -> bool:
        h = max(_height(a), _height(b))
        (ax, ay), (bx, by) = _center(a), _center(b)
        return float(np.hypot(ax - bx, ay - by)) / h <= radius

    # ------------------------------------------------------------------ binding
    def _bind(self, tr: _Track, pid: str, now: float, reason: str, score: float) -> None:
        person = self._people[pid]
        if person.bound_track and person.bound_track != tr.cid:
            ghost = self._tracks.get(person.bound_track)
            if ghost is not None and ghost.person == pid:
                ghost.person = None
                ghost.pending.clear()
                ghost.pending_since = None
                ghost.pending_first_sample = None
                ghost.reason = f"superseded_by_T{tr.local_id}"
                # Positional hand-over only: if the ghost is re-detected at its own
                # spot right away, it may take the P## back (_try_reclaim).
                if reason.startswith(("continuity", "sticky")):
                    ghost.superseded_pid = pid
                    ghost.superseded_at = now
            logger.info("%s moved from lost track %s to T%s (%s)", pid, person.bound_track, tr.local_id, reason)
        person.bound_track = tr.cid
        person.last_holder_cid = tr.cid
        tr.superseded_pid = None
        if person.anchor_local_id is None:
            person.anchor_local_id = tr.local_id
        person.last_bbox = tr.last_bbox
        person.last_seen = now
        # Samples that produced a re-identification pass the strict (probation) gates
        # before entering the bank: a borderline match must not reshape the person.
        for emb in tr.pending:
            self._maybe_add(person, emb, trusted=(person.enroll_track == tr.cid))
        # Gallery features are votes: pending samples of an appearance-verified bind
        # count toward this person; a position-only bind must first pass a VLM check.
        positional = reason.startswith(("continuity", "sticky"))
        check = (positional and self._hybrid
                 and bool(getattr(self.config, "vlm_identity_veto", False))
                 and bool(getattr(self.config, "vlm_positional_check", True))
                 and person.enroll_track != tr.cid)
        if tr.pending_features and self._hybrid and not check:
            self._add_person_votes(person, tr.pending_features)
        tr.person = pid
        tr.bound_since = now
        tr.pending.clear()
        tr.vlm_check_pending = check
        if not check:
            tr.pending_features.clear()
            tr.pending_feature_hashes.clear()
            tr.pending_feature_meta.clear()
            tr.vlm_dup_skips = 0
        tr.vlm_votes_pushed = 0
        tr.pending_since = None
        tr.pending_first_sample = None
        tr.contradictions = 0
        tr.released = False
        tr.sticky_verify = False
        tr.sticky_from = None
        tr.sticky_prev = None
        tr.reason = reason
        tr.score = score

    def _enroll(self, tr: _Track, now: float, reason: str) -> str:
        from core.vlm.vlm_features import consensus_embedding, empty_features, majority_vote

        # Live monitoring: a new P## exists only for someone at a marked workstation.
        # Matching an existing P## is unchanged, including after they leave the desk.
        if getattr(self, "enroll_requires_desk", False) and not tr.desk:
            tr.reason = "not_at_workstation"
            return ""

        pid = f"P{self._next_index:02d}"
        self._next_index += 1
        # Prefer the most-consistent embedding among the collected samples.
        consensus = consensus_embedding(tr.pending)
        bank = list(tr.pending[-max(1, self.config.bank_size):])
        if consensus is not None:
            bank = [consensus] + [e for e in bank if e is not consensus][: max(0, self.config.bank_size - 1)]
        # Drain any finished VLM worker before snapshotting features (async race).
        self._collect_vlm_future(tr)
        feats = empty_features()
        self._people[pid] = _Person(
            person_id=pid,
            bank=[bank[0]],
            anchor_count=1,
            last_bbox=tr.last_bbox,
            last_seen=now,
            anchor_local_id=tr.local_id,
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            samples=1,
            enroll_track=tr.cid,
            features=feats,
        )
        person = self._people[pid]
        if tr.pending_features:
            self._add_person_votes(person, tr.pending_features)
            feats = person.features
        for emb in bank[1:]:
            self._add_to_bank(person, emb)
        person.anchor_count = min(len(person.bank), self.config.max_anchors)
        tr.pending = []
        tr.pending_features = []
        tr.pending_feature_hashes = []
        tr.pending_feature_meta = []
        tr.vlm_dup_skips = 0
        self._bind(tr, pid, now, reason, 0.0)
        logger.info(
            "Enrolled %s from T%s (%s; gallery=%d; features=%s)",
            pid, tr.local_id, reason, len(self._people),
            {k: v for k, v in feats.items() if v != "unknown"} or "{}",
        )
        return pid

    def _gallery_has_vlm(self, person: _Person) -> bool:
        from core.vlm.vlm_features import UNKNOWN

        gallery = person.features or {}
        return bool(gallery and any(str(v) != UNKNOWN for v in gallery.values()))

    def _vlm_allows(self, tr: _Track, person: _Person) -> bool:
        """Optional hybrid gate: gallery VLM features can block embedding rematch.

        Disabled by default (``vlm_identity_veto=False``) until labeled CCTV evaluation
        shows Moondream reliability is acceptable. When enabled:

        * Gallery empty / all-unknown → emb-only (cannot compare clothing).
        * Track still collecting VLM samples → ``False`` (wait; do not attach on emb alone).
        * Both ready → every comparable key must agree (``features_compatible``).
        """
        return self._vlm_verdict(tr, person) == "ok"

    def _vlm_verdict(self, tr: _Track, person: _Person, now: float | None = None) -> str:
        """``ok`` | ``wait`` | ``mismatch`` for reusing ``person`` on this track.

        Track side needs ``vlm_veto_min_samples`` unique samples and a ≥2-vote majority
        per key (a single noisy crop never vetoes). One conflicting key is tolerated
        when ``vlm_veto_agree_override`` keys agree. ``now`` given: waiting longer than
        ``vlm_max_wait_seconds`` since calculation started falls back to emb (``ok``).
        """
        if not self._hybrid:
            return "ok"
        if not bool(getattr(self.config, "vlm_identity_veto", False)):
            return "ok"
        if not self._gallery_has_vlm(person):
            return "ok"
        if not self._vlm_veto_ready(tr):
            if now is not None and self._vlm_wait_expired(tr, now):
                return "ok"
            return "wait"
        n_comp, n_agree = self._vlm_track_agreement(tr, person)
        if n_comp < self.config.vlm_min_comparable:
            # Not enough clothing/attrs to judge — fall back to emb (VLM uninformative).
            return "ok"
        conflicts = n_comp - n_agree
        if conflicts == 0:
            return "ok"
        if conflicts == 1 and n_agree >= int(getattr(self.config, "vlm_veto_agree_override", 3)):
            return "ok"
        return "mismatch"

    def _vlm_track_agreement(self, tr: _Track, person: _Person) -> tuple[int, int]:
        from core.vlm.vlm_features import feature_agreement, majority_vote

        voted = majority_vote(
            tr.pending_features,
            majority_min=int(self.config.vlm_majority_min),
            allow_single=False,
        )
        return feature_agreement(voted, person.features or {})

    def _vlm_veto_ready(self, tr: _Track) -> bool:
        need = min(int(getattr(self.config, "vlm_veto_min_samples", 3)),
                   int(self.config.consensus_samples))
        if len(tr.pending_features) >= max(1, need):
            return True
        return self._vlm_unique_exhausted(tr)

    def _vlm_wait_expired(self, tr: _Track, now: float) -> bool:
        limit = float(getattr(self.config, "vlm_max_wait_seconds", 20.0) or 0.0)
        start = tr.calc_started if tr.calc_started is not None else tr.first_seen
        return limit > 0 and now - start >= limit

    def _vlm_enroll_ready(self, tr: _Track, now: float) -> bool:
        """Hybrid: a new P## needs some VLM evidence so its gallery is not blank/guessed."""
        if not self._hybrid or self.feature_extractor is None:
            return True
        need = min(int(getattr(self.config, "vlm_enroll_min_samples", 2)),
                   int(self.config.consensus_samples))
        if len(tr.pending_features) >= max(0, need):
            return True
        return self._vlm_unique_exhausted(tr) or self._vlm_wait_expired(tr, now)

    def _add_person_votes(self, person: _Person, samples: Sequence[Mapping[str, str]]) -> None:
        """Accumulate VLM samples into the person's votes and recompute stable features."""
        from core.vlm.vlm_features import add_feature_votes, stable_features

        if not samples:
            return
        for s in samples:
            add_feature_votes(person.feature_votes, s)
        person.feature_samples += len(samples)
        min_votes = min(int(getattr(self.config, "vlm_gallery_min_votes", 2)),
                        int(self.config.consensus_samples))
        before = dict(person.features)
        person.features = stable_features(
            person.feature_votes, person.features or None,
            min_votes=max(1, min_votes),
            min_share=float(getattr(self.config, "vlm_gallery_min_share", 0.6)),
            color_min_share=float(
                getattr(self.config, "vlm_gallery_color_min_share", 0.55)
            ),
            color_min_votes=int(
                getattr(self.config, "vlm_gallery_color_min_votes", 3)
            ),
        )
        changed = {k: (before.get(k, "unknown"), v) for k, v in person.features.items()
                   if before.get(k, "unknown") != v}
        if changed:
            logger.info("%s gallery features updated %s (votes from %d samples)",
                        person.person_id, changed, person.feature_samples)

    def _finish_positional_check(self, tr: _Track, person: _Person, now: float) -> None:
        """Continuity / sticky bind got its VLM samples: keep it, or release on conflict.

        Position-only hand-overs skip the rematch veto; a neighbour stepping into the
        spot of a just-lost person must not keep that P## when clothing clearly differs.
        """
        pid = person.person_id
        n_comp, n_agree = self._vlm_track_agreement(tr, person)
        conflicts = n_comp - n_agree
        need = max(1, int(getattr(self.config, "vlm_release_min_conflicts", 2)))
        if self._gallery_has_vlm(person) and conflicts >= need:
            logger.info(
                "T%s VLM conflicts with %s on %d/%d features after %s — releasing",
                tr.local_id, pid, conflicts, n_comp, tr.reason,
            )
            if person.bound_track == tr.cid:
                person.bound_track = None
            if tr.sticky_prev is not None:
                person.last_bbox, person.last_seen = tr.sticky_prev
            tr.person = None
            tr.vlm_rejected.add(pid)
            tr.vlm_check_pending = False
            tr.sticky_verify = False
            tr.sticky_from = None
            tr.sticky_prev = None
            tr.contradictions = 0
            tr.pending = []
            tr.pending_since = now
            tr.pending_first_sample = None
            tr.calc_started = None
            tr.reason = f"vlm_release from {pid}"
            return
        # Accepted: its samples now count as this person's evidence.
        tr.vlm_check_pending = False
        if tr.pending_features:
            self._add_person_votes(person, tr.pending_features)
        tr.vlm_votes_pushed = len(tr.pending_features)

    def _consensus_ready(self, tr: _Track) -> bool:
        """Ready to decide once ReID samples are in — do not wait on slow VLM."""
        need = self.config.consensus_samples if self._hybrid else self.config.enroll_samples
        return len(tr.pending) >= need

    def _vlm_ready(self, tr: _Track) -> bool:
        if not self._hybrid:
            return True
        return self._vlm_samples_sufficient(tr)

    def _query_vectors(self, tr: _Track) -> list[np.ndarray]:
        """Embeddings used for gallery scoring (consensus vector when hybrid)."""
        if not tr.pending:
            return []
        if self._hybrid and len(tr.pending) >= self.config.consensus_samples:
            from core.vlm.vlm_features import consensus_embedding

            c = consensus_embedding(tr.pending)
            return [c] if c is not None else list(tr.pending[-5:])
        return list(tr.pending)

    # ------------------------------------------------------------------ main update
    def update(
        self,
        frame_bgr: np.ndarray,
        timestamp_seconds: float,
        tracks: Sequence[Any],
        removed_track_ids: Iterable[str] = (),
        *,
        today: date | None = None,
        desks: Mapping[str, str] | None = None,
        frame_index: int | None = None,
    ) -> dict[str, IdentityAssignment]:
        """Advance one tracking frame. ``tracks`` are ``PersonTrack``-like objects.

        ``desks`` (optional): composite track id → workstation id for tracks currently
        seated at a desk. Used only as a prior for returnees; omit to disable.
        ``frame_index`` (optional): source video/stream frame id for VLM provenance;
        when omitted, an internal counter is used.
        """
        now = float(timestamp_seconds)
        if not self.config.enabled:
            return {}
        self._ensure_day(today or date.today())
        for cid in removed_track_ids:
            self.forget_track(cid)

        fh, fw = frame_bgr.shape[:2]
        self._frame_wh = (fw, fh)
        if frame_index is not None:
            self._frame_index = int(frame_index)
        else:
            self._frame_index += 1
        hold = max(1, int(self.config.sticky_hold_frames))
        self._sticky = [s for s in self._sticky if self._frame_index - s.frame_index <= hold]
        desks = desks or {}
        frame_dt = 0.0
        if self._last_update_time is not None:
            frame_dt = min(1.0, max(0.0, now - self._last_update_time))
        self._last_update_time = now
        observed: list[Any] = []
        observed_tracks: list[_Track] = []
        for t in tracks:
            cid = t.composite_track_id
            bbox = tuple(int(v) for v in t.bbox)
            tr = self._tracks.get(cid)
            is_new = tr is None
            if tr is None:
                tr = _Track(cid, int(t.local_track_id), now, bbox, now)  # type: ignore[arg-type]
                tr.last_observed_frame = self._frame_index
                self._tracks[cid] = tr
            if tr.discarded:
                continue
            if t.is_predicted:
                continue
            prev_bbox = tr.last_bbox
            tr.last_bbox = bbox  # type: ignore[assignment]
            tr.last_observed = now
            tr.last_observed_frame = self._frame_index
            tr.desk = desks.get(cid)
            # Lost holder back at its own spot → take the P## back from a positional bind.
            if not tr.person and tr.superseded_pid:
                self._try_reclaim(tr, prev_bbox, now)
            # New (or unbound) track near a just-lost person → reuse P/T instantly
            # (allowed during warm-up: rematch is free, not a calculating restart).
            if (is_new or not tr.person) and not tr.released:
                self._try_sticky_bind(tr, now)
            observed.append(t)
            observed_tracks.append(tr)
            if tr.person and tr.person in self._people:
                person = self._people[tr.person]
                person.last_bbox = tr.last_bbox
                person.last_seen = now
                dt = min(1.0, max(0.0, now - tr.last_desk_update))
                person.seen_seconds += dt
                if tr.desk:
                    person.desk_seconds[tr.desk] = person.desk_seconds.get(tr.desk, 0.0) + dt
                tr.last_desk_update = now

        # Two separate boxes seen together are two people (used to veto P## hand-overs).
        self._update_covisibility(observed_tracks, frame_dt)

        live = {t.composite_track_id for t in tracks}
        gone = [c for c in self._tracks if c not in live]
        for cid in gone:
            self.forget_track(cid)
        if gone:
            self._prune_covisibility()
        self._lost_pending = [lp for lp in self._lost_pending
                              if now - lp.last_observed <= self.config.continuity_seconds]

        # ---- 1. sample embeddings for tracks that need one now
        boxes = {t.composite_track_id: self._tracks[t.composite_track_id].last_bbox for t in observed}
        want: list[str] = []
        for t in observed:
            tr = self._tracks[t.composite_track_id]
            if tr.discarded:
                continue
            lifecycle = getattr(getattr(t, "lifecycle", None), "value", "CONFIRMED")
            if lifecycle == "TENTATIVE":
                continue
            # A provisional sticky bind is checked on the first crop of usable size, even
            # partly occluded / low-confidence: that sample only decides accept/reject and
            # never enters the bank (the track is on probation right after binding).
            verifying = tr.sticky_verify
            # During cool-down we still sample ReID/VLM so gallery rematch can attach a P##.
            # New enrollment is blocked until stable (see _resolve_pending allow_enroll).
            if not verifying and float(getattr(t, "confidence", 1.0)) < self.config.min_identity_confidence:
                continue
            if verifying:
                interval = 0.0
            elif tr.person:
                interval = self.config.update_interval_seconds
            else:
                interval = self.config.pending_sample_interval_seconds
            if now - tr.last_sample < interval:
                continue
            others = [] if verifying else [b for c, b in boxes.items() if c != tr.cid]
            if not self._quality_ok(tr.last_bbox, others):
                if verifying:
                    logger.debug("T%s sticky %s check waiting: crop %s not usable",
                                 tr.local_id, tr.person, tr.last_bbox)
                continue
            want.append(tr.cid)
        embs: dict[str, np.ndarray] = {}
        if want and self.encoder is not None:
            crops = []
            for cid in want:
                x1, y1, x2, y2 = self._tracks[cid].last_bbox
                crops.append(frame_bgr[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)])
            try:
                for cid, emb in zip(want, self.encoder.embed_many(crops)):
                    embs[cid] = emb
                    self._tracks[cid].last_sample = now
            except Exception as exc:  # noqa: BLE001 — identity is best-effort; tracking continues
                logger.warning("ReID embedding failed: %s", exc)

        # ---- 2. bound tracks: refresh bank / detect tracker identity switches
        for cid, emb in embs.items():
            tr = self._tracks[cid]
            if not tr.person or tr.person not in self._people:
                continue
            person = self._people[tr.person]
            anchor_s, s = self._gate_score(emb, person)
            tr.score = s
            _, rival_s = self._best_rival(emb, person)
            switched_to_rival = (rival_s >= s + self.config.rival_switch_margin
                                 and rival_s >= self.config.strong_match_threshold)
            if tr.sticky_verify:
                verify_s = max(anchor_s, s)
                # Someone else explains this crop about as well → position guess is unsafe.
                ambiguous = (verify_s < self.config.strong_match_threshold
                             and rival_s > verify_s - self.config.match_margin / 2)
                if verify_s < self.config.sticky_verify_threshold or switched_to_rival or ambiguous:
                    self._reject_sticky(tr, emb, verify_s, now)
                    continue
                logger.debug("T%s sticky %s verified (%.2f, rival %.2f)", tr.local_id, tr.person, verify_s, rival_s)
                tr.sticky_verify = False
                tr.sticky_from = None
                tr.sticky_prev = None
            trusted = person.enroll_track == cid
            on_probation = (not trusted
                            and now - tr.bound_since < self.config.probation_seconds)
            agrees = max(anchor_s, s) >= self.config.drift_threshold
            if switched_to_rival:
                added = False
            elif on_probation:
                added = agrees  # evidence counts, but nothing enters the bank yet
            else:
                added = self._maybe_add(person, emb, trusted=trusted)
            if added:
                tr.contradictions = 0
            elif switched_to_rival or not agrees:
                tr.contradictions += 1
                if tr.contradictions >= self.config.drift_samples:
                    logger.info("T%s no longer looks like %s (%d contradicting samples) — re-resolving",
                                tr.local_id, tr.person, tr.contradictions)
                    if person.bound_track == cid:
                        person.bound_track = None
                    tr.person = None
                    tr.contradictions = 0
                    tr.pending = [emb]
                    tr.pending_since = now
                    tr.pending_first_sample = now
                    tr.reason = "identity_switch"
                    tr.released = True
                    # The box now holds someone else: its co-visibility history belonged
                    # to the previous occupant and must not veto who this one can be.
                    tr.covis.clear()
            else:
                tr.contradictions = max(0, tr.contradictions - 1)

        # ---- 2b. bound tracks: keep voting gallery VLM features from trusted tracks
        # (corrects early wrong values) and VLM-check position-only hand-overs.
        if self._hybrid and self.feature_extractor is not None and self._vlm_pool is not None:
            max_votes = int(getattr(self.config, "vlm_gallery_max_votes", 15) or 0)
            for t in observed:
                tr = self._tracks[t.composite_track_id]
                if not tr.person or tr.person not in self._people:
                    continue
                person = self._people[tr.person]
                if tr.vlm_check_pending:
                    self._collect_vlm_future(tr)
                    if self._vlm_veto_ready(tr):
                        self._finish_positional_check(tr, person, now)
                    else:
                        self._maybe_queue_vlm(tr, frame_bgr, now,
                            [b for c, b in boxes.items() if c != tr.cid])
                    continue
                trusted = (person.enroll_track == tr.cid
                           or (not tr.sticky_verify
                               and now - tr.bound_since >= self.config.probation_seconds))
                if not trusted:
                    continue
                self._collect_vlm_future(tr)
                fresh = tr.pending_features[tr.vlm_votes_pushed:]
                if fresh:
                    self._add_person_votes(person, fresh)
                    tr.vlm_votes_pushed = len(tr.pending_features)
                if len(tr.pending_features) >= self.config.consensus_samples:
                    # Batch full: start a new one (hash memory resets with it).
                    tr.pending_features.clear()
                    tr.pending_feature_hashes.clear()
                    tr.pending_feature_meta.clear()
                    tr.vlm_votes_pushed = 0
                    tr.vlm_dup_skips = 0
                if max_votes <= 0 or person.feature_samples < max_votes:
                    self._maybe_queue_vlm(tr, frame_bgr, now,
                        [b for c, b in boxes.items() if c != tr.cid])

        # ---- 3. unresolved tracks: cool-down rematch → enroll after stable → or drop ghosts
        pending_match: list[_Track] = []   # cool-down: gallery rematch only
        pending_full: list[_Track] = []    # stable: rematch + enroll
        max_calc = float(self.config.max_calculating_seconds or 0.0)
        for t in observed:
            tr = self._tracks[t.composite_track_id]
            if tr.person or tr.discarded:
                continue
            if tr.calc_started is None:
                tr.calc_started = now
            if max_calc > 0 and (now - tr.calc_started) >= max_calc:
                self._discard_calculating(tr, now)
                continue
            if tr.pending_since is None:
                tr.pending_since = now
            emb = embs.get(tr.cid)
            if emb is not None and not tr.pending and not tr.released:
                self._inherit_lost_pending(tr, emb, now)
            if emb is not None and (not tr.pending or tr.pending[-1] is not emb):
                tr.pending.append(emb)
                tr.pending = tr.pending[-self.config.pending_keep:]
                if tr.pending_first_sample is None:
                    tr.pending_first_sample = now
            # Structured VLM features in background workers (does not block the stream).
            # Crops come from clean ``frame_bgr`` (pre-visualization). Duplicate hashes
            # are rejected so consensus requires distinct source frames.
            if self._hybrid and self.feature_extractor is not None and self._vlm_pool is not None:
                self._collect_vlm_future(tr)
                if emb is not None:
                    self._maybe_queue_vlm(tr, frame_bgr, now,
                        [b for c, b in boxes.items() if c != tr.cid])
            need = self.config.consensus_samples if self._hybrid else self.config.enroll_samples
            stable = self._is_stable(tr, now)
            if not stable:
                age = now - tr.first_seen
                cool = float(self.config.min_stable_seconds or 0.0)
                tr.reason = (
                    f"cooling ({age:.0f}/{cool:.0f}s, emb {len(tr.pending)}/{need})"
                )
                pending_match.append(tr)
            else:
                if self._hybrid:
                    tr.reason = (
                        f"calculating (emb {len(tr.pending)}/{need}, "
                        f"vlm {len(tr.pending_features)}/{need})"
                    )
                else:
                    tr.reason = f"calculating (emb {len(tr.pending)}/{need})"
                pending_full.append(tr)
        if pending_match:
            self._resolve_pending(pending_match, observed, now, allow_enroll=False)
        if pending_full:
            self._resolve_pending(pending_full, observed, now, allow_enroll=True)
        self._absorb_stale_duplicates(now)

        # ---- 4. report
        out: dict[str, IdentityAssignment] = {}
        for t in tracks:
            tr = self._tracks.get(t.composite_track_id)
            if tr is None:
                continue
            if tr.discarded:
                out[tr.cid] = IdentityAssignment(
                    None, None, "drop", tr.score, tr.reason or "calculating_timeout",
                )
                continue
            if t.is_predicted:
                state = "ghost"
            elif tr.person and tr.sticky_verify:
                # Provisional position sticky — do not show P## until ReID verifies.
                state = "pending"
            elif tr.person:
                state = "bound"
            elif not self._is_stable(tr, now):
                state = "warming"  # cool-down: rematch allowed, new enroll blocked
            else:
                state = "pending"
            # Hide P## while sticky_verify so overlay does not flicker wrong IDs.
            show_pid = None if (not tr.person or tr.sticky_verify) else tr.person
            out[tr.cid] = IdentityAssignment(
                show_pid, self.display_track_id(show_pid), state, tr.score, tr.reason,
            )
        return out

    def _append_vlm_sample(
        self,
        tr: _Track,
        feats: dict[str, str],
        *,
        sha1: str,
        frame_index: int | None,
        timestamp_seconds: float | None,
    ) -> bool:
        """Accept one consensus sample if crop hash is new. Returns True if kept."""
        if not feats:
            return False
        if len(tr.pending_features) >= self.config.consensus_samples:
            return False
        if sha1 in tr.pending_feature_hashes:
            logger.debug(
                "T%s VLM consensus skip duplicate crop sha=%s frame=%s",
                tr.local_id, sha1[:12], frame_index,
            )
            return False
        if frame_index is not None and any(
            m.get("frame_index") == frame_index for m in tr.pending_feature_meta
        ):
            logger.debug(
                "T%s VLM consensus skip duplicate frame_index=%s sha=%s",
                tr.local_id, frame_index, sha1[:12],
            )
            return False
        tr.pending_features.append(feats)
        tr.pending_feature_hashes.append(sha1)
        tr.pending_feature_meta.append({
            "track_id": tr.local_id,
            "frame_index": frame_index,
            "timestamp_seconds": timestamp_seconds,
            "crop_sha1": sha1,
        })
        tr.vlm_dup_skips = 0
        logger.info(
            "T%s VLM consensus sample %d/%d frame=%s t=%s sha=%s",
            tr.local_id,
            len(tr.pending_features),
            self.config.consensus_samples,
            frame_index,
            f"{timestamp_seconds:.2f}" if timestamp_seconds is not None else "?",
            sha1[:12],
        )
        return True

    def _vlm_unique_exhausted(self, tr: _Track) -> bool:
        """True when ≥1 unique sample exists and further crops keep hashing the same.

        Static / seated employees rarely yield ``consensus_samples`` distinct images;
        commit the incomplete signature rather than waiting forever or re-querying GPU.
        """
        limit = int(getattr(self.config, "vlm_dup_skip_limit", 3) or 3)
        return bool(tr.pending_features) and tr.vlm_dup_skips >= max(1, limit)

    def _vlm_samples_sufficient(self, tr: _Track) -> bool:
        if len(tr.pending_features) >= self.config.consensus_samples:
            return True
        return self._vlm_unique_exhausted(tr)

    def _collect_vlm_future(self, tr: _Track) -> None:
        """Drain a finished VLM future; reject hash mismatches / duplicates."""
        if tr.vlm_future is None or not tr.vlm_future.done():
            return
        expect = tr.vlm_inflight_sha1
        tr.vlm_inflight_sha1 = None
        try:
            result = tr.vlm_future.result(timeout=0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM features failed for T%s: %s", tr.local_id, exc)
            tr.vlm_future = None
            return
        tr.vlm_future = None
        from core.vlm.vlm_features import VlmExtractResult

        if isinstance(result, VlmExtractResult):
            feats = result.features
            sha = result.crop_sha1
            frame_index = result.frame_index
            ts = result.timestamp_seconds
        elif isinstance(result, dict):
            feats = result
            sha = expect or ""
            frame_index = self._frame_index
            ts = None
        else:
            logger.warning("VLM features unexpected result type for T%s: %s", tr.local_id, type(result))
            return
        if expect and sha and sha != expect:
            logger.warning(
                "T%s VLM async crop mismatch: submitted %s got %s — discarding",
                tr.local_id, expect[:12], sha[:12],
            )
            return
        if not sha:
            logger.warning("T%s VLM sample missing crop hash — discarding", tr.local_id)
            return
        self._append_vlm_sample(
            tr, feats, sha1=sha, frame_index=frame_index, timestamp_seconds=ts,
        )

    def _maybe_queue_vlm(
        self,
        tr: _Track,
        frame_bgr: np.ndarray,
        now: float,
        others: Sequence[tuple[int, int, int, int]] = (),
    ) -> None:
        """Submit one VLM extract if interval elapsed and crop hash is new.

        ``others``: boxes of other people this frame; a crop they cover is skipped.
        """
        if self._vlm_pool is None or self.feature_extractor is None:
            return
        if tr.vlm_future is not None:
            return
        if self._vlm_samples_sufficient(tr):
            return
        if now - tr.last_vlm_sample < self.config.pending_sample_interval_seconds:
            return
        from core.vlm.vlm_features import crop_sha1, expand_person_bbox

        fh, fw = frame_bgr.shape[:2]
        # Pad from the clean source frame (not a resized crop) so head ROI keeps context.
        x1, y1, x2, y2 = expand_person_bbox(
            tr.last_bbox,
            (fw, fh),
            pad_x_fraction=float(getattr(self.config, "vlm_crop_pad_x", 0.15) or 0.0),
            pad_top_fraction=float(getattr(self.config, "vlm_crop_pad_top", 0.08) or 0.0),
        )
        max_overlap = float(getattr(self.config, "vlm_max_overlap", 0.08) or 0.0)
        torso_max = float(getattr(self.config, "vlm_torso_max_overlap", 0.05) or 0.0)
        if others and (max_overlap > 0 or torso_max > 0):
            area = max(1, (x2 - x1) * (y2 - y1))
            # Torso band in frame coords (same fractions as VLM torso crop).
            bx1, by1, bx2, by2 = (int(v) for v in tr.last_bbox)
            bh = max(1, by2 - by1)
            ty1 = by1 + int(round(bh * 0.18))
            ty2 = by1 + int(round(bh * 0.58))
            t_area = max(1, (bx2 - bx1) * max(1, ty2 - ty1))
            for ox1, oy1, ox2, oy2 in others:
                if max_overlap > 0:
                    iw = min(x2, ox2) - max(x1, ox1)
                    ih = min(y2, oy2) - max(y1, oy1)
                    if iw > 0 and ih > 0 and iw * ih / area > max_overlap:
                        logger.debug("T%s skip VLM: crop overlapped by another person", tr.local_id)
                        return
                if torso_max > 0:
                    tw = min(bx2, ox2) - max(bx1, ox1)
                    th = min(ty2, oy2) - max(ty1, oy1)
                    if tw > 0 and th > 0 and tw * th / t_area > torso_max:
                        logger.debug("T%s skip VLM: torso overlapped by another person", tr.local_id)
                        return
        crop = frame_bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return
        sha = crop_sha1(crop)
        if sha in tr.pending_feature_hashes:
            # Same pixels — do not burn GPU; count toward static-person exhaustion.
            tr.last_vlm_sample = now
            if tr.pending_features:
                tr.vlm_dup_skips += 1
            logger.debug(
                "T%s skip VLM submit: duplicate crop sha=%s frame=%s skips=%d",
                tr.local_id, sha[:12], self._frame_index, tr.vlm_dup_skips,
            )
            return
        tr.last_vlm_sample = now
        tr.vlm_inflight_sha1 = sha
        tr.vlm_future = self._submit_vlm(crop, tr, now, expect_sha1=sha)

    def _submit_vlm(
        self,
        crop: np.ndarray,
        tr: _Track,
        now: float,
        *,
        expect_sha1: str | None = None,
    ) -> Future:
        """Background Moondream call with track/frame context for audit logs."""
        assert self._vlm_pool is not None and self.feature_extractor is not None
        extractor = self.feature_extractor
        # Prefer extract_detailed (returns provenance); fall back to extract for fakes/tests.
        if hasattr(extractor, "extract_detailed"):
            return self._vlm_pool.submit(
                extractor.extract_detailed,
                crop,
                track_id=tr.local_id,
                frame_index=self._frame_index,
                timestamp_seconds=now,
                expect_sha1=expect_sha1,
            )
        return self._vlm_pool.submit(
            extractor.extract,
            crop,
            track_id=tr.local_id,
            frame_index=self._frame_index,
            timestamp_seconds=now,
        )

    def _discard_calculating(self, tr: _Track, now: float) -> None:
        """P## never resolved after max_calculating_seconds — drop box + stop GPU work."""
        if tr.discarded:
            return
        if tr.vlm_future is not None and not tr.vlm_future.done():
            tr.vlm_future.cancel()
        tr.vlm_future = None
        tr.vlm_inflight_sha1 = None
        tr.pending.clear()
        tr.pending_features.clear()
        tr.pending_feature_hashes.clear()
        tr.pending_feature_meta.clear()
        tr.vlm_dup_skips = 0
        tr.discarded = True
        tr.reason = "calculating_timeout"
        tr.person = None
        self._drop_cids.append(tr.cid)
        logger.info(
            "T%s still calculating after %.0fs — removing box/tracker (likely empty object)",
            tr.local_id, float(self.config.max_calculating_seconds or 0.0),
        )

    def _absorb_stale_duplicates(self, now: float) -> None:
        """Retire weak orphan IDs that a richer live ID explains.

        Someone enrolled from a short, dim doorway track (3–5 crops) may score only
        0.4–0.5 against that reference on their next appearance and be enrolled again.
        Once the second ID has a rich bank and the first has not been seen since, and
        the first's samples match the second's bank well, they are the same person: the
        first is absorbed (its samples join the live bank through the normal gates). The
        ID currently shown on screen never changes; only the stale one disappears.
        """
        cfg = self.config
        if now - self._last_absorb < cfg.absorb_interval_seconds:
            return
        self._last_absorb = now
        live_rich = [p for p in self._people.values()
                     if p.bound_track is not None and p.bound_track in self._tracks
                     and p.seen_seconds >= cfg.absorb_weak_seconds * 2]
        if not live_rich:
            return
        for weak in list(self._people.values()):
            if weak.seen_seconds > cfg.absorb_weak_seconds or weak.bound_track is not None:
                continue
            best, best_s = None, -1.0
            for live in live_rich:
                if live is weak or live.last_seen <= weak.last_seen:
                    continue  # live must be the later ID (weak unseen since)
                s = self._score(weak.bank, live)
                if s > best_s:
                    best, best_s = live, s
            if best is None or best_s < cfg.absorb_threshold:
                continue
            # The orphan must not explain anyone else nearly as well.
            others = [self._score(weak.bank, p) for p in self._people.values() if p is not weak and p is not best]
            if others and best_s - max(others) < cfg.match_margin:
                continue
            logger.info("%s absorbed into %s (stale duplicate, %.2f)", weak.person_id, best.person_id, best_s)
            for emb in weak.bank:
                self._maybe_add(best, emb)
            best.samples += weak.samples
            best.seen_seconds += weak.seen_seconds
            for desk, secs in weak.desk_seconds.items():
                best.desk_seconds[desk] = best.desk_seconds.get(desk, 0.0) + secs
            self.aliases[weak.person_id] = best.person_id
            del self._people[weak.person_id]

    def _inherit_lost_pending(self, tr: _Track, emb: np.ndarray, now: float) -> None:
        """First sample of a new unresolved track: continue a just-lost unresolved track
        that was nearby and looks the same (tracker break while walking in)."""
        cfg = self.config
        best, best_s = None, -1.0
        for lp in self._lost_pending:
            if now - lp.last_observed > cfg.continuity_seconds or not lp.pending:
                continue
            if not self._near(lp.last_bbox, tr.last_bbox, cfg.continuity_radius):
                continue
            s = float(np.sort(np.stack(lp.pending) @ emb)[-min(3, len(lp.pending)):].mean())
            if s > best_s:
                best, best_s = lp, s
        if best is not None and best_s >= cfg.continuity_min_score:
            tr.pending = list(best.pending)
            tr.pending_features = list(best.pending_features)
            tr.pending_feature_hashes = list(best.pending_feature_hashes)
            tr.pending_feature_meta = list(best.pending_feature_meta)
            tr.pending_since = min(tr.pending_since if tr.pending_since is not None else now, best.pending_since)
            tr.pending_first_sample = best.pending_first_sample
            tr.reason = f"continuing T{best.local_id} ({best_s:.2f})"
            self._lost_pending.remove(best)

    def _holder_blocks(self, person: _Person, claimant: _Track, now: float) -> bool:
        """Is ``person`` still held by another track against ``claimant``?

        A holder that was seen this frame always blocks. A holder seen within
        ``hold_seconds`` keeps blocking only if it was still seen after the claimant
        appeared — that is a duplicate/alternating track, not a hand-over. This stops
        P## ping-pong on one-frame detection misses.
        """
        if person.bound_track is None or person.bound_track == claimant.cid:
            return False
        holder = self._tracks.get(person.bound_track)
        if holder is None or holder.person != person.person_id:
            return False
        if holder.last_observed >= now:
            return True
        recent = now - holder.last_observed < self._hold_for(holder)
        return recent and holder.last_observed >= claimant.first_seen

    def _hold_for(self, holder: _Track) -> float:
        """Seconds a lost holder keeps its P## against tracks that pre-date the loss.

        A seated person does not move: a detector miss of a few seconds at the desk must
        not free the P## for a neighbour who was visible the whole time.
        """
        hold = float(self.config.hold_seconds)
        if holder.desk:
            hold = max(hold, float(self.config.seated_hold_seconds or 0.0))
        return hold

    def _resolve_pending(
        self,
        pending: list[_Track],
        observed: list[Any],
        now: float,
        *,
        allow_enroll: bool = True,
    ) -> None:
        cfg = self.config
        people = list(self._people.values())
        held = [p for p in people if p.bound_track is not None and p.bound_track in self._tracks
                and now - self._tracks[p.bound_track].last_observed
                < self._hold_for(self._tracks[p.bound_track])]
        held_boxes = [self._tracks[p.bound_track].last_bbox for p in held]

        candidates = people
        n, m = len(pending), len(candidates)
        app = np.full((n, max(m, 1)), -1.0, dtype=np.float32)
        cont = np.zeros((n, max(m, 1)), dtype=bool)
        feas = np.zeros((n, max(m, 1)), dtype=bool)
        blocked = np.zeros((n, max(m, 1)), dtype=bool)
        # Blocked person whose holder box is close: this track may be a duplicate/partial
        # box of them. A holder far away is someone else standing elsewhere right now.
        near_holder = np.zeros((n, max(m, 1)), dtype=bool)
        # This track was seen alongside the person's holder as a separate box: it is a
        # different physical person — never a candidate, never a duplicate box.
        coex = np.zeros((n, max(m, 1)), dtype=bool)
        for i, tr in enumerate(pending):
            qvecs = self._query_vectors(tr)
            for j, person in enumerate(candidates):
                if qvecs:
                    app[i, j] = self._score(qvecs, person)
                if self._coexisted(person, tr) or person.person_id in tr.vlm_rejected:
                    coex[i, j] = True
                    continue
                if self._holder_blocks(person, tr, now):
                    blocked[i, j] = True
                    hb = self._tracks[person.bound_track].last_bbox  # type: ignore[index]
                    h = max(_height(hb), _height(tr.last_bbox))
                    (hx, hy), (tx, ty) = _center(hb), _center(tr.last_bbox)
                    near_holder[i, j] = float(np.hypot(hx - tx, hy - ty)) / h <= 1.0
                    continue
                ok, is_cont = self._feasible(person, tr.last_bbox, now)
                feas[i, j] = ok
                cont[i, j] = is_cont

        # Desk prior: the pending track is seated at a candidate's learned home desk.
        desk = np.zeros((n, max(m, 1)), dtype=bool)
        for i, tr in enumerate(pending):
            if tr.desk:
                for j, person in enumerate(candidates):
                    desk[i, j] = person.home_desk(cfg.desk_home_share, cfg.desk_home_min_seconds) == tr.desk

        assigned: dict[int, int] = {}
        if m:
            utility = np.where(feas, app + cont * cfg.continuity_bonus + desk * cfg.desk_bonus, -10.0)
            try:
                from scipy.optimize import linear_sum_assignment

                rows, cols = linear_sum_assignment(-utility)
            except ImportError:  # greedy fallback
                rows, cols, used = [], [], set()
                for i in np.argsort(-utility.max(axis=1)):
                    j = int(np.argmax(np.where([k in used for k in range(m)], -99, utility[i])))
                    rows.append(i)
                    cols.append(j)
                    used.add(j)
            assigned = {int(i): int(j) for i, j in zip(rows, cols) if utility[i, j] > -5.0}

        for i, tr in enumerate(pending):
            n_samples = len(tr.pending)
            row = app[i, :m] if m else np.array([])
            j = assigned.get(i)
            best = float(row[j]) if j is not None else -1.0
            # Runner-up over people this track could be (someone seated elsewhere right
            # now cannot be the runner-up; a near duplicate box still counts).
            others = np.array([float(row[k]) for k in range(m)
                               if k != j and not coex[i, k]
                               and (not blocked[i, k] or near_holder[i, k])]) if j is not None else np.array([])
            second = float(others.max()) if others.size else -1.0
            margin = best - second if second > -1.0 else 1.0
            pending_age = now - (tr.pending_since if tr.pending_since is not None else now)

            span = now - (tr.pending_first_sample if tr.pending_first_sample is not None else now)
            ready = self._consensus_ready(tr)

            if j is not None:
                person = candidates[j]
                pid = person.person_id
                # (a) Tracker break at the spot: fast hand-over (spatial continuity).
                # VLM not required — same physical track hand-off.
                if (cont[i, j] and n_samples >= 1 and best >= cfg.continuity_min_score
                        and margin >= cfg.match_margin / 2):
                    self._bind(tr, pid, now, f"continuity ({best:.2f})", best)
                    continue
                # Appearance rematch: hybrid requires emb AND VLM when the gallery
                # already has clothing/attrs for that person (no emb-only shortcut).
                if ready:
                    verdict = self._vlm_verdict(tr, person, now)
                    if verdict == "wait":
                        tr.reason = (
                            f"calculating (emb {best:.2f}, "
                            f"vlm {len(tr.pending_features)}/{cfg.consensus_samples})"
                        )
                    elif verdict == "mismatch":
                        tr.reason = f"vlm_mismatch vs {pid} (emb {best:.2f})"
                    else:
                        # (b) Unmistakable appearance (VLM agreed or gallery has no VLM yet).
                        if (n_samples >= cfg.strong_match_samples
                                and best >= cfg.strong_match_threshold
                                and margin >= cfg.match_margin):
                            self._bind(tr, pid, now, f"appearance {best:.2f} (+{margin:.2f})", best)
                            continue
                        # (c) Seated at this person's own desk.
                        if (desk[i, j] and n_samples >= cfg.strong_match_samples
                                and best >= cfg.desk_match_threshold
                                and margin >= cfg.match_margin / 2):
                            self._bind(tr, pid, now, f"appearance {best:.2f} at home desk {tr.desk}", best)
                            continue
                        # (d) Plain appearance with whole-track evidence.
                        need = cfg.match_threshold if m > 1 else cfg.sole_match_threshold
                        if (n_samples >= cfg.min_match_samples
                                and span >= cfg.min_match_span_seconds
                                and best >= need and margin >= cfg.match_margin):
                            self._bind(tr, pid, now, f"appearance {best:.2f} (+{margin:.2f})", best)
                            continue

            best_any = float(row.max()) if row.size else -1.0
            # Looks like someone already visible on another track → probably a duplicate box.
            occ_best = max((float(app[i, k]) for k in range(m) if blocked[i, k] and near_holder[i, k]),
                           default=-1.0)
            # Best score among people this track could actually be (excludes people
            # visible right now somewhere else) — used for "is this a new person?".
            best_possible = max((float(app[i, k]) for k in range(m)
                                 if not coex[i, k] and (not blocked[i, k] or near_holder[i, k])), default=-1.0)
            duplicate_box = any(_iou(tr.last_bbox, b) >= cfg.duplicate_iou for b in held_boxes)
            tr.score = best_any
            if duplicate_box:
                tr.reason = "duplicate_box"
                continue
            if not ready:
                need = cfg.consensus_samples if self._hybrid else cfg.enroll_samples
                if self._hybrid:
                    tr.reason = (
                        f"calculating (emb {n_samples}/{need}, "
                        f"vlm {len(tr.pending_features)}/{need})"
                    )
                else:
                    tr.reason = f"calculating (emb {n_samples}/{need})"
                continue
            clearly_new = (
                allow_enroll
                and n_samples >= cfg.enroll_samples
                and span >= cfg.min_enroll_span_seconds
                and best_possible < cfg.new_person_threshold
                and occ_best < cfg.sole_match_threshold
            )
            vlm_enroll_ok = self._vlm_enroll_ready(tr, now)
            vlm_wait_reason = (f"calculating (best {best_any:.2f}, "
                               f"waiting vlm {len(tr.pending_features)})")
            if clearly_new:
                if not vlm_enroll_ok:
                    tr.reason = vlm_wait_reason
                    continue
                self._enroll(tr, now, f"new (best known {best_possible:.2f})")
                continue
            if pending_age >= cfg.max_pending_seconds and n_samples >= cfg.strong_match_samples:
                need_t = (cfg.timeout_match_threshold if m > 1
                          else max(cfg.timeout_match_threshold, cfg.sole_match_threshold))
                verdict_j = self._vlm_verdict(tr, candidates[j], now) if j is not None else "ok"
                if (j is not None and best >= need_t and margin >= cfg.match_margin
                        and verdict_j == "ok"):
                    self._bind(tr, candidates[j].person_id, now, f"timeout match {best:.2f}", best)
                elif tr.released and best_possible >= cfg.new_person_threshold:
                    tr.reason = f"released, ambiguous (best {best_possible:.2f})"
                elif (allow_enroll and best_possible < cfg.timeout_new_threshold
                        and occ_best < cfg.sole_match_threshold):
                    # Same guard as clearly_new: a box that looks like a person held by
                    # an adjacent track is not a new person (duplicate / displaced holder).
                    if not vlm_enroll_ok:
                        tr.reason = vlm_wait_reason
                        continue
                    self._enroll(tr, now, f"timeout new (best known {best_possible:.2f})")
                elif pending_age >= cfg.ambiguous_enroll_seconds and occ_best < cfg.sole_match_threshold:
                    if (j is not None and best >= cfg.match_threshold
                            and verdict_j == "ok"):
                        self._bind(tr, candidates[j].person_id, now,
                                   f"ambiguous match {best:.2f} (+{margin:.2f})", best)
                    elif (j is not None and best >= cfg.match_threshold
                            and verdict_j == "wait") or not vlm_enroll_ok:
                        # Plausible returnee whose VLM evidence is still in flight:
                        # a new P## now would duplicate them.
                        tr.reason = vlm_wait_reason
                    elif allow_enroll:
                        # Embedding alone is not enough in hybrid: enroll new if VLM
                        # does not confirm a previous person.
                        self._enroll(tr, now, f"ambiguous new (best known {best_possible:.2f})")
                    else:
                        tr.reason = f"cooling rematch (best {best_any:.2f})"
                else:
                    tr.reason = f"ambiguous (best {best_any:.2f}, visible {occ_best:.2f})"
                continue
            if not allow_enroll:
                cool = float(cfg.min_stable_seconds or 0.0)
                age = now - tr.first_seen
                tr.reason = f"cooling ({age:.0f}/{cool:.0f}s, best {best_any:.2f})"
                continue
            tr.reason = f"calculating (emb {n_samples}/{cfg.enroll_samples}, best {best_any:.2f})"
