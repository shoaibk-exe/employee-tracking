"""Persistent global identity registry.

Global IDs are independent of local BoT-SORT track IDs and survive camera
disconnects, track expiry, and process restarts (via the database repository).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np

from backend.association.global_associator import AssociationResult, AssociatorConfig, GlobalAssociator, IdentityHint
from backend.association.identity_constraints import ActiveObservation, ConstraintContext
from backend.identity.identity_lifecycle import next_activity_status
from backend.recognition.face_gallery import FaceGallery
from backend.reid.embedding_gallery import EmbeddingGallery
from backend.types import (
    AssociationDecision,
    IdentityActivityStatus,
    Tracklet,
    VerificationStatus,
)
from backend.utils.colors import color_for_global_id, provisional_color, unknown_color

logger = logging.getLogger(__name__)

FACE_SOURCE = "face"  # legacy ArcFace vectors
ATTR_SOURCE = "vlm_attr"  # CLIP attribute / small-VLM identity signatures
ANCHOR_SOURCES = frozenset({FACE_SOURCE, ATTR_SOURCE})  # kept out of the body ReID gallery


class IdentityRepository(Protocol):
    def next_global_id(self, prefix: str, width: int) -> str: ...
    def upsert_identity(self, record: dict[str, Any]) -> None: ...
    def add_embedding(self, record: dict[str, Any]) -> None: ...
    def add_tracklet(self, record: dict[str, Any]) -> None: ...
    def add_event(self, record: dict[str, Any]) -> None: ...
    def add_association(self, record: dict[str, Any]) -> None: ...
    def list_identities(self) -> list[dict[str, Any]]: ...
    def list_embeddings(self, active_only: bool = True) -> list[dict[str, Any]]: ...
    def get_identity(self, global_id: str) -> dict[str, Any] | None: ...
    def list_events(self, global_id: str, limit: int = 100) -> list[dict[str, Any]]: ...


@dataclass
class IdentityRecord:
    global_id: str
    activity_status: IdentityActivityStatus
    verification_status: VerificationStatus
    first_seen: float
    last_seen: float
    last_camera_id: str | None = None
    cameras_seen: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    display_name: str | None = None  # never auto-filled from appearance


@dataclass
class Binding:
    compound_key: str
    global_id: str | None
    status: str  # confirmed | provisional | unknown | calculating
    confidence: float = 0.0
    updated_at: float = 0.0
    camera_id: str = ""


class InMemoryRepository:
    """Fallback repository used when DB is unavailable (tests)."""

    def __init__(self) -> None:
        self._seq = 0
        self._identities: dict[str, dict[str, Any]] = {}
        self._embeddings: list[dict[str, Any]] = []
        self._tracklets: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._associations: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def next_global_id(self, prefix: str, width: int) -> str:
        with self._lock:
            self._seq += 1
            return f"{prefix}{self._seq:0{width}d}"

    def upsert_identity(self, record: dict[str, Any]) -> None:
        self._identities[record["global_id"]] = dict(record)

    def add_embedding(self, record: dict[str, Any]) -> None:
        self._embeddings.append(dict(record))

    def reassign_embeddings(self, source_gid: str, target_gid: str) -> int:
        moved = 0
        for r in self._embeddings:
            if r.get("global_id") == source_gid:
                r["global_id"] = target_gid
                moved += 1
        return moved

    def add_tracklet(self, record: dict[str, Any]) -> None:
        self._tracklets.append(dict(record))

    def add_event(self, record: dict[str, Any]) -> None:
        self._events.append(dict(record))

    def add_association(self, record: dict[str, Any]) -> None:
        self._associations.append(dict(record))

    def list_identities(self) -> list[dict[str, Any]]:
        return list(self._identities.values())

    def list_embeddings(self, active_only: bool = True) -> list[dict[str, Any]]:
        rows = self._embeddings
        if active_only:
            rows = [r for r in rows if r.get("active", True)]
        return list(rows)

    def get_identity(self, global_id: str) -> dict[str, Any] | None:
        return self._identities.get(global_id)

    def list_events(self, global_id: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = [e for e in self._events if e.get("global_id") == global_id]
        return rows[-limit:]


class GlobalIdentityManager:
    def __init__(
        self,
        associator: GlobalAssociator,
        gallery: EmbeddingGallery,
        repository: IdentityRepository | None = None,
        *,
        id_prefix: str = "G",
        id_width: int = 6,
        allow_gallery_update_on_provisional: bool = False,
    ) -> None:
        self.associator = associator
        self.gallery = gallery
        self.repo = repository or InMemoryRepository()
        self.id_prefix = id_prefix
        self.id_width = id_width
        self.allow_gallery_update_on_provisional = allow_gallery_update_on_provisional
        self._lock = threading.RLock()
        self._identities: dict[str, IdentityRecord] = {}
        self._bindings: dict[str, Binding] = {}
        self._provisional_seq = 0
        self.unresolved_count = 0
        self.face_gallery = FaceGallery()
        self._load_from_repo()

    def _load_from_repo(self) -> None:
        try:
            for row in self.repo.list_identities():
                self._identities[row["global_id"]] = IdentityRecord(
                    global_id=row["global_id"],
                    activity_status=IdentityActivityStatus(row.get("identity_status", "INACTIVE")),
                    verification_status=VerificationStatus(row.get("verification_status", "UNVERIFIED")),
                    first_seen=float(row.get("first_seen_ts", time.time())),
                    last_seen=float(row.get("last_seen_ts", time.time())),
                    last_camera_id=row.get("last_camera_id"),
                    cameras_seen=list(row.get("cameras_seen") or []),
                    created_at=str(row.get("created_at", "")),
                    updated_at=str(row.get("updated_at", "")),
                    display_name=row.get("display_name"),
                )
            emb_rows = self.repo.list_embeddings(active_only=False)
            face_rows = [r for r in emb_rows if r.get("verification_source") in ANCHOR_SOURCES]
            emb_rows = [r for r in emb_rows if r.get("verification_source") not in ANCHOR_SOURCES]
            if emb_rows:
                self.gallery.load_entries(emb_rows)
            if face_rows:
                logger.info("Loaded %d face embeddings", self.face_gallery.load(face_rows))
            logger.info(
                "Loaded %d identities and %d embeddings from repository",
                len(self._identities),
                len(emb_rows),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Identity repository load skipped: %s", exc)

    def color_for(self, global_id: str | None, status: str) -> tuple[int, int, int]:
        if not global_id:
            return unknown_color()
        if status == "provisional" or global_id.startswith("PROVISIONAL_"):
            return provisional_color()
        return color_for_global_id(global_id)

    def get_binding(self, compound_key: str) -> Binding | None:
        with self._lock:
            return self._bindings.get(compound_key)

    def bind_live_track(
        self,
        compound_key: str,
        global_id: str | None,
        status: str,
        confidence: float = 0.0,
        *,
        timestamp: float | None = None,
        camera_id: str | None = None,
    ) -> None:
        with self._lock:
            self._bindings[compound_key] = Binding(
                compound_key=compound_key,
                global_id=global_id,
                status=status,
                confidence=confidence,
                updated_at=time.monotonic() if timestamp is None else float(timestamp),
                camera_id=camera_id or compound_key.split(":", 1)[0],
            )

    def process_tracklets(self, tracklets: list[Tracklet], now: float | None = None) -> list[AssociationResult]:
        now = time.monotonic() if now is None else now
        with self._lock:
            hints = {
                gid: IdentityHint(
                    global_id=gid,
                    camera_id=rec.last_camera_id or "",
                    last_seen=rec.last_seen,
                    activity_status=rec.activity_status.value,
                    verification_status=rec.verification_status.value,
                )
                for gid, rec in self._identities.items()
            }
            # Only currently-fresh bindings count as simultaneous occupants.
            # Stale bindings must not block cross-camera handoffs/returns.
            simultaneous_horizon = 1.5
            ctx = ConstraintContext(
                active=[
                    ActiveObservation(
                        global_id=b.global_id,
                        camera_id=b.camera_id or b.compound_key.split(":", 1)[0],
                        timestamp=b.updated_at,
                        compound_key=b.compound_key,
                    )
                    for b in self._bindings.values()
                    if b.global_id
                    and b.status in {"confirmed", "provisional"}
                    and abs(now - b.updated_at) <= simultaneous_horizon
                ],
                last_seen={
                    gid: (rec.last_camera_id or "", rec.last_seen)
                    for gid, rec in self._identities.items()
                    if rec.last_camera_id
                },
            )
            self.associator.set_active_observations(ctx)
            results = self.associator.associate(tracklets, hints)
            self.unresolved_count = 0
            for tl, result in zip(tracklets, results):
                self._apply_result(tl, result, now)
            self._refresh_activity(now)
            return results

    def _apply_result(self, tl: Tracklet, result: AssociationResult, now: float) -> None:
        decision = result.decision
        gid = result.global_id

        if decision == AssociationDecision.NEW:
            gid = self._create_identity(tl, now, verification=VerificationStatus.UNVERIFIED)
            decision = AssociationDecision.CONFIRMED
            result.global_id = gid
            result.decision = decision
        elif decision == AssociationDecision.CONFIRMED and gid:
            self._touch_identity(gid, tl, now, VerificationStatus.UNVERIFIED)
            if tl.embedding is not None:
                self._add_embedding(gid, tl, confirmed=True)
        elif decision == AssociationDecision.PROVISIONAL:
            if gid is None:
                self._provisional_seq += 1
                gid = f"PROVISIONAL_{self._provisional_seq:03d}"
                result.global_id = gid
            else:
                self._touch_identity(gid, tl, now, VerificationStatus.PROVISIONAL)
                if tl.embedding is not None and self.allow_gallery_update_on_provisional:
                    self._add_embedding(gid, tl, confirmed=False)
            self.unresolved_count += 1
        elif decision in {AssociationDecision.UNKNOWN, AssociationDecision.REJECTED}:
            self._provisional_seq += 1
            gid = f"PROVISIONAL_{self._provisional_seq:03d}"
            result.global_id = gid
            result.decision = AssociationDecision.UNKNOWN
            self.unresolved_count += 1

        status = {
            AssociationDecision.CONFIRMED: "confirmed",
            AssociationDecision.PROVISIONAL: "provisional",
            AssociationDecision.UNKNOWN: "unknown",
            AssociationDecision.NEW: "confirmed",
            AssociationDecision.REJECTED: "unknown",
        }.get(result.decision, "unknown")

        self._bindings[tl.compound_key] = Binding(
            compound_key=tl.compound_key,
            global_id=result.global_id,
            status=status,
            confidence=result.confidence,
            updated_at=tl.end_time,
            camera_id=tl.camera_id,
        )
        tl.global_id = result.global_id
        tl.assignment_confidence = result.confidence
        tl.decision = result.decision

        self.repo.add_tracklet(
            {
                "tracklet_id": tl.tracklet_id,
                "camera_id": tl.camera_id,
                "tracker_session_id": tl.tracker_session_id,
                "local_track_id": tl.local_track_id,
                "start_time": tl.start_time,
                "end_time": tl.end_time,
                "global_id": result.global_id,
                "assignment_confidence": result.confidence,
            }
        )
        self.repo.add_association(
            {
                "association_id": f"as_{uuid.uuid4().hex[:10]}",
                "source_tracklet_id": tl.tracklet_id,
                "destination_global_id": result.global_id,
                "evidence": result.evidence,
                "decision": result.decision.value,
                "confidence": result.confidence,
                "model_version": self.gallery.model_version,
                "review_status": "pending" if status != "confirmed" else "auto",
            }
        )
        if result.global_id and not str(result.global_id).startswith("PROVISIONAL_"):
            self.repo.add_event(
                {
                    "event_id": f"ev_{uuid.uuid4().hex[:10]}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": f"ASSOCIATED_{result.decision.value}",
                    "global_id": result.global_id,
                    "camera_id": tl.camera_id,
                    "metadata": {"reason": result.reason, "confidence": result.confidence},
                }
            )

    def _create_identity(
        self,
        tl: Tracklet,
        now: float,
        verification: VerificationStatus,
    ) -> str:
        gid = self.repo.next_global_id(self.id_prefix, self.id_width)
        # Ensure uniqueness against in-memory map if repo restarted sequence.
        while gid in self._identities:
            gid = self.repo.next_global_id(self.id_prefix, self.id_width)
        iso = datetime.now(timezone.utc).isoformat()
        rec = IdentityRecord(
            global_id=gid,
            activity_status=IdentityActivityStatus.ACTIVE,
            verification_status=verification,
            first_seen=tl.start_time,
            last_seen=tl.end_time,
            last_camera_id=tl.camera_id,
            cameras_seen=[tl.camera_id],
            created_at=iso,
            updated_at=iso,
        )
        self._identities[gid] = rec
        self._persist_identity(rec)
        if tl.embedding is not None:
            self._add_embedding(gid, tl, confirmed=True)
        logger.info("Created global identity %s from %s", gid, tl.compound_key)
        return gid

    def _touch_identity(
        self,
        gid: str,
        tl: Tracklet,
        now: float,
        verification: VerificationStatus,
    ) -> None:
        rec = self._identities.get(gid)
        if rec is None:
            iso = datetime.now(timezone.utc).isoformat()
            rec = IdentityRecord(
                global_id=gid,
                activity_status=IdentityActivityStatus.ACTIVE,
                verification_status=verification,
                first_seen=tl.start_time,
                last_seen=tl.end_time,
                last_camera_id=tl.camera_id,
                cameras_seen=[tl.camera_id],
                created_at=iso,
                updated_at=iso,
            )
            self._identities[gid] = rec
        else:
            rec.last_seen = tl.end_time
            rec.last_camera_id = tl.camera_id
            rec.activity_status = IdentityActivityStatus.ACTIVE
            if tl.camera_id not in rec.cameras_seen:
                rec.cameras_seen.append(tl.camera_id)
            if verification == VerificationStatus.PROVISIONAL and rec.verification_status == VerificationStatus.UNVERIFIED:
                rec.verification_status = VerificationStatus.PROVISIONAL
            rec.updated_at = datetime.now(timezone.utc).isoformat()
        self._persist_identity(rec)

    def _add_embedding(self, gid: str, tl: Tracklet, *, confirmed: bool) -> None:
        assert tl.embedding is not None
        emb_id = f"emb_{uuid.uuid4().hex[:10]}"
        quality = float(np.mean([o.quality_score for o in tl.observations if o.embedding is not None]) or 0.5)
        self.gallery.add(
            gid,
            tl.embedding,
            camera_id=tl.camera_id,
            timestamp=tl.end_time,
            quality_score=quality,
            embedding_id=emb_id,
            confirmed=confirmed,
        )
        self.repo.add_embedding(
            {
                "embedding_id": emb_id,
                "global_id": gid,
                "model_version": self.gallery.model_version,
                "embedding": tl.embedding.tolist(),
                "camera_id": tl.camera_id,
                "timestamp": tl.end_time,
                "quality_score": quality,
                "verification_source": "reid",
                "active": confirmed,
            }
        )

    def _persist_identity(self, rec: IdentityRecord) -> None:
        self.repo.upsert_identity(
            {
                "global_id": rec.global_id,
                "identity_status": rec.activity_status.value,
                "verification_status": rec.verification_status.value,
                "first_seen": datetime.fromtimestamp(rec.first_seen, tz=timezone.utc).isoformat()
                if rec.first_seen > 1e6
                else datetime.now(timezone.utc).isoformat(),
                "last_seen": datetime.fromtimestamp(rec.last_seen, tz=timezone.utc).isoformat()
                if rec.last_seen > 1e6
                else datetime.now(timezone.utc).isoformat(),
                "first_seen_ts": rec.first_seen,
                "last_seen_ts": rec.last_seen,
                "last_camera_id": rec.last_camera_id,
                "cameras_seen": rec.cameras_seen,
                "created_at": rec.created_at,
                "updated_at": rec.updated_at,
                "display_name": rec.display_name,
            }
        )

    def _refresh_activity(self, now: float) -> None:
        for rec in self._identities.values():
            observed = any(
                b.global_id == rec.global_id and b.status in {"confirmed", "provisional"}
                for b in self._bindings.values()
            )
            rec.activity_status = next_activity_status(
                rec.activity_status,
                observed=observed,
                seconds_since_seen=max(0.0, now - rec.last_seen),
            )

    def release_track(self, compound_key: str) -> None:
        with self._lock:
            self._bindings.pop(compound_key, None)

    # ------------------------------------------------------------------
    # Live (per-track) identity helpers used by LiveIdentityResolver.
    # Timestamps passed here are wall-clock epoch seconds.
    # ------------------------------------------------------------------
    def create_live_identity(
        self,
        camera_id: str,
        timestamp: float,
        *,
        verification: VerificationStatus = VerificationStatus.UNVERIFIED,
        reason: str = "",
    ) -> str:
        with self._lock:
            gid = self.repo.next_global_id(self.id_prefix, self.id_width)
            while gid in self._identities:
                gid = self.repo.next_global_id(self.id_prefix, self.id_width)
            iso = datetime.now(timezone.utc).isoformat()
            rec = IdentityRecord(
                global_id=gid,
                activity_status=IdentityActivityStatus.ACTIVE,
                verification_status=verification,
                first_seen=timestamp,
                last_seen=timestamp,
                last_camera_id=camera_id,
                cameras_seen=[camera_id],
                created_at=iso,
                updated_at=iso,
            )
            self._identities[gid] = rec
            self._persist_identity(rec)
        self.record_event("IDENTITY_CREATED", gid, camera_id, {"reason": reason})
        logger.info("Created global identity %s on %s (%s)", gid, camera_id, reason)
        return gid

    def touch_live_identity(self, gid: str, camera_id: str, timestamp: float, *, persist: bool = True) -> None:
        with self._lock:
            rec = self._identities.get(gid)
            if rec is None:
                return
            rec.last_seen = max(rec.last_seen, timestamp)
            rec.last_camera_id = camera_id
            rec.activity_status = IdentityActivityStatus.ACTIVE
            if camera_id not in rec.cameras_seen:
                rec.cameras_seen.append(camera_id)
                persist = True
            rec.updated_at = datetime.now(timezone.utc).isoformat()
            if persist:
                self._persist_identity(rec)

    def add_live_embedding(
        self,
        gid: str,
        embedding: np.ndarray,
        *,
        camera_id: str,
        timestamp: float,
        quality: float,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Add a confirmed embedding to the gallery; optionally persist it now.

        Returns the repository record so callers can persist it later.
        """
        emb_id = f"emb_{uuid.uuid4().hex[:10]}"
        self.gallery.add(
            gid,
            embedding,
            camera_id=camera_id,
            timestamp=timestamp,
            quality_score=quality,
            embedding_id=emb_id,
            confirmed=True,
        )
        record = {
            "embedding_id": emb_id,
            "global_id": gid,
            "model_version": self.gallery.model_version,
            "embedding": np.asarray(embedding, dtype=np.float32).tolist(),
            "camera_id": camera_id,
            "timestamp": timestamp,
            "quality_score": quality,
            "verification_source": "reid",
            "active": True,
        }
        if persist:
            self.repo.add_embedding(record)
        return record

    def add_face_embedding(
        self,
        gid: str,
        embedding: np.ndarray,
        *,
        camera_id: str,
        timestamp: float,
        quality: float,
        persist: bool = True,
        source: str = ATTR_SOURCE,
        model_version: str = "clip-attr-v1",
    ) -> bool:
        """Store an identity-anchor vector (CLIP attributes or face). No images. Returns False if redundant."""
        if source not in ANCHOR_SOURCES:
            source = ATTR_SOURCE
        emb_id = f"{'face' if source == FACE_SOURCE else 'attr'}_{uuid.uuid4().hex[:10]}"
        if not self.face_gallery.add(gid, embedding, camera_id=camera_id, timestamp=timestamp,
                                     quality=quality, embedding_id=emb_id):
            return False
        if persist:
            self.repo.add_embedding({
                "embedding_id": emb_id,
                "global_id": gid,
                "model_version": model_version,
                "embedding": np.asarray(embedding, dtype=np.float32).tolist(),
                "camera_id": camera_id,
                "timestamp": timestamp,
                "quality_score": quality,
                "verification_source": source,
                "active": True,
            })
        return True

    def persist_embedding_record(self, record: dict[str, Any]) -> None:
        self.repo.add_embedding(record)

    def record_event(self, event_type: str, gid: str | None, camera_id: str | None, metadata: dict[str, Any]) -> None:
        try:
            self.repo.add_event(
                {
                    "event_id": f"ev_{uuid.uuid4().hex[:10]}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": event_type,
                    "global_id": gid,
                    "camera_id": camera_id,
                    "metadata": metadata,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Event persist failed (%s): %s", event_type, exc)

    def record_tracklet(self, record: dict[str, Any], association: dict[str, Any] | None = None) -> None:
        try:
            self.repo.add_tracklet(record)
            if association is not None:
                self.repo.add_association(association)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tracklet persist failed: %s", exc)

    def archive_identity(self, gid: str, *, merged_into: str | None = None, reason: str = "") -> None:
        """Retire an identity (e.g. a duplicate created moments ago). Reversible via audit log."""
        with self._lock:
            rec = self._identities.get(gid)
            if rec is None:
                return
            rec.activity_status = IdentityActivityStatus.ARCHIVED
            rec.updated_at = datetime.now(timezone.utc).isoformat()
            self._persist_identity(rec)
            target = self._identities.get(merged_into) if merged_into else None
            if target is not None:
                for cam in rec.cameras_seen:
                    if cam not in target.cameras_seen:
                        target.cameras_seen.append(cam)
                target.first_seen = min(target.first_seen, rec.first_seen)
                self._persist_identity(target)
        self.record_event(
            "IDENTITY_MERGED" if merged_into else "IDENTITY_ARCHIVED",
            gid,
            None,
            {"merged_into": merged_into, "reason": reason},
        )

    def merge_identity_into(self, source_gid: str, target_gid: str, *, reason: str = "") -> int:
        """Merge an established duplicate: move gallery + stored embeddings, archive the source."""
        if source_gid == target_gid or not self.is_live_identity(source_gid) or not self.is_live_identity(target_gid):
            return 0
        moved = self.gallery.reassign_identity(source_gid, target_gid)
        self.face_gallery.reassign(source_gid, target_gid)
        reassign = getattr(self.repo, "reassign_embeddings", None)
        if reassign is not None:
            try:
                reassign(source_gid, target_gid)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Embedding reassignment %s -> %s failed: %s", source_gid, target_gid, exc)
        self.archive_identity(source_gid, merged_into=target_gid, reason=reason)
        return moved

    def is_live_identity(self, gid: str) -> bool:
        with self._lock:
            rec = self._identities.get(gid)
            return rec is not None and rec.activity_status != IdentityActivityStatus.ARCHIVED

    def identity_last_seen(self, gid: str) -> tuple[str | None, float | None]:
        with self._lock:
            rec = self._identities.get(gid)
            if rec is None:
                return None, None
            return rec.last_camera_id, rec.last_seen

    def refresh_activity(self, now: float) -> None:
        with self._lock:
            self._refresh_activity(now)

    def set_unresolved_count(self, count: int) -> None:
        self.unresolved_count = int(count)

    def stats(self) -> dict[str, int]:
        with self._lock:
            active = sum(
                1
                for r in self._identities.values()
                if r.activity_status in {
                    IdentityActivityStatus.ACTIVE,
                    IdentityActivityStatus.TEMPORARILY_LOST,
                    IdentityActivityStatus.PROVISIONAL,
                }
            )
            return {
                "active_global_identities": active,
                "total_retained_identities": len(self._identities),
                "unresolved_tracks": self.unresolved_count,
            }

    def list_identities(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "global_id": r.global_id,
                    "status": r.activity_status.value,
                    "verification_status": r.verification_status.value,
                    "first_seen": r.first_seen,
                    "last_seen": r.last_seen,
                    "current_camera": r.last_camera_id,
                    "previous_cameras": r.cameras_seen,
                    "display_name": r.display_name,
                }
                for r in sorted(self._identities.values(), key=lambda x: x.global_id)
            ]

    def identity_detail(self, global_id: str) -> dict[str, Any] | None:
        with self._lock:
            rec = self._identities.get(global_id)
            if rec is None:
                row = self.repo.get_identity(global_id)
                return row
            events = self.repo.list_events(global_id, limit=100)
            return {
                "global_id": rec.global_id,
                "status": rec.activity_status.value,
                "verification_status": rec.verification_status.value,
                "first_seen": rec.first_seen,
                "last_seen": rec.last_seen,
                "current_camera": rec.last_camera_id,
                "previous_cameras": rec.cameras_seen,
                "confidence": None,
                "identity_evidence": "appearance_reid",
                "events": events,
                "display_name": rec.display_name,
            }

    def manual_correct(self, tracklet_or_key: str, global_id: str) -> None:
        """Manual operator correction — reversible via audit event."""
        with self._lock:
            self._bindings[tracklet_or_key] = Binding(
                compound_key=tracklet_or_key,
                global_id=global_id,
                status="confirmed",
                confidence=1.0,
                updated_at=time.monotonic(),
                camera_id=tracklet_or_key.split(":", 1)[0],
            )
            self.repo.add_event(
                {
                    "event_id": f"ev_{uuid.uuid4().hex[:10]}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "MANUAL_CORRECTION",
                    "global_id": global_id,
                    "camera_id": tracklet_or_key.split(":", 1)[0],
                    "metadata": {"compound_key": tracklet_or_key},
                }
            )
