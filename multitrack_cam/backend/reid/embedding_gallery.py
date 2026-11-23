"""Short-term and long-term appearance galleries with cosine search.

Uses NumPy brute-force search by default. Optionally uses FAISS when installed.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.utils.geometry import cosine_similarity, l2_normalize

logger = logging.getLogger(__name__)


@dataclass
class GalleryEntry:
    global_id: str
    embedding: np.ndarray
    camera_id: str
    timestamp: float
    quality_score: float
    model_version: str
    verification_source: str = "reid"
    active: bool = True
    embedding_id: str | None = None


@dataclass
class SearchHit:
    global_id: str
    score: float
    embedding_id: str | None
    camera_id: str
    timestamp: float


class EmbeddingGallery:
    def __init__(
        self,
        *,
        max_embeddings_per_identity: int = 32,
        short_term_window_seconds: float = 600.0,
        model_version: str = "reid-v1",
    ) -> None:
        self.max_embeddings_per_identity = max_embeddings_per_identity
        self.short_term_window_seconds = short_term_window_seconds
        self.model_version = model_version
        self._lock = threading.RLock()
        self._entries: list[GalleryEntry] = []
        self._faiss_index = None
        self._faiss_ids: list[str] = []
        self._try_init_faiss()

    def _try_init_faiss(self) -> None:
        try:
            import faiss  # type: ignore

            self._faiss = faiss
            self._use_faiss = True
            logger.info("FAISS available for gallery search")
        except Exception:  # noqa: BLE001
            self._faiss = None
            self._use_faiss = False

    def add(
        self,
        global_id: str,
        embedding: np.ndarray,
        *,
        camera_id: str,
        timestamp: float,
        quality_score: float,
        verification_source: str = "reid",
        embedding_id: str | None = None,
        confirmed: bool = True,
    ) -> None:
        """Add embedding. Unconfirmed/provisional embeddings should pass confirmed=False
        so callers can keep them out of the long-term confirmed set if desired.
        """
        if embedding is None:
            return
        if not confirmed:
            # Keep provisional samples only in a soft short-term bank tagged inactive=False.
            active = False
        else:
            active = True
        entry = GalleryEntry(
            global_id=global_id,
            embedding=l2_normalize(embedding),
            camera_id=camera_id,
            timestamp=timestamp,
            quality_score=float(quality_score),
            model_version=self.model_version,
            verification_source=verification_source,
            active=active,
            embedding_id=embedding_id,
        )
        with self._lock:
            self._entries.append(entry)
            self._invalidate()
            self._prune_identity(global_id)
            self._rebuild_faiss_if_needed()

    def _prune_identity(self, global_id: str) -> None:
        idxs = [i for i, e in enumerate(self._entries) if e.global_id == global_id and e.active]
        if len(idxs) <= self.max_embeddings_per_identity:
            return
        # Drop lowest quality among oldest half.
        ranked = sorted(idxs, key=lambda i: (self._entries[i].quality_score, self._entries[i].timestamp))
        drop = set(ranked[: len(idxs) - self.max_embeddings_per_identity])
        self._entries = [e for i, e in enumerate(self._entries) if i not in drop]
        self._invalidate()

    def search(
        self,
        embedding: np.ndarray,
        *,
        top_k: int = 5,
        now: float | None = None,
        short_term_only: bool = False,
        include_inactive: bool = False,
    ) -> list[SearchHit]:
        if embedding is None:
            return []
        query = l2_normalize(embedding)
        now = time.monotonic() if now is None else now
        with self._lock:
            candidates = []
            for e in self._entries:
                if not include_inactive and not e.active:
                    continue
                if short_term_only and (now - e.timestamp) > self.short_term_window_seconds:
                    continue
                candidates.append(e)
            if not candidates:
                return []

            # Aggregate best score per global_id.
            best: dict[str, SearchHit] = {}
            for e in candidates:
                score = cosine_similarity(query, e.embedding)
                prev = best.get(e.global_id)
                if prev is None or score > prev.score:
                    best[e.global_id] = SearchHit(
                        global_id=e.global_id,
                        score=score,
                        embedding_id=e.embedding_id,
                        camera_id=e.camera_id,
                        timestamp=e.timestamp,
                    )
            hits = sorted(best.values(), key=lambda h: h.score, reverse=True)
            return hits[:top_k]

    def _signature(self) -> tuple[int, int]:
        # Entries are only mutated via add/load/prune/remove/reassign, all of
        # which change the list identity, its length, or call _invalidate().
        return (id(self._entries), len(self._entries))

    def _invalidate(self) -> None:
        self._matrix_cache = None

    def identity_matrices(self, *, include_inactive: bool = False) -> dict[str, np.ndarray]:
        """Return {global_id: (n, d) matrix of L2-normalized embeddings} (cached)."""
        with self._lock:
            cache = getattr(self, "_matrix_cache", None)
            key = (self._signature(), include_inactive)
            if cache is not None and cache[0] == key:
                return cache[1]
            groups: dict[str, list[np.ndarray]] = {}
            for e in self._entries:
                if not include_inactive and not e.active:
                    continue
                groups.setdefault(e.global_id, []).append(e.embedding)
            mats = {gid: np.stack(v).astype(np.float32) for gid, v in groups.items()}
            self._matrix_cache = (key, mats)
            return mats

    def camera_matrices(self) -> dict[str, dict[str, np.ndarray]]:
        """Active embeddings grouped as {camera_id: {global_id: (n, d) matrix}}."""
        with self._lock:
            groups: dict[str, dict[str, list[np.ndarray]]] = {}
            for e in self._entries:
                if e.active:
                    groups.setdefault(e.camera_id, {}).setdefault(e.global_id, []).append(e.embedding)
            return {
                cam: {gid: np.stack(v).astype(np.float32) for gid, v in by_gid.items()}
                for cam, by_gid in groups.items()
            }

    def identity_cameras(self) -> dict[str, set[str]]:
        """Cameras that contributed active embeddings to each identity."""
        with self._lock:
            out: dict[str, set[str]] = {}
            for e in self._entries:
                if e.active:
                    out.setdefault(e.global_id, set()).add(e.camera_id)
            return out

    def _prototypes(self) -> dict[str, np.ndarray]:
        with self._lock:
            mats = self.identity_matrices()
            cache = getattr(self, "_proto_cache", None)
            if cache is not None and cache[0] is mats:
                return cache[1]
            protos = {}
            for gid, mat in mats.items():
                p = mat.mean(axis=0)
                protos[gid] = p / max(float(np.linalg.norm(p)), 1e-12)
            self._proto_cache = (mats, protos)
            return protos

    def score_bank(
        self,
        queries: np.ndarray,
        *,
        candidates: set[str] | None = None,
        max_weight: float = 0.5,
    ) -> dict[str, float]:
        """Robust bank-vs-identity similarity.

        ``queries`` is an (m, d) bank of embeddings from one live track. Score =
        max_weight * mean_q(max_e cos(q, e)) + (1 - max_weight) * cos(mean(q), mean(e)).
        The first term rewards matching any stored viewpoint, the second term
        penalizes identities that only match one lucky sample.
        """
        q = np.asarray(queries, dtype=np.float32).reshape(-1, queries.shape[-1])
        if q.size == 0:
            return {}
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        q_proto = q.mean(axis=0)
        q_proto /= max(float(np.linalg.norm(q_proto)), 1e-12)
        out: dict[str, float] = {}
        mats = self.identity_matrices()
        protos = self._prototypes()
        for gid, mat in mats.items():
            if candidates is not None and gid not in candidates:
                continue
            sims = q @ mat.T
            best_per_q = float(sims.max(axis=1).mean())
            out[gid] = max_weight * best_per_q + (1.0 - max_weight) * float(q_proto @ protos[gid])
        return out

    def max_similarity(self, global_id: str, embedding: np.ndarray) -> float:
        """Highest cosine between ``embedding`` and any active entry of ``global_id``."""
        mat = self.identity_matrices().get(global_id)
        if mat is None:
            return -1.0
        return float((mat @ l2_normalize(embedding)).max())

    def reassign_identity(self, source_gid: str, target_gid: str) -> int:
        """Move all entries of ``source_gid`` to ``target_gid``. Returns count moved."""
        moved = 0
        with self._lock:
            for e in self._entries:
                if e.global_id == source_gid:
                    e.global_id = target_gid
                    moved += 1
            self._invalidate()
            self._prune_identity(target_gid)
        return moved

    def remove_identity(self, global_id: str) -> None:
        with self._lock:
            self._entries = [e for e in self._entries if e.global_id != global_id]
            self._invalidate()
            self._rebuild_faiss_if_needed()

    def identities(self) -> set[str]:
        with self._lock:
            return {e.global_id for e in self._entries if e.active}

    def export_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "global_id": e.global_id,
                    "embedding": e.embedding.tolist(),
                    "camera_id": e.camera_id,
                    "timestamp": e.timestamp,
                    "quality_score": e.quality_score,
                    "model_version": e.model_version,
                    "verification_source": e.verification_source,
                    "active": e.active,
                    "embedding_id": e.embedding_id,
                }
                for e in self._entries
            ]

    def load_entries(self, rows: list[dict[str, Any]]) -> None:
        with self._lock:
            self._entries = []
            self._invalidate()
            for row in rows:
                self._entries.append(
                    GalleryEntry(
                        global_id=str(row["global_id"]),
                        embedding=l2_normalize(np.asarray(row["embedding"], dtype=np.float32)),
                        camera_id=str(row.get("camera_id", "")),
                        timestamp=float(row.get("timestamp", 0.0)),
                        quality_score=float(row.get("quality_score", 0.0)),
                        model_version=str(row.get("model_version", self.model_version)),
                        verification_source=str(row.get("verification_source", "reid")),
                        active=bool(row.get("active", True)),
                        embedding_id=row.get("embedding_id"),
                    )
                )
            self._rebuild_faiss_if_needed()

    def _rebuild_faiss_if_needed(self) -> None:
        # Brute-force NumPy is fine for typical CCTV gallery sizes; FAISS reserved for scale.
        return
