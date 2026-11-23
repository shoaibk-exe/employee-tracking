"""Per-identity face embedding memory (ArcFace vectors, never face images)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FaceEntry:
    embedding: np.ndarray
    camera_id: str
    timestamp: float
    quality: float
    embedding_id: str | None = None


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class FaceGallery:
    def __init__(self, max_per_identity: int = 24, diversity_max_similarity: float = 0.92) -> None:
        self.max_per_identity = max_per_identity
        self.diversity_max_similarity = diversity_max_similarity
        self._lock = threading.RLock()
        self._entries: dict[str, list[FaceEntry]] = {}
        self._mats: dict[str, np.ndarray] | None = None

    def add(self, gid: str, embedding: np.ndarray, *, camera_id: str, timestamp: float, quality: float,
            embedding_id: str | None = None, force: bool = False) -> bool:
        """Store a face sample; skipped when it adds no new view (near-duplicate)."""
        v = _unit(embedding)
        with self._lock:
            entries = self._entries.setdefault(gid, [])
            if entries and not force:
                sims = np.stack([e.embedding for e in entries]) @ v
                if float(sims.max()) >= self.diversity_max_similarity:
                    return False
            entries.append(FaceEntry(v, camera_id, timestamp, quality, embedding_id))
            if len(entries) > self.max_per_identity:
                entries.sort(key=lambda e: (e.quality, e.timestamp))
                del entries[0]
            self._mats = None
            return True

    def load(self, rows: list[dict[str, Any]]) -> int:
        n = 0
        for row in rows:
            if not row.get("active", True):
                continue
            self.add(
                str(row["global_id"]), np.asarray(row["embedding"], dtype=np.float32),
                camera_id=str(row.get("camera_id") or ""), timestamp=float(row.get("timestamp") or 0.0),
                quality=float(row.get("quality_score") or 0.0), embedding_id=row.get("embedding_id"), force=True,
            )
            n += 1
        return n

    def matrices(self) -> dict[str, np.ndarray]:
        with self._lock:
            if self._mats is None:
                self._mats = {g: np.stack([e.embedding for e in es]) for g, es in self._entries.items() if es}
            return self._mats

    def score(self, queries: np.ndarray, candidates: set[str] | None = None) -> dict[str, float]:
        """Mean over query faces of the best cosine to each identity's stored faces."""
        q = np.asarray(queries, dtype=np.float32).reshape(-1, np.asarray(queries).shape[-1])
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        out: dict[str, float] = {}
        for gid, mat in self.matrices().items():
            if candidates is not None and gid not in candidates:
                continue
            out[gid] = float((q @ mat.T).max(axis=1).mean())
        return out

    def count(self, gid: str) -> int:
        with self._lock:
            return len(self._entries.get(gid, []))

    def identities(self) -> set[str]:
        with self._lock:
            return {g for g, es in self._entries.items() if es}

    def reassign(self, source_gid: str, target_gid: str) -> int:
        with self._lock:
            moved = self._entries.pop(source_gid, [])
            if moved:
                self._entries.setdefault(target_gid, []).extend(moved)
                es = self._entries[target_gid]
                if len(es) > self.max_per_identity:
                    es.sort(key=lambda e: (e.quality, e.timestamp))
                    del es[: len(es) - self.max_per_identity]
                self._mats = None
            return len(moved)
