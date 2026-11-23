"""Long-term re-identification helpers after extended absences."""

from __future__ import annotations

from dataclasses import dataclass

from backend.reid.embedding_gallery import EmbeddingGallery, SearchHit


@dataclass
class RecoveryPolicy:
    long_gap_seconds: float = 1800.0
    day_gap_seconds: float = 20 * 3600.0
    match_threshold: float = 0.55
    long_gap_boost: float = 0.08
    day_gap_boost: float = 0.12
    match_margin: float = 0.08


def adjusted_threshold(delta_seconds: float, policy: RecoveryPolicy) -> float:
    thr = policy.match_threshold
    if delta_seconds >= policy.day_gap_seconds:
        return thr + policy.day_gap_boost
    if delta_seconds >= policy.long_gap_seconds:
        return thr + policy.long_gap_boost
    return thr


def recover_candidates(
    gallery: EmbeddingGallery,
    embedding,
    *,
    now: float,
    policy: RecoveryPolicy | None = None,
) -> list[SearchHit]:
    """Search retained identities for a returning person. Conservative filtering only."""
    policy = policy or RecoveryPolicy()
    hits = gallery.search(embedding, top_k=10, now=now, short_term_only=False)
    if not hits:
        return []
    best = hits[0]
    second = hits[1].score if len(hits) > 1 else 0.0
    delta = now - best.timestamp
    thr = adjusted_threshold(delta, policy)
    if best.score < thr:
        return []
    if best.score - second < policy.match_margin:
        return []  # ambiguity: leave unresolved
    return hits
