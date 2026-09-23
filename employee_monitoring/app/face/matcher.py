"""Cosine matching against a multi-template gallery.

One threshold is not enough. Ali at 0.69 and Ahmed at 0.68 must stay UNKNOWN:
a wrong attendance mark is worse than a missed frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import MatchConfig


@dataclass
class MatchResult:
    employee_id: str | None
    similarity_score: float
    second_best_score: float
    recognition_confidence: float
    accepted: bool


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return vec
    return vec / norm


def match_embedding(
    embedding: np.ndarray,
    gallery: dict[str, np.ndarray],
    rules: MatchConfig,
) -> MatchResult:
    query = l2_normalize(embedding)
    scored: list[tuple[str, float]] = []
    for employee_id, templates in gallery.items():
        matrix = np.asarray(templates, dtype=np.float32)
        if matrix.size == 0:
            continue
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        matrix = matrix / norms
        best_template = float(np.max(matrix @ query))
        scored.append((employee_id, best_template))
    scored.sort(key=lambda item: item[1], reverse=True)
    if not scored:
        return MatchResult(None, 0.0, 0.0, 0.0, False)
    best_id, best_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    margin = best_score - second_score
    # A single enrolled employee has no rival identity, so margin is not meaningful.
    margin_ok = len(scored) == 1 or margin >= rules.min_margin
    accepted = best_score >= rules.min_similarity and margin_ok
    if not accepted:
        return MatchResult(None, best_score, second_score, 0.0, False)
    return MatchResult(
        employee_id=best_id,
        similarity_score=best_score,
        second_best_score=second_score,
        recognition_confidence=best_score,
        accepted=True,
    )
