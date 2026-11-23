"""Global cross-camera association engine.

Conservative matching: uncertain evidence yields PROVISIONAL/UNKNOWN, never a forced merge.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from backend.association.camera_topology import CameraTopology
from backend.association.identity_constraints import ConstraintContext, IdentityConstraints
from backend.association.matching_cost import CostWeights, association_score, similarity_to_cost
from backend.reid.embedding_gallery import EmbeddingGallery, SearchHit
from backend.types import AssociationDecision, AssociationResult, Tracklet

logger = logging.getLogger(__name__)


@dataclass
class AssociatorConfig:
    match_threshold: float = 0.55
    strong_match_threshold: float = 0.65
    match_margin: float = 0.08
    new_person_threshold: float = 0.30
    long_gap_caution_seconds: float = 1800.0
    long_gap_threshold_boost: float = 0.08
    appearance_weight: float = 0.70
    temporal_weight: float = 0.15
    topology_weight: float = 0.15


@dataclass
class IdentityHint:
    global_id: str
    camera_id: str
    last_seen: float
    activity_status: str
    verification_status: str


class GlobalAssociator:
    def __init__(
        self,
        gallery: EmbeddingGallery,
        topology: CameraTopology,
        config: AssociatorConfig | None = None,
    ) -> None:
        self.gallery = gallery
        self.topology = topology
        self.config = config or AssociatorConfig()
        self.constraints = IdentityConstraints(topology)
        self.weights = CostWeights(
            appearance=self.config.appearance_weight,
            temporal=self.config.temporal_weight,
            topology=self.config.topology_weight,
        )
        self.last_latency_ms = 0.0
        self._ctx = ConstraintContext()

    def set_active_observations(self, ctx: ConstraintContext) -> None:
        self._ctx = ctx

    def associate(
        self,
        tracklets: list[Tracklet],
        identity_hints: dict[str, IdentityHint] | None = None,
    ) -> list[AssociationResult]:
        t0 = time.perf_counter()
        identity_hints = identity_hints or {}
        if not tracklets:
            self.last_latency_ms = 0.0
            return []

        # Stage A/B/C: build candidate scores per tracklet.
        candidate_maps: list[dict[str, tuple[float, dict[str, Any]]]] = []
        for tl in tracklets:
            candidate_maps.append(self._candidates_for_tracklet(tl, identity_hints))

        # Stage D: global bipartite assignment over union of candidate IDs.
        all_ids: list[str] = sorted({gid for cmap in candidate_maps for gid in cmap})
        results: list[AssociationResult | None] = [None] * len(tracklets)

        if all_ids:
            cost = np.ones((len(tracklets), len(all_ids)), dtype=np.float64)
            valid = np.zeros_like(cost, dtype=bool)
            for i, cmap in enumerate(candidate_maps):
                for j, gid in enumerate(all_ids):
                    if gid in cmap:
                        score, _ = cmap[gid]
                        cost[i, j] = similarity_to_cost(score)
                        valid[i, j] = True

            # Pad with dummy columns so tracklets may remain unassigned.
            n_dummy = len(tracklets)
            if n_dummy > 0:
                dummy = np.full((len(tracklets), n_dummy), 0.85, dtype=np.float64)
                cost = np.hstack([cost, dummy])
                valid = np.hstack([valid, np.zeros((len(tracklets), n_dummy), dtype=bool)])

            rows, cols = linear_sum_assignment(cost)
            assigned_gids: set[str] = set()
            for r, c in zip(rows, cols):
                if c >= len(all_ids):
                    continue  # dummy / unassigned
                if not valid[r, c]:
                    continue
                gid = all_ids[c]
                if gid in assigned_gids:
                    continue
                score, evidence = candidate_maps[r][gid]
                decision, conf, reason = self._verify(score, evidence, candidate_maps[r], gid)
                if decision in {AssociationDecision.CONFIRMED, AssociationDecision.PROVISIONAL}:
                    assigned_gids.add(gid)
                    results[r] = AssociationResult(
                        tracklet_id=tracklets[r].tracklet_id,
                        decision=decision,
                        global_id=gid,
                        confidence=conf,
                        evidence=evidence,
                        reason=reason,
                    )

        # Stage E/F: unresolved tracklets -> NEW or UNKNOWN/PROVISIONAL local.
        out: list[AssociationResult] = []
        for i, tl in enumerate(tracklets):
            if results[i] is not None:
                out.append(results[i])  # type: ignore[arg-type]
                continue
            cmap = candidate_maps[i]
            best = max(cmap.values(), key=lambda x: x[0]) if cmap else None
            best_score = best[0] if best else 0.0
            if best_score < self.config.new_person_threshold:
                out.append(
                    AssociationResult(
                        tracklet_id=tl.tracklet_id,
                        decision=AssociationDecision.NEW,
                        global_id=None,
                        confidence=1.0 - best_score,
                        evidence={"best_score": best_score},
                        reason="no_close_gallery_match",
                    )
                )
            else:
                out.append(
                    AssociationResult(
                        tracklet_id=tl.tracklet_id,
                        decision=AssociationDecision.UNKNOWN,
                        global_id=None,
                        confidence=best_score,
                        evidence={"best_score": best_score, "candidates": list(cmap.keys())},
                        reason="ambiguous_or_below_accept",
                    )
                )

        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return out

    def _candidates_for_tracklet(
        self,
        tl: Tracklet,
        identity_hints: dict[str, IdentityHint],
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        if tl.embedding is None:
            return {}

        hits = self.gallery.search(tl.embedding, top_k=8, now=tl.end_time)
        # Also search long-term (include older) already covered by default search.
        scored: dict[str, tuple[float, dict[str, Any]]] = {}
        for hit in hits:
            ok, reason = self.constraints.can_assign(
                global_id=hit.global_id,
                camera_id=tl.camera_id,
                timestamp=tl.end_time,
                ctx=self._ctx,
                compound_key=tl.compound_key,
            )
            if not ok:
                continue

            hint = identity_hints.get(hit.global_id)
            src_cam = hint.camera_id if hint else hit.camera_id
            last_ts = hint.last_seen if hint else hit.timestamp
            delta = tl.end_time - last_ts
            score, evidence = association_score(
                appearance_similarity=hit.score,
                src_camera=src_cam,
                dst_camera=tl.camera_id,
                delta_seconds=delta,
                topology=self.topology,
                weights=self.weights,
                long_gap_caution_seconds=self.config.long_gap_caution_seconds,
            )
            # Long-gap caution: require stronger appearance.
            if abs(delta) >= self.config.long_gap_caution_seconds:
                evidence["long_gap_caution"] = True
            evidence["constraint"] = reason
            evidence["gallery_score"] = hit.score
            scored[hit.global_id] = (score, evidence)
        return scored

    def _verify(
        self,
        score: float,
        evidence: dict[str, Any],
        all_scores: dict[str, tuple[float, dict[str, Any]]],
        gid: str,
    ) -> tuple[AssociationDecision, float, str]:
        ranked = sorted(all_scores.items(), key=lambda kv: kv[1][0], reverse=True)
        best_gid, (best_score, _) = ranked[0]
        second = ranked[1][1][0] if len(ranked) > 1 else 0.0
        margin = best_score - second

        threshold = self.config.match_threshold
        if evidence.get("long_gap_caution"):
            threshold += self.config.long_gap_threshold_boost

        if best_gid != gid:
            return AssociationDecision.REJECTED, score, "not_best_after_assignment"

        if best_score >= self.config.strong_match_threshold and margin >= self.config.match_margin:
            return AssociationDecision.CONFIRMED, best_score, "strong_match"

        if best_score >= threshold and margin >= self.config.match_margin:
            return AssociationDecision.CONFIRMED, best_score, "accepted_match"

        # Near-threshold but ambiguous: keep as provisional *candidate* for an existing ID.
        if best_score >= threshold:
            return AssociationDecision.PROVISIONAL, best_score, "margin_insufficient"

        # Below acceptance threshold: never attach to an existing identity.
        # Caller will emit UNKNOWN or NEW.
        return AssociationDecision.REJECTED, best_score, "below_confirm_threshold"
