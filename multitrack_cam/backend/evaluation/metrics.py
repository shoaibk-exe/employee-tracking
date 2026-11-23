"""Lightweight identity evaluation helpers.

Full HOTA/IDF1 on video require labeled MOT-style ground truth. This module
computes assignment-level metrics from event logs for regression tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class EvalSummary:
    total_assignments: int
    false_merges: int
    recovered_returns: int
    provisional_count: int
    notes: str


def evaluate_identity_events(
    predicted: Iterable[dict],
    ground_truth: Iterable[dict],
) -> EvalSummary:
    """Compare predicted {tracklet_id -> global_id} against GT person labels.

    predicted/ground_truth items: {"tracklet_id", "person_label" or "global_id"}
    """
    gt = {str(x["tracklet_id"]): str(x.get("person_label") or x.get("global_id")) for x in ground_truth}
    pred = {str(x["tracklet_id"]): str(x.get("global_id")) for x in predicted if x.get("global_id")}

    # Build mapping from predicted global -> set of GT labels.
    clusters: dict[str, set[str]] = {}
    for tid, gid in pred.items():
        if tid not in gt:
            continue
        clusters.setdefault(gid, set()).add(gt[tid])

    false_merges = sum(1 for labels in clusters.values() if len(labels) > 1)
    provisional_count = sum(1 for gid in pred.values() if str(gid).startswith("PROVISIONAL_"))

    # Returning person recovery: same GT label mapped to same predicted gid across >1 tracklets.
    by_gt: dict[str, set[str]] = {}
    for tid, label in gt.items():
        if tid in pred:
            by_gt.setdefault(label, set()).add(pred[tid])
    recovered = sum(1 for gids in by_gt.values() if len(gids) == 1 and len([t for t, l in gt.items() if l in by_gt and t in pred]) >= 1)

    return EvalSummary(
        total_assignments=len(pred),
        false_merges=false_merges,
        recovered_returns=recovered,
        provisional_count=provisional_count,
        notes="Use labeled CCTV clips for HOTA/IDF1; this summary is assignment-level only.",
    )
