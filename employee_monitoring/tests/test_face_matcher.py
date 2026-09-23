from datetime import datetime, timedelta, timezone

import numpy as np

from app.config import MatchConfig
from app.face.matcher import match_embedding
from app.tracking.track_manager import TrackManager


def _unit(x: float, y: float) -> np.ndarray:
    vector = np.array([x, y], dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_clear_match_is_accepted():
    query = _unit(1, 0)
    gallery = {
        "EMP001": _unit(0.95, 0.3122).reshape(1, -1),
        "EMP002": _unit(0.2, 0.9798).reshape(1, -1),
    }
    result = match_embedding(query, gallery, MatchConfig(min_similarity=0.45, min_margin=0.08))
    assert result.accepted
    assert result.employee_id == "EMP001"
    assert result.second_best_score < result.similarity_score


def test_close_scores_stay_unknown():
    query = _unit(1, 0)
    gallery = {
        "EMP001": _unit(0.69, (1 - 0.69**2) ** 0.5).reshape(1, -1),
        "EMP002": _unit(0.68, (1 - 0.68**2) ** 0.5).reshape(1, -1),
    }
    result = match_embedding(query, gallery, MatchConfig(min_similarity=0.45, min_margin=0.08))
    assert result.accepted is False
    assert result.employee_id is None
    assert result.similarity_score > result.second_best_score


def test_templates_of_the_same_person_are_not_a_second_identity():
    query = _unit(1, 0)
    gallery = {"EMP001": np.stack([_unit(0.8, 0.6), _unit(0.7, 0.7141)], axis=0)}
    result = match_embedding(query, gallery, MatchConfig(min_similarity=0.45, min_margin=0.08))
    assert result.accepted
    assert result.employee_id == "EMP001"
    assert result.second_best_score == 0


def test_identity_needs_three_observations_inside_two_seconds():
    manager = TrackManager(min_observations=3, window_seconds=2)
    start = datetime(2026, 9, 22, 9, 0, tzinfo=timezone.utc)
    assert manager.observe(17, "EMP001", 0.72, start) is None
    assert manager.observe(17, "EMP001", 0.75, start + timedelta(seconds=0.4)) is None
    assert manager.observe(17, None, 0.0, start + timedelta(seconds=0.8)) is None
    confirmed = manager.observe(17, "EMP001", 0.77, start + timedelta(seconds=1.2))
    assert confirmed == "EMP001"
    assert manager.observe(17, "EMP005", 0.9, start + timedelta(seconds=1.4)) == "EMP001"


def test_separate_tracks_keep_separate_identities():
    manager = TrackManager(min_observations=3, window_seconds=2)
    start = datetime(2026, 9, 22, 9, 0, tzinfo=timezone.utc)
    for offset in (0, 0.3, 0.6):
        manager.observe(14, "EMP001", 0.8, start + timedelta(seconds=offset))
        manager.observe(15, "EMP005", 0.8, start + timedelta(seconds=offset))
    assert manager.confirmed_employee(14) == "EMP001"
    assert manager.confirmed_employee(15) == "EMP005"
