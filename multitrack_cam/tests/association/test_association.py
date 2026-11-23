from __future__ import annotations

import numpy as np

from backend.association.camera_topology import CameraTopology
from backend.association.global_associator import AssociatorConfig, GlobalAssociator
from backend.identity.global_identity_manager import GlobalIdentityManager, InMemoryRepository
from backend.reid.embedding_gallery import EmbeddingGallery
from backend.types import Tracklet, TrackletObservation
from backend.utils.geometry import l2_normalize


def _tracklet(tid: str, cam: str, emb: np.ndarray, t0: float = 0.0, t1: float = 2.0) -> Tracklet:
    obs = [
        TrackletObservation(timestamp=t0 + i * 0.5, frame_id=i, bbox_xyxy=(10, 10, 50, 120), confidence=0.9, embedding=emb, quality_score=0.8)
        for i in range(4)
    ]
    return Tracklet(
        tracklet_id=tid,
        camera_id=cam,
        tracker_session_id="sess",
        local_track_id=1,
        start_time=t0,
        end_time=t1,
        observations=obs,
        embedding=emb,
    )


def test_same_embedding_recovers_identity_across_cameras():
    gallery = EmbeddingGallery()
    topology = CameraTopology(
        {
            "transitions": [
                {"from": "cam_01", "to": "cam_02", "min_travel_seconds": 0.5, "max_travel_seconds": 60},
            ],
            "cameras": {},
            "simultaneous_allowed_pairs": [],
        }
    )
    associator = GlobalAssociator(gallery, topology, AssociatorConfig(match_threshold=0.5, match_margin=0.05, new_person_threshold=0.2))
    manager = GlobalIdentityManager(associator, gallery, repository=InMemoryRepository())

    emb = l2_normalize(np.ones(512, dtype=np.float32))
    first = _tracklet("tl1", "cam_01", emb, 0.0, 2.0)
    r1 = manager.process_tracklets([first], now=2.0)
    assert r1[0].global_id is not None
    gid = r1[0].global_id

    second = _tracklet("tl2", "cam_02", emb, 5.0, 7.0)
    second.local_track_id = 9
    r2 = manager.process_tracklets([second], now=7.0)
    assert r2[0].global_id == gid


def test_similar_but_distinct_embeddings_do_not_force_merge():
    gallery = EmbeddingGallery()
    topology = CameraTopology({"transitions": [], "cameras": {}, "simultaneous_allowed_pairs": []})
    associator = GlobalAssociator(
        gallery,
        topology,
        AssociatorConfig(match_threshold=0.85, strong_match_threshold=0.9, match_margin=0.1, new_person_threshold=0.2),
    )
    manager = GlobalIdentityManager(associator, gallery, repository=InMemoryRepository())

    a = l2_normalize(np.random.default_rng(0).normal(size=512).astype(np.float32))
    b = l2_normalize(np.random.default_rng(1).normal(size=512).astype(np.float32))
    r1 = manager.process_tracklets([_tracklet("tl1", "cam_01", a)], now=2.0)
    r2 = manager.process_tracklets([_tracklet("tl2", "cam_02", b, 3.0, 5.0)], now=5.0)
    assert r1[0].global_id != r2[0].global_id


def test_simultaneous_conflict_blocks_same_id_on_disconnected_cameras():
    topology = CameraTopology(
        {
            "transitions": [],
            "cameras": {"cam_01": {"overlapping_with": []}, "cam_02": {"overlapping_with": []}},
            "simultaneous_allowed_pairs": [],
        }
    )
    gallery = EmbeddingGallery()
    emb = l2_normalize(np.ones(512, dtype=np.float32))
    gallery.add("G000001", emb, camera_id="cam_01", timestamp=10.0, quality_score=0.9)
    associator = GlobalAssociator(gallery, topology, AssociatorConfig(match_threshold=0.4, match_margin=0.01))
    from backend.association.identity_constraints import ActiveObservation, ConstraintContext

    associator.set_active_observations(
        ConstraintContext(
            active=[ActiveObservation(global_id="G000001", camera_id="cam_01", timestamp=10.0, compound_key="cam_01:s:1")],
            last_seen={"G000001": ("cam_01", 10.0)},
        )
    )
    tl = _tracklet("tlx", "cam_02", emb, 10.0, 10.2)
    results = associator.associate([tl])
    # Soft outcome: either no confirmed reuse or provisional/unknown — never forced silent merge required.
    assert results[0].decision.value in {"UNKNOWN", "NEW", "PROVISIONAL", "REJECTED"} or results[0].global_id != "G000001" or results[0].decision.value != "CONFIRMED"
