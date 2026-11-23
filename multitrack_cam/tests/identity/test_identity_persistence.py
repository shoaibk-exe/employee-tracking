from __future__ import annotations

import numpy as np

from backend.association.camera_topology import CameraTopology
from backend.association.global_associator import AssociatorConfig, GlobalAssociator
from backend.database.repository import DatabaseRepository
from backend.identity.global_identity_manager import GlobalIdentityManager
from backend.reid.embedding_gallery import EmbeddingGallery
from backend.types import Tracklet, TrackletObservation
from backend.utils.geometry import l2_normalize


def _tl(gid_seed: int = 0) -> Tracklet:
    emb = l2_normalize(np.full(64, float(gid_seed + 1), dtype=np.float32))
    obs = [
        TrackletObservation(timestamp=float(i), frame_id=i, bbox_xyxy=(1, 1, 20, 40), confidence=0.9, embedding=emb, quality_score=0.7)
        for i in range(3)
    ]
    return Tracklet(
        tracklet_id=f"tl_{gid_seed}",
        camera_id="cam_01",
        tracker_session_id="abc",
        local_track_id=gid_seed + 1,
        start_time=0.0,
        end_time=2.0,
        observations=obs,
        embedding=emb,
    )


def test_identities_survive_manager_reload(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    repo1 = DatabaseRepository(db_url)
    gallery1 = EmbeddingGallery()
    assoc1 = GlobalAssociator(gallery1, CameraTopology({"transitions": [], "cameras": {}}), AssociatorConfig())
    mgr1 = GlobalIdentityManager(assoc1, gallery1, repository=repo1)
    result = mgr1.process_tracklets([_tl(0)], now=2.0)
    gid = result[0].global_id
    assert gid and gid.startswith("G")

    # Simulate restart.
    repo2 = DatabaseRepository(db_url)
    gallery2 = EmbeddingGallery()
    assoc2 = GlobalAssociator(gallery2, CameraTopology({"transitions": [], "cameras": {}}), AssociatorConfig())
    mgr2 = GlobalIdentityManager(assoc2, gallery2, repository=repo2)
    ids = {row["global_id"] for row in mgr2.list_identities()}
    assert gid in ids
    assert len(gallery2.identities()) >= 1


def test_merged_identity_stays_merged_after_reload(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'merge.db'}"

    def manager():
        gallery = EmbeddingGallery()
        assoc = GlobalAssociator(gallery, CameraTopology({"transitions": [], "cameras": {}}), AssociatorConfig())
        return GlobalIdentityManager(assoc, gallery, repository=DatabaseRepository(db_url)), gallery

    mgr1, _ = manager()
    keep = mgr1.create_live_identity("cam_02", 1.0)
    dup = mgr1.create_live_identity("cam_03", 2.0)
    mgr1.add_live_embedding(keep, l2_normalize(np.ones(64, dtype=np.float32)), camera_id="cam_02", timestamp=1.0, quality=0.8)
    mgr1.add_live_embedding(
        dup, l2_normalize(np.concatenate([np.ones(32), -np.ones(32)]).astype(np.float32)),
        camera_id="cam_03", timestamp=2.0, quality=0.8,
    )
    assert keep != dup
    assert mgr1.merge_identity_into(dup, keep, reason="test") > 0
    assert not mgr1.is_live_identity(dup)

    mgr2, gallery2 = manager()
    assert not mgr2.is_live_identity(dup)
    assert dup not in gallery2.identities() and keep in gallery2.identities()
