"""Offline, audited merge of verified duplicate identities.

Run with the server stopped (it keeps identities in memory). Backs up the DB first.
Only embeddings captured on --verified-camera are moved; the duplicate's other
embeddings are deactivated (kept, not deleted) because they were not verified.

    python scripts/merge_identities.py --pair G000013:G000001 --pair G000020:G000002 \
        --verified-camera cam_03 --reason "seat-anchor verified"
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.association.camera_topology import CameraTopology  # noqa: E402
from backend.association.global_associator import AssociatorConfig, GlobalAssociator  # noqa: E402
from backend.database.repository import DatabaseRepository  # noqa: E402
from backend.identity.global_identity_manager import GlobalIdentityManager  # noqa: E402
from backend.reid.embedding_gallery import EmbeddingGallery  # noqa: E402


def _server_running(url: str) -> bool:
    try:
        urllib.request.urlopen(f"{url}/api/health", timeout=2)
        return True
    except Exception:  # noqa: BLE001
        return False


def _backup(db: Path) -> Path:
    dst = db.parent / "backups" / f"{db.stem}_pre_merge_{time.strftime('%Y%m%d_%H%M%S')}.db"
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(db)
    out = sqlite3.connect(dst)
    with out:
        src.backup(out)
    src.close()
    out.close()
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "data/store/mtmct.db")
    ap.add_argument("--pair", action="append", required=True, help="DUPLICATE:SURVIVOR")
    ap.add_argument("--verified-camera", required=True)
    ap.add_argument("--reason", default="manual verified merge")
    ap.add_argument("--server", default="http://localhost:8090")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if _server_running(args.server):
        sys.exit("Server is running; stop it first (identities are held in memory).")
    pairs = [tuple(p.split(":")) for p in args.pair]

    con = sqlite3.connect(args.db)
    for src, dst in pairs:
        rows = con.execute(
            "select camera_id, count(*) from identity_embeddings where global_id=? and active=1 group by camera_id",
            (src,),
        ).fetchall()
        status = con.execute("select identity_status from global_identities where global_id=?", (dst,)).fetchone()
        print(f"{src} -> {dst}: active embeddings {dict(rows)}; survivor status {status}")
        if status is None or status[0] == "ARCHIVED":
            sys.exit(f"Survivor {dst} missing or archived; aborting.")
    if args.dry_run:
        return

    print("backup:", _backup(args.db))
    deactivated = {}
    with con:
        for src, _ in pairs:
            cur = con.execute(
                "update identity_embeddings set active=0 where global_id=? and active=1 and camera_id != ?",
                (src, args.verified_camera),
            )
            deactivated[src] = cur.rowcount
    con.close()

    repo = DatabaseRepository(f"sqlite:///{args.db}")
    gallery = EmbeddingGallery()
    assoc = GlobalAssociator(gallery, CameraTopology({"transitions": [], "cameras": {}}), AssociatorConfig())
    mgr = GlobalIdentityManager(assoc, gallery, repository=repo)
    for src, dst in pairs:
        if deactivated[src]:
            mgr.record_event("EMBEDDINGS_DEACTIVATED", src, None, {
                "count": deactivated[src], "reason": f"unverified (not {args.verified_camera}) before merge into {dst}",
            })
        moved = mgr.merge_identity_into(src, dst, reason=args.reason)
        print(f"merged {src} -> {dst}: moved {moved} gallery embeddings, deactivated {deactivated[src]} unverified")


if __name__ == "__main__":
    main()
