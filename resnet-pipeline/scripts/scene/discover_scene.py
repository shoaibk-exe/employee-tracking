"""Build a workstation scene map from administrator-drawn ROIs.

Seed from an administrator-drawn ROI file (recommended production path):

    python -m scripts.scene.discover_scene --from-rois config/workstations.json

Optional source override for the reference frame:

    python -m scripts.scene.discover_scene --from-rois config/workstations.json --source testing/input2.mp4

Outputs ``config/scene_map.json`` (versioned; previous kept as ``scene_map.v<N>.json``), the
reference frame ``config/scene_map_reference.jpg`` used for camera-alignment checks, and a
preview ``config/scene_map_preview.jpg``. Review with ``python -m scripts.scene.review_scene_map``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import ConfigError, load_settings  # noqa: E402
from core.occupancy.roi_manager import ROIManager  # noqa: E402
from core.scene.scene_mapper import (  # noqa: E402
    SOURCE_ADMIN, STATUS_CONFIRMED, STATUS_NEEDS_REVIEW, SceneMap, SceneMapError,
    carry_over_ids, iou, now_utc, workstations_from_admin_rois,
)
from core.io.video_reader import FramePacket, VideoReader  # noqa: E402

logger = logging.getLogger("discover_scene")

STATUS_COLORS = {STATUS_CONFIRMED: (0, 200, 0), STATUS_NEEDS_REVIEW: (0, 200, 255), "rejected": (0, 0, 255)}


def draw_map(frame: np.ndarray, smap: SceneMap, compare: dict[str, tuple] | None = None) -> np.ndarray:
    out = frame.copy()
    for roi_id, box in (compare or {}).items():
        cv2.rectangle(out, box[:2], box[2:], (255, 0, 255), 1)
        cv2.putText(out, roi_id, (box[0] + 2, box[3] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
    for w in smap.workstations:
        color = STATUS_COLORS.get(w.status, (200, 200, 200))
        cv2.rectangle(out, w.context_box[:2], w.context_box[2:], color, 1)
        cv2.rectangle(out, w.seat_box[:2], w.seat_box[2:], color, 3)
        label = f"{w.workstation_id} {w.confidence:.2f} {w.status}"
        cv2.putText(out, label, (w.seat_box[0] + 3, w.seat_box[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(out, label, (w.seat_box[0] + 3, w.seat_box[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    return out


def compare_to_rois(smap: SceneMap, rois: dict[str, tuple]) -> dict[str, object]:
    """One-to-one greedy match of map context boxes to admin ROIs by IoU."""
    pairs = sorted(((iou(w.context_box, box), w.workstation_id, roi_id)
                    for w in smap.active_workstations for roi_id, box in rois.items()), reverse=True)
    used_w, used_r, matches = set(), set(), []
    for value, ws_id, roi_id in pairs:
        if value < 0.3 or ws_id in used_w or roi_id in used_r:
            continue
        used_w.add(ws_id)
        used_r.add(roi_id)
        matches.append((ws_id, roi_id, round(value, 3)))
    return {
        "matched": sorted(matches, key=lambda m: m[1]),
        "missed_admin_rois": sorted(set(rois) - used_r),
        "extra_proposals": sorted({w.workstation_id for w in smap.active_workstations} - used_w),
        "mean_iou": round(float(np.mean([m[2] for m in matches])), 3) if matches else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config/settings.yaml")
    parser.add_argument("--source", help="MP4 / RTSP / camera index (default: camera.source)")
    parser.add_argument("--output", type=Path, default=None, help="Scene map path (default: scene.map)")
    parser.add_argument("--from-rois", type=Path, required=True,
                        help="Create the map from admin-drawn ROIs (required)")
    parser.add_argument("--compare-rois", type=Path, help="Evaluate the map against another ROI file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        overrides = {"camera": {"source": args.source}} if args.source else {}
        settings = load_settings(args.config, overrides)
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 2
    output = args.output or settings.scene_map_path
    previous = None
    if output.is_file():
        try:
            previous = SceneMap.load(output)
        except SceneMapError as exc:
            logger.warning("Ignoring unreadable previous map: %s", exc)

    reader = VideoReader(settings.source, inference_interval_seconds=1.0,
                         tracking_interval_seconds=None, rotation=settings.rotation)
    frame_size = (reader.width, reader.height)
    reference: np.ndarray | None = None
    try:
        for item in reader:
            if isinstance(item, FramePacket):
                reference = item.frame
                break
        rois = ROIManager.from_json(args.from_rois)
        rois.check_rotation(settings.rotation)
        rois.configure_for_frame(*frame_size, allow_scaling=False)
        workstations = workstations_from_admin_rois({w.workstation_id: w.bbox for w in rois.workstations})
        notes = [f"seeded from admin ROIs {args.from_rois}"]
        source = SOURCE_ADMIN
        workstations = carry_over_ids(workstations, previous)
    finally:
        reader.close()
    if reference is None:
        logger.error("No frames read from %s", settings.source)
        return 1

    smap = SceneMap(
        tenant_id=settings.tenant_id, camera_id=settings.camera_id,
        version=(previous.version + 1) if previous else 1, frame_size=frame_size, rotation=settings.rotation,
        workstations=workstations, created_utc=now_utc(), source=source,
        reference_frame=output.with_name(f"{output.stem}_reference.jpg").name, notes=notes,
    )
    smap.save(output)
    ref_path = output.with_name(smap.reference_frame)
    cv2.imwrite(str(ref_path), reference)
    preview = output.with_name(f"{output.stem}_preview.jpg")
    compare = None
    if args.compare_rois:
        compare_rois = ROIManager.from_json(args.compare_rois)
        compare_rois.configure_for_frame(*frame_size, allow_scaling=False)
        compare = {w.workstation_id: w.bbox for w in compare_rois.workstations}
        logger.info("ROI comparison: %s", compare_to_rois(smap, compare))
    cv2.imwrite(str(preview), draw_map(reference, smap, compare))
    logger.info("Wrote %s (v%d, %d workstations), %s, %s",
                output, smap.version, len(smap.workstations), ref_path, preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
