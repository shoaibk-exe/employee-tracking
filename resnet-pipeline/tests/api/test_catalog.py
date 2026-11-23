"""Catalog persistence for cameras and workstation ROIs."""

from pathlib import Path

import pytest

from api.store import Catalog, CatalogError


def test_new_stream_always_needs_own_roi(tmp_path: Path):
    catalog = Catalog(tmp_path)
    first = catalog.add_camera("Hall", "rtsp://10.0.0.8/live", rotate=True)
    assert first["needs_roi"] is True

    saved = catalog.save_rois(
        first["id"],
        640,
        360,
        180,
        [{"name": "W01", "bbox": [20, 30, 120, 160]}],
        "rtsp://10.0.0.8/live",
    )
    assert saved["rotation"] == 180
    assert saved["reference_resolution"] == [640, 360]
    assert saved["workstations"]["W01"]["bbox"] == [20, 30, 120, 160]

    # Same RTSP / rotation must not copy workstation boxes onto the new camera.
    second = catalog.add_camera("Hall copy", "rtsp://10.0.0.8/live", rotate=True)
    assert second["needs_roi"] is True
    assert catalog.read_rois(second["id"]) is None

    rotated_differently = catalog.add_camera("Hall upright", "rtsp://10.0.0.8/live", rotate=False)
    assert rotated_differently["needs_roi"] is True


def test_saved_roi_bbox_matches_drawn_pixels(tmp_path: Path):
    catalog = Catalog(tmp_path)
    camera = catalog.add_camera("Desk cam", "rtsp://10.0.0.10/live", rotate=True)
    boxes = [
        {"name": "W01", "bbox": [100, 200, 400, 500]},
        {"name": "W02", "bbox": [450, 210, 780, 520]},
    ]
    saved = catalog.save_rois(
        camera["id"],
        1280,
        720,
        180,
        boxes,
        camera["rtsp"],
    )
    assert saved["workstations"]["W01"]["bbox"] == [100, 200, 400, 500]
    assert saved["workstations"]["W02"]["bbox"] == [450, 210, 780, 520]
    assert catalog.read_rois(camera["id"])["workstations"]["W01"]["bbox"] == [100, 200, 400, 500]


def test_boxes_must_lie_inside_the_snapshot(tmp_path: Path):
    catalog = Catalog(tmp_path)
    camera = catalog.add_camera("Door", "rtsp://10.0.0.9/live", rotate=False)
    with pytest.raises(CatalogError):
        catalog.save_rois(
            camera["id"],
            100,
            80,
            0,
            [{"name": "W01", "bbox": [90, 10, 120, 40]}],
            camera["rtsp"],
        )


def test_camera_ids_do_not_reuse_after_delete(tmp_path: Path):
    catalog = Catalog(tmp_path)
    first = catalog.add_camera("A", "rtsp://10.0.0.1/live", rotate=False)
    second = catalog.add_camera("B", "rtsp://10.0.0.2/live", rotate=False)
    assert first["id"] == 1
    assert second["id"] == 2
    catalog.delete_camera(second["id"])
    third = catalog.add_camera("C", "rtsp://10.0.0.3/live", rotate=False)
    assert third["id"] == 3


def test_enroll_rejects_non_image_bytes(tmp_path: Path):
    catalog = Catalog(tmp_path)
    with pytest.raises(CatalogError, match="JPEG or PNG"):
        catalog.enroll("Alex", "face.bin", b"not-an-image")


def test_add_camera_requires_hostname(tmp_path: Path):
    catalog = Catalog(tmp_path)
    with pytest.raises(CatalogError, match="hostname"):
        catalog.add_camera("Bad", "rtsp:///nohost", rotate=False)
