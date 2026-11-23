from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from backend.cameras.stream_manager import StreamManager
from backend.config import AppConfig, CameraSpec


def _write_video(path: Path, frames: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(frames):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[:] = (i * 5) % 255
        w.write(img)
    w.release()


def test_four_cameras_produce_frames(tmp_path):
    videos = []
    for i in range(4):
        p = tmp_path / f"c{i}.mp4"
        _write_video(p)
        videos.append(p)

    cameras = [
        CameraSpec(camera_id=f"cam_0{i+1}", display_name=f"C{i+1}", location="t", source=str(videos[i]))
        for i in range(4)
    ]
    cfg = AppConfig(
        raw={
            "cameras": {
                "target_fps": 8,
                "process_width": 320,
                "process_height": 240,
                "queue_size": 1,
                "drop_policy": "latest",
            }
        },
        cameras=cameras,
        topology={},
    )
    mgr = StreamManager(cfg)
    mgr.start()
    try:
        deadline = time.time() + 5.0
        seen = set()
        while time.time() < deadline and len(seen) < 4:
            frames = mgr.latest_frames()
            seen.update(frames.keys())
            time.sleep(0.05)
        assert len(seen) == 4
        statuses = mgr.statuses()
        assert all("password" not in str(s.get("source", "")).lower() for s in statuses)
    finally:
        mgr.stop()
