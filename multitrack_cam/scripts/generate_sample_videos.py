#!/usr/bin/env python3
"""Generate four synthetic multi-camera videos for offline pipeline testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def draw_person(frame: np.ndarray, cx: int, cy: int, color: tuple[int, int, int], scale: float = 1.0) -> None:
    h = int(90 * scale)
    w = int(40 * scale)
    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = cx + w // 2, cy + h // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.circle(frame, (cx, y1 + int(12 * scale)), int(10 * scale), (30, 30, 30), -1)


def generate_camera(path: Path, camera_index: int, frames: int = 120, fps: int = 10, size=(960, 540)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    w, h = size
    # Two people with distinct colors; they hand off across cameras over time.
    colors = [(40, 80, 220), (40, 180, 80)]
    for i in range(frames):
        frame = np.full((h, w, 3), 35 + camera_index * 8, dtype=np.uint8)
        cv2.putText(frame, f"CAM {camera_index + 1}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)

        t = i / max(1, frames - 1)
        # Person A appears on cam0 early, cam1 mid, cam2 later, cam3 late.
        active_cam_a = min(3, int(t * 4))
        if camera_index == active_cam_a:
            x = int(80 + (w - 160) * ((t * 4) % 1.0))
            y = int(h * 0.55)
            draw_person(frame, x, y, colors[0], scale=1.0)

        # Person B opposite schedule.
        active_cam_b = 3 - min(3, int(t * 4))
        if camera_index == active_cam_b:
            x = int(w - 80 - (w - 160) * ((t * 4) % 1.0))
            y = int(h * 0.60)
            draw_person(frame, x, y, colors[1], scale=1.05)

        writer.write(frame)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="sample_data")
    parser.add_argument("--frames", type=int, default=150)
    args = parser.parse_args()
    out = Path(args.out)
    for i in range(4):
        generate_camera(out / f"cam_0{i + 1}.mp4", i, frames=args.frames)
        print(f"wrote {out / f'cam_0{i + 1}.mp4'}")


if __name__ == "__main__":
    main()
