"""Click a desk polygon on a live frame and save it into configs/desks.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import yaml

from app.config import CONFIG_DIR, load_config
from app.streams.rtsp_reader import RTSPReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a desk polygon")
    parser.add_argument("--camera", default="office_01")
    parser.add_argument("--desk-id", required=True)
    parser.add_argument("--employee-id", required=True)
    args = parser.parse_args()
    config = load_config()
    camera = config.camera(args.camera)
    if camera is None:
        raise SystemExit(f"Unknown camera {args.camera}")
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    reader = RTSPReader(
        camera_id=camera.camera_id,
        source=camera.rtsp_url,
        buffer_size=config.stream.buffer_size,
        open_timeout_seconds=config.stream.open_timeout_seconds,
    )
    reader.start()
    window = "draw desk  (s save, z undo, q quit)"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            packet = reader.latest()
            frame = None if packet is None else packet.frame.copy()
            if frame is None:
                frame = cv2.zeros((480, 854, 3), dtype="uint8")
            for index, point in enumerate(points):
                cv2.circle(frame, point, 4, (0, 220, 255), -1)
                if index:
                    cv2.line(frame, points[index - 1], point, (0, 220, 255), 2)
            if len(points) >= 3:
                cv2.line(frame, points[-1], points[0], (0, 220, 255), 1)
            cv2.putText(
                frame,
                f"{args.desk_id} -> {args.employee_id}  points={len(points)}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, frame)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            if key == ord("z") and points:
                points.pop()
            if key == ord("s"):
                if len(points) < 3:
                    print("need at least 3 points")
                    continue
                _save(args.desk_id, args.employee_id, args.camera, points)
                print(f"saved {args.desk_id}")
                break
    finally:
        reader.stop()
        cv2.destroyAllWindows()


def _save(desk_id: str, employee_id: str, camera_id: str, points: list[tuple[int, int]]) -> None:
    path = CONFIG_DIR / "desks.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data = data or {}
    desks = data.setdefault("desks", {})
    desks[desk_id] = {
        "employee_id": employee_id,
        "camera_id": camera_id,
        "polygon": [list(point) for point in points],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
