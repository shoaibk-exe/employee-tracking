"""Show capture health for one camera. No models run here."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from app.config import load_config
from app.streams.rtsp_reader import RTSPReader, mask_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview a configured camera")
    parser.add_argument("--camera", default="entrance_01")
    args = parser.parse_args()
    config = load_config()
    camera = config.camera(args.camera)
    if camera is None:
        raise SystemExit(f"Unknown camera {args.camera}")
    reader = RTSPReader(
        camera_id=camera.camera_id,
        source=camera.rtsp_url,
        buffer_size=config.stream.buffer_size,
        reconnect_base_seconds=config.stream.reconnect_base_seconds,
        reconnect_max_seconds=config.stream.reconnect_max_seconds,
        offline_after_seconds=config.stream.offline_after_seconds,
        open_timeout_seconds=config.stream.open_timeout_seconds,
    )
    reader.start()
    last_number = -1
    processed = 0
    window_start = time.monotonic()
    process_fps = 0.0
    print(f"opening {camera.camera_id} {mask_url(camera.rtsp_url)}  (q to quit)")
    try:
        while True:
            packet = reader.latest()
            health = reader.health_snapshot()
            frame = None if packet is None else packet.frame
            if frame is None:
                frame = cv2.zeros((480, 854, 3), dtype="uint8")
            else:
                if packet.frame_number != last_number:
                    last_number = packet.frame_number
                    processed += 1
                elapsed = time.monotonic() - window_start
                if elapsed >= 1:
                    process_fps = processed / elapsed
                    processed = 0
                    window_start = time.monotonic()
            height, width = frame.shape[:2]
            stamp = "" if packet is None else packet.received_timestamp.strftime("%H:%M:%S")
            lines = [
                f"{camera.camera_id}  {width}x{height}",
                f"state={health.connection_status.value}  capture_fps={health.fps:.1f}  process_fps={process_fps:.1f}",
                f"reconnects={health.reconnect_count}  time={stamp}  {health.detail}",
            ]
            y = 28
            for line in lines:
                cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
                y += 28
            cv2.imshow(camera.camera_id, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
