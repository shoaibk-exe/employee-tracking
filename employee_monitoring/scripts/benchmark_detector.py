"""Time the sitting/standing detector. This does not score desk-time error."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from app.detection.employee_detector import EmployeeDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a YOLO weights file")
    parser.add_argument("--model", default="models/sitting_model/best.pt")
    parser.add_argument("--source", required=True, help="Image or video path")
    parser.add_argument("--frames", type=int, default=50)
    args = parser.parse_args()
    model_path = args.model
    if not Path(model_path).is_absolute():
        model_path = str(ROOT / model_path)
    detector = EmployeeDetector(model_path)
    source = Path(args.source)
    frames = []
    if source.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        image = cv2.imread(str(source))
        if image is None:
            raise SystemExit("could not read image")
        frames = [image] * args.frames
    else:
        cap = cv2.VideoCapture(str(source))
        while len(frames) < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
    if not frames:
        raise SystemExit("no frames")
    detector.detect(frames[0])
    started = time.perf_counter()
    detections = 0
    for frame in frames:
        detections += len(detector.detect(frame))
    elapsed = time.perf_counter() - started
    print(f"frames={len(frames)} seconds={elapsed:.2f} fps={len(frames) / elapsed:.2f} detections={detections}")


if __name__ == "__main__":
    main()
