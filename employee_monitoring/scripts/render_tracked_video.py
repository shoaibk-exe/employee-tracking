"""Write a video with the person number and the Working / Notworking label on every box.

The number is the tracker id. It should stay the same when two people cross.
The class label can still change between Working and Notworking.
Boxes under 50% confidence are not drawn.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from app.tracking.tracker import UltralyticsByteTracker
from app.utils.timestamps import utc_now


def _label(class_name: str) -> str:
    key = "".join(ch for ch in class_name.lower() if ch.isalnum())
    if key == "working":
        return "Working"
    if key == "notworking":
        return "Notworking"
    return class_name


def _color(track_id: int) -> tuple[int, int, int]:
    swatch = cv2.cvtColor(np.uint8([[[((track_id * 47) % 180), 210, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(swatch[0]), int(swatch[1]), int(swatch[2])


def _draw_label(frame: np.ndarray, lines: list[str], x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    sizes = [cv2.getTextSize(line, font, scale, thickness) for line in lines]
    width = max(size[0][0] for size in sizes) + 10
    line_h = max(size[0][1] + size[1] for size in sizes) + 6
    top = max(0, y - line_h * len(lines) - 4)
    cv2.rectangle(frame, (x, top), (x + width, top + line_h * len(lines)), color, -1)
    for index, line in enumerate(lines):
        baseline = sizes[index][1]
        text_y = top + line_h * index + sizes[index][0][1] + max(0, (line_h - sizes[index][0][1] - baseline) // 2)
        cv2.putText(frame, line, (x + 4, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render(source: Path, model: Path, output: Path, confidence: float, device: str | None) -> None:
    if not model.is_file():
        raise SystemExit(f"weights missing: {model}")
    if not source.is_file():
        raise SystemExit(f"video missing: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tracker = UltralyticsByteTracker(
        str(model),
        image_size=640,
        confidence=confidence,
        class_filter=None,
        tracker_config="configs/trackers/botsort_reid.yaml",
        device=device,
    )
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"could not open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"could not open writer {output}")
    print(f"model={model} device={device or 'gpu'}", flush=True)
    print(f"writing {output} {width}x{height} @ {fps:.1f} conf>={confidence}", flush=True)
    frame_i = 0
    seen: set[int] = set()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tracks = tracker.track(frame, utc_now())
        for track in tracks:
            seen.add(track.track_id)
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            color = _color(track.track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            _draw_label(
                frame,
                [f"Person {track.track_id}", f"{_label(track.class_name)}  {track.confidence:.0%}"],
                x1,
                y1,
                color,
            )
        cv2.putText(
            frame,
            f"people={len(tracks)}  person numbers seen={len(seen)}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_i += 1
        if frame_i % 50 == 0:
            print(f"frame={frame_i} boxes={len(tracks)} person_numbers={len(seen)}", flush=True)
    cap.release()
    writer.release()
    print(f"DONE frames={frame_i} person_numbers={len(seen)} output={output}", flush=True)


def main() -> None:
    repo = ROOT.parent
    parser = argparse.ArgumentParser(description="Draw person number and Working/Notworking on a video.")
    parser.add_argument("--source", type=Path, default=repo / "input_videos" / "vid1.mp4")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "sitting_model" / "best.pt")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument(
        "--device",
        default=None,
        help="Leave empty to use the GPU. Pass cpu to run on the CPU.",
    )
    args = parser.parse_args()
    output = args.output or (repo / "output_videos" / f"{args.source.stem}_ids.mp4")
    render(args.source, args.model, output, args.conf, args.device)


if __name__ == "__main__":
    main()
