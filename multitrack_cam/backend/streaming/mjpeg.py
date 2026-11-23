"""MJPEG helpers for browser-compatible live preview."""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Callable, Iterator

import cv2
import numpy as np


def encode_jpeg(frame: np.ndarray, quality: int = 70) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def mjpeg_generator(
    frame_fn: Callable[[], np.ndarray | None],
    *,
    fps: float = 8.0,
    quality: int = 70,
) -> Iterator[bytes]:
    interval = 1.0 / max(0.5, fps)
    boundary = b"--frame\r\n"
    while True:
        t0 = time.monotonic()
        frame = frame_fn()
        if frame is not None:
            jpg = encode_jpeg(frame, quality=quality)
            yield (
                boundary
                + b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
                + jpg
                + b"\r\n"
            )
        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, interval - elapsed))


async def mjpeg_live_generator(
    jpeg_fn: Callable[[], tuple[object, bytes] | None],
    *,
    max_fps: float = 20.0,
    keepalive_seconds: float = 2.0,
) -> AsyncIterator[bytes]:
    """Push each new pre-encoded JPEG once, as soon as it exists.

    ``jpeg_fn`` returns ``(key, jpeg)`` and may block (annotate/encode), so it runs in a
    worker thread; the event loop only polls. The last frame is re-sent every
    ``keepalive_seconds`` so browsers do not drop a stalled camera's connection.
    """
    poll = 1.0 / max(1.0, max_fps)
    boundary = b"--frame\r\n"
    last_key: object = None
    last_sent = 0.0
    while True:
        item = await asyncio.to_thread(jpeg_fn)
        now = time.monotonic()
        if item is not None and (item[0] != last_key or now - last_sent >= keepalive_seconds):
            last_key, jpg = item
            last_sent = now
            yield (
                boundary
                + b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
                + jpg
                + b"\r\n"
            )
        await asyncio.sleep(poll)
