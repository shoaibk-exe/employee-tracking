"""One full-detection process for a camera.

The API starts this and starts it again if it exits. A corrupt frame, a
dropped RTSP connection, or a model error stays inside this process. When the
process dies, the GPU memory it held is released and the API opens the stream
again.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # Set before torch is imported. Reduces the allocator failures seen when the card is nearly full.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("api.worker")

    launch = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    jpeg_path = Path(launch["jpeg"])
    jpeg_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = jpeg_path.with_name("progress.json")

    from core.config import load_settings
    from main import FullApp

    settings = load_settings(Path(launch["settings"]), launch["overrides"])
    holder: dict[str, FullApp] = {}

    def stop(_signum: int, _frame: object) -> None:
        app = holder.get("app")
        if app is not None:
            app.request_stop()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    from core.scalability import AsyncJpegPublisher

    jpeg_publisher: AsyncJpegPublisher | None = None

    def write_jpeg_bytes(payload: bytes) -> None:
        app = holder.get("app")
        if app is not None and app._stop_requested:
            return
        temporary = jpeg_path.with_suffix(".jpg.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, jpeg_path)

    def publish_sync(frame) -> None:
        import cv2

        app = holder.get("app")
        if app is not None and app._stop_requested:
            return
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return
        write_jpeg_bytes(encoded.tobytes())

    scale = settings.scalability
    if scale.enabled and scale.async_encode:
        jpeg_publisher = AsyncJpegPublisher(write_jpeg_bytes, quality=70)

        def publish(frame) -> None:
            app = holder.get("app")
            if app is not None and app._stop_requested:
                return
            jpeg_publisher.submit(frame)
    else:
        publish = publish_sync

    def publish_progress(payload: dict) -> None:
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, progress_path)

    app = FullApp(settings, on_frame=publish, on_progress=publish_progress)
    holder["app"] = app
    try:
        app.run()
    except Exception:
        log.exception("Detection process failed")
        raise
    finally:
        if jpeg_publisher is not None:
            jpeg_publisher.close()


if __name__ == "__main__":
    main()
