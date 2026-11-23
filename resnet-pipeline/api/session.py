"""Keep full detection running on one RTSP camera via the shared multi-cam hub.

The picture is the same pipeline as ``python main.py --mode full --show``.
Nothing is saved as a video. Cameras share VideoMAE + ResNet inside ``api.hub``;
identical RTSP URLs also share one TCP session via ``SharedCapture``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from api.hub import ACTIVE_DIR, ensure_hub, hub_dirs

logger = logging.getLogger("api.session")

StatusCallback = Callable[[str, str, str], None]
ROOT = Path(__file__).resolve().parents[1]


class DetectionSession:
    def __init__(
        self,
        camera_id: str,
        source: str,
        rotation: int,
        roi_path: Path,
        settings_path: Path,
        output_dir: Path,
        on_status: StatusCallback,
    ) -> None:
        self.camera_id = str(camera_id)
        self.source = source
        self.rotation = rotation
        self.roi_path = Path(roi_path)
        self.settings_path = Path(settings_path)
        self.output_dir = Path(output_dir)
        self._on_status = on_status
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._marked_online = False
        self._jpeg_mtime = 0.0
        self.jpeg: bytes | None = None
        self.progress: dict[str, Any] = {}
        self.error = ""

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"detect-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, wait: float = 8.0) -> None:
        self._stop.set()
        self._unregister()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=wait)

    def refresh_launch(self) -> None:
        """Rewrite the hub launch file so a ROI change is picked up without tearing down this session."""
        self._register()
        # Catalog may have been set to "reconnecting"; require the next fresh frame
        # (or seeded preview) to mark online again.
        self._marked_online = False

    def reload_preview(self) -> None:
        """Pick up a freshly seeded live.jpg for the View detection page."""
        self._seed_jpeg_from_disk()
        if self.latest_jpeg():
            self._marked_online = True
            self.error = ""
            self._on_status(self.camera_id, "online", "")

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self.jpeg

    def latest_progress(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.progress)

    def _launch_path(self) -> Path:
        hub_dirs()
        return ACTIVE_DIR / f"{self.camera_id}.json"

    def _register(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        launch = {
            "camera_id": self.camera_id,
            "settings": str(self.settings_path),
            "jpeg": str(self.output_dir / "live.jpg"),
            "overrides": {
                "pipeline": {"mode": "full"},
                "camera": {
                    "source": self.source,
                    "camera_id": self.camera_id,
                    "rotation": self.rotation,
                },
                "rois": {"config": str(self.roi_path)},
                "visualization": {"show_window": False, "save_video": False},
                "output": {"directory": str(self.output_dir)},
            },
        }
        path = self._launch_path()
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(launch), encoding="utf-8")
        temporary.replace(path)
        (self.output_dir / "launch.json").write_text(json.dumps(launch), encoding="utf-8")
        ensure_hub()

    def _unregister(self) -> None:
        try:
            self._launch_path().unlink(missing_ok=True)
        except OSError:
            pass

    def _run(self) -> None:
        """Register once with the hub and keep the JPEG subscription alive.

        Do **not** delete/recreate the hub launch file on a stall — that kills the
        detection thread while VideoReader is already reconnecting, which is how
        streams get stuck in a reconnect fight.
        """
        self._on_status(self.camera_id, "reconnecting", "Waiting for RTSP / next good frame.")
        self._seed_jpeg_from_disk()
        self._register()
        while not self._stop.is_set():
            # Hub may have dropped us after a crash — put the launch file back.
            if not self._launch_path().is_file():
                logger.warning("Camera %s missing from hub; re-registering", self.camera_id)
                self._on_status(self.camera_id, "reconnecting", "Re-registering with hub.")
                self._register()
                self._marked_online = False

            fresh = self._pull_jpeg()
            if fresh:
                # Clear reconnecting as soon as frames are moving again.
                if (not self._marked_online) or self.error:
                    self._marked_online = True
                    self.error = ""
                    self._on_status(self.camera_id, "online", "")
            elif self._jpeg_stale(45.0):
                if self.error != "Stream stalled; waiting for next good frame.":
                    self.error = "Stream stalled; waiting for next good frame."
                    self._marked_online = False
                    logger.warning("Camera %s %s", self.camera_id, self.error)
                    self._on_status(self.camera_id, "reconnecting", self.error)
            self._stop.wait(0.25)
        self._unregister()
        self._on_status(self.camera_id, "offline", "")

    def _jpeg_stale(self, seconds: float) -> bool:
        if self._jpeg_mtime <= 0:
            return self.latest_jpeg() is not None
        return (time.time() - self._jpeg_mtime) > seconds

    def _seed_jpeg_from_disk(self) -> None:
        """Keep showing the last good frame while RTSP is down."""
        path = self.output_dir / "live.jpg"
        try:
            data = path.read_bytes()
            mtime = path.stat().st_mtime
        except OSError:
            return
        if not data:
            return
        self._jpeg_mtime = mtime
        with self._lock:
            self.jpeg = data

    def _pull_jpeg(self) -> bool:
        path = self.output_dir / "live.jpg"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return False
        if mtime == self._jpeg_mtime:
            return False
        try:
            data = path.read_bytes()
        except OSError:
            return False
        if not data:
            return False
        # Ignore files that did not actually change content after a rewrite of the same bytes.
        self._jpeg_mtime = mtime
        with self._lock:
            self.jpeg = data
        return True
