"""Capture thread. Its only job is a fresh frame and an honest connection status.

Inference must not run in this thread. If it does, a slow model stalls the socket and the
pipeline starts reasoning about video that is already seconds old.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable

import numpy as np

from app.health.camera_health import CameraHealth, CameraHealthMonitor
from app.streams.frame_buffer import FramePacket, LatestFrameBuffer
from app.streams.reconnect_manager import ReconnectManager
from app.utils.timestamps import utc_now

logger = logging.getLogger(__name__)


def _configure_ffmpeg(is_rtsp: bool) -> None:
    if not is_rtsp:
        return
    # Process-wide. TCP plus a short socket timeout avoids a silent UDP stall.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000|stimeout;5000000"
    )


def mask_url(source: str) -> str:
    if source.startswith("rtsp://") and "@" in source:
        scheme, rest = source.split("://", 1)
        creds, host = rest.split("@", 1)
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return source


class RTSPReader:
    def __init__(
        self,
        camera_id: str,
        source: str,
        buffer_size: int = 2,
        reconnect_base_seconds: float = 2.0,
        reconnect_max_seconds: float = 15.0,
        offline_after_seconds: float = 5.0,
        open_timeout_seconds: float = 8.0,
        flush_on_connect_seconds: float = 0.3,
        degraded_below_fps: float = 2.0,
        opener: Callable[[str], object] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.source = (source or "").strip()
        self.is_rtsp = self.source.lower().startswith(("rtsp://", "rtsps://"))
        self.open_timeout_seconds = open_timeout_seconds
        self.flush_on_connect_seconds = flush_on_connect_seconds
        self._opener = opener
        self.buffer: LatestFrameBuffer[FramePacket] = LatestFrameBuffer(buffer_size)
        self.reconnect = ReconnectManager(reconnect_base_seconds, reconnect_max_seconds)
        self.health = CameraHealthMonitor(
            camera_id,
            offline_after_seconds=offline_after_seconds,
            degraded_below_fps=degraded_below_fps,
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture = None
        self._frame_number = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"capture-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.open_timeout_seconds + 2)
            self._thread = None
        self._release()

    def latest(self) -> FramePacket | None:
        return self.buffer.get_latest()

    def health_snapshot(self) -> CameraHealth:
        return self.health.snapshot()

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self.source:
                self.health.mark_offline("RTSP URL is empty")
                logger.error(
                    "camera source missing",
                    extra={"camera": self.camera_id, "event": "CAMERA_OFFLINE"},
                )
                if self._stop.wait(5):
                    return
                continue
            if not self._ensure_open():
                delay = self.reconnect.fail()
                self.health.mark_reconnect_attempt()
                self.health.mark_reconnecting(self.health.snapshot().detail or "open failed")
                logger.warning(
                    "camera reconnect scheduled",
                    extra={
                        "camera": self.camera_id,
                        "event": "CAMERA_RECONNECT",
                        "reconnect_count": self.reconnect.count,
                        "delay_s": f"{delay:.1f}",
                    },
                )
                if self._stop.wait(delay):
                    return
                continue
            packet = self._read_one()
            if packet is None:
                self._release()
                delay = self.reconnect.fail()
                self.health.mark_reconnect_attempt()
                if self._stop.wait(delay):
                    return
                continue
            self.reconnect.reset()
            self.health.mark_frame(packet.received_timestamp)
            self.buffer.put(packet)

    def _ensure_open(self) -> bool:
        if self._capture is not None:
            return True
        self.health.mark_reconnecting("opening")
        _configure_ffmpeg(self.is_rtsp)
        capture = self._open_with_timeout()
        if capture is None:
            return False
        self._capture = capture
        self._flush_backlog()
        self.health.mark_connected()
        logger.info(
            "camera connected",
            extra={"camera": self.camera_id, "event": "CAMERA_CONNECTED", "source": mask_url(self.source)},
        )
        return True

    def _open_with_timeout(self) -> object | None:
        if self._opener is not None:
            try:
                capture = self._opener(self.source)
            except Exception as exc:  # noqa: BLE001
                self.health.mark_reconnecting(str(exc))
                return None
            return capture

        import cv2

        holder: dict[str, object] = {}

        def _connect() -> None:
            backend = cv2.CAP_FFMPEG if self.is_rtsp else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.source, backend)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            holder["cap"] = cap

        worker = threading.Thread(target=_connect, daemon=True)
        worker.start()
        worker.join(self.open_timeout_seconds)
        if worker.is_alive():
            self.health.mark_reconnecting("open timed out")
            return None
        cap = holder.get("cap")
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            self.health.mark_reconnecting("open failed")
            return None
        return cap

    def _flush_backlog(self) -> None:
        """Some RTSP backends hand back a burst of old frames right after connect."""
        cap = self._capture
        if cap is None or self.flush_on_connect_seconds <= 0:
            return
        deadline = time.monotonic() + self.flush_on_connect_seconds
        while time.monotonic() < deadline and not self._stop.is_set():
            grabbed = cap.grab()
            if not grabbed:
                break

    def _read_one(self) -> FramePacket | None:
        cap = self._capture
        if cap is None:
            return None
        ok, frame = cap.read()
        if not ok or not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim != 3:
            self.health.mark_reconnecting("read failed")
            return None
        # Both stamps are this server's clock. Camera presentation timestamps are not trusted.
        now = utc_now()
        with self._lock:
            self._frame_number += 1
            number = self._frame_number
        return FramePacket(
            camera_id=self.camera_id,
            frame=frame,
            frame_number=number,
            capture_timestamp=now,
            received_timestamp=now,
        )

    def _release(self) -> None:
        cap = self._capture
        self._capture = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                logger.warning(
                    "camera release failed",
                    extra={"camera": self.camera_id, "event": "CAMERA_RELEASE_FAILED"},
                )
