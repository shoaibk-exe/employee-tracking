"""Independent camera decode worker with bounded queue and reconnect.

RTSP/HEVC streams use PyAV (OpenCV's FFmpeg backend often returns green/corrupt
frames on these cameras — notably cam_04 / 2k_megnetic).
File and USB sources still use OpenCV.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Iterator

import cv2
import numpy as np

from backend.config import CameraSpec
from backend.types import CameraStatus, FramePacket

logger = logging.getLogger(__name__)

ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
LIVE_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://")


def is_live_source(source: str) -> bool:
    return source.lower().startswith(LIVE_PREFIXES) or source.isdigit()


def redact_source(source: str) -> str:
    """Hide credentials in RTSP URLs for logs/API."""
    if "://" not in source:
        return source
    scheme, rest = source.split("://", 1)
    if "@" in rest and ":" in rest.split("@", 1)[0]:
        creds, host = rest.split("@", 1)
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return source


def is_corrupt_frame(frame: np.ndarray) -> bool:
    """Detect classic green/HEVC-corrupt frames from bad OpenCV decodes."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return True
    h, w = frame.shape[:2]
    if h < 16 or w < 16:
        return True
    # Sample a grid for speed.
    sample = frame[:: max(1, h // 32), :: max(1, w // 32)]
    bgr_mean = sample.reshape(-1, 3).mean(axis=0)
    b, g, r = map(float, bgr_mean)
    std = float(sample.std())
    if std < 2.5:
        return True
    # Solid green / green-dominant corruption.
    if g > 150 and g > (r + 40) and g > (b + 40):
        return True
    if g > 200 and r < 50 and b < 50:
        return True
    return False


class CameraWorker:
    """Decode one camera feed on a dedicated thread."""

    def __init__(
        self,
        spec: CameraSpec,
        *,
        queue_size: int = 2,
        drop_policy: str = "latest",
        target_fps: float = 10.0,
        process_width: int = 1280,
        process_height: int = 720,
        rtsp_transport: str = "tcp",
        reconnect_delay_seconds: float = 2.0,
        max_reconnect_delay_seconds: float = 30.0,
        frozen_timeout_seconds: float = 5.0,
        open_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 10.0,
    ) -> None:
        self.spec = spec
        self.queue_size = max(1, queue_size)
        self.drop_policy = drop_policy
        self.target_fps = max(0.1, float(target_fps))
        self.process_width = process_width
        self.process_height = process_height
        self.rtsp_transport = rtsp_transport
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.max_reconnect_delay_seconds = max_reconnect_delay_seconds
        self.frozen_timeout_seconds = frozen_timeout_seconds
        self.open_timeout_seconds = open_timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds

        self._queue: queue.Queue[FramePacket | None] = queue.Queue(maxsize=self.queue_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self.status = CameraStatus.DISABLED if not spec.enabled else CameraStatus.DISCONNECTED
        self.input_fps = 0.0
        self.latency_ms = 0.0
        self.last_frame_id = 0
        self._frames_published = 0
        self.last_timestamp: float | None = None
        self.reconnect_count = 0
        self.error_message: str | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_packet: FramePacket | None = None
        self.skip_until_keyframe = True
        self.corrupt_frames = 0

    @property
    def camera_id(self) -> str:
        return self.spec.camera_id

    def start(self) -> None:
        if not self.spec.enabled:
            self.status = CameraStatus.DISABLED
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"cam-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def get_latest(self, timeout: float = 0.0) -> FramePacket | None:
        """Return the newest frame, discarding older queued frames."""
        packet: FramePacket | None = None
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            remaining = deadline - time.monotonic()
            try:
                item = self._queue.get(timeout=remaining if remaining > 0 else 0.0)
            except queue.Empty:
                break
            if item is None:
                break
            packet = item
            if timeout <= 0:
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        break
                    packet = item
                break
        if packet is None:
            with self._lock:
                return self._latest_packet
        return packet

    def peek_latest(self) -> FramePacket | None:
        """Newest decoded packet without consuming the processing queue (display path)."""
        with self._lock:
            return self._latest_packet

    def snapshot(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def public_status(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "display_name": self.spec.display_name,
            "location": self.spec.location,
            "status": self.status.value,
            "source": redact_source(self.spec.source),
            "input_fps": round(self.input_fps, 2),
            "latency_ms": round(self.latency_ms, 1),
            "last_frame_id": self.last_frame_id,
            "last_timestamp": self.last_timestamp,
            "reconnect_count": self.reconnect_count,
            "error_message": self.error_message,
            "corrupt_frames_dropped": self.corrupt_frames,
        }

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """Downscale to fit process box. Never upscale."""
        h, w = frame.shape[:2]
        if self.process_width <= 0 or self.process_height <= 0:
            return frame
        if w == self.process_width and h == self.process_height:
            return frame
        if w <= self.process_width and h <= self.process_height:
            return frame
        scale = min(self.process_width / float(w), self.process_height / float(h))
        nw = max(2, int(round(w * scale)))
        nh = max(2, int(round(h * scale)))
        nw -= nw % 2
        nh -= nh % 2
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray | None:
        if frame is None or frame.size == 0:
            return None
        # Heuristic green/flat check targets broken live HEVC decodes only;
        # local files/test clips may legitimately be flat.
        if is_live_source(self.spec.source) and is_corrupt_frame(frame):
            return None
        if self.spec.rotation in ROTATE_CODES:
            frame = cv2.rotate(frame, ROTATE_CODES[self.spec.rotation])
        h, w = frame.shape[:2]
        if h < 16 or w < 16:
            return None
        return self._resize(frame)

    def _publish(self, packet: FramePacket) -> None:
        with self._lock:
            self._latest_frame = packet.frame
            self._latest_packet = packet
            self.last_frame_id = packet.frame_id
            self.last_timestamp = packet.timestamp
            self._frames_published += 1
        if self.drop_policy == "latest":
            while True:
                try:
                    self._queue.put_nowait(packet)
                    break
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break
        else:
            try:
                self._queue.put(packet, timeout=0.5)
            except queue.Full:
                logger.debug("%s frame queue full; dropping", self.camera_id)

    def _emit_loop_metrics(self, frame_id_ref: list[int], fps_state: dict[str, float], frame: np.ndarray, source_fps: float | None) -> None:
        now = time.monotonic()
        prepared = self._prepare_frame(frame)
        if prepared is None:
            return
        frame_id_ref[0] += 1
        fps_state["count"] = fps_state.get("count", 0) + 1
        if now - fps_state["t0"] >= 1.0:
            self.input_fps = fps_state["count"] / (now - fps_state["t0"])
            fps_state["t0"] = now
            fps_state["count"] = 0
        packet = FramePacket(
            camera_id=self.camera_id,
            frame=prepared,
            frame_id=frame_id_ref[0],
            timestamp=now + float(self.spec.time_offset_seconds),
            wall_clock=datetime.now(timezone.utc),
            source_fps=source_fps,
            after_reconnect=self.reconnect_count > 0 and frame_id_ref[0] == 1,
        )
        self.latency_ms = (time.monotonic() - now) * 1000.0
        self.status = CameraStatus.LIVE
        self._publish(packet)

    def _iter_pyav_frames(self) -> Iterator[tuple[np.ndarray, float | None]]:
        import av

        options = {
            "rtsp_transport": self.rtsp_transport,
            "stimeout": str(int(self.open_timeout_seconds * 1_000_000)),
            "max_delay": "5000000",
            "fflags": "nobuffer",
            "flags": "low_delay",
        }
        container = av.open(self.spec.source, options=options, timeout=self.open_timeout_seconds)
        try:
            if not container.streams.video:
                raise RuntimeError("No video stream")
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            # Prefer latest frames over perfect continuity for live CCTV.
            try:
                stream.codec_context.skip_frame = "NONKEY"  # type: ignore[attr-defined]
                # Above is too aggressive; reset to default DEFAULT.
                stream.codec_context.skip_frame = "DEFAULT"  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            source_fps = None
            if stream.average_rate is not None:
                try:
                    source_fps = float(stream.average_rate)
                except Exception:  # noqa: BLE001
                    source_fps = None

            min_interval = 1.0 / self.target_fps
            next_emit = 0.0
            awaiting_keyframe = False
            for frame in container.decode(video=0):
                if self._stop.is_set():
                    break
                # After packet loss HEVC references are broken until the next
                # keyframe; those frames are smeared and poison detection/ReID.
                if getattr(frame, "is_corrupt", False):
                    awaiting_keyframe = self.skip_until_keyframe
                    self.corrupt_frames += 1
                    continue
                if awaiting_keyframe:
                    if not getattr(frame, "key_frame", False):
                        self.corrupt_frames += 1
                        continue
                    awaiting_keyframe = False
                now = time.monotonic()
                if now < next_emit:
                    continue
                next_emit = now + min_interval
                img = frame.to_ndarray(format="bgr24")
                yield img, source_fps
        finally:
            try:
                container.close()
            except Exception:  # noqa: BLE001
                pass

    def _run_rtsp_pyav(self) -> None:
        frame_id_ref = [0]
        fps_state = {"t0": time.monotonic(), "count": 0}
        got_any = False
        last_ok = time.monotonic()
        for img, source_fps in self._iter_pyav_frames():
            if not got_any:
                self.status = CameraStatus.LIVE
                self.error_message = None
                logger.info(
                    "Camera %s connected via PyAV (%s) native=%sx%s",
                    self.camera_id,
                    redact_source(self.spec.source),
                    img.shape[1],
                    img.shape[0],
                )
                got_any = True
            if is_corrupt_frame(img):
                if time.monotonic() - last_ok > self.frozen_timeout_seconds:
                    raise RuntimeError("Stream producing corrupt/green frames")
                continue
            last_ok = time.monotonic()
            self._emit_loop_metrics(frame_id_ref, fps_state, img, source_fps)
        if not got_any:
            raise RuntimeError(f"No frames from {redact_source(self.spec.source)}")
        if time.monotonic() - last_ok > self.frozen_timeout_seconds:
            raise RuntimeError("Stream frozen / read failed")

    def _open_capture(self) -> cv2.VideoCapture:
        source = self.spec.source
        if source.isdigit():
            return cv2.VideoCapture(int(source))
        params: list[int] = []
        if is_live_source(source):
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                params += [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.open_timeout_seconds * 1000)]
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                params += [cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.read_timeout_seconds * 1000)]
        if params:
            return cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
        return cv2.VideoCapture(source)

    def _run_opencv_file(self) -> None:
        cap = self._open_capture()
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open source {redact_source(self.spec.source)}")
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or None
            self.status = CameraStatus.LIVE
            self.error_message = None
            logger.info("Camera %s connected (%s)", self.camera_id, redact_source(self.spec.source))

            frame_id_ref = [0]
            fps_state = {"t0": time.monotonic(), "count": 0}
            min_interval = 1.0 / self.target_fps
            next_emit = 0.0
            while not self._stop.is_set():
                ok, frame = cap.read()
                now = time.monotonic()
                if not ok or frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                if now < next_emit:
                    time.sleep(max(0.0, next_emit - now))
                next_emit = time.monotonic() + min_interval
                self._emit_loop_metrics(frame_id_ref, fps_state, frame, source_fps)
        finally:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass

    def _run(self) -> None:
        delay = self.reconnect_delay_seconds
        live = is_live_source(self.spec.source) and not self.spec.source.isdigit()

        while not self._stop.is_set():
            self.status = CameraStatus.CONNECTING if self.reconnect_count == 0 else CameraStatus.RECONNECTING
            frames_before = self._frames_published
            try:
                if live and self.spec.source.lower().startswith("rtsp"):
                    self._run_rtsp_pyav()
                    # Clean demux end → reconnect.
                    raise RuntimeError("RTSP demux ended")
                self._run_opencv_file()
            except Exception as exc:  # noqa: BLE001
                self.error_message = str(exc)
                self.status = CameraStatus.ERROR
                logger.warning("Camera %s error: %s", self.camera_id, exc)

            if self._stop.is_set():
                break
            if self._frames_published - frames_before >= 25:
                delay = self.reconnect_delay_seconds  # the stream worked: a new drop starts a fresh backoff
            self.reconnect_count += 1
            self.status = CameraStatus.RECONNECTING
            logger.info("Camera %s reconnecting in %.1fs", self.camera_id, delay)
            self._stop.wait(delay)
            delay = min(delay * 1.5, self.max_reconnect_delay_seconds)

        self.status = CameraStatus.DISCONNECTED
