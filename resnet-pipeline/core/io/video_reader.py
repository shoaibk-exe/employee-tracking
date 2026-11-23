"""Video ingestion for MP4 files and live RTSP/camera sources.

Timestamps:

* Files: ``timestamp_seconds`` is the media presentation timestamp of each frame
  (``CAP_PROP_POS_MSEC`` read after ``grab()``, which the FFmpeg backend fills from
  the packet PTS, so variable-frame-rate files are handled). If the backend returns
  non-increasing values, the reader switches to ``previous + 1/fps`` and marks
  packets with ``timestamp_source="fps_estimate"``. If that happens on files you
  care about, re-mux them with FFmpeg (``ffmpeg -i in.mp4 -c copy out.mp4``) or
  decode them with PyAV, which exposes frame PTS directly.
* Live: ``timestamp_seconds`` is monotonic time since the session started, sampled
  when the frame was read (not when it was processed). ``wall_clock_utc`` is the
  session's UTC start plus that monotonic offset, so NTP jumps cannot distort
  durations.

Live outages are reported with a ``SourceOutage`` item so callers can stop
crediting the previous state; the first frame after reconnecting has
``after_reconnect=True``.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ROTATE_CODES = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
LIVE_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://")


class VideoSourceError(RuntimeError):
    """The video source cannot be opened or read."""


@dataclass
class FramePacket:
    frame: np.ndarray
    frame_index: int
    timestamp_seconds: float
    wall_clock_utc: datetime | None
    timestamp_source: str
    run_inference: bool = True
    run_tracking: bool = False
    after_reconnect: bool = False


@dataclass(frozen=True)
class SourceOutage:
    last_frame_seconds: float | None
    detected_seconds: float
    wall_clock_utc: datetime | None
    reason: str


@dataclass(frozen=True)
class ReconnectPolicy:
    read_failure_tolerance: int = 5
    reconnect_delay_seconds: float = 2.0
    max_reconnect_delay_seconds: float = 30.0
    max_reconnect_attempts: int | None = None
    open_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 10.0


def is_live_source(source: str) -> bool:
    return source.lower().startswith(LIVE_PREFIXES) or source.isdigit()


def rotate(frame: np.ndarray, rotation: int) -> np.ndarray:
    return frame if rotation == 0 else cv2.rotate(frame, ROTATE_CODES[rotation])


def raw_capture_factory(source: str, policy: ReconnectPolicy) -> Any:
    """Open a dedicated OpenCV capture (one TCP session)."""
    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
        if capture is not None and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:  # noqa: BLE001
                pass
        return capture
    params: list[int] = []
    if source.lower().startswith("rtsp"):
        # UDP reopen often fails after a corrupt frame. TCP is the transport that stays up.
        current = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
        if "rtsp_transport" not in current:
            option = "rtsp_transport;tcp"
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"{current}|{option}" if current else option
    if is_live_source(source):
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params += [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(policy.open_timeout_seconds * 1000)]
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params += [cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(policy.read_timeout_seconds * 1000)]
    capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params) if params else cv2.VideoCapture(source)
    # Tiny decoder buffer: prefer the newest frame over backlog (live only).
    if capture is not None and is_live_source(source) and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001
            pass
    return capture


def default_capture_factory(source: str, policy: ReconnectPolicy) -> Any:
    """Prefer a shared RTSP broker so duplicate camera URLs do not open N TCP sessions."""
    share = os.environ.get("EMPTRACKER_SHARE_RTSP", "1").strip() not in {"0", "false", "False"}
    if share and source.lower().startswith(("rtsp://", "rtsps://")):
        from core.io.shared_live import SharedCapture

        return SharedCapture.open(source, policy, raw_capture_factory)
    return raw_capture_factory(source, policy)


class _LiveGrabber(threading.Thread):
    """Continuously reads a live source, keeping only the newest frame; reconnects on failure."""

    def __init__(self, source: str, policy: ReconnectPolicy, factory: Callable[[str, ReconnectPolicy], Any],
                 capture: Any) -> None:
        super().__init__(name="live-grabber", daemon=True)
        self.source, self.policy, self.factory = source, policy, factory
        self.capture = capture
        self.cond = threading.Condition()
        self.stop_event = threading.Event()
        self.latest: tuple[np.ndarray, float] | None = None
        self.seq = 0
        self.connected = True
        self.lost = False
        self.disconnects = 0
        self.attempts_since_frame = 0

    def run(self) -> None:
        failures = 0
        while not self.stop_event.is_set():
            ok, frame = self.capture.read()
            # Skip corrupt / empty HEVC leftovers without tearing the stream down.
            if ok and frame is not None and frame.size and frame.ndim == 3 and min(frame.shape[:2]) >= 2:
                failures = 0
                self.attempts_since_frame = 0
                with self.cond:
                    self.latest = (frame, time.monotonic())
                    self.seq += 1
                    self.connected = True
                    self.cond.notify_all()
                continue
            failures += 1
            if failures < max(8, self.policy.read_failure_tolerance * 2):
                # Bad frame / brief blip — wait briefly and keep the socket.
                time.sleep(0.05)
                continue
            failures = 0
            with self.cond:
                self.connected = False
                self.disconnects += 1
                self.cond.notify_all()
            logger.warning("Live source %s stopped delivering frames; reconnecting", self.source)
            # SharedCapture.release() drops the broker ref — only release dedicated captures.
            from core.io.shared_live import SharedCapture

            if isinstance(self.capture, SharedCapture):
                # Broker reconnects itself; wait for the next good frame.
                time.sleep(self.policy.reconnect_delay_seconds)
                continue
            self.capture.release()
            if not self._reconnect():
                with self.cond:
                    self.lost = True
                    self.cond.notify_all()
                return

    def _reconnect(self) -> bool:
        """Reconnect with exponential backoff; attempts only reset once a frame is actually received."""
        delay = self.policy.reconnect_delay_seconds
        while not self.stop_event.is_set():
            self.attempts_since_frame += 1
            attempt = self.attempts_since_frame
            if self.policy.max_reconnect_attempts is not None and attempt > self.policy.max_reconnect_attempts:
                logger.error("Giving up on %s after %d reconnect attempts without a frame", self.source, attempt - 1)
                return False
            if self.stop_event.wait(delay):
                return False
            capture = self.factory(self.source, self.policy)
            if capture is not None and capture.isOpened():
                self.capture = capture
                logger.info("Reconnected to %s (attempt %d)", self.source, attempt)
                return True
            if capture is not None:
                capture.release()
            logger.warning("Reconnect attempt %d to %s failed; retrying in %.1fs", attempt, self.source, delay)
            delay = min(max(delay * 2, 0.1), self.policy.max_reconnect_delay_seconds)
        return False

    def stop(self) -> None:
        self.stop_event.set()
        with self.cond:
            self.cond.notify_all()


class VideoReader:
    """Iterates ``FramePacket`` (and, for live sources, ``SourceOutage``) items."""

    def __init__(
        self,
        source: str,
        inference_interval_seconds: float = 1.0,
        tracking_interval_seconds: float | None = None,
        rotation: int = 0,
        yield_all_frames: bool = False,
        recording_start_utc: datetime | None = None,
        reconnect: ReconnectPolicy | None = None,
        capture_factory: Callable[[str, ReconnectPolicy], Any] = default_capture_factory,
        inference_phase_offset_seconds: float = 0.0,
    ) -> None:
        if rotation not in (0, *ROTATE_CODES):
            raise ValueError(f"rotation must be 0, 90, 180, or 270, got {rotation}")
        if inference_interval_seconds < 0:
            raise ValueError("inference_interval_seconds must be >= 0")
        if tracking_interval_seconds is not None and tracking_interval_seconds < 0:
            raise ValueError("tracking_interval_seconds must be >= 0")
        if inference_phase_offset_seconds < 0:
            raise ValueError("inference_phase_offset_seconds must be >= 0")
        self.source = str(source)
        self.interval = float(inference_interval_seconds)
        # Stagger ResNet/inference ticks across cameras; tracking cadence is unchanged.
        self.inference_phase = float(inference_phase_offset_seconds)
        # None disables tracking cadence. 0.0 = track every decoded frame (required for BoT-SORT continuity).
        self.tracking_enabled = tracking_interval_seconds is not None
        self.tracking_interval = 0.0 if tracking_interval_seconds is None else float(tracking_interval_seconds)
        self.rotation = rotation
        self.yield_all_frames = bool(
            yield_all_frames or (self.tracking_enabled and self.tracking_interval == 0.0)
        )
        self.policy = reconnect or ReconnectPolicy()
        self.is_live = is_live_source(self.source)
        self._factory = capture_factory
        self._grabber: _LiveGrabber | None = None
        self.source_lost = False
        self.timestamp_fallback_used = False

        if not self.is_live and not Path(self.source).is_file():
            raise VideoSourceError(f"Video file not found: {self.source}")
        self._capture = None
        self._first_frame: np.ndarray | None = None
        open_delay = self.policy.reconnect_delay_seconds
        while True:
            capture = capture_factory(self.source, self.policy)
            if capture is not None and capture.isOpened():
                self._capture = capture
                break
            if capture is not None:
                capture.release()
            if not self.is_live:
                raise VideoSourceError(f"Could not open video source: {self.source}")
            # Live: keep retrying instead of killing the detection thread.
            logger.warning(
                "Could not open live source %s; retrying in %.1fs",
                self.source, open_delay,
            )
            time.sleep(open_delay)
            open_delay = min(open_delay * 2.0, self.policy.max_reconnect_delay_seconds)

        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps: float | None = fps if math.isfinite(fps) and 0 < fps < 1000 else None
        count = float(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        self.frame_count: int | None = int(count) if not self.is_live and count > 0 else None
        raw_w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        raw_h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if raw_w <= 0 or raw_h <= 0:
            # Wait for the first good frame (shared RTSP may still be connecting).
            deadline = time.monotonic() + max(30.0, self.policy.open_timeout_seconds * 3)
            frame = None
            while time.monotonic() < deadline:
                ok, frame = self._capture.read()
                if ok and frame is not None and getattr(frame, "size", 0):
                    break
                if not self.is_live:
                    raise VideoSourceError(f"Could not read a frame from {self.source}")
                time.sleep(0.2)
            if frame is None or not getattr(frame, "size", 0):
                if not self.is_live:
                    raise VideoSourceError(f"Could not read a frame from {self.source}")
                # Leave placeholders; live grabber will fill size after reconnect.
                raw_w, raw_h = 1920, 1080
                logger.warning("%s: using placeholder size until first frame arrives", self.source)
            else:
                raw_h, raw_w = frame.shape[:2]
                if not self.is_live:
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    self._first_frame = frame
        self.width, self.height = (raw_h, raw_w) if rotation in (90, 270) else (raw_w, raw_h)

        self._monotonic_start = time.monotonic()
        self.session_start_utc = datetime.now(timezone.utc) if self.is_live else None
        self.origin_utc = self.session_start_utc if self.is_live else recording_start_utc
        logger.info(
            "Opened %s source %s: %dx%d (after rotation %d), fps=%s, frames=%s",
            "live" if self.is_live else "file", self.source, self.width, self.height, rotation,
            f"{self.fps:.3f}" if self.fps else "unknown", self.frame_count,
        )

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[FramePacket | SourceOutage]:
        return self._iter_live() if self.is_live else self._iter_file()

    def _wall_clock(self, ts: float) -> datetime | None:
        return self.origin_utc + timedelta(seconds=ts) if self.origin_utc is not None else None

    def _due(
        self,
        ts: float,
        next_due: float | None,
        interval: float,
        *,
        phase_offset: float = 0.0,
    ) -> tuple[bool, float | None]:
        if interval <= 0:
            return True, None
        if next_due is None:
            # Align the first tick onto the phase grid so multi-cam peaks do not collide.
            phase = max(0.0, float(phase_offset))
            if ts + 1e-6 < phase:
                return False, phase
            base = phase
            while base <= ts + 1e-6:
                base += interval
            return True, base
        if ts >= next_due - 1e-6:
            base = next_due
            while base <= ts + 1e-6:
                base += interval
            return True, base
        return False, next_due

    def _iter_file(self) -> Iterator[FramePacket]:
        cap = self._capture
        index, last_ts = 0, None
        next_infer, next_track = None, None
        use_fallback, retrieve_failures = False, 0
        while True:
            if not cap.grab():
                if self.frame_count and index < self.frame_count - 1:
                    logger.warning("File ended at frame %d of %d reported frames", index, self.frame_count)
                logger.info("End of file after %d frames (last timestamp %s)", index, last_ts)
                return
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not use_fallback and (not math.isfinite(pts) or pts < 0 or (last_ts is not None and pts <= last_ts)):
                if self.fps is None:
                    raise VideoSourceError(
                        f"{self.source}: backend gives untrustworthy timestamps and no FPS; "
                        "re-mux the file or decode it with PyAV"
                    )
                use_fallback = self.timestamp_fallback_used = True
                logger.warning("%s: non-increasing PTS at frame %d (%.4f <= %s); falling back to 1/fps "
                               "increments from here", self.source, index, pts, last_ts)
            if use_fallback:
                ts = (last_ts + 1.0 / self.fps) if last_ts is not None else 0.0  # type: ignore[operator]
            else:
                ts = pts
            due_infer, next_infer = self._due(
                ts, next_infer, self.interval, phase_offset=self.inference_phase,
            )
            if self.tracking_enabled:
                due_track, next_track = self._due(ts, next_track, self.tracking_interval)
            else:
                due_track = False
            if due_infer or due_track or self.yield_all_frames:
                ok, frame = cap.retrieve()
                if not ok or frame is None or frame.size == 0:
                    retrieve_failures += 1
                    logger.warning("Failed to decode frame %d at %.3fs", index, ts)
                    if retrieve_failures > 25:
                        raise VideoSourceError(f"{self.source}: too many consecutive decode failures")
                else:
                    retrieve_failures = 0
                    yield FramePacket(
                        rotate(frame, self.rotation), index, ts, self._wall_clock(ts),
                        "fps_estimate" if use_fallback else "pts",
                        run_inference=due_infer, run_tracking=due_track,
                    )
            last_ts = ts
            index += 1

    def _iter_live(self) -> Iterator[FramePacket | SourceOutage]:
        grabber = _LiveGrabber(self.source, self.policy, self._factory, self._capture)
        self._grabber = grabber
        start = self._monotonic_start
        if self._first_frame is not None:
            grabber.latest, grabber.seq = (self._first_frame, start), 1
        grabber.start()
        seen, next_infer, next_track, last_ts = 0, None, None, None
        outage_open, reconnect_pending = False, False
        stall_timeout = self.policy.read_timeout_seconds + 1.0
        while True:
            with grabber.cond:
                grabber.cond.wait_for(
                    lambda: grabber.seq != seen or grabber.lost or grabber.stop_event.is_set()
                    or (not grabber.connected and not outage_open), timeout=stall_timeout,
                )
                seq, latest, connected, lost = grabber.seq, grabber.latest, grabber.connected, grabber.lost
            if grabber.stop_event.is_set():
                return
            now = time.monotonic() - start
            stalled = seq == seen and connected and not lost
            if (lost or not connected or stalled) and not outage_open:
                outage_open, reconnect_pending = True, True
                next_infer = next_track = None
                yield SourceOutage(last_ts, now, self._wall_clock(now),
                                   "stream_stalled" if stalled else "stream_disconnected")
            if lost:
                self.source_lost = True
                return
            if seq == seen or latest is None:
                continue
            frame, captured = latest
            seen = seq
            ts = captured - start
            if last_ts is not None and ts <= last_ts:
                continue
            outage_open = False
            due_infer, next_infer = self._due(
                ts, next_infer, self.interval, phase_offset=self.inference_phase,
            )
            if self.tracking_enabled:
                due_track, next_track = self._due(ts, next_track, self.tracking_interval)
            else:
                due_track = False
            if due_infer or due_track or self.yield_all_frames:
                yield FramePacket(
                    rotate(frame, self.rotation), seq - 1, ts, self._wall_clock(ts), "monotonic",
                    run_inference=due_infer, run_tracking=due_track, after_reconnect=reconnect_pending,
                )
                reconnect_pending = False
            last_ts = ts

    def close(self) -> None:
        if self._grabber is not None:
            self._grabber.stop()
            self._grabber.join(timeout=self.policy.read_timeout_seconds + 2.0)
            capture = self._grabber.capture
            self._grabber = None
            if capture is not None:
                capture.release()
        if self._capture is not None:
            self._capture.release()
