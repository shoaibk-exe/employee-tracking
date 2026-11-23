"""One RTSP TCP session shared by every camera pointing at the same URL.

Opening the same MediaMTX / camera URL three times fights the server and
freezes all viewers. This broker keeps a single OpenCV capture and fans the
newest good frame out to every subscriber.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

_BROKERS: dict[str, "SharedRtspBroker"] = {}
_BROKERS_LOCK = threading.Lock()


def _frame_ok(frame: np.ndarray | None) -> bool:
    if frame is None:
        return False
    try:
        if frame.size == 0 or frame.ndim != 3 or frame.shape[0] < 2 or frame.shape[1] < 2:
            return False
        # Reject all-zero / decode-failed planes that OpenCV sometimes returns.
        return bool(frame.strides) and frame.dtype.kind in "iu"
    except Exception:  # noqa: BLE001
        return False


class SharedRtspBroker(threading.Thread):
    """Background reader for one live URL."""

    def __init__(
        self,
        source: str,
        policy: Any,
        factory: Callable[[str, Any], Any],
    ) -> None:
        super().__init__(name=f"shared-rtsp:{source[-24:]}", daemon=True)
        self.source = source
        self.policy = policy
        self.factory = factory
        self.refs = 0
        self.stop_event = threading.Event()
        self.cond = threading.Condition()
        self.latest: tuple[np.ndarray, float] | None = None
        self.seq = 0
        self.connected = False
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self._capture: Any = None

    def run(self) -> None:
        delay = float(self.policy.reconnect_delay_seconds)
        failures = 0
        while not self.stop_event.is_set():
            if self._capture is None or not self._capture.isOpened():
                capture = self.factory(self.source, self.policy)
                if capture is None or not capture.isOpened():
                    if capture is not None:
                        capture.release()
                    with self.cond:
                        self.connected = False
                        self.cond.notify_all()
                    logger.warning(
                        "Shared RTSP open failed for %s; retry in %.1fs",
                        self.source, delay,
                    )
                    if self.stop_event.wait(delay):
                        break
                    delay = min(delay * 2.0, float(self.policy.max_reconnect_delay_seconds))
                    continue
                import cv2

                self._capture = capture
                self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                self.fps = fps if fps > 0 else 0.0
                # First good frame fills size if backend reported 0x0.
                if self.width <= 0 or self.height <= 0:
                    ok0, frame0 = capture.read()
                    if ok0 and _frame_ok(frame0):
                        self.height, self.width = frame0.shape[:2]
                        with self.cond:
                            self.latest = (frame0.copy(), time.monotonic())
                            self.seq += 1
                delay = float(self.policy.reconnect_delay_seconds)
                failures = 0
                with self.cond:
                    self.connected = True
                    self.cond.notify_all()
                logger.info("Shared RTSP connected %s (%dx%d)", self.source, self.width, self.height)

            ok, frame = self._capture.read()
            if ok and _frame_ok(frame):
                failures = 0
                with self.cond:
                    self.latest = (frame.copy(), time.monotonic())
                    self.seq += 1
                    self.connected = True
                    self.cond.notify_all()
                continue

            # Bad / empty / corrupt decode — skip without tearing the session down immediately.
            failures += 1
            if failures < max(3, int(self.policy.read_failure_tolerance)):
                time.sleep(0.02)
                continue
            logger.warning(
                "Shared RTSP %s: %d bad/empty frames; reconnecting",
                self.source, failures,
            )
            failures = 0
            try:
                self._capture.release()
            except Exception:  # noqa: BLE001
                pass
            self._capture = None
            with self.cond:
                self.connected = False
                self.cond.notify_all()

        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:  # noqa: BLE001
                pass
            self._capture = None

    def acquire(self) -> None:
        self.refs += 1

    def release(self) -> None:
        self.refs = max(0, self.refs - 1)
        if self.refs == 0:
            self.stop_event.set()
            with self.cond:
                self.cond.notify_all()


class SharedCapture:
    """Minimal VideoCapture stand-in backed by ``SharedRtspBroker``."""

    def __init__(self, broker: SharedRtspBroker) -> None:
        self._broker = broker
        self._seen = 0
        self._released = False
        broker.acquire()

    @classmethod
    def open(cls, source: str, policy: Any, factory: Callable[[str, Any], Any]) -> "SharedCapture":
        with _BROKERS_LOCK:
            broker = _BROKERS.get(source)
            if broker is None or not broker.is_alive():
                # Real open must not recurse into SharedCapture.
                broker = SharedRtspBroker(source, policy, factory)
                _BROKERS[source] = broker
                broker.start()
            return cls(broker)

    def isOpened(self) -> bool:
        # True once the broker thread exists; open may still be retrying.
        return self._broker.is_alive() and not self._broker.stop_event.is_set()

    def read(self) -> tuple[bool, np.ndarray | None]:
        broker = self._broker
        with broker.cond:
            broker.cond.wait_for(
                lambda: broker.seq != self._seen or broker.stop_event.is_set(),
                timeout=float(broker.policy.read_timeout_seconds),
            )
            if broker.seq == self._seen or broker.latest is None:
                return False, None
            self._seen = broker.seq
            frame, _ = broker.latest
            return True, frame.copy()

    def grab(self) -> bool:
        ok, _ = self.read()
        return ok

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        return self.read()

    def get(self, prop: int) -> float:
        import cv2

        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._broker.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._broker.height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self._broker.fps or 20.0)
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 0.0
        return 0.0

    def set(self, _prop: int, _value: float) -> bool:
        return False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        broker = self._broker
        with _BROKERS_LOCK:
            broker.release()
            if broker.refs == 0:
                broker.join(timeout=float(broker.policy.read_timeout_seconds) + 2.0)
                if _BROKERS.get(broker.source) is broker:
                    _BROKERS.pop(broker.source, None)
