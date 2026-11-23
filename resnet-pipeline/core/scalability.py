"""Multi-camera scalability helpers (previous-pipeline discipline).

These utilities keep capture/display live under load without changing the
detection math when the system is healthy:

- phase offsets stagger expensive ticks across cameras
- coalescing workers keep at most 1 running + 1 pending (newest wins)
- overload shedding skips OPTIONAL work only (never YOLO / ResNet / association)
- async JPEG publish keeps encode off the detection hot path
- runtime speed flags enable TF32 / cuDNN benchmark once per process

Defaults in settings keep single-camera behaviour equivalent to today.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class ScalabilityConfig:
    """Opt-in multi-cam scalability knobs. Safe defaults preserve detection."""

    enabled: bool = True
    phase_offset: bool = True
    async_encode: bool = True
    runtime_speed_flags: bool = True
    # VideoMAE only — YOLO / ResNet / association stay on the sync detection path.
    async_videomae: bool = True
    overload_shedding: bool = True
    overload_budget_seconds: float = 1.5
    overload_consecutive: int = 8


def phase_offset_seconds(camera_id: str, period: float) -> float:
    """Deterministic stagger in ``[0, period)`` so cameras do not peak together."""
    if period <= 0:
        return 0.0
    # Salted digest spreads short ids ("1","2","3") instead of clustering.
    digest = hashlib.md5(f"emptracker-phase:{camera_id}".encode("utf-8")).digest()
    frac = int.from_bytes(digest[:8], "big") / float(2**64)
    return frac * float(period)


def apply_runtime_speed_flags() -> None:
    """Enable common CUDA speed flags once. No-op on CPU / missing torch."""
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info("CUDA speed flags ON (TF32 + cudnn.benchmark)")
    except Exception:  # noqa: BLE001 — never block startup on backend flags
        logger.warning("Could not enable CUDA speed flags", exc_info=True)


class OverloadGuard:
    """Skip optional work after sustained over-budget loops; resume when recovered."""

    def __init__(self, budget_seconds: float, consecutive: int) -> None:
        self.budget = max(0.05, float(budget_seconds))
        self.need = max(1, int(consecutive))
        self._over = 0
        self.shedding = False

    def record(self, loop_seconds: float) -> bool:
        if loop_seconds > self.budget:
            self._over += 1
        else:
            self._over = 0
            if self.shedding:
                logger.info("Overload recovered (loop %.3fs <= budget %.3fs)", loop_seconds, self.budget)
            self.shedding = False
        if self._over >= self.need and not self.shedding:
            self.shedding = True
            logger.warning(
                "Overload shedding ON (loop %.3fs > %.3fs for %d ticks); skipping optional VideoMAE",
                loop_seconds, self.budget, self._over,
            )
        return self.shedding

    def should_shed(self) -> bool:
        return self.shedding


class CoalescingWorker(Generic[T]):
    """Long-lived worker: at most one running job + one pending replacement."""

    def __init__(self, name: str, handler: Callable[[T], None]) -> None:
        self._handler = handler
        self._lock = threading.Lock()
        self._pending: T | None = None
        self._running = False
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._running or self._pending is not None

    def submit(self, job: T) -> None:
        if self._stop.is_set():
            return
        with self._lock:
            self._pending = job
            self._wake.set()

    def close(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            while not self._stop.is_set():
                with self._lock:
                    job = self._pending
                    self._pending = None
                    if job is None:
                        self._running = False
                        break
                    self._running = True
                try:
                    self._handler(job)
                except Exception:  # noqa: BLE001
                    logger.exception("Coalescing worker %s failed", self._thread.name)
                finally:
                    with self._lock:
                        self._running = False
                        if self._pending is not None:
                            self._wake.set()


class AsyncJpegPublisher:
    """Encode + write JPEG off the detection thread; newest frame wins."""

    def __init__(self, write_jpeg: Callable[[bytes], None], quality: int = 70) -> None:
        self._write = write_jpeg
        self._quality = int(quality)
        self._worker = CoalescingWorker[Any]("async-jpeg", self._encode_write)

    def submit(self, frame: Any) -> None:
        # Copy so the detection thread can reuse / overwrite the source buffer.
        try:
            payload = frame.copy()
        except Exception:  # noqa: BLE001
            payload = frame
        self._worker.submit(payload)

    def close(self) -> None:
        self._worker.close()

    def _encode_write(self, frame: Any) -> None:
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        if not ok:
            return
        self._write(encoded.tobytes())


_SPEED_FLAGS_DONE = False
_SPEED_FLAGS_LOCK = threading.Lock()


def ensure_runtime_speed_flags(enabled: bool = True) -> None:
    """Idempotent process-level CUDA speed flags."""
    global _SPEED_FLAGS_DONE
    if not enabled:
        return
    with _SPEED_FLAGS_LOCK:
        if _SPEED_FLAGS_DONE:
            return
        apply_runtime_speed_flags()
        _SPEED_FLAGS_DONE = True
