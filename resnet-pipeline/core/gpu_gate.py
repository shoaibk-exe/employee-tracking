"""One GPU job at a time — across threads AND camera worker processes.

Detector, ReID, and VideoMAE take this lock around their GPU work.
Without a cross-process lock, each ``api.worker`` / hub thread can allocate
activations at once and the 12GB card OOMs even with one heavy model.
"""

from __future__ import annotations

import atexit
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_thread_gate = threading.RLock()
_LOCK_PATH = Path(os.environ.get(
    "EMPTRACKER_GPU_LOCK",
    "/tmp/emptracker_gpu.lock",
))
_file_handle: object | None = None
_file_lock = threading.Lock()


def _acquire_file_lock() -> None:
    """Exclusive flock so separate camera processes take turns on the GPU."""
    global _file_handle
    import fcntl

    with _file_lock:
        _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        if _file_handle is None:
            _file_handle = open(_LOCK_PATH, "a+", encoding="utf-8")
        handle = _file_handle
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release_file_lock() -> None:
    import fcntl

    with _file_lock:
        handle = _file_handle
    if handle is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _close_file_lock() -> None:
    global _file_handle
    with _file_lock:
        if _file_handle is not None:
            try:
                _file_handle.close()
            except Exception:  # noqa: BLE001
                pass
            _file_handle = None


atexit.register(_close_file_lock)


@contextmanager
def gpu_gate() -> Iterator[None]:
    with _thread_gate:
        _acquire_file_lock()
        try:
            yield
        finally:
            _release_file_lock()
