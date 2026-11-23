"""Write-behind wrapper so database commits never block the video loop.

Reads and ``next_global_id`` (which must be transactional and immediate) go
straight to the wrapped repository; inserts/upserts are queued and applied in
order by one background thread.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any

logger = logging.getLogger(__name__)

_WRITE_METHODS = (
    "upsert_identity", "add_embedding", "add_tracklet", "add_event", "add_association", "upsert_camera",
    "reassign_embeddings",
)


class WriteBehindRepository:
    def __init__(self, inner: Any, max_queue: int = 20000) -> None:
        self._inner = inner
        self._queue: queue.Queue[tuple[str, tuple, dict] | None] = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._run, name="db-writer", daemon=True)
        self.dropped = 0
        self._thread.start()

    def __getattr__(self, name: str) -> Any:
        if name in _WRITE_METHODS:
            def enqueue(*args: Any, **kwargs: Any) -> None:
                try:
                    self._queue.put_nowait((name, args, kwargs))
                except queue.Full:
                    self.dropped += 1
                    if self.dropped % 100 == 1:
                        logger.warning("DB write queue full; dropped %d writes", self.dropped)
            return enqueue
        return getattr(self._inner, name)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            name, args, kwargs = item
            try:
                getattr(self._inner, name)(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("DB write %s failed: %s", name, exc)
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = 10.0) -> None:
        done = threading.Event()

        def waiter() -> None:
            self._queue.join()
            done.set()

        threading.Thread(target=waiter, daemon=True).start()
        if not done.wait(timeout):
            logger.warning("DB write-behind flush timed out with ~%d pending", self._queue.qsize())

    def close(self, timeout: float = 10.0) -> None:
        self.flush(timeout)
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
