"""In-memory face gallery. Several templates per employee, never a single photo."""

from __future__ import annotations

import threading

import numpy as np


class FaceGallery:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._templates: dict[str, np.ndarray] = {}
        self._names: dict[str, str] = {}

    def replace(
        self,
        templates: dict[str, np.ndarray],
        names: dict[str, str] | None = None,
    ) -> None:
        with self._lock:
            self._templates = templates
            self._names = dict(names or {})

    def snapshot(self) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        with self._lock:
            return dict(self._templates), dict(self._names)

    def name_for(self, employee_id: str) -> str:
        with self._lock:
            return self._names.get(employee_id, employee_id)
