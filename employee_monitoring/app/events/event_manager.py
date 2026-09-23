"""Publish state-change events once. Callers do not log every processed frame."""

from __future__ import annotations

import logging
from typing import Callable

from app.events.event_types import Event
from app.logging_config import log_event

logger = logging.getLogger(__name__)

Listener = Callable[[Event], None]


class EventManager:
    def __init__(self, listener: Listener | None = None) -> None:
        self._listener = listener
        self.published: list[Event] = []

    def publish(self, event: Event) -> Event:
        self.published.append(event)
        log_event(
            logger,
            event.event_type.value,
            event=event.event_type.value,
            employee=event.employee_id,
            camera=event.camera_id,
            confidence=None if event.confidence is None else f"{event.confidence:.3f}",
        )
        if self._listener is not None:
            self._listener(event)
        return event
