"""Structured logs for state changes, recognition decisions, and failures.

Frame-by-frame logs are intentionally omitted. A camera at 10 FPS would hide the events that matter.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


_STANDARD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class StructuredFormatter(logging.Formatter):
    """Append explicit extra fields so logs stay grep-friendly."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras: list[str] = []
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_"):
                continue
            if value is None:
                continue
            extras.append(f"{key}={value}")
        if extras:
            return f"{base} {' '.join(extras)}"
        return base


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        StructuredFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    """Log one decision. Callers pass business fields, never embeddings or images."""
    logger.info(message, extra=fields)
