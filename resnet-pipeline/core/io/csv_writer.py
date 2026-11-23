"""Incremental, append-only CSV reports.

Rows are appended and flushed regularly, so a crash loses at most the last
``flush_interval_seconds`` of records. Each run has its own ``session_id``, so
several runs can share the same files without overwriting one another.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.occupancy.classifier import Prediction
from core.occupancy.state_manager import StateTransition
from core.occupancy.time_tracker import Interval

logger = logging.getLogger(__name__)

PREDICTION_FIELDS = ["camera_id", "session_id", "timestamp_seconds", "wall_clock_utc", "workstation_id",
                     "predicted_class", "confidence", "probabilities_json"]
EVENT_FIELDS = ["camera_id", "session_id", "workstation_id", "previous_state", "new_state",
                "transition_timestamp", "confirmation_timestamp", "reason"]
INTERVAL_FIELDS = ["camera_id", "session_id", "workstation_id", "state", "start_timestamp", "end_timestamp",
                   "duration_seconds", "end_reason"]
SUMMARY_FIELDS = ["camera_id", "session_id", "workstation_id", "occupied_seconds", "empty_seconds",
                  "standing_near_seconds", "unknown_seconds", "observed_seconds", "transition_count"]
DAILY_FIELDS = ["camera_id", "session_id", "report_date", "workstation_id", "state", "duration_seconds"]


class CsvSchemaError(RuntimeError):
    """An existing CSV file has a different header than this version writes."""


class CsvAppender:
    def __init__(self, path: Path, fieldnames: Sequence[str], flush_interval_seconds: float = 5.0) -> None:
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.flush_interval = flush_interval_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.is_file() and self.path.stat().st_size > 0
        if exists:
            with self.path.open(newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle), [])
            if header != self.fieldnames:
                raise CsvSchemaError(
                    f"{self.path} has columns {header}, expected {self.fieldnames}. "
                    "Move the old file aside or choose another output directory."
                )
        self._handle = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
        if not exists:
            self._writer.writeheader()
            self._handle.flush()
        self._last_flush = time.monotonic()
        self.rows = 0

    def write(self, row: Mapping[str, Any]) -> None:
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self.rows += 1
        if time.monotonic() - self._last_flush >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._last_flush = time.monotonic()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()


def _ts(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


class ReportWriter:
    """Owns predictions/events/intervals/summary CSVs for one session."""

    def __init__(self, directory: Path, camera_id: str, session_id: str, flush_interval_seconds: float = 5.0) -> None:
        self.directory = Path(directory)
        self.camera_id = camera_id
        self.session_id = session_id
        flush = flush_interval_seconds
        self.predictions = CsvAppender(self.directory / "predictions.csv", PREDICTION_FIELDS, flush)
        self.events = CsvAppender(self.directory / "events.csv", EVENT_FIELDS, flush)
        self.intervals = CsvAppender(self.directory / "intervals.csv", INTERVAL_FIELDS, flush)
        self._base = {"camera_id": camera_id, "session_id": session_id}

    def write_prediction(self, ts: float, wall_clock: datetime | None, workstation_id: str,
                         prediction: Prediction) -> None:
        self.predictions.write({
            **self._base,
            "timestamp_seconds": _ts(ts),
            "wall_clock_utc": wall_clock.isoformat(timespec="milliseconds") if wall_clock else "",
            "workstation_id": workstation_id,
            "predicted_class": prediction.predicted_class,
            "confidence": f"{prediction.confidence:.4f}",
            "probabilities_json": json.dumps({k: round(v, 4) for k, v in prediction.probabilities.items()}),
        })

    def write_event(self, transition: StateTransition) -> None:
        self.events.write({
            **self._base,
            "workstation_id": transition.workstation_id,
            "previous_state": transition.previous_state or "",
            "new_state": transition.new_state,
            "transition_timestamp": _ts(transition.transition_timestamp),
            "confirmation_timestamp": _ts(transition.confirmation_timestamp),
            "reason": transition.reason,
        })

    def write_interval(self, interval: Interval) -> None:
        self.intervals.write({
            **self._base,
            "workstation_id": interval.workstation_id,
            "state": interval.state,
            "start_timestamp": _ts(interval.start),
            "end_timestamp": _ts(interval.end),
            "duration_seconds": _ts(interval.duration),
            "end_reason": interval.end_reason,
        })

    def write_summaries(self, rows: Iterable[tuple[str, Mapping[str, Any]]], extra_states: Sequence[str] = ()) -> Path:
        fields = SUMMARY_FIELDS[:-2] + [f"{s}_seconds" for s in extra_states] + SUMMARY_FIELDS[-2:]
        path = self.directory / "summary.csv" if not extra_states else self.directory / "summary_extended.csv"
        appender = CsvAppender(path, fields, 0.0)
        try:
            for workstation_id, row in rows:
                appender.write({**self._base, "workstation_id": workstation_id,
                                **{k: row.get(k, 0.0) for k in fields if k.endswith(("_seconds", "_count"))}})
        finally:
            appender.close()
        return path

    def write_daily(self, rows: Iterable[tuple[str, date, str, float]]) -> Path:
        path = self.directory / "summary_daily.csv"
        appender = CsvAppender(path, DAILY_FIELDS, 0.0)
        try:
            for workstation_id, day, state, seconds in rows:
                appender.write({**self._base, "report_date": day.isoformat(), "workstation_id": workstation_id,
                                "state": state, "duration_seconds": _ts(seconds)})
        finally:
            appender.close()
        return path

    def flush(self) -> None:
        for appender in (self.predictions, self.events, self.intervals):
            appender.flush()

    def close(self) -> None:
        for appender in (self.predictions, self.events, self.intervals):
            appender.close()
