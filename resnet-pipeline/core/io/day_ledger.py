"""Per-camera calendar-day totals that survive detection restarts.

Each detection start is a new session and its own counters begin at zero.
This file keeps every session's seconds under the local calendar day they
belong to. Publishing replaces that session's row, so a restart cannot
double-count it, and a later session is added on top.

A session that runs across midnight is split at midnight in the report
timezone. Yesterday's file keeps yesterday's seconds only.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

SESSION_START = re.compile(r"^(\d{8}T\d{6})Z")
SKIP_SECONDS = {"observed_seconds"}


def session_start_utc(session_id: str) -> datetime | None:
    match = SESSION_START.match(session_id or "")
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def _num(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


class DayLedger:
    def __init__(self, directory: str | Path, report_timezone: ZoneInfo | Any) -> None:
        self.directory = Path(directory) / "days"
        self.timezone = report_timezone

    def _tz_name(self) -> str:
        return str(getattr(self.timezone, "key", None) or self.timezone)

    def _path(self, day: date) -> Path:
        return self.directory / f"{day.isoformat()}.json"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        handle = (self.directory / ".lock").open("a", encoding="utf-8")
        try:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_UN)
            finally:
                handle.close()

    def _empty(self, day: date) -> dict[str, Any]:
        return {"version": 1, "date": day.isoformat(), "timezone": self._tz_name(), "sessions": {}}

    def _read(self, day: date) -> dict[str, Any]:
        path = self._path(day)
        if not path.is_file():
            return self._empty(day)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable day file %s: %s", path, exc)
            return self._empty(day)
        if not isinstance(data, dict) or not isinstance(data.get("sessions"), dict):
            logger.warning("Ignoring day file %s with an unexpected shape", path)
            return self._empty(day)
        return data

    def _write(self, day: date, data: dict[str, Any]) -> None:
        path = self._path(day)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)

    def record_session(
        self,
        day: date,
        session_id: str,
        workstations: Mapping[str, Mapping[str, Any]],
        *,
        closed: bool,
    ) -> None:
        """Replace this session's seconds for ``day``. A later call updates the same row."""
        with self._locked():
            data = self._read(day)
            sessions = data.setdefault("sessions", {})
            previous = sessions.get(session_id) or {}
            prev_ws = previous.get("workstations") or {}
            merged: dict[str, dict[str, Any]] = {}
            for ws_id, row in workstations.items():
                previous_row = prev_ws.get(str(ws_id)) or {}
                item = {key: value for key, value in previous_row.items() if not str(key).endswith("_seconds")}
                for key, value in row.items():
                    if str(key).endswith("_seconds"):
                        item[key] = round(_num(value), 3)
                if row.get("bbox"):
                    item["bbox"] = [int(v) for v in row["bbox"]]
                if row.get("state"):
                    item["state"] = row["state"]
                if row.get("confidence") is not None:
                    item["confidence"] = row["confidence"]
                if row.get("predicted_class"):
                    item["predicted_class"] = row["predicted_class"]
                for key in ("activity_class", "activity_confidence", "yolo_activity"):
                    if row.get(key) is not None:
                        item[key] = row[key]
                merged[str(ws_id)] = item
            sessions[session_id] = {
                "closed": bool(closed) or bool(previous.get("closed")),
                "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "workstations": merged,
            }
            self._write(day, data)

    def import_summary(self, path: str | Path) -> int:
        """Add finished sessions that are not already stored. Does not replace a live row."""
        path = Path(path)
        if not path.is_file():
            return 0
        grouped: dict[str, dict[str, dict[str, float]]] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                session_id = str(row.get("session_id") or "")
                ws_id = str(row.get("workstation_id") or "")
                start = session_start_utc(session_id)
                if start is None or not ws_id:
                    continue
                seconds = {
                    key: round(_num(value), 3)
                    for key, value in row.items()
                    if key.endswith("_seconds") and key not in SKIP_SECONDS
                }
                grouped.setdefault(session_id, {})[ws_id] = seconds
        added = 0
        with self._locked():
            for session_id, workstations in grouped.items():
                start = session_start_utc(session_id)
                assert start is not None
                day = start.astimezone(self.timezone).date()
                data = self._read(day)
                sessions = data.setdefault("sessions", {})
                if session_id in sessions:
                    continue
                sessions[session_id] = {
                    "closed": True,
                    "updated_utc": start.isoformat(timespec="seconds"),
                    "source": "summary",
                    "workstations": workstations,
                }
                self._write(day, data)
                added += 1
        if added:
            logger.info("Added %d finished sessions to the daily totals in %s", added, self.directory)
        return added

    def combined(self, day: date) -> dict[str, Any]:
        """Sum sessions for ``day``.

        Only workstations present in the newest session are returned (avoids ghost desks
        from older ROI layouts). Seconds for those IDs are still summed across sessions.
        The newest session supplies live state / VideoMAE / YOLO labels.
        """
        with self._locked():
            data = self._read(day)
        sessions = data.get("sessions") or {}
        ordered = sorted(sessions.items(), key=lambda item: (item[1].get("updated_utc") or "", item[0]))
        latest_ids: set[str] = set()
        if ordered:
            latest_ids = {
                str(ws_id) for ws_id in (ordered[-1][1].get("workstations") or {})
            }
        acc: dict[str, dict[str, Any]] = {}
        for _session_id, session in ordered:
            for ws_id, row in (session.get("workstations") or {}).items():
                key = str(ws_id)
                if latest_ids and key not in latest_ids:
                    continue
                slot = acc.setdefault(key, {
                    "workstation_id": key,
                    "state": "unknown",
                    "confidence": None,
                    "bbox": [],
                })
                for field, value in row.items():
                    if str(field).endswith("_seconds"):
                        slot[field] = round(_num(slot.get(field)) + _num(value), 3)
                if row.get("bbox"):
                    slot["bbox"] = [int(v) for v in row["bbox"]]
                if row.get("state"):
                    slot["state"] = row["state"]
                if row.get("confidence") is not None:
                    slot["confidence"] = row["confidence"]
                if row.get("predicted_class"):
                    slot["predicted_class"] = row["predicted_class"]
                for field in ("activity_class", "activity_confidence", "yolo_activity"):
                    if row.get(field) is not None:
                        slot[field] = row[field]
        return {
            "date": day.isoformat(),
            "timezone": self._tz_name(),
            "session_count": len(sessions),
            "workstations": [acc[key] for key in sorted(acc)],
        }
