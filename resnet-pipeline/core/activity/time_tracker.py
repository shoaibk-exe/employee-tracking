"""Accumulate VideoMAE class time per workstation for calendar-day reporting."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone, tzinfo
from typing import Iterable

from core.occupancy.time_tracker import split_by_day

ACTIVITY_CLASSES = ("computer_interaction", "neither", "phone_using")


class ActivityTimeTracker:
    """Counts source-time seconds spent in each VideoMAE desk class.

    Every ``observe`` call closes the open interval since the previous call, so
    day totals keep growing even when the predicted class stays the same.
    """

    def __init__(
        self,
        workstation_id: str,
        class_names: Iterable[str] | None = None,
        origin_utc: datetime | None = None,
        report_tz: tzinfo = timezone.utc,
    ) -> None:
        self.workstation_id = workstation_id
        self.class_names = list(class_names or ACTIVITY_CLASSES)
        self.totals: dict[str, float] = {name: 0.0 for name in self.class_names}
        self.daily: dict[tuple[date, str], float] = defaultdict(float)
        self.origin_utc = origin_utc
        self.report_tz = report_tz
        self.current: str | None = None
        self.last_ts: float | None = None

    def _seconds_key(self, name: str) -> str:
        return f"{name}_seconds"

    def _credit(self, start: float, end: float, label: str) -> None:
        if end <= start or label not in self.totals:
            return
        duration = end - start
        self.totals[label] += duration
        if self.origin_utc is not None:
            for day, seconds in split_by_day(start, end, self.origin_utc, self.report_tz):
                self.daily[(day, label)] += seconds

    def observe(self, ts: float, predicted_class: str | None) -> None:
        """Credit elapsed time to the previous class, then remember the new class."""
        if predicted_class is not None and predicted_class not in self.totals:
            self.totals[predicted_class] = 0.0
            if predicted_class not in self.class_names:
                self.class_names.append(predicted_class)
        if self.last_ts is not None and self.current is not None and ts > self.last_ts:
            self._credit(self.last_ts, ts, self.current)
        self.current = predicted_class
        self.last_ts = ts

    def totals_at(self, ts: float) -> dict[str, float]:
        """Totals including any time after the last observe up to ``ts``."""
        out = dict(self.totals)
        if self.current is not None and self.last_ts is not None and ts > self.last_ts:
            out[self.current] = out.get(self.current, 0.0) + (ts - self.last_ts)
        return out

    def seconds_by_day(self, ts: float) -> dict[date, dict[str, float]]:
        """Per-calendar-day seconds, including time after the last observe."""
        by_day: dict[date, dict[str, float]] = defaultdict(dict)
        for (day, label), seconds in self.daily.items():
            by_day[day][label] = by_day[day].get(label, 0.0) + seconds
        if (
            self.current is not None
            and self.last_ts is not None
            and self.origin_utc is not None
            and ts > self.last_ts
        ):
            for day, seconds in split_by_day(self.last_ts, ts, self.origin_utc, self.report_tz):
                by_day[day][self.current] = by_day[day].get(self.current, 0.0) + seconds
        # Ensure every known class key exists for stable CSV / JSON fields.
        for day, values in by_day.items():
            for name in self.class_names:
                values.setdefault(name, 0.0)
        return {day: dict(values) for day, values in by_day.items()}

    def as_seconds_fields(self, totals: dict[str, float]) -> dict[str, float]:
        return {self._seconds_key(name): round(float(totals.get(name, 0.0) or 0.0), 3) for name in self.class_names}
