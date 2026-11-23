"""Per-workstation state interval and duration accounting.

Intervals are contiguous and non-overlapping: each confirmed transition closes the
open interval at the transition timestamp and opens the next one there, so every
second between the first observation and the last valid observation is counted in
exactly one state. Durations use source timestamps (media time or monotonic elapsed
time), never frame counts, so midnight or wall-clock adjustments cannot affect them.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Iterable

from core.occupancy.state_manager import UNKNOWN, StateTransition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Interval:
    workstation_id: str
    state: str
    start: float
    end: float
    end_reason: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def truncated(self) -> bool:
        return self.end_reason != "transition"


def split_by_day(start: float, end: float, origin_utc: datetime, tz: tzinfo) -> list[tuple[date, float]]:
    """Split ``[start, end)`` seconds-since-origin into per-local-calendar-day durations."""
    if end <= start:
        return []
    parts: list[tuple[date, float]] = []
    cursor = origin_utc + timedelta(seconds=start)
    stop = origin_utc + timedelta(seconds=end)
    while cursor < stop:
        local = cursor.astimezone(tz)
        next_midnight_local = datetime.combine(local.date() + timedelta(days=1), datetime.min.time(), tz)
        boundary = min(stop, next_midnight_local.astimezone(timezone.utc))
        parts.append((local.date(), (boundary - cursor).total_seconds()))
        cursor = boundary
    return parts


class WorkstationTimeTracker:
    def __init__(
        self,
        workstation_id: str,
        states: Iterable[str],
        origin_utc: datetime | None = None,
        report_tz: tzinfo = timezone.utc,
    ) -> None:
        self.workstation_id = workstation_id
        self.states = list(dict.fromkeys([*states, UNKNOWN]))
        self.totals: dict[str, float] = {s: 0.0 for s in self.states}
        self.daily: dict[tuple[date, str], float] = defaultdict(float)
        self.origin_utc = origin_utc
        self.report_tz = report_tz
        self.open_state: str | None = None
        self.open_start: float | None = None
        self.transition_count = 0
        self.last_observation: float | None = None
        self.finalized = False

    def _close(self, end: float, reason: str) -> Interval | None:
        assert self.open_state is not None and self.open_start is not None
        interval = Interval(self.workstation_id, self.open_state, self.open_start, end, reason)
        self.totals[interval.state] = self.totals.get(interval.state, 0.0) + interval.duration
        if self.origin_utc is not None:
            for day, seconds in split_by_day(interval.start, interval.end, self.origin_utc, self.report_tz):
                self.daily[(day, interval.state)] += seconds
        return interval if interval.duration > 0 else None

    def observe(self, ts: float) -> None:
        """Record a valid observation time; used to truncate the final interval."""
        if self.last_observation is None or ts > self.last_observation:
            self.last_observation = ts

    def apply(self, transition: StateTransition) -> Interval | None:
        """Apply a state transition; returns the interval it closed (None if none/zero-length)."""
        if self.finalized:
            raise RuntimeError(f"{self.workstation_id}: tracker already finalized")
        ts = transition.transition_timestamp
        closed = None
        if self.open_state is not None:
            assert self.open_start is not None
            if ts < self.open_start:
                raise ValueError(
                    f"{self.workstation_id}: transition at {ts} precedes open interval start {self.open_start}"
                )
            closed = self._close(ts, "transition")
        if transition.previous_state is not None:
            self.transition_count += 1
        self.open_state, self.open_start = transition.new_state, ts
        if transition.new_state not in self.totals:
            self.totals[transition.new_state] = 0.0
            self.states.append(transition.new_state)
        return closed

    def finalize(self, reason: str, end_ts: float | None = None) -> Interval | None:
        """Close the open interval at the last valid observation (never beyond it)."""
        if self.finalized:
            return None
        self.finalized = True
        if self.open_state is None or self.open_start is None:
            return None
        end = self.last_observation if end_ts is None else end_ts
        if self.last_observation is not None:
            end = min(end, self.last_observation) if end is not None else self.last_observation
        end = max(end if end is not None else self.open_start, self.open_start)
        interval = self._close(end, f"truncated_{reason}")
        self.open_state = self.open_start = None
        return interval

    def current_duration(self, now: float) -> float:
        if self.open_start is None:
            return 0.0
        return max(0.0, now - self.open_start)

    def totals_at(self, now: float | None = None) -> dict[str, float]:
        """Totals including the open interval up to ``now`` (or the last observation)."""
        result = dict(self.totals)
        if self.open_state is not None and self.open_start is not None:
            until = self.last_observation if now is None else now
            if until is not None:
                result[self.open_state] = result.get(self.open_state, 0.0) + max(0.0, until - self.open_start)
        return result

    def seconds_by_day(self, now: float | None = None) -> dict[date, dict[str, float]]:
        """Closed time plus the open interval, split at midnights in the report timezone.

        Keys are state names (``occupied``, ``empty``, …). The sum across days equals
        ``totals_at(now)`` when ``origin_utc`` is set. Without an origin, the caller
        assigns the whole total to one calendar day.
        """
        if self.origin_utc is None:
            return {}
        by_day: dict[date, dict[str, float]] = {}
        for (day, state), seconds in self.daily.items():
            bucket = by_day.setdefault(day, {})
            bucket[state] = bucket.get(state, 0.0) + float(seconds)
        if self.open_state is not None and self.open_start is not None:
            until = self.last_observation if now is None else now
            if until is not None and until > self.open_start:
                for day, seconds in split_by_day(self.open_start, until, self.origin_utc, self.report_tz):
                    bucket = by_day.setdefault(day, {})
                    bucket[self.open_state] = bucket.get(self.open_state, 0.0) + float(seconds)
        return by_day

    def summary(self, now: float | None = None) -> dict[str, float | int]:
        totals = self.totals_at(now)
        observed = sum(v for k, v in totals.items() if k != UNKNOWN)
        row: dict[str, float | int] = {f"{state}_seconds": round(seconds, 3) for state, seconds in totals.items()}
        row["observed_seconds"] = round(observed, 3)
        row["transition_count"] = self.transition_count
        return row
