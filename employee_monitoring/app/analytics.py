"""Day totals from sessions. Office presence and desk presence stay separate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from app.utils.timestamps import ensure_utc, to_local


@dataclass
class TimeSpan:
    start: datetime
    end: datetime | None = None


def _closed(span: TimeSpan, now: datetime) -> tuple[datetime, datetime]:
    start = ensure_utc(span.start)
    end = ensure_utc(span.end) if span.end is not None else ensure_utc(now)
    if end < start:
        end = start
    return start, end


def _overlap(a: tuple[datetime, datetime], b: tuple[datetime, datetime]) -> float:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    return max(0.0, (end - start).total_seconds())


def _day_window(day: date, timezone_name: str) -> tuple[datetime, datetime]:
    start_local = datetime.combine(day, time.min).replace(tzinfo=to_local(datetime.now(timezone.utc), timezone_name).tzinfo)
    # Rebuild with the named zone so DST rules apply.
    from zoneinfo import ZoneInfo

    zone = ZoneInfo(timezone_name)
    start = datetime.combine(day, time.min, zone)
    return start.astimezone(timezone.utc), (start + timedelta(days=1)).astimezone(timezone.utc)


def spans_on_day(spans: list[TimeSpan], day: date, timezone_name: str, now: datetime) -> list[tuple[datetime, datetime]]:
    window = _day_window(day, timezone_name)
    selected: list[tuple[datetime, datetime]] = []
    for span in spans:
        closed = _closed(span, now)
        if _overlap(closed, window) <= 0 and to_local(closed[0], timezone_name).date() != day:
            continue
        # Sessions are attributed to the local day they started, including a shift that passes midnight.
        if to_local(closed[0], timezone_name).date() == day:
            selected.append(closed)
    return selected


def total_seconds(spans: list[tuple[datetime, datetime]]) -> int:
    return int(round(sum((end - start).total_seconds() for start, end in spans)))


def away_seconds(office: list[tuple[datetime, datetime]], desk: list[tuple[datetime, datetime]]) -> int:
    """Away-from-desk while inside the office. Time outside the office is not away time."""
    away = 0.0
    for block in office:
        covered = sum(_overlap(block, seat) for seat in desk)
        away += max(0.0, (block[1] - block[0]).total_seconds() - covered)
    return int(round(away))


def longest_away_seconds(office: list[tuple[datetime, datetime]], desk: list[tuple[datetime, datetime]]) -> int:
    longest = 0.0
    for block in office:
        cursor = block[0]
        pieces = sorted(
            (max(block[0], seat[0]), min(block[1], seat[1]))
            for seat in desk
            if _overlap(block, seat) > 0
        )
        for start, end in pieces:
            if start > cursor:
                longest = max(longest, (start - cursor).total_seconds())
            cursor = max(cursor, end)
        if block[1] > cursor:
            longest = max(longest, (block[1] - cursor).total_seconds())
    return int(round(longest))
