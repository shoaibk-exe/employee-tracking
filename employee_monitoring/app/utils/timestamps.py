"""Server-clock timestamps.

Camera clocks drift. Event time is the time this process received the frame, stored in UTC.
Local timezone is applied only when a response is formatted for people.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def to_local(value: datetime, timezone_name: str) -> datetime:
    return ensure_utc(value).astimezone(ZoneInfo(timezone_name))


def format_clock(value: datetime, timezone_name: str) -> str:
    return to_local(value, timezone_name).strftime("%H:%M:%S")


def local_date_iso(value: datetime, timezone_name: str) -> str:
    return to_local(value, timezone_name).date().isoformat()
