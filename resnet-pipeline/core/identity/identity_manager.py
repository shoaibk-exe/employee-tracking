"""Verified employee identity for person tracks.

Rules:
* A tracker ID is never an employee ID. Every track starts as ``UNKNOWN_EMPLOYEE``.
* An identity is assigned only from *verified* evidence produced by an external source
  (badge reader, administrator label, separately consented and audited verification
  service). This module performs no biometric processing and stores no embeddings.
* Evidence below ``min_confidence``, for unknown/inactive employees, or from another tenant
  is rejected and logged.
* Conflicts (one employee on two live tracks, or two employees on one track) set every
  involved track back to ``UNKNOWN_EMPLOYEE``; nothing is guessed.
* Assignments end when the track is removed or the tracking session resets.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Protocol

logger = logging.getLogger(__name__)

UNKNOWN_EMPLOYEE = "UNKNOWN_EMPLOYEE"


class IdentityError(ValueError):
    """Malformed employee directory or identity evidence."""


@dataclass(frozen=True)
class Employee:
    employee_id: str
    active: bool = True
    display_name: str | None = None


class EmployeeDirectory:
    def __init__(self, tenant_id: str, employees: Iterable[Employee] = ()) -> None:
        self.tenant_id = tenant_id
        self._employees: dict[str, Employee] = {}
        for employee in employees:
            if employee.employee_id in self._employees:
                raise IdentityError(f"Duplicate employee_id {employee.employee_id}")
            if employee.employee_id == UNKNOWN_EMPLOYEE:
                raise IdentityError(f"{UNKNOWN_EMPLOYEE} is reserved")
            self._employees[employee.employee_id] = employee

    @classmethod
    def from_json(cls, path: str | Path, tenant_id: str) -> "EmployeeDirectory":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IdentityError(f"Cannot read employee directory {path}: {exc}") from exc
        if data.get("tenant_id") != tenant_id:
            raise IdentityError(f"Employee directory {path} belongs to tenant {data.get('tenant_id')!r}")
        return cls(tenant_id, [Employee(str(e["employee_id"]), bool(e.get("active", True)), e.get("display_name"))
                               for e in data.get("employees", [])])

    def get(self, employee_id: str) -> Employee | None:
        return self._employees.get(employee_id)

    def __len__(self) -> int:
        return len(self._employees)


@dataclass(frozen=True)
class IdentityEvidence:
    tenant_id: str
    composite_track_id: str
    employee_id: str
    source: str               # e.g. "badge", "admin_label", "verification_service"
    timestamp_seconds: float
    confidence: float
    verified: bool
    reference: str = ""       # opaque audit reference (never raw biometrics)


@dataclass(frozen=True)
class IdentityEvent:
    kind: str                 # assigned | ended | rejected | conflict
    composite_track_id: str
    employee_id: str | None
    timestamp_seconds: float
    source: str
    reason: str


@dataclass
class _Assignment:
    employee_id: str
    since: float
    source: str
    last_seen: float


class IdentitySource(Protocol):
    def poll(self, timestamp_seconds: float) -> list[IdentityEvidence]: ...


class NullIdentitySource:
    """No verified identity integration configured: every track stays UNKNOWN_EMPLOYEE."""

    def poll(self, timestamp_seconds: float) -> list[IdentityEvidence]:
        return []


@dataclass
class IdentityManager:
    tenant_id: str
    directory: EmployeeDirectory
    min_confidence: float = 0.90
    _by_track: dict[str, _Assignment] = field(default_factory=dict)
    _by_employee: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.directory.tenant_id != self.tenant_id:
            raise IdentityError("Employee directory tenant does not match identity manager tenant")
        if not 0 < self.min_confidence <= 1:
            raise IdentityError("identity.min_confidence must be in (0, 1]")

    def employee_of(self, composite_track_id: str) -> str:
        assignment = self._by_track.get(composite_track_id)
        return assignment.employee_id if assignment else UNKNOWN_EMPLOYEE

    def assignments(self) -> Mapping[str, str]:
        return {tid: a.employee_id for tid, a in self._by_track.items()}

    def submit(self, evidence: IdentityEvidence, live_tracks: Iterable[str]) -> list[IdentityEvent]:
        tid, emp, ts = evidence.composite_track_id, evidence.employee_id, evidence.timestamp_seconds
        live = set(live_tracks)

        def reject(reason: str) -> list[IdentityEvent]:
            logger.warning("Identity evidence rejected (%s): track=%s employee=%s source=%s",
                           reason, tid, emp, evidence.source)
            return [IdentityEvent("rejected", tid, emp, ts, evidence.source, reason)]

        if evidence.tenant_id != self.tenant_id:
            return reject("tenant_mismatch")
        if not evidence.verified:
            return reject("not_verified")
        if evidence.confidence < self.min_confidence:
            return reject(f"confidence {evidence.confidence:.2f} < {self.min_confidence:.2f}")
        employee = self.directory.get(emp)
        if employee is None:
            return reject("unknown_employee")
        if not employee.active:
            return reject("inactive_employee")
        if tid not in live:
            return reject("track_not_live")

        current = self._by_track.get(tid)
        if current is not None and current.employee_id == emp:
            current.last_seen = ts
            return []
        events: list[IdentityEvent] = []
        other_track = self._by_employee.get(emp)
        if current is not None or (other_track is not None and other_track in live and other_track != tid):
            reason = (f"track already {current.employee_id}" if current is not None
                      else f"employee already on {other_track}")
            events.append(IdentityEvent("conflict", tid, emp, ts, evidence.source, reason))
            for involved in {tid, other_track} - {None}:
                events += self._end(involved, ts, "identity_conflict")  # type: ignore[arg-type]
            logger.warning("Identity conflict for %s / %s: %s; both set to %s", tid, emp, reason, UNKNOWN_EMPLOYEE)
            return events
        if other_track is not None:
            events += self._end(other_track, ts, "superseded_stale_track")
        self._by_track[tid] = _Assignment(emp, ts, evidence.source, ts)
        self._by_employee[emp] = tid
        events.append(IdentityEvent("assigned", tid, emp, ts, evidence.source, "verified_evidence"))
        return events

    def track_removed(self, composite_track_id: str, timestamp: float) -> list[IdentityEvent]:
        return self._end(composite_track_id, timestamp, "track_removed")

    def reset(self, timestamp: float, reason: str = "tracking_session_reset") -> list[IdentityEvent]:
        events: list[IdentityEvent] = []
        for tid in list(self._by_track):
            events += self._end(tid, timestamp, reason)
        return events

    def _end(self, composite_track_id: str, timestamp: float, reason: str) -> list[IdentityEvent]:
        assignment = self._by_track.pop(composite_track_id, None)
        if assignment is None:
            return []
        if self._by_employee.get(assignment.employee_id) == composite_track_id:
            del self._by_employee[assignment.employee_id]
        return [IdentityEvent("ended", composite_track_id, assignment.employee_id, timestamp, assignment.source, reason)]
