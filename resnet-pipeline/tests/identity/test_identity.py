"""Identity manager rules and employee-level time attribution."""

from __future__ import annotations

import json

import pytest

from core.association.association_engine import AssociationEvent
from core.identity.identity_manager import (
    UNKNOWN_EMPLOYEE, Employee, EmployeeDirectory, IdentityError, IdentityEvidence, IdentityManager,
)
from core.association.presence_time import PresenceTimeTracker

T1, T2 = "acme:CAM01:s1:1", "acme:CAM01:s1:2"


@pytest.fixture
def manager():
    directory = EmployeeDirectory("acme", [Employee("E100"), Employee("E200"), Employee("E900", active=False)])
    return IdentityManager("acme", directory, min_confidence=0.9)


def ev(track, emp, ts=1.0, conf=0.99, verified=True, tenant="acme"):
    return IdentityEvidence(tenant, track, emp, "badge", ts, conf, verified, "audit-1")


def test_tracks_are_unknown_until_verified(manager):
    assert manager.employee_of(T1) == UNKNOWN_EMPLOYEE
    events = manager.submit(ev(T1, "E100"), [T1])
    assert [e.kind for e in events] == ["assigned"]
    assert manager.employee_of(T1) == "E100"


@pytest.mark.parametrize("evidence,reason", [
    (ev(T1, "E100", verified=False), "not_verified"),
    (ev(T1, "E100", conf=0.5), "confidence"),
    (ev(T1, "E404"), "unknown_employee"),
    (ev(T1, "E900"), "inactive_employee"),
    (ev(T1, "E100", tenant="other"), "tenant_mismatch"),
])
def test_rejections(manager, evidence, reason):
    events = manager.submit(evidence, [T1])
    assert events[0].kind == "rejected" and reason in events[0].reason
    assert manager.employee_of(T1) == UNKNOWN_EMPLOYEE


def test_evidence_for_dead_track_is_rejected(manager):
    assert manager.submit(ev(T1, "E100"), [T2])[0].reason == "track_not_live"


def test_conflicts_reset_both_tracks_to_unknown(manager):
    manager.submit(ev(T1, "E100"), [T1, T2])
    events = manager.submit(ev(T2, "E100", ts=2.0), [T1, T2])
    assert "conflict" in [e.kind for e in events]
    assert manager.employee_of(T1) == UNKNOWN_EMPLOYEE and manager.employee_of(T2) == UNKNOWN_EMPLOYEE
    manager.submit(ev(T1, "E200", ts=3.0), [T1, T2])
    events = manager.submit(ev(T1, "E100", ts=4.0), [T1, T2])
    assert events[0].kind == "conflict" and manager.employee_of(T1) == UNKNOWN_EMPLOYEE


def test_stale_track_assignment_is_superseded(manager):
    manager.submit(ev(T1, "E100"), [T1])
    events = manager.submit(ev(T2, "E100", ts=5.0), [T2])
    assert [e.kind for e in events] == ["ended", "assigned"]
    assert manager.employee_of(T2) == "E100" and manager.employee_of(T1) == UNKNOWN_EMPLOYEE


def test_removal_and_reset_end_assignments(manager):
    manager.submit(ev(T1, "E100"), [T1])
    assert manager.track_removed(T1, 9.0)[0].reason == "track_removed"
    manager.submit(ev(T2, "E200"), [T2])
    assert manager.reset(10.0)[0].reason == "tracking_session_reset"
    assert not manager.assignments()


def test_directory_tenant_checks(tmp_path):
    path = tmp_path / "employees.json"
    path.write_text(json.dumps({"tenant_id": "acme", "employees": [{"employee_id": "E1"}]}))
    assert len(EmployeeDirectory.from_json(path, "acme")) == 1
    with pytest.raises(IdentityError, match="tenant"):
        EmployeeDirectory.from_json(path, "beta")
    with pytest.raises(IdentityError, match="reserved"):
        EmployeeDirectory("acme", [Employee(UNKNOWN_EMPLOYEE)])


def test_employee_time_only_covers_verified_window(manager):
    presence = PresenceTimeTracker()
    presence.apply(AssociationEvent("start", T1, 1, "WS01", 0.0, 3.0, "confirmed_seated", 0.9))
    for event in manager.submit(ev(T1, "E100", ts=10.0), [T1]):
        presence.identity_event(event)
    for event in manager.track_removed(T1, 40.0):
        presence.identity_event(event)
    parts = presence.apply(AssociationEvent("end", T1, 1, "WS01", 50.0, 52.0, "track_removed", 0.9))
    assert [(p.employee_id, p.start, p.end) for p in parts] == [
        (UNKNOWN_EMPLOYEE, 0.0, 10.0), ("E100", 10.0, 40.0), (UNKNOWN_EMPLOYEE, 40.0, 50.0)]
    totals = presence.employee_totals()
    assert totals["E100"] == {"WS01": 30.0}
    assert totals[UNKNOWN_EMPLOYEE] == {"WS01": 20.0}
    assert presence.workstation_totals() == {"WS01": 50.0}
