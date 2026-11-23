"""Tenant isolation, idempotency, crash tolerance and privacy guards of the repository."""

from __future__ import annotations

import pytest

from storage.repository import JsonlRepository, RepositoryError


def test_put_is_idempotent_across_restarts(tmp_path):
    repo = JsonlRepository(tmp_path, "acme")
    assert repo.put("seated_intervals", {"session": "s1", "start": 1.0}, {"duration": 5.0})
    assert not repo.put("seated_intervals", {"session": "s1", "start": 1.0}, {"duration": 5.0})
    repo.close()
    reopened = JsonlRepository(tmp_path, "acme")
    assert not reopened.put("seated_intervals", {"session": "s1", "start": 1.0}, {"duration": 5.0})
    assert len(reopened.query("seated_intervals")) == 1


def test_tenants_are_isolated(tmp_path):
    a, b = JsonlRepository(tmp_path, "acme"), JsonlRepository(tmp_path, "beta")
    a.put("sessions", {"id": 1}, {"session_id": "s1"})
    assert b.query("sessions") == []
    with pytest.raises(RepositoryError, match="tenant"):
        a.put("sessions", {"id": 2}, {"tenant_id": "beta"})
    for bad in ("", "../x", "a/b", "x" * 80):
        with pytest.raises(RepositoryError):
            JsonlRepository(tmp_path, bad)


def test_foreign_rows_are_detected(tmp_path):
    repo = JsonlRepository(tmp_path, "acme")
    (tmp_path / "acme" / "sessions.jsonl").write_text('{"record_id": "x", "tenant_id": "beta"}\n')
    with pytest.raises(RepositoryError, match="Foreign"):
        repo.query("sessions")


def test_torn_trailing_line_is_skipped(tmp_path):
    repo = JsonlRepository(tmp_path, "acme")
    repo.put("sessions", {"id": 1}, {"session_id": "s1"})
    repo.close()
    with (tmp_path / "acme" / "sessions.jsonl").open("a") as handle:
        handle.write('{"record_id": "tor')
    assert len(JsonlRepository(tmp_path, "acme").query("sessions")) == 1


def test_sensitive_fields_and_unknown_tables_rejected(tmp_path):
    repo = JsonlRepository(tmp_path, "acme")
    with pytest.raises(RepositoryError, match="sensitive"):
        repo.put("identity_events", {"id": 1}, {"embedding": [0.1, 0.2]})
    with pytest.raises(RepositoryError, match="Unknown table"):
        repo.put("frames", {"id": 1}, {})
