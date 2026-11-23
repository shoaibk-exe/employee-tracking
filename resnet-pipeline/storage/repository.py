"""Tenant-scoped, idempotent persistence for monitoring history.

``Repository`` is the interface the pipeline writes through. ``JsonlRepository`` is the
file-backed implementation for development and single-node deployments:

* one directory per tenant (``<root>/<tenant_id>/<table>.jsonl``); tenant IDs are validated
* every record carries ``tenant_id`` and a deterministic ``record_id`` (hash of its natural
  key), so replays and restarts never duplicate rows
* append-only with fsync on flush; a torn last line after a crash is skipped on load
* no frames, crops, or biometric data are ever written

A PostgreSQL implementation must satisfy the same interface (``record_id`` as primary key
with ``ON CONFLICT DO NOTHING``, and ``tenant_id`` in every query predicate).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

logger = logging.getLogger(__name__)

TENANT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
TABLES = (
    "sessions", "scene_maps", "workstation_intervals", "workstation_events", "seated_intervals",
    "employee_intervals", "identity_events", "fusion_disagreements",
)
FORBIDDEN_KEYS = {"frame", "image", "crop", "embedding", "embeddings", "face", "token", "password"}


class RepositoryError(RuntimeError):
    """Invalid tenant, table, or record."""


def record_id(table: str, key: Mapping[str, Any]) -> str:
    payload = json.dumps({"table": table, **key}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


class Repository(Protocol):
    tenant_id: str

    def put(self, table: str, key: Mapping[str, Any], record: Mapping[str, Any]) -> bool: ...

    def query(self, table: str, **filters: Any) -> list[dict[str, Any]]: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class JsonlRepository:
    def __init__(self, root: str | Path, tenant_id: str) -> None:
        if not TENANT_PATTERN.match(tenant_id or ""):
            raise RepositoryError(f"Invalid tenant_id {tenant_id!r}")
        self.tenant_id = tenant_id
        self.directory = Path(root) / tenant_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self._ids: dict[str, set[str]] = {}
        self._handles: dict[str, Any] = {}

    def _path(self, table: str) -> Path:
        if table not in TABLES:
            raise RepositoryError(f"Unknown table {table!r}")
        return self.directory / f"{table}.jsonl"

    def _load(self, table: str) -> list[dict[str, Any]]:
        path = self._path(table)
        rows: list[dict[str, Any]] = []
        if not path.is_file():
            return rows
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed line %d in %s (torn write?)", number, path)
                    continue
                if row.get("tenant_id") != self.tenant_id:
                    raise RepositoryError(f"Foreign tenant row in {path} line {number}")
                rows.append(row)
        return rows

    def _known_ids(self, table: str) -> set[str]:
        if table not in self._ids:
            self._ids[table] = {row["record_id"] for row in self._load(table) if "record_id" in row}
        return self._ids[table]

    def put(self, table: str, key: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
        """Insert once; returns False if a record with the same natural key already exists."""
        if record.get("tenant_id", self.tenant_id) != self.tenant_id:
            raise RepositoryError(f"Record tenant {record.get('tenant_id')!r} != repository tenant {self.tenant_id!r}")
        bad = FORBIDDEN_KEYS & {k.lower() for k in record}
        if bad:
            raise RepositoryError(f"Refusing to persist sensitive fields {sorted(bad)}")
        rid = record_id(table, {"tenant_id": self.tenant_id, **key})
        ids = self._known_ids(table)
        if rid in ids:
            return False
        row = {"record_id": rid, "tenant_id": self.tenant_id, **record}
        handle = self._handles.get(table)
        if handle is None:
            handle = self._path(table).open("a", encoding="utf-8")
            self._handles[table] = handle
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        ids.add(rid)
        return True

    def put_many(self, table: str, items: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> int:
        return sum(1 for key, record in items if self.put(table, key, record))

    def query(self, table: str, **filters: Any) -> list[dict[str, Any]]:
        self.flush()
        return [row for row in self._load(table) if all(row.get(k) == v for k, v in filters.items())]

    def flush(self) -> None:
        for handle in self._handles.values():
            handle.flush()
            os.fsync(handle.fileno())

    def close(self) -> None:
        self.flush()
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
