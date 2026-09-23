"""SQLite store for employees, embeddings, attendance, and occupancy events."""

from __future__ import annotations

import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline.config import DB_PATH, PHOTO_DIR

_SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL,
    vector BLOB NOT NULL,
    dim INTEGER NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL,
    work_date TEXT NOT NULL,
    clock_in TEXT NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,
    UNIQUE (employee_id, work_date),
    FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS occupancy_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    chair_track_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_sec REAL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_employee ON embeddings(employee_id);
CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(work_date);
CREATE INDEX IF NOT EXISTS idx_occ_day ON occupancy_events(camera_id, chair_track_id, started_at);
"""


class Store:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        PHOTO_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def get_or_create_employee(self, name: str) -> int:
        name = name.strip()
        if not name:
            raise ValueError("Employee name is required")
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM employees WHERE name = ?", (name,)
            ).fetchone()
            if row:
                return int(row["id"])
            cur = self._conn.execute(
                "INSERT INTO employees (name, created_at) VALUES (?, ?)",
                (name, self._now()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def add_embedding(self, employee_id: int, vector: np.ndarray) -> None:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        with self._lock:
            self._conn.execute(
                "INSERT INTO embeddings (employee_id, vector, dim) VALUES (?, ?, ?)",
                (employee_id, vec.tobytes(), int(vec.size)),
            )
            self._conn.commit()

    def list_employees(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT e.id, e.name, e.created_at, COUNT(x.id) AS embedding_count
                FROM employees e
                LEFT JOIN embeddings x ON x.employee_id = e.id
                GROUP BY e.id
                ORDER BY e.name COLLATE NOCASE
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def employee_names(self) -> dict[int, str]:
        with self._lock:
            rows = self._conn.execute("SELECT id, name FROM employees").fetchall()
        return {int(r["id"]): str(r["name"]) for r in rows}

    def load_gallery(self) -> dict[int, np.ndarray]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT employee_id, vector, dim FROM embeddings"
            ).fetchall()
        gallery: dict[int, list[np.ndarray]] = {}
        for row in rows:
            dim = int(row["dim"])
            vec = np.frombuffer(row["vector"], dtype=np.float32)
            if vec.size != dim:
                continue
            gallery.setdefault(int(row["employee_id"]), []).append(vec)
        return {eid: np.stack(vecs, axis=0) for eid, vecs in gallery.items() if vecs}

    def has_attendance(self, employee_id: int, work_date: Optional[str] = None) -> bool:
        work_date = work_date or date.today().isoformat()
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM attendance WHERE employee_id = ? AND work_date = ?",
                (employee_id, work_date),
            ).fetchone()
        return row is not None

    def mark_first_seen(
        self,
        employee_id: int,
        confidence: float,
        source: str,
        work_date: Optional[str] = None,
    ) -> Optional[str]:
        """Insert today's Present row. Returns clock_in ISO string, or None if already marked."""
        work_date = work_date or date.today().isoformat()
        clock_in = self._now()
        with self._lock:
            existing = self._conn.execute(
                "SELECT clock_in FROM attendance WHERE employee_id = ? AND work_date = ?",
                (employee_id, work_date),
            ).fetchone()
            if existing:
                return None
            self._conn.execute(
                """
                INSERT INTO attendance (employee_id, work_date, clock_in, confidence, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                (employee_id, work_date, clock_in, float(confidence), source),
            )
            self._conn.commit()
        return clock_in

    def attendance_sheet(self, work_date: Optional[str] = None) -> list[dict]:
        work_date = work_date or date.today().isoformat()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    e.id AS employee_id,
                    e.name,
                    a.clock_in,
                    a.confidence,
                    a.source
                FROM employees e
                LEFT JOIN attendance a
                    ON a.employee_id = e.id AND a.work_date = ?
                ORDER BY e.name COLLATE NOCASE
                """,
                (work_date,),
            ).fetchall()
        sheet = []
        for row in rows:
            present = row["clock_in"] is not None
            sheet.append(
                {
                    "employee_id": int(row["employee_id"]),
                    "name": row["name"],
                    "status": "Present" if present else "Absent",
                    "clock_in": row["clock_in"] or "",
                    "confidence": float(row["confidence"]) if present else None,
                    "source": row["source"] or "",
                    "work_date": work_date,
                }
            )
        return sheet

    def open_away_event(self, camera_id: str, chair_track_id: int) -> int:
        started = self._now()
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO occupancy_events
                    (camera_id, chair_track_id, status, started_at, ended_at, duration_sec)
                VALUES (?, ?, 'away', ?, NULL, NULL)
                """,
                (camera_id, int(chair_track_id), started),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def close_open_away_events(self, camera_id: str) -> None:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id FROM occupancy_events
                WHERE camera_id = ? AND ended_at IS NULL
                """,
                (camera_id,),
            ).fetchall()
        for row in rows:
            self.close_away_event(int(row["id"]))

    def close_away_event(self, event_id: int) -> float:
        ended = datetime.now()
        with self._lock:
            row = self._conn.execute(
                "SELECT started_at FROM occupancy_events WHERE id = ?",
                (event_id,),
            ).fetchone()
            if not row:
                return 0.0
            started = datetime.fromisoformat(row["started_at"])
            duration = max(0.0, (ended - started).total_seconds())
            self._conn.execute(
                """
                UPDATE occupancy_events
                SET ended_at = ?, duration_sec = ?
                WHERE id = ?
                """,
                (ended.isoformat(timespec="seconds"), duration, event_id),
            )
            self._conn.commit()
        return duration

    def away_seconds_today(self, camera_id: str, chair_track_id: int) -> float:
        day = date.today().isoformat()
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COALESCE(SUM(
                    CASE
                        WHEN duration_sec IS NOT NULL THEN duration_sec
                        ELSE (julianday('now', 'localtime') - julianday(started_at)) * 86400.0
                    END
                ), 0) AS total
                FROM occupancy_events
                WHERE camera_id = ?
                  AND chair_track_id = ?
                  AND status = 'away'
                  AND started_at >= ?
                """,
                (camera_id, int(chair_track_id), day),
            ).fetchone()
        return float(row["total"] if row else 0.0)

    def occupancy_today(self, camera_id: str) -> list[dict]:
        day = date.today().isoformat()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    chair_track_id,
                    COUNT(*) AS away_events,
                    COALESCE(SUM(
                        CASE
                            WHEN duration_sec IS NOT NULL THEN duration_sec
                            ELSE (julianday('now', 'localtime') - julianday(started_at)) * 86400.0
                        END
                    ), 0) AS away_seconds
                FROM occupancy_events
                WHERE camera_id = ? AND status = 'away' AND started_at >= ?
                GROUP BY chair_track_id
                ORDER BY chair_track_id
                """,
                (camera_id, day),
            ).fetchall()
        return [dict(r) for r in rows]
