"""SQLite persistence for batches, original reports and accepted retests."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .analysis import Report, SampleUse

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "moisture.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    revision    INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id      INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    report_no     TEXT NOT NULL,
    position      INTEGER NOT NULL,
    low           REAL NOT NULL,
    high          REAL NOT NULL,
    original_low  REAL NOT NULL,
    original_high REAL NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    UNIQUE (batch_id, report_no)
);

CREATE TABLE IF NOT EXISTS report_samples (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id  INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    sample_no  TEXT NOT NULL,
    grams      REAL NOT NULL,
    UNIQUE (report_id, sample_no)
);

CREATE TABLE IF NOT EXISTS retests (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id   INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    report_id  INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    report_no  TEXT NOT NULL,
    old_low    REAL NOT NULL,
    old_high   REAL NOT NULL,
    new_low    REAL NOT NULL,
    new_high   REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reports_batch ON reports(batch_id);
CREATE INDEX IF NOT EXISTS idx_samples_report ON report_samples(report_id);
CREATE INDEX IF NOT EXISTS idx_retests_batch ON retests(batch_id);
"""


class BatchChanged(Exception):
    """Raised when the batch revision no longer matches the submitted page."""


class RetestRejected(Exception):
    """Raised when a submitted retest must not be persisted."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class StoredReport:
    report_no: str
    low: float
    high: float
    original_low: float
    original_high: float
    samples: tuple[SampleUse, ...]

    def to_domain(self) -> Report:
        return Report(
            report_no=self.report_no,
            samples=self.samples,
            low=self.low,
            high=self.high,
        )


@dataclass(frozen=True)
class StoredBatch:
    id: int
    name: str
    revision: int
    created_at: str
    updated_at: str
    reports: tuple[StoredReport, ...]

    def to_domain(self) -> list[Report]:
        return [report.to_domain() for report in self.reports]


@dataclass(frozen=True)
class RetestEvent:
    report_no: str
    old_low: float
    old_high: float
    new_low: float
    new_high: float
    created_at: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_path() -> Path:
    return Path(os.environ.get("DATABASE_PATH", str(DEFAULT_DB_PATH)))


def connect(path: Path | None = None) -> sqlite3.Connection:
    target = Path(path) if path is not None else db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Each request owns its connection; FastAPI may create the dependency in a
    # worker thread and then run the handler on the event loop, so the
    # per-thread guard must be relaxed.  A connection is never shared between
    # concurrent requests; cross-process safety comes from SQLite locking.
    conn = sqlite3.connect(
        target, timeout=30, isolation_level=None, check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def create_batch(
    conn: sqlite3.Connection,
    name: str,
    reports: Sequence[Report],
) -> int:
    stamp = now_iso()
    conn.execute("BEGIN IMMEDIATE")
    try:
        cursor = conn.execute(
            "INSERT INTO batches (name, revision, created_at, updated_at) "
            "VALUES (?, 1, ?, ?)",
            (name, stamp, stamp),
        )
        batch_id = int(cursor.lastrowid)
        for position, report in enumerate(reports):
            cursor = conn.execute(
                "INSERT INTO reports "
                "(batch_id, report_no, position, low, high, original_low, "
                " original_high, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    batch_id,
                    report.report_no,
                    position,
                    report.low,
                    report.high,
                    report.low,
                    report.high,
                    stamp,
                    stamp,
                ),
            )
            report_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT INTO report_samples (report_id, sample_no, grams) "
                "VALUES (?, ?, ?)",
                [
                    (report_id, use.sample_no, use.grams)
                    for use in report.samples
                ],
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return batch_id


def list_batches(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT b.id, b.name, b.revision, b.created_at, b.updated_at,
               COUNT(DISTINCT r.id) AS report_count,
               COUNT(DISTINCT s.sample_no) AS sample_count,
               COUNT(DISTINCT t.id) AS retest_count
        FROM batches b
        LEFT JOIN reports r ON r.batch_id = b.id
        LEFT JOIN report_samples s ON s.report_id = r.id
        LEFT JOIN retests t ON t.batch_id = b.id
        GROUP BY b.id
        ORDER BY b.id DESC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def get_batch(conn: sqlite3.Connection, batch_id: int) -> StoredBatch | None:
    row = conn.execute(
        "SELECT id, name, revision, created_at, updated_at FROM batches WHERE id = ?",
        (batch_id,),
    ).fetchone()
    if row is None:
        return None
    reports = _load_reports(conn, batch_id)
    return StoredBatch(
        id=int(row["id"]),
        name=row["name"],
        revision=int(row["revision"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        reports=tuple(reports),
    )


def _load_reports(conn: sqlite3.Connection, batch_id: int) -> list[StoredReport]:
    report_rows = conn.execute(
        "SELECT id, report_no, low, high, original_low, original_high "
        "FROM reports WHERE batch_id = ? ORDER BY position, id",
        (batch_id,),
    ).fetchall()
    result: list[StoredReport] = []
    for report in report_rows:
        samples = conn.execute(
            "SELECT sample_no, grams FROM report_samples "
            "WHERE report_id = ? ORDER BY id",
            (report["id"],),
        ).fetchall()
        result.append(
            StoredReport(
                report_no=report["report_no"],
                low=float(report["low"]),
                high=float(report["high"]),
                original_low=float(report["original_low"]),
                original_high=float(report["original_high"]),
                samples=tuple(
                    SampleUse(sample_no=s["sample_no"], grams=float(s["grams"]))
                    for s in samples
                ),
            )
        )
    return result


def list_retests(conn: sqlite3.Connection, batch_id: int) -> list[RetestEvent]:
    rows = conn.execute(
        "SELECT report_no, old_low, old_high, new_low, new_high, created_at "
        "FROM retests WHERE batch_id = ? ORDER BY id",
        (batch_id,),
    ).fetchall()
    return [
        RetestEvent(
            report_no=row["report_no"],
            old_low=float(row["old_low"]),
            old_high=float(row["old_high"]),
            new_low=float(row["new_low"]),
            new_high=float(row["new_high"]),
            created_at=row["created_at"],
        )
        for row in rows
    ]


def apply_retest(
    conn: sqlite3.Connection,
    batch_id: int,
    expected_revision: int,
    report_no: str,
    new_low: float,
    new_high: float,
    verdict,
) -> None:
    """Atomically validate and store a retest.

    ``verdict(reports, report_no, low, high)`` returns ``None`` when the
    replacement makes the batch compatible, or a rejection code.  It runs while
    the write lock is held so a concurrent change cannot slip between
    validation and storage.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT revision FROM batches WHERE id = ?", (batch_id,)
        ).fetchone()
        if row is None:
            raise RetestRejected("missing_batch")
        if int(row["revision"]) != int(expected_revision):
            raise RetestRejected("stale")
        reports = tuple(_load_reports(conn, batch_id))
        domain = [report.to_domain() for report in reports]
        code = verdict(domain, report_no, new_low, new_high)
        if code is not None:
            raise RetestRejected(code)

        target = next(r for r in reports if r.report_no == report_no)
        stamp = now_iso()
        row = conn.execute(
            "SELECT id FROM reports WHERE batch_id = ? AND report_no = ?",
            (batch_id, report_no),
        ).fetchone()
        report_id = int(row["id"])
        conn.execute(
            "UPDATE reports SET low = ?, high = ?, updated_at = ? WHERE id = ?",
            (new_low, new_high, stamp, report_id),
        )
        conn.execute(
            "INSERT INTO retests (batch_id, report_id, report_no, old_low, "
            "old_high, new_low, new_high, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                batch_id,
                report_id,
                report_no,
                target.low,
                target.high,
                new_low,
                new_high,
                stamp,
            ),
        )
        conn.execute(
            "UPDATE batches SET revision = revision + 1, updated_at = ? "
            "WHERE id = ?",
            (stamp, batch_id),
        )
        conn.execute("COMMIT")
    except RetestRejected:
        conn.execute("ROLLBACK")
        raise
    except Exception:
        conn.execute("ROLLBACK")
        raise


