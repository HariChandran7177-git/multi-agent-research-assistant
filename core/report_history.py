"""
core/report_history.py
----------------------
Stores and retrieves past research reports in a local SQLite database.
Gives users a history of all their research queries and outputs.
"""

import sqlite3
import json
import time
import os
from typing import Optional, List, Dict, Any
from core.logger import get_logger

logger = get_logger(__name__)

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "report_history.sqlite"))


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the reports table if it doesn t exist."""
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                query       TEXT NOT NULL,
                report      TEXT NOT NULL,
                confidence  REAL DEFAULT 0.0,
                iterations  INTEGER DEFAULT 0,
                tone        TEXT DEFAULT 'professional',
                user_id     TEXT DEFAULT 'default_user',
                created_at  REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_user_id ON reports(user_id)")
        conn.commit()
    finally:
        conn.close()


def save_report(query, report, confidence=0.0, iterations=0, tone="professional", user_id="default_user"):
    """Save a completed report. Returns the new record ID."""
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "INSERT INTO reports (query, report, confidence, iterations, tone, user_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (query, report, confidence, iterations, tone, user_id, time.time()),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def list_reports(user_id=None, limit=20):
    """List recent reports, optionally filtered by user_id."""
    conn = _get_conn()
    try:
        if user_id:
            rows = conn.execute(
                "SELECT id, query, confidence, iterations, tone, user_id, created_at FROM reports WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, query, confidence, iterations, tone, user_id, created_at FROM reports ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_report(report_id):
    """Fetch a single report by ID."""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_report(report_id):
    """Delete a report by ID. Returns True if deleted."""
    conn = _get_conn()
    try:
        cursor = conn.execute("DELETE FROM reports WHERE id=?", (report_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# Initialize DB on import
init_db()
