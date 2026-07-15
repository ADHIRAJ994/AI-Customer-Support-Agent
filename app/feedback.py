"""
Simple SQLite storage for user feedback (thumbs up/down + optional
comment) on assistant answers.
"""
import os
import sqlite3
import time
from contextlib import contextmanager

from app.config import settings


def _ensure_dir():
    d = os.path.dirname(settings.feedback_db)
    if d:
        os.makedirs(d, exist_ok=True)


@contextmanager
def _conn():
    _ensure_dir()
    conn = sqlite3.connect(settings.feedback_db)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                rating TEXT NOT NULL,
                comment TEXT,
                created_at REAL NOT NULL
            )
            """
        )


def save_feedback(message_id: str, thread_id: str, rating: str, comment: str | None) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO feedback (message_id, thread_id, rating, comment, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (message_id, thread_id, rating, comment, time.time()),
        )


def get_stats() -> dict:
    with _conn() as conn:
        cur = conn.execute(
            "SELECT rating, COUNT(*) FROM feedback GROUP BY rating"
        )
        rows = dict(cur.fetchall())
    return {"up": rows.get("up", 0), "down": rows.get("down", 0)}