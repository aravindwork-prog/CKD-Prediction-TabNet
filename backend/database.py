"""Small SQLite helpers for storing CKD prediction requests."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DATABASE_PATH = Path(__file__).resolve().parent / "predictions.db"


def init_db() -> None:
    """Create the prediction log table when it does not already exist."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                input_json TEXT,
                prediction TEXT,
                confidence REAL,
                risk_level TEXT
            )
        """)


def log_prediction(input_data: dict, prediction: str, confidence: float, risk_level: str) -> None:
    """Insert one prediction request and its result into SQLite."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            "INSERT INTO predictions (timestamp, input_json, prediction, confidence, risk_level) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), json.dumps(input_data), prediction, confidence, risk_level),
        )


def get_recent_predictions(limit: int) -> list[dict]:
    """Return the newest prediction log entries first."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id, timestamp, input_json, prediction, confidence, risk_level "
            "FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]
