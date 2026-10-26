import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).with_name("gateway.db")


def connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with closing(connect()) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    success INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(requests)")
            }
            if "success" not in columns:
                connection.execute(
                    "ALTER TABLE requests "
                    "ADD COLUMN success INTEGER NOT NULL DEFAULT 1"
                )
                connection.execute(
                    "UPDATE requests SET success = 0 "
                    "WHERE model_used = 'unavailable'"
                )


def log_request(
    prompt: str,
    provider: str,
    model_used: str,
    cache_hit: bool,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    latency_ms: float,
    success: bool = True,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with closing(connect()) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO requests (
                    timestamp, prompt, provider, model_used, cache_hit,
                    input_tokens, output_tokens, cost, latency_ms, success
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    prompt,
                    provider,
                    model_used,
                    int(cache_hit),
                    input_tokens,
                    output_tokens,
                    cost,
                    latency_ms,
                    int(success),
                ),
            )


def get_totals() -> tuple[int, int, float, int, int, int, float]:
    with closing(connect()) as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*),
                COALESCE(SUM(cache_hit), 0),
                COALESCE(SUM(cost), 0),
                COALESCE(SUM(input_tokens), 0),
                COALESCE(SUM(output_tokens), 0),
                COALESCE(SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END), 0),
                COALESCE(AVG(latency_ms), 0)
            FROM requests
            """
        ).fetchone()

    return (
        int(row[0]),
        int(row[1]),
        float(row[2]),
        int(row[3]),
        int(row[4]),
        int(row[5]),
        float(row[6]),
    )
