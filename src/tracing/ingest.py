"""Ingest logs/raw/*.jsonl into logs/traces.db (SQLite).

Idempotent per run_id: re-ingesting a run_id deletes its prior rows first, so
re-running an experiment and re-ingesting never leaves stale/duplicate rows.
The JSONL files remain the source of truth; this DB can be deleted and rebuilt
at any time with `python -m src.tracing.ingest`.
"""

import json
import sqlite3
from pathlib import Path

from .util import canonical_json

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_LOG_DIR = BASE_DIR / "logs" / "raw"
DB_PATH = BASE_DIR / "logs" / "traces.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text())
    return conn


def _clear_run(conn: sqlite3.Connection, run_id: str) -> None:
    for table in ("tool_calls", "state_sequence", "plan_steps", "runs"):
        conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))


def ingest_file(conn: sqlite3.Connection, path: Path) -> None:
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not events:
        return
    run_id = events[0]["run_id"]
    _clear_run(conn, run_id)

    run_row = {
        "run_id": run_id,
        "task": None,
        "model": None,
        "condition": None,
        "run_index": None,
        "task_input_hash": None,
        "success": None,
        "error": None,
        "final_output_json": None,
        "total_tokens": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "latency_ms": None,
        "started_at": None,
        "ended_at": None,
    }
    plan_rows, state_rows, tool_rows = [], [], []
    last_state = "START"

    for ev in events:
        kind = ev["event"]
        if kind == "run_start":
            run_row.update(
                task=ev["task"],
                model=ev["model"],
                condition=ev["condition"],
                run_index=ev["run_index"],
                task_input_hash=ev["task_input_hash"],
                started_at=ev["timestamp"],
            )
        elif kind == "plan":
            plan_rows = [(run_id, i, step) for i, step in enumerate(ev["steps"])]
        elif kind == "state_transition":
            last_state = ev["to_state"]
            state_rows.append((run_id, len(state_rows), last_state))
        elif kind == "tool_call":
            tool_rows.append(
                (
                    run_id,
                    ev["call_index"],
                    ev["tool_name"],
                    canonical_json(ev["args"]),
                    ev["args_hash"],
                    canonical_json(ev["result"]),
                    ev["result_hash"],
                    ev.get("state"),
                )
            )
        elif kind == "run_end":
            run_row.update(
                success=int(bool(ev["success"])),
                error=ev.get("error"),
                final_output_json=canonical_json(ev["final_output"]),
                total_tokens=ev.get("total_tokens"),
                prompt_tokens=ev.get("prompt_tokens"),
                completion_tokens=ev.get("completion_tokens"),
                latency_ms=ev.get("latency_ms"),
                ended_at=ev["timestamp"],
            )

    conn.execute(
        """INSERT INTO runs (run_id, task, model, condition, run_index, task_input_hash,
               success, error, final_output_json, total_tokens, prompt_tokens,
               completion_tokens, latency_ms, started_at, ended_at)
           VALUES (:run_id, :task, :model, :condition, :run_index, :task_input_hash,
               :success, :error, :final_output_json, :total_tokens, :prompt_tokens,
               :completion_tokens, :latency_ms, :started_at, :ended_at)""",
        run_row,
    )
    conn.executemany(
        "INSERT INTO plan_steps (run_id, step_index, step_id) VALUES (?, ?, ?)", plan_rows
    )
    conn.executemany(
        "INSERT INTO state_sequence (run_id, step_index, state_name) VALUES (?, ?, ?)",
        state_rows,
    )
    conn.executemany(
        """INSERT INTO tool_calls (run_id, call_index, tool_name, args_json, args_hash,
               result_json, result_hash, state) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        tool_rows,
    )


def ingest_all() -> int:
    conn = _connect()
    count = 0
    for path in sorted(RAW_LOG_DIR.glob("*.jsonl")):
        ingest_file(conn, path)
        count += 1
    conn.commit()
    conn.close()
    return count


if __name__ == "__main__":
    n = ingest_all()
    print(f"Ingested {n} run(s) into {DB_PATH}")
