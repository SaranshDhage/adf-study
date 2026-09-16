-- Trace database schema — Draft v1. See docs/trace_schema.md for rationale.
-- Built by ingest.py from logs/raw/<run_id>.jsonl; safe to drop and rebuild at any time.

CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    task                TEXT NOT NULL,
    model               TEXT NOT NULL,
    condition           TEXT NOT NULL,      -- 'baseline' | 'harness' | 'ablation_<name>'
    run_index           INTEGER NOT NULL,
    task_input_hash     TEXT NOT NULL,
    success             INTEGER,            -- 0/1, NULL if run_end never observed
    error               TEXT,
    final_output_json   TEXT,               -- canonical JSON string of final_output
    total_tokens        INTEGER,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    latency_ms          REAL,
    started_at          REAL,
    ended_at            REAL
);

CREATE TABLE IF NOT EXISTS plan_steps (
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    step_index  INTEGER NOT NULL,
    step_id     TEXT NOT NULL,
    PRIMARY KEY (run_id, step_index)
);

CREATE TABLE IF NOT EXISTS state_sequence (
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    step_index  INTEGER NOT NULL,
    state_name  TEXT NOT NULL,
    PRIMARY KEY (run_id, step_index)
);

CREATE TABLE IF NOT EXISTS tool_calls (
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    call_index  INTEGER NOT NULL,
    tool_name   TEXT NOT NULL,
    args_json   TEXT NOT NULL,
    args_hash   TEXT NOT NULL,
    result_json TEXT,
    result_hash TEXT,
    state       TEXT,                       -- FSM state active at call time, NULL under baseline
    PRIMARY KEY (run_id, call_index)
);

CREATE INDEX IF NOT EXISTS idx_runs_task_model_condition
    ON runs(task, model, condition);
