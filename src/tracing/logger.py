"""Append-only JSONL trace logger — one file per run, source of truth.

Usage:
    logger = TraceLogger(task="finance_ecl", model="llama3",
                          condition="baseline", run_index=7,
                          task_input_hash=hash_obj(task_input))
    logger.log_plan(["LOAD_DATA", "VALIDATE_DATA", "CALCULATE", "GENERATE_REPORT"])
    logger.log_state_transition("START", "LOAD_DATA")
    logger.log_tool_call("data_loader", args={...}, result={...}, state="LOAD_DATA")
    logger.log_run_end(final_output={...}, success=True, total_tokens=1234,
                        prompt_tokens=900, completion_tokens=334, latency_ms=2100.5)

See docs/trace_schema.md for the event schema this writes.
"""

import time
from pathlib import Path
from typing import Any, Optional

from .util import canonical_json, hash_obj

RAW_LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "raw"


def sanitize_model_name(model: str) -> str:
    """Model slugs from API providers (e.g. OpenRouter's "qwen/qwen-2.5-7b-instruct")
    contain characters illegal in filenames/run_ids — flatten them."""
    return model.replace("/", "--").replace(":", "-")


def make_run_id(task: str, model: str, condition: str, run_index: int) -> str:
    return f"{task}__{sanitize_model_name(model)}__{condition}__{run_index:03d}"


class TraceLogger:
    def __init__(
        self,
        task: str,
        model: str,
        condition: str,
        run_index: int,
        task_input_hash: str,
        log_dir: Path = RAW_LOG_DIR,
    ):
        self.run_id = make_run_id(task, model, condition, run_index)
        self._call_index = 0
        log_dir.mkdir(parents=True, exist_ok=True)
        self._path = log_dir / f"{self.run_id}.jsonl"
        # Overwrite on construction: re-running a (task, model, condition, run_index)
        # tuple replaces the prior attempt rather than appending duplicate events.
        self._fh = open(self._path, "w", encoding="utf-8")
        self._write(
            "run_start",
            task=task,
            model=model,
            condition=condition,
            run_index=run_index,
            task_input_hash=task_input_hash,
        )

    def _write(self, event: str, **fields: Any) -> None:
        record = {"event": event, "run_id": self.run_id, "timestamp": time.time(), **fields}
        self._fh.write(canonical_json(record) + "\n")
        self._fh.flush()

    def log_plan(self, steps: list[str]) -> None:
        self._write("plan", steps=steps)

    def log_state_transition(self, from_state: str, to_state: str) -> None:
        self._write("state_transition", from_state=from_state, to_state=to_state)

    def log_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: Any,
        state: Optional[str] = None,
    ) -> None:
        self._write(
            "tool_call",
            call_index=self._call_index,
            tool_name=tool_name,
            args=args,
            args_hash=hash_obj(args),
            result=result,
            result_hash=hash_obj(result),
            state=state,
        )
        self._call_index += 1

    def log_run_end(
        self,
        final_output: Any,
        success: bool,
        error: Optional[str] = None,
        total_tokens: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        self._write(
            "run_end",
            final_output=final_output,
            success=success,
            error=error,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
        self._fh.close()
