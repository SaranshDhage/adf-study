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

    def log_adf_config(
        self,
        cfg_dict: dict,
        adf_total: float,
        adf_rate: float,
        n_decision_points: int,
        per_layer: dict,
        proxy: int,
    ) -> None:
        """Log the HarnessConfig and its computed ADF at run start.
        Emitted once per run, immediately after run_start.
        """
        self._write(
            "adf_config",
            harness_config=cfg_dict,
            adf_total=adf_total,
            adf_rate=adf_rate,
            n_decision_points=n_decision_points,
            per_layer=per_layer,
            proxy=proxy,
        )

    def log_routing_decision(
        self,
        step_index: int,
        state: str,
        mode: str,
        agent_name: str,
        candidates: list,
    ) -> None:
        """Log the routing decision for one step (fixed or model-chosen)."""
        self._write(
            "routing_decision",
            step_index=step_index,
            state=state,
            mode=mode,
            agent_name=agent_name,
            candidates=candidates,
        )

    def log_skill_load(
        self,
        agent_name: str,
        skill_level: str,
        skill_hash: str,
    ) -> None:
        """Log which skill file was loaded for an agent."""
        self._write(
            "skill_load",
            agent_name=agent_name,
            skill_level=skill_level,
            skill_hash=skill_hash,
        )

    def log_step_attempt(
        self,
        step_index: int,
        state: str,
        attempt: int,
        tool_called: Optional[str],
        args: dict,
        result: Any,
        valid: bool,
        error: Optional[str],
    ) -> None:
        """Log a single tool-call attempt within a step (all attempts retained,
        including failures — see loop_eng.md §1 rule 5).
        """
        self._write(
            "step_attempt",
            step_index=step_index,
            state=state,
            attempt=attempt,
            tool_called=tool_called,
            args=args,
            args_hash=hash_obj(args),
            result=result,
            result_hash=hash_obj(result),
            valid=valid,
            error=error,
        )

    def log_state_choice(
        self,
        step_index: int,
        mode: str,
        chosen: str,
        candidates: list,
        raw_response: str = "",
    ) -> None:
        """Log a state-selection decision (model_chosen mode)."""
        self._write(
            "state_choice",
            step_index=step_index,
            mode=mode,
            chosen=chosen,
            candidates=candidates,
            raw_response=raw_response[:500],  # truncate to keep log manageable
        )

    def log_codegen_attempt(
        self,
        step_index: int,
        state: str,
        attempt: int,
        code: str,
        result: Any,
        error: Optional[str],
    ) -> None:
        """Log a codegen substrate attempt: code written + execution outcome."""
        self._write(
            "codegen_attempt",
            step_index=step_index,
            state=state,
            attempt=attempt,
            code=code,
            code_hash=hash_obj(code),
            result=result,
            result_hash=hash_obj(result),
            error=error,
        )

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
