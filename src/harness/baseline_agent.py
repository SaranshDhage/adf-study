"""Vanilla (unconstrained) baseline agent — Phase 1 of the harness-engineering
experiment. No FSM, no tool allow-lists, no structured-planning schema: the
model is free to plan and call tools however it likes. Phase 3 will run the
harnessed counterpart against the same tasks/models for comparison.

Flow per run:
  1. Plan turn — ask the model for a free-form ordered plan (logged, but not
     enforced) so Plan Stability (docs/determinism_index.md) has something to
     compare against Tool Path Consistency.
  2. Execution loop — standard tool-calling loop with no restrictions on
     which tool can be called when. A run ends when the model calls the
     task's report_generator tool (final_output = that call's result) or
     when it stops calling tools / hits MAX_ITERATIONS (treated as failure).

Every state transition, tool call, and the final outcome are logged via
TraceLogger (src/tracing/) for later ingestion and metric computation.
"""

import json
import time
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.harness.llm_utils import accumulate_usage, make_llm, parse_plan
from src.harness.tools_finance import REPORT_TOOL_NAME as FINANCE_REPORT_TOOL
from src.harness.tools_finance import make_finance_tools
from src.harness.tools_legal import REPORT_TOOL_NAME as LEGAL_REPORT_TOOL
from src.harness.tools_legal import make_legal_tools
from src.tasks import finance_ecl, legal_clause
from src.tracing import TraceLogger, hash_obj

MAX_ITERATIONS = 12

TASKS: dict[str, dict[str, Any]] = {
    "finance_ecl": {
        "module": finance_ecl,
        "make_tools": make_finance_tools,
        "report_tool": FINANCE_REPORT_TOOL,
    },
    "legal_clause": {
        "module": legal_clause,
        "make_tools": make_legal_tools,
        "report_tool": LEGAL_REPORT_TOOL,
    },
}


def run_baseline(
    task_name: str,
    model: str,
    run_index: int,
    condition: str = "baseline",
    temperature: float = 0.7,
    llm: Optional[Any] = None,
) -> dict[str, Any]:
    """`llm` is an injection point for tests (see tests/test_agents.py) — a
    scripted fake chat model can stand in for the real OpenRouter-backed
    model so control flow (retries, error paths) is verifiable without live
    inference."""
    cfg = TASKS[task_name]
    module = cfg["module"]
    session: dict[str, Any] = {}
    tools = cfg["make_tools"](session)
    tools_by_name = {t.name: t for t in tools}
    report_tool_name = cfg["report_tool"]

    logger = TraceLogger(
        task=task_name,
        model=model,
        condition=condition,
        run_index=run_index,
        task_input_hash=hash_obj(str(module.DATA_PATH)),
    )

    llm = llm or make_llm(model, temperature=temperature)
    llm_with_tools = llm.bind_tools(tools)
    task_prompt = module.TASK_PROMPT.format(data_path=str(module.DATA_PATH))

    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    final_output: Optional[Any] = None
    success = False
    error: Optional[str] = None
    last_state = "START"
    start = time.time()

    try:
        plan_messages = [
            SystemMessage(content=task_prompt),
            HumanMessage(
                content="Before taking any action, output ONLY a JSON array of "
                "short step names describing your intended plan, nothing else."
            ),
        ]
        plan_response = llm.invoke(plan_messages)
        accumulate_usage(plan_response, totals)
        logger.log_plan(parse_plan(plan_response.content))

        messages = [
            SystemMessage(content=task_prompt),
            HumanMessage(content="Begin executing the task now using the available tools."),
        ]
        for _ in range(MAX_ITERATIONS):
            response = llm_with_tools.invoke(messages)
            accumulate_usage(response, totals)
            messages.append(response)

            if not response.tool_calls:
                final_output = {"raw_text": response.content}
                error = "agent stopped without calling report_generator"
                break

            report_result = None
            for call in response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                tool_fn = tools_by_name.get(tool_name)
                if tool_fn is None:
                    result = {"error": f"unknown tool {tool_name}"}
                else:
                    try:
                        result = tool_fn.invoke(tool_args)
                    except Exception as exc:  # noqa: BLE001 - captured into trace, not swallowed
                        result = {"error": str(exc)}

                logger.log_state_transition(last_state, tool_name)
                last_state = tool_name
                logger.log_tool_call(tool_name, args=tool_args, result=result, state=tool_name)
                messages.append(
                    ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"])
                )
                if tool_name == report_tool_name:
                    report_result = result

            if report_result is not None:
                final_output = report_result
                break
        else:
            error = "max iterations reached without calling report_generator"

        if final_output is not None:
            success = hash_obj(final_output) == hash_obj(module.GROUND_TRUTH)
    except Exception as exc:  # noqa: BLE001 - run-level failure must still be logged, not raised
        error = f"run failed: {exc}"
    finally:
        latency_ms = (time.time() - start) * 1000
        logger.log_run_end(
            final_output=final_output,
            success=success,
            error=error,
            total_tokens=totals["total_tokens"],
            prompt_tokens=totals["prompt_tokens"],
            completion_tokens=totals["completion_tokens"],
            latency_ms=latency_ms,
        )

    return {
        "run_id": logger.run_id,
        "success": success,
        "error": error,
        "final_output": final_output,
    }


if __name__ == "__main__":
    import sys

    task_arg = sys.argv[1] if len(sys.argv) > 1 else "finance_ecl"
    model_arg = sys.argv[2] if len(sys.argv) > 2 else "qwen/qwen-2.5-7b-instruct"
    result = run_baseline(task_arg, model_arg, run_index=0)
    print(json.dumps(result, indent=2, default=str))
