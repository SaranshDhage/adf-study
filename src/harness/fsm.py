"""Deterministic Execution Harness (Phase 2) — FSM executor.

Constrains the baseline agent (baseline_agent.py) with the mechanisms from
draft-1.md §7:
  - Finite-state execution: fixed state order, no skipping/reordering.
  - Constrained tool selection: exactly one tool bound per state.
  - Structured planning: the agent still emits a plan (for comparability
    with the baseline's Plan Stability metric), but the plan does NOT
    influence execution — actual state order is fixed by design, so State
    Transition Stability is 1.0 by construction under harness (see
    docs/determinism_index.md's discussion of STS as most informative as a
    *baseline* metric).
  - Validation layer: each tool's output is checked against a per-state
    validator before the FSM advances.
  - Execution policy: bounded retries per state on validation/tool failure;
    exhausting retries aborts the run (escalation, not a silent skip).
"""

import json
import re
import time
from typing import Any, Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.harness.llm_utils import accumulate_usage, make_llm, parse_plan
from src.harness.tools_finance import make_finance_tools
from src.harness.tools_legal import make_legal_tools
from src.tasks import finance_ecl, legal_clause
from src.tracing import TraceLogger, hash_obj

MAX_RETRIES_PER_STATE = 2
MAX_PLAN_RETRIES = 2


def _is_list_of_dicts(value: Any, required_keys: set[str]) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, dict) and required_keys <= item.keys() for item in value
    )


FINANCE_STATE_TOOL = {
    "LOAD_DATA": "data_loader",
    "VALIDATE_DATA": "validation_tool",
    "CALCULATE": "calculator_tool",
    "GENERATE_REPORT": "report_generator",
}
FINANCE_STATE_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "LOAD_DATA": lambda r: _is_list_of_dicts(r, {"loan_id", "balance", "pd", "lgd"}),
    "VALIDATE_DATA": lambda r: isinstance(r, dict)
    and isinstance(r.get("valid"), list)
    and isinstance(r.get("excluded"), list),
    "CALCULATE": lambda r: isinstance(r, dict)
    and isinstance(r.get("per_loan"), list)
    and isinstance(r.get("total_ecl"), (int, float)),
    "GENERATE_REPORT": lambda r: isinstance(r, dict)
    and {"total_ecl", "valid_count", "excluded", "per_loan"} <= r.keys(),
}
LEGAL_STATE_TOOL = {
    "LOAD_DOC": "document_loader",
    "EXTRACT_CLAUSES": "clause_extractor",
    "CLASSIFY": "clause_classifier",
    "GENERATE_REPORT": "report_generator",
}
LEGAL_STATE_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "LOAD_DOC": lambda r: isinstance(r, str) and len(r.strip()) > 0,
    "EXTRACT_CLAUSES": lambda r: _is_list_of_dicts(r, {"clause_id", "text"}),
    "CLASSIFY": lambda r: _is_list_of_dicts(r, {"clause_id", "category", "snippet"}),
    "GENERATE_REPORT": lambda r: isinstance(r, dict)
    and {"clause_count", "category_counts", "clauses"} <= r.keys(),
}
TASKS: dict[str, dict[str, Any]] = {
    "finance_ecl": {
        "module": finance_ecl,
        "make_tools": make_finance_tools,
        "states": ["LOAD_DATA", "VALIDATE_DATA", "CALCULATE", "GENERATE_REPORT"],
        "state_tool": FINANCE_STATE_TOOL,
        "validators": FINANCE_STATE_VALIDATORS,
        "report_state": "GENERATE_REPORT",
    },
    "legal_clause": {
        "module": legal_clause,
        "make_tools": make_legal_tools,
        "states": ["LOAD_DOC", "EXTRACT_CLAUSES", "CLASSIFY", "GENERATE_REPORT"],
        "state_tool": LEGAL_STATE_TOOL,
        "validators": LEGAL_STATE_VALIDATORS,
        "report_state": "GENERATE_REPORT",
    },
}


def _structured_plan_prompt(states: list[str], state_tool: dict[str, str]) -> str:
    mapping = ", ".join(f"{s} -> {state_tool[s]}" for s in states)
    return (
        "Before taking any action, output ONLY a JSON array committing to your "
        "exact execution plan — nothing else. Each element must be an object "
        'with exactly two keys: "state" and "intended_tool". Include exactly '
        f"these states in this exact order: {', '.join(states)}. Each state's "
        f"intended_tool must be: {mapping}."
    )


def parse_structured_plan(content: str) -> Optional[list]:
    """Best-effort JSON-array extraction, same approach as parse_plan (models
    often wrap the array in prose/markdown fences despite instructions)."""
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def validate_structured_plan(
    plan: Optional[list], states: list[str], state_tool: dict[str, str]
) -> tuple[bool, str]:
    """Gate for Structured Planning: the emitted plan must commit to exactly
    the FSM's legal state graph (§ this module's docstring) — same states, in
    the same fixed order, each paired with its one legal tool. This is the
    mechanism the reproducibility-diagnostic (checkpoint.md Session 4) found
    missing: unlike tool/state execution, plan text was previously
    unconstrained free text and was the dominant source of measured
    Reproducibility Rate variance."""
    if not isinstance(plan, list):
        return False, "response was not a JSON array"
    if len(plan) != len(states):
        return False, f"expected {len(states)} plan steps, got {len(plan)}"
    for expected_state, entry in zip(states, plan):
        if not isinstance(entry, dict):
            return False, f"step for {expected_state} is not an object"
        if entry.get("state") != expected_state:
            return False, f"expected state {expected_state!r}, got {entry.get('state')!r}"
        expected_tool = state_tool[expected_state]
        if entry.get("intended_tool") != expected_tool:
            return (
                False,
                f"state {expected_state} expected intended_tool {expected_tool!r}, "
                f"got {entry.get('intended_tool')!r}",
            )
    return True, ""


def run_harness(
    task_name: str,
    model: str,
    run_index: int,
    condition: str = "harness",
    temperature: float = 0.7,
    llm: Optional[Any] = None,
    max_retries_per_state: int = MAX_RETRIES_PER_STATE,
    use_validation: bool = True,
    tool_choice: Optional[str] = None,
    structured_planning: bool = False,
    max_plan_retries: int = MAX_PLAN_RETRIES,
) -> dict[str, Any]:
    """`llm` is an injection point for tests (see tests/test_agents.py) — a
    scripted fake chat model can stand in for the real OpenRouter-backed
    model so control flow (retries, error paths) is verifiable without live
    inference.

    `max_retries_per_state`/`use_validation` are the Phase 5 ablation knobs:
    defaults reproduce the full/"combined" harness. Passing
    max_retries_per_state=0, use_validation=False isolates the pure effect of
    fixed state order + tool binding with no safety net (single attempt per
    state, no output validation gate) — see condition "ablation_fsm_only" in
    src/run_experiment.py.

    `tool_choice` is a per-model calibration knob, NOT a harness-design
    parameter: the harness always presents exactly one tool per state (the
    substantive "constrained tool selection" mechanism), but *forcing* that
    tool via the API's tool_choice="required" is only needed for some models.
    Discovered empirically: google/gemma-3-27b-it sometimes returns no tool
    call at all without forcing; qwen/qwen-2.5-7b-instruct worked reliably
    without it, and forcing it there triggered a provider-side bug (the
    "Phala" backend serving this model corrupts its own JSON output when
    tool_choice="required" is set — reproduced 3/3, confirmed independent of
    our code by testing the raw OpenAI-compatible client directly). Default
    None leaves tool_choice unset (matches Qwen's working configuration);
    pass "required" explicitly per model that needs it (see src/run_experiment.py).

    `structured_planning`/`max_plan_retries` add the Structured Planning
    mechanism (condition "harness_structured_planning" in
    src/run_experiment.py): instead of a free-text plan (logged but
    unconstrained), the model must commit to the FSM's exact state/tool
    graph as a schema-validated JSON array before execution starts, with
    bounded retries and escalation on repeated failure — see
    validate_structured_plan.
    """
    cfg = TASKS[task_name]
    module = cfg["module"]
    session: dict[str, Any] = {}
    tools_by_name = {t.name: t for t in cfg["make_tools"](session)}
    states = cfg["states"]

    logger = TraceLogger(
        task=task_name,
        model=model,
        condition=condition,
        run_index=run_index,
        task_input_hash=hash_obj(str(module.DATA_PATH)),
    )

    llm = llm or make_llm(model, temperature=temperature)
    task_prompt = module.TASK_PROMPT.format(data_path=str(module.DATA_PATH))

    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    final_output: Optional[Any] = None
    success = False
    error: Optional[str] = None
    last_state = "START"
    start = time.time()

    try:
        if structured_planning:
            # Structured Planning gate: the plan must commit to the FSM's
            # exact legal state graph (states, in order, each with its one
            # legal tool) before execution starts — reject/retry on
            # mismatch, escalate if never satisfied. This constrains the
            # one axis (free-text plan wording) that tool/state/output
            # determinism doesn't touch — see validate_structured_plan.
            plan_messages: list[Any] = [
                SystemMessage(content=task_prompt),
                HumanMessage(content=_structured_plan_prompt(states, cfg["state_tool"])),
            ]
            plan_error: Optional[str] = None
            for _attempt in range(max_plan_retries + 1):
                plan_response = llm.invoke(plan_messages)
                accumulate_usage(plan_response, totals)
                plan_messages.append(plan_response)

                parsed_plan = parse_structured_plan(plan_response.content)
                valid, reason = validate_structured_plan(parsed_plan, states, cfg["state_tool"])
                if valid:
                    plan_error = None
                    break
                plan_error = reason
                plan_messages.append(
                    HumanMessage(content=f"Invalid plan: {reason}. Try again, output ONLY the JSON array.")
                )

            if plan_error:
                error = f"structured plan validation failed after {max_plan_retries} retries: {plan_error}"
                logger.log_plan([])
            else:
                logger.log_plan([f"{s}:{cfg['state_tool'][s]}" for s in states])
        else:
            # Logged for comparability with the baseline's Plan Stability metric.
            # This plan does NOT influence execution — see module docstring.
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

        history: list[Any] = [SystemMessage(content=task_prompt)]

        for state in states if error is None else []:
            tool_name = cfg["state_tool"][state]
            tool_fn = tools_by_name[tool_name]
            validator = cfg["validators"][state] if use_validation else (lambda _r: True)
            bind_kwargs = {"tool_choice": tool_choice} if tool_choice else {}
            llm_state_bound = llm.bind_tools([tool_fn], **bind_kwargs)

            instruction = (
                f"You are now in state {state}. Call the '{tool_name}' tool "
                f"(it takes no arguments — it operates on data already in the "
                f"working context from earlier steps). This is the only tool "
                f"available in this state."
            )
            state_messages = history + [HumanMessage(content=instruction)]

            result: Any = None
            call: Optional[dict[str, Any]] = None
            state_error: Optional[str] = None

            for _attempt in range(max_retries_per_state + 1):
                response = llm_state_bound.invoke(state_messages)
                accumulate_usage(response, totals)
                state_messages.append(response)

                if not response.tool_calls:
                    state_error = f"model did not call {tool_name}"
                    state_messages.append(
                        HumanMessage(content=f"You must call the '{tool_name}' tool. Try again.")
                    )
                    continue

                call = response.tool_calls[0]
                if call["name"] != tool_name:
                    state_error = f"model called {call['name']} instead of {tool_name}"
                    state_messages.append(
                        ToolMessage(
                            content=f"'{call['name']}' is not available in state {state}.",
                            tool_call_id=call["id"],
                        )
                    )
                    state_messages.append(
                        HumanMessage(content=f"You must call the '{tool_name}' tool. Try again.")
                    )
                    continue

                try:
                    result = tool_fn.invoke(call["args"])
                except Exception as exc:  # noqa: BLE001 - captured into trace, not swallowed
                    result = {"error": str(exc)}

                state_messages.append(
                    ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"])
                )

                if validator(result):
                    state_error = None
                    break

                state_error = f"{tool_name} output failed validation"
                state_messages.append(
                    HumanMessage(
                        content=f"That output failed validation. Call '{tool_name}' again."
                    )
                )

            logger.log_state_transition(last_state, state)
            last_state = state
            logger.log_tool_call(
                tool_name, args=(call["args"] if call else {}), result=result, state=state
            )

            if state_error:
                error = (
                    f"state {state} failed after {max_retries_per_state} retries: {state_error}"
                )
                break

            history.append(
                HumanMessage(content=f"{tool_name} result: {json.dumps(result, default=str)}")
            )

            if state == cfg["report_state"]:
                final_output = result

        if error is None and final_output is not None:
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
    result = run_harness(task_arg, model_arg, run_index=0)
    print(json.dumps(result, indent=2, default=str))
