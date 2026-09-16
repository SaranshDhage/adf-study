"""Configurable harness executor — Phase 2.

Accepts a HarnessConfig and a TaskDefinition; every layer (plan / state /
routing / tool / arg / substrate) is independently togglable.  The harness
must run with *any* legal combination of HarnessConfig fields (loop_eng.md §3).

Design principles
-----------------
- Reuses src/harness/fsm.py logic for plan parsing/validation; does NOT
  duplicate it.
- Each run emits a per-layer TraceLogger event stream.  Failed steps are
  retained in the trace (loop_eng.md §1 rule 5).
- ADF is computed from (cfg, task_spec) before execution begins and logged
  as an "adf_config" event.
- codegen substrate: implemented with exec() in a controlled namespace.
  Not production-safe; correct for a research harness.
- arg_mode is realised through LangChain tool schema variation: the model
  actually receives tools with different argument schemas, so the measured
  freedom is real, not merely claimed.
- skill_level and retry_policy contribute 0 bits to ADF (spec §2.3) and
  do not affect the control-flow graph of this module; they are logged only.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from src.adf.metric import ADFResult, FINANCE_ECL, HarnessConfig, TaskSpec, compute_adf
from src.harness.fsm import (
    MAX_PLAN_RETRIES,
    MAX_RETRIES_PER_STATE,
    _structured_plan_prompt,
    parse_structured_plan,
    validate_structured_plan,
)
from src.harness.llm_utils import accumulate_usage, make_llm, parse_plan
from src.tracing import TraceLogger, hash_obj

# Maximum steps the model-chosen state loop may run before the run is
# terminated (prevents infinite loops on converging tasks).
_MAX_MODEL_CHOSEN_STEPS = 30


# ---------------------------------------------------------------------------
# Arg-mode tool schema helpers
# ---------------------------------------------------------------------------

def _make_arg_variant(
    name: str,
    description: str,
    body_fn: Callable[[], Any],
    arg_mode: str,
    enum_arity: int = 4,
) -> BaseTool:
    """Wrap a zero-arg session tool with the schema appropriate for arg_mode.

    body_fn is a plain callable (no LangChain decoration) that reads/writes
    session state and returns the tool result.  The wrapper adds an argument
    of the right schema so the model actually faces the configured degree of
    freedom at the argument layer.
    """
    if arg_mode == "fixed":
        # No arguments — model has 1 choice (call or not, but the harness
        # forces the call, so effectively 0 bits).
        class _NoArgs(BaseModel):
            pass

        def _no_arg_fn(**_: Any) -> Any:
            return body_fn()

        return StructuredTool.from_function(
            func=_no_arg_fn,
            name=name,
            description=description,
            args_schema=_NoArgs,
        )

    if arg_mode == "enum_strict":
        opts = [f"mode_{i}" for i in range(enum_arity)]
        opts_str = ", ".join(opts)

        class _EnumArgs(BaseModel):
            invocation_mode: str = Field(
                default="mode_0",
                description=f"Processing mode. Must be exactly one of: {opts_str}."
            )

        def _enum_fn(invocation_mode: str = "mode_0", **_: Any) -> Any:
            return body_fn()

        return StructuredTool.from_function(
            func=_enum_fn,
            name=name,
            description=f"{description}  [enum_strict: choose one of {opts_str}]",
            args_schema=_EnumArgs,
        )

    if arg_mode == "typed":
        class _TypedArgs(BaseModel):
            invocation_note: str = Field(
                default="",
                description="Briefly describe how you intend to execute this step."
            )

        def _typed_fn(invocation_note: str = "", **_: Any) -> Any:
            return body_fn()

        return StructuredTool.from_function(
            func=_typed_fn,
            name=name,
            description=description,
            args_schema=_TypedArgs,
        )

    # free_form (also used as the normalised value when codegen overrides arg_mode)
    class _FreeArgs(BaseModel):
        invocation_spec: str = Field(
            default="",
            description="Specify anything you want for this invocation. Any format accepted."
        )

    def _free_fn(invocation_spec: str = "", **_: Any) -> Any:
        return body_fn()

    return StructuredTool.from_function(
        func=_free_fn,
        name=name,
        description=description,
        args_schema=_FreeArgs,
    )


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

@dataclass
class TaskDefinition:
    """Everything the configurable harness needs to execute one task family.

    make_raw_tools(session) returns a dict of {tool_name: zero_arg_callable}.
    The harness wraps each callable with _make_arg_variant() to apply the
    configured arg schema before presenting the tool to the model.

    skill_texts is keyed by skill_level ("none" | "vague" | "precise"); each
    value is appended to the worker's system prompt when that skill level is
    active.  Empty string → no extra guidance.
    """
    name: str
    task_spec: TaskSpec
    states: list                    # nominal state order (for fsm_fixed)
    state_tool_map: dict            # state -> forced tool name
    state_tool_subsets: dict        # state -> [tool names] for tool_mode=subset
    all_tool_names: list            # all M tool names (for tool_mode=free_choice)
    tool_descriptions: dict         # tool_name -> description string
    make_raw_tools: Callable        # (session: dict) -> dict[str, Callable]
    validators: dict                # state -> validator Callable
    report_state: str               # state that produces final_output
    worker_map: dict                # state -> agent_name (fixed routing)
    all_workers: list               # all K worker names
    skill_texts: dict               # skill_level -> str
    ground_truth: Any
    task_prompt: str
    codegen_env_factory: Callable
    linear_states: Any = None  # Optional[list]: if set, fsm_fixed uses this shorter
                               # sequence instead of `states`.  Used by branching tasks
                               # to enforce a linear path at ADF≈0, which cannot handle
                               # data-dependent branches — the mechanism for H2 failure
                               # at the over-constrained end.   # (session) -> dict of names for exec() env


# ---------------------------------------------------------------------------
# Per-run data structures (returned in-memory; separately logged to JSONL)
# ---------------------------------------------------------------------------

@dataclass
class ToolAttempt:
    attempt: int
    tool_called: Optional[str]
    args: dict
    result: Any
    valid: bool
    error: Optional[str]


@dataclass
class StepTrace:
    step_index: int
    state: str
    agent_name: str
    routing_mode: str
    tools_available: list
    tool_mode: str
    tool_selected: Optional[str]
    arg_mode: str
    substrate: str
    attempts: list           # list[ToolAttempt]
    final_result: Any
    step_success: bool       # did the step pass its validator?
    error: Optional[str]


@dataclass
class RunResult:
    run_id: str
    harness_config: HarnessConfig
    adf_total: float
    adf_rate: float
    plan_mode: str
    plan_content: Any
    plan_valid: bool
    plan_retries: int
    steps: list              # list[StepTrace]
    final_output: Any
    task_success: bool
    error: Optional[str]
    realized_T: int
    tokens: dict
    latency_ms: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _plan_phase(
    cfg: HarnessConfig,
    task_def: TaskDefinition,
    llm: Any,
    logger: TraceLogger,
    totals: dict,
) -> tuple[Any, bool, int]:
    """Execute the plan phase.  Returns (plan_content, plan_valid, retries)."""
    import pathlib as _pl
    # For schema_validated plans on branching tasks, validate against the
    # linear_states sequence (the FSM the constrained harness actually runs).
    states = (
        task_def.linear_states
        if (cfg.state_mode == "fsm_fixed" and task_def.linear_states is not None)
        else task_def.states
    )
    state_tool = task_def.state_tool_map
    _data_root = _pl.Path(__file__).resolve().parents[2] / "data"
    try:
        resolved = task_def.task_prompt.format(data_path=str(_data_root))
    except KeyError:
        resolved = task_def.task_prompt
    system_msg = SystemMessage(content=resolved)

    if cfg.plan_mode == "schema_validated":
        # Reuse the FSM's structured planning gate verbatim (spec §3: "reuse
        # the existing FSM, validation gate, and Structured Planning").
        plan_messages: list = [
            system_msg,
            HumanMessage(content=_structured_plan_prompt(states, state_tool)),
        ]
        plan_error: Optional[str] = None
        retries = 0
        for attempt in range(MAX_PLAN_RETRIES + 1):
            resp = llm.invoke(plan_messages)
            accumulate_usage(resp, totals)
            plan_messages.append(resp)
            parsed = parse_structured_plan(resp.content)
            valid, reason = validate_structured_plan(parsed, states, state_tool)
            if valid:
                plan_error = None
                logger.log_plan([f"{s}:{state_tool[s]}" for s in states])
                return parsed, True, retries
            plan_error = reason
            retries = attempt + 1
            plan_messages.append(
                HumanMessage(
                    content=f"Invalid plan: {reason}. Output ONLY the JSON array."
                )
            )
        # Exhausted retries — plan validation failed; this is DATA not a discard.
        logger.log_plan([])
        return None, False, MAX_PLAN_RETRIES

    else:  # free_text
        plan_messages = [
            system_msg,
            HumanMessage(
                content=(
                    "Before taking any action, output ONLY a JSON array of short "
                    "step names describing your intended plan, nothing else."
                )
            ),
        ]
        resp = llm.invoke(plan_messages)
        accumulate_usage(resp, totals)
        parsed = parse_plan(resp.content)
        logger.log_plan(parsed)
        return parsed, True, 0  # free_text plans are always "valid" (unconstrained)


def _routing_phase(
    cfg: HarnessConfig,
    task_def: TaskDefinition,
    state: str,
    step_index: int,
    history: list,
    llm: Any,
    logger: TraceLogger,
    totals: dict,
) -> str:
    """Determine which agent handles this step.  Returns agent_name."""
    if cfg.routing_mode == "fixed_map":
        agent = task_def.worker_map.get(state, "default_worker")
        logger.log_routing_decision(
            step_index=step_index,
            state=state,
            mode="fixed_map",
            agent_name=agent,
            candidates=task_def.all_workers,
        )
        return agent

    # model_chosen: manager picks the agent
    candidates = task_def.all_workers
    routing_prompt = (
        f"You are managing a team of specialists. You need to handle the task "
        f"step: {state}.\n"
        f"Available specialists: {candidates}.\n"
        "Respond with ONLY a JSON object: {\"agent\": \"<name>\"}"
    )
    messages = history + [HumanMessage(content=routing_prompt)]
    resp = llm.invoke(messages)
    accumulate_usage(resp, totals)

    # Parse the agent name from the response
    agent = candidates[0]  # default
    match = re.search(r'"agent"\s*:\s*"([^"]+)"', resp.content)
    if match and match.group(1) in candidates:
        agent = match.group(1)

    logger.log_routing_decision(
        step_index=step_index,
        state=state,
        mode="model_chosen",
        agent_name=agent,
        candidates=candidates,
    )
    return agent


def _build_lc_tools(
    cfg: HarnessConfig,
    task_def: TaskDefinition,
    session: dict,
    state: str,
) -> list:
    """Build the LangChain tool list for one step, honouring tool_mode and
    arg_mode (including the codegen normalisation of arg_mode to free_form)."""
    cfg = cfg.normalized()  # ensure codegen → free_form override is applied
    raw_tools = task_def.make_raw_tools(session)
    descriptions = task_def.tool_descriptions

    def _wrap(name: str) -> BaseTool:
        return _make_arg_variant(
            name=name,
            description=descriptions.get(name, name),
            body_fn=raw_tools[name],
            arg_mode=cfg.arg_mode,
            enum_arity=task_def.task_spec.enum_arity,
        )

    if cfg.tool_mode == "forced_single":
        forced = task_def.state_tool_map[state]
        return [_wrap(forced)]

    if cfg.tool_mode == "subset":
        subset = task_def.state_tool_subsets.get(state, [task_def.state_tool_map[state]])
        return [_wrap(n) for n in subset]

    # free_choice: all M tools + answer-without-tool handled via no bind
    return [_wrap(n) for n in task_def.all_tool_names]


def _tool_call_attempt(
    step_index: int,
    state: str,
    attempt: int,
    lc_tools: list,
    tool_mode: str,
    forced_tool: Optional[str],
    state_messages: list,
    llm: Any,
    validator: Callable,
    totals: dict,
    logger: TraceLogger,
) -> tuple[ToolAttempt, list]:
    """Execute one tool-calling attempt.  Returns (ToolAttempt, updated_messages)."""
    bind_kwargs: dict = {}
    if tool_mode == "forced_single":
        # Bedrock Converse API: "any" = must call one of the provided tools.
        # (OpenAI used "required"; LangChain-AWS maps "any" → toolChoice={"any":{}})
        bind_kwargs["tool_choice"] = "any"
    llm_bound = llm.bind_tools(lc_tools, **bind_kwargs)

    resp = llm_bound.invoke(state_messages)
    accumulate_usage(resp, totals)
    state_messages = state_messages + [resp]

    if not resp.tool_calls:
        ta = ToolAttempt(attempt=attempt, tool_called=None, args={},
                         result=None, valid=False,
                         error="model did not emit a tool call")
        logger.log_step_attempt(step_index, state, attempt, None, {}, None,
                                False, ta.error)
        state_messages = state_messages + [
            HumanMessage(content="You must call a tool. Please try again.")
        ]
        return ta, state_messages

    call = resp.tool_calls[0]
    tool_name = call["name"]
    tool_args = call.get("args", {})

    # Locate the matching tool object
    tool_obj = next((t for t in lc_tools if t.name == tool_name), None)
    if tool_obj is None:
        available = [t.name for t in lc_tools]
        err = f"model called '{tool_name}', not in available set {available}"
        ta = ToolAttempt(attempt=attempt, tool_called=tool_name, args=tool_args,
                         result=None, valid=False, error=err)
        logger.log_step_attempt(step_index, state, attempt, tool_name, tool_args,
                                None, False, err)
        state_messages = state_messages + [
            ToolMessage(content=f"'{tool_name}' is not available.", tool_call_id=call["id"]),
            HumanMessage(content=f"Tool not available. Choose from: {available}."),
        ]
        return ta, state_messages

    try:
        result = tool_obj.invoke(tool_args)
    except Exception as exc:  # noqa: BLE001
        result = {"error": str(exc)}

    valid = validator(result)
    error = None if valid else f"{tool_name} output failed validation"

    ta = ToolAttempt(attempt=attempt, tool_called=tool_name, args=tool_args,
                     result=result, valid=valid, error=error)
    logger.log_step_attempt(step_index, state, attempt, tool_name, tool_args,
                            result, valid, error)

    state_messages = state_messages + [
        ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"])
    ]
    if not valid:
        state_messages = state_messages + [
            HumanMessage(content=f"Output failed validation. Call a tool again.")
        ]
    return ta, state_messages


def _exec_code_sandbox(code: str, env: dict) -> tuple[Any, Optional[str]]:
    """Execute model-generated code in a restricted namespace.

    The exec() namespace is populated with task-module functions and the
    current session dict.  The code is expected to assign its result to
    `_result` in the local namespace.

    Safety note: this uses plain exec() — acceptable for a local research
    harness, not for production.
    """
    local_ns: dict = {"_result": None}
    local_ns.update(env)
    stdout_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, local_ns)  # noqa: S102
        result = local_ns.get("_result")
        if result is None:
            captured = stdout_buf.getvalue().strip()
            result = captured if captured else None
        return result, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _codegen_attempt(
    step_index: int,
    state: str,
    attempt: int,
    state_messages: list,
    llm: Any,
    validator: Callable,
    codegen_env: dict,
    totals: dict,
    logger: TraceLogger,
) -> tuple[ToolAttempt, list]:
    """Execute one codegen substrate attempt.  Model writes Python; we exec it."""
    resp = llm.invoke(state_messages)
    accumulate_usage(resp, totals)
    state_messages = state_messages + [resp]

    # Extract code block from the model response
    code = resp.content
    fence_match = re.search(r"```(?:python)?\s*(.*?)```", code, re.DOTALL)
    if fence_match:
        code = fence_match.group(1).strip()

    result, exec_error = _exec_code_sandbox(code, codegen_env)
    valid = exec_error is None and validator(result)
    error = exec_error or (None if valid else "codegen output failed validation")

    logger.log_codegen_attempt(step_index, state, attempt, code, result, error)

    ta = ToolAttempt(attempt=attempt, tool_called="__codegen__", args={"code": code},
                     result=result, valid=valid, error=error)

    if not valid:
        feedback = error or "output failed validation"
        state_messages = state_messages + [
            HumanMessage(
                content=f"Execution error or validation failure: {feedback}. "
                        "Please rewrite the code and try again."
            )
        ]
    return ta, state_messages


def _execute_step(
    cfg: HarnessConfig,
    task_def: TaskDefinition,
    session: dict,
    state: str,
    step_index: int,
    agent_name: str,
    history: list,
    llm: Any,
    logger: TraceLogger,
    totals: dict,
) -> StepTrace:
    """Execute one task step under the configured substrate, tool, and arg modes.

    All attempts are logged and retained regardless of outcome (loop_eng.md §1.5).
    """
    cfg = cfg.normalized()
    validator = task_def.validators.get(state, lambda _: True)
    lc_tools = _build_lc_tools(cfg, task_def, session, state)
    tool_names_available = [t.name for t in lc_tools]
    forced_tool = (
        task_def.state_tool_map[state] if cfg.tool_mode == "forced_single" else None
    )

    # Skill content prepended to the step instruction
    skill_text = task_def.skill_texts.get(cfg.skill_level, "")
    skill_suffix = f"\n\nGuidance:\n{skill_text}" if skill_text else ""
    logger.log_skill_load(agent_name, cfg.skill_level, hash_obj(skill_text))

    base_instruction = (
        f"You are now executing step '{state}'. "
        f"Agent role: {agent_name}."
        + (f" Tool to use: '{forced_tool}'." if forced_tool else "")
        + skill_suffix
    )

    if cfg.action_substrate == "codegen":
        codegen_instruction = (
            f"{base_instruction}\n\n"
            "Write Python code to accomplish this step.  Your code will be "
            "executed directly.  Assign the result to `_result`.  "
            "Wrap the code in ```python ... ``` fences."
        )
        step_msgs = list(history) + [HumanMessage(content=codegen_instruction)]
        codegen_env = task_def.codegen_env_factory(session)

    elif cfg.action_substrate == "hybrid":
        # Model may call a tool OR write code.
        base_instruction = (
            base_instruction
            + "\nYou may either call one of the available tools OR write Python "
            "code (in ```python ...``` fences) to complete this step."
        )

    attempts: list = []
    final_result: Any = None
    step_error: Optional[str] = None

    max_retries = (
        MAX_RETRIES_PER_STATE
        if cfg.retry_policy in ("bounded", "bounded_with_feedback")
        else 0
    )

    step_msgs = list(history) + [HumanMessage(content=base_instruction)]

    for attempt_i in range(max_retries + 1):
        if cfg.action_substrate == "codegen":
            ta, step_msgs = _codegen_attempt(
                step_index, state, attempt_i, step_msgs, llm,
                validator, codegen_env, totals, logger,
            )
        elif cfg.action_substrate == "hybrid":
            # Try to detect whether the model writes a code block or makes a
            # tool call.  First, let the model respond freely (no tool_choice).
            resp = llm.bind_tools(lc_tools).invoke(step_msgs)
            accumulate_usage(resp, totals)
            step_msgs = step_msgs + [resp]

            if resp.tool_calls:
                # Route through tool-call path
                ta, step_msgs = _tool_call_attempt(
                    step_index, state, attempt_i, lc_tools,
                    cfg.tool_mode, forced_tool, list(step_msgs[:-1]) + [resp],
                    llm, validator, totals, logger,
                )
            else:
                # Try codegen path
                code = resp.content
                fence_match = re.search(r"```(?:python)?\s*(.*?)```", code, re.DOTALL)
                if fence_match:
                    code = fence_match.group(1).strip()
                    result, exec_err = _exec_code_sandbox(
                        code, task_def.codegen_env_factory(session)
                    )
                    valid = exec_err is None and validator(result)
                    error = exec_err or (None if valid else "output failed validation")
                    logger.log_codegen_attempt(step_index, state, attempt_i,
                                               code, result, error)
                    ta = ToolAttempt(attempt=attempt_i, tool_called="__codegen__",
                                     args={"code": code}, result=result,
                                     valid=valid, error=error)
                    if not valid:
                        step_msgs = step_msgs + [
                            HumanMessage(content=f"Failed: {error}. Try again.")
                        ]
                else:
                    # No code fence, no tool call — prompt again
                    ta = ToolAttempt(attempt=attempt_i, tool_called=None, args={},
                                     result=None, valid=False,
                                     error="no tool call and no code block detected")
                    logger.log_step_attempt(step_index, state, attempt_i, None,
                                            {}, None, False, ta.error)
                    step_msgs = step_msgs + [
                        HumanMessage(
                            content="Please either call a tool or write Python code "
                            "in ```python ...``` fences."
                        )
                    ]
        else:  # tool_call
            ta, step_msgs = _tool_call_attempt(
                step_index, state, attempt_i, lc_tools,
                cfg.tool_mode, forced_tool, step_msgs,
                llm, validator, totals, logger,
            )

        attempts.append(ta)

        if ta.valid:
            final_result = ta.result
            step_error = None
            break

        step_error = ta.error

    # Log the canonical tool_call event (for DI computation compatibility
    # with existing analysis code that reads tool_call events).
    last_valid = next((a for a in reversed(attempts) if a.valid), None)
    best = last_valid or (attempts[-1] if attempts else None)
    if best:
        logger.log_tool_call(
            tool_name=best.tool_called or "__none__",
            args=best.args,
            result=best.result,
            state=state,
        )

    return StepTrace(
        step_index=step_index,
        state=state,
        agent_name=agent_name,
        routing_mode=cfg.routing_mode,
        tools_available=tool_names_available,
        tool_mode=cfg.tool_mode,
        tool_selected=best.tool_called if best else None,
        arg_mode=cfg.arg_mode,
        substrate=cfg.action_substrate,
        attempts=attempts,
        final_result=final_result,
        step_success=step_error is None,
        error=step_error,
    )


def _get_next_state_model_chosen(
    step_index: int,
    history: list,
    all_states: list,
    terminal_state: str,
    llm: Any,
    logger: TraceLogger,
    totals: dict,
) -> str:
    """Ask the model to choose the next state from the legal state set."""
    candidates = all_states + ["END"]
    prompt = (
        f"You have completed step {step_index}. "
        f"Choose the next state to execute. Available states: {candidates}.\n"
        'Respond with ONLY a JSON object: {"next_state": "<state_name>"}\n'
        'Choose "END" when the task is complete.'
    )
    messages = list(history) + [HumanMessage(content=prompt)]
    resp = llm.invoke(messages)
    accumulate_usage(resp, totals)

    chosen = "END"  # default: terminate
    match = re.search(r'"next_state"\s*:\s*"([^"]+)"', resp.content)
    if match:
        candidate = match.group(1)
        if candidate in candidates:
            chosen = candidate

    logger.log_state_choice(
        step_index=step_index,
        mode="model_chosen",
        chosen=chosen,
        candidates=candidates,
        raw_response=resp.content,
    )
    return chosen


# ---------------------------------------------------------------------------
# Task definition factories (F1 linear task families)
# ---------------------------------------------------------------------------

def _make_f1_finance_task_def() -> TaskDefinition:
    """Build the TaskDefinition for finance_ecl (F1 linear family)."""
    from src.tasks import finance_ecl as task
    from src.harness.fsm import FINANCE_STATE_VALIDATORS

    STATES = ["LOAD_DATA", "VALIDATE_DATA", "CALCULATE", "GENERATE_REPORT"]
    STATE_TOOL = {
        "LOAD_DATA": "data_loader",
        "VALIDATE_DATA": "validation_tool",
        "CALCULATE": "calculator_tool",
        "GENERATE_REPORT": "report_generator",
    }
    ALL_TOOLS = ["data_loader", "validation_tool", "calculator_tool", "report_generator"]
    DESCRIPTIONS = {
        "data_loader": (
            "Loads the loan portfolio into the working context and returns the rows, "
            "each with loan_id, balance, pd, and lgd."
        ),
        "validation_tool": (
            "Validates the loaded loan rows: balance > 0, pd in [0,1], lgd in [0,1]. "
            "Returns {valid: [...], excluded: [{loan_id, reason}, ...]}."
        ),
        "calculator_tool": (
            "Computes ECL = balance * pd * lgd for each valid loan and sums them. "
            "Returns {per_loan: [{loan_id, ecl}, ...], total_ecl: float}."
        ),
        "report_generator": (
            "Assembles the final report: total_ecl, valid_count, excluded, per_loan. "
            "Call this last."
        ),
    }
    # Subset: two-tool windows per phase (models the partial-constraint rung).
    STATE_SUBSETS = {
        "LOAD_DATA":      ["data_loader", "validation_tool"],
        "VALIDATE_DATA":  ["data_loader", "validation_tool"],
        "CALCULATE":      ["validation_tool", "calculator_tool"],
        "GENERATE_REPORT":["calculator_tool", "report_generator"],
    }
    WORKER_MAP = {
        "LOAD_DATA": "data_worker",
        "VALIDATE_DATA": "data_worker",
        "CALCULATE": "compute_worker",
        "GENERATE_REPORT": "report_worker",
    }
    SKILL_TEXTS = {
        "none": "",
        "vague": (
            "You are a credit risk specialist. Use your domain knowledge to execute "
            "each step correctly."
        ),
        "precise": (
            "IFRS 9 ECL pipeline:\n"
            "1. LOAD_DATA: call data_loader — reads portfolio CSV into session.\n"
            "2. VALIDATE_DATA: call validation_tool — rejects rows where balance≤0, "
            "pd∉[0,1], or lgd∉[0,1].\n"
            "3. CALCULATE: call calculator_tool — ECL=balance×pd×lgd per valid row, "
            "rounded to 2dp; sum for total_ecl.\n"
            "4. GENERATE_REPORT: call report_generator — assembles total_ecl, "
            "valid_count, excluded list, per_loan list.\n"
            "Execute states in the order given above. Do not skip steps."
        ),
    }

    def make_raw_tools(session: dict) -> dict:
        def data_loader_fn():
            session["rows"] = task.load_data()
            return session["rows"]

        def validation_tool_fn():
            if "rows" not in session:
                raise ValueError("data_loader must be called before validation_tool")
            valid, excluded = task.validate_data(session["rows"])
            session["valid"] = valid
            session["excluded"] = excluded
            return {"valid": valid, "excluded": excluded}

        def calculator_tool_fn():
            if "valid" not in session:
                raise ValueError("validation_tool must be called before calculator_tool")
            calc = task.calculate_ecl(session["valid"])
            session["calc"] = calc
            return calc

        def report_generator_fn():
            if "calc" not in session or "excluded" not in session:
                raise ValueError(
                    "calculator_tool and validation_tool must be called before report_generator"
                )
            return task.generate_report(session["calc"], session["excluded"])

        return {
            "data_loader": data_loader_fn,
            "validation_tool": validation_tool_fn,
            "calculator_tool": calculator_tool_fn,
            "report_generator": report_generator_fn,
        }

    def codegen_env_factory(session: dict) -> dict:
        return {
            "load_data": task.load_data,
            "validate_data": task.validate_data,
            "calculate_ecl": task.calculate_ecl,
            "generate_report": task.generate_report,
            "session": session,
        }

    return TaskDefinition(
        name="finance_ecl",
        task_spec=FINANCE_ECL,
        states=STATES,
        state_tool_map=STATE_TOOL,
        state_tool_subsets=STATE_SUBSETS,
        all_tool_names=ALL_TOOLS,
        tool_descriptions=DESCRIPTIONS,
        make_raw_tools=make_raw_tools,
        validators=FINANCE_STATE_VALIDATORS,
        report_state="GENERATE_REPORT",
        worker_map=WORKER_MAP,
        all_workers=["data_worker", "compute_worker", "report_worker"],
        skill_texts=SKILL_TEXTS,
        ground_truth=task.GROUND_TRUTH,
        task_prompt=task.TASK_PROMPT,
        codegen_env_factory=codegen_env_factory,
    )


def _make_f1_legal_task_def() -> TaskDefinition:
    """Build the TaskDefinition for legal_clause (F1 linear family)."""
    from src.tasks import legal_clause as task
    from src.harness.fsm import LEGAL_STATE_VALIDATORS

    STATES = ["LOAD_DOC", "EXTRACT_CLAUSES", "CLASSIFY", "GENERATE_REPORT"]
    STATE_TOOL = {
        "LOAD_DOC": "document_loader",
        "EXTRACT_CLAUSES": "clause_extractor",
        "CLASSIFY": "clause_classifier",
        "GENERATE_REPORT": "report_generator",
    }
    ALL_TOOLS = ["document_loader", "clause_extractor", "clause_classifier", "report_generator"]
    DESCRIPTIONS = {
        "document_loader": "Loads the raw contract text into session.",
        "clause_extractor": (
            "Splits the loaded document into numbered clauses. "
            "Returns a list of {clause_id, text} objects."
        ),
        "clause_classifier": (
            "Classifies each extracted clause using the fixed-priority keyword taxonomy. "
            "Returns a list of {clause_id, category, snippet}."
        ),
        "report_generator": (
            "Assembles the final clause report: clause_count, category_counts, clauses list."
        ),
    }
    STATE_SUBSETS = {
        "LOAD_DOC":       ["document_loader", "clause_extractor"],
        "EXTRACT_CLAUSES":["document_loader", "clause_extractor"],
        "CLASSIFY":       ["clause_extractor", "clause_classifier"],
        "GENERATE_REPORT":["clause_classifier", "report_generator"],
    }
    WORKER_MAP = {
        "LOAD_DOC": "data_worker",
        "EXTRACT_CLAUSES": "data_worker",
        "CLASSIFY": "compute_worker",
        "GENERATE_REPORT": "report_worker",
    }
    SKILL_TEXTS = {
        "none": "",
        "vague": (
            "You are a contract review specialist. Use your legal domain knowledge "
            "to extract and classify clauses accurately."
        ),
        "precise": (
            "Contract review pipeline:\n"
            "1. LOAD_DOC: call document_loader — reads contract text into session.\n"
            "2. EXTRACT_CLAUSES: call clause_extractor — splits on 'Clause N:' pattern.\n"
            "3. CLASSIFY: call clause_classifier — fixed-priority taxonomy: Force Majeure > "
            "Confidentiality > Indemnification > Termination > Payment Terms > "
            "Governing Law > Limitation of Liability > Other.\n"
            "4. GENERATE_REPORT: call report_generator — assembles counts and clause list.\n"
            "Execute in order. Do not skip steps."
        ),
    }

    def make_raw_tools(session: dict) -> dict:
        def document_loader_fn():
            session["document"] = task.load_document()
            return session["document"]

        def clause_extractor_fn():
            if "document" not in session:
                raise ValueError("document_loader must be called before clause_extractor")
            session["clauses"] = task.extract_clauses(session["document"])
            return session["clauses"]

        def clause_classifier_fn():
            if "clauses" not in session:
                raise ValueError("clause_extractor must be called before clause_classifier")
            session["classified"] = task.classify_clauses(session["clauses"])
            return session["classified"]

        def report_generator_fn():
            if "classified" not in session:
                raise ValueError("clause_classifier must be called before report_generator")
            return task.generate_report(session["classified"])

        return {
            "document_loader": document_loader_fn,
            "clause_extractor": clause_extractor_fn,
            "clause_classifier": clause_classifier_fn,
            "report_generator": report_generator_fn,
        }

    def codegen_env_factory(session: dict) -> dict:
        return {
            "load_document": task.load_document,
            "extract_clauses": task.extract_clauses,
            "classify_clauses": task.classify_clauses,
            "generate_report": task.generate_report,
            "session": session,
        }

    # legal_clause has the same task dimensions as finance_ecl.
    import dataclasses as _dc
    from src.adf.metric import FINANCE_ECL
    legal_spec = _dc.replace(FINANCE_ECL, name="legal_clause")

    return TaskDefinition(
        name="legal_clause",
        task_spec=legal_spec,
        states=STATES,
        state_tool_map=STATE_TOOL,
        state_tool_subsets=STATE_SUBSETS,
        all_tool_names=ALL_TOOLS,
        tool_descriptions=DESCRIPTIONS,
        make_raw_tools=make_raw_tools,
        validators=LEGAL_STATE_VALIDATORS,
        report_state="GENERATE_REPORT",
        worker_map=WORKER_MAP,
        all_workers=["data_worker", "compute_worker", "report_worker"],
        skill_texts=SKILL_TEXTS,
        ground_truth=task.GROUND_TRUTH,
        task_prompt=task.TASK_PROMPT,
        codegen_env_factory=codegen_env_factory,
    )


def _make_f2_branching_task_def(instance_path: Optional[str] = None) -> TaskDefinition:
    """Build the TaskDefinition for branching_ecl (F2 family).

    instance_path: path to a specific generated portfolio CSV.  If None, uses
    the first available instance in data/branching/.
    """
    from src.tasks import branching_ecl as task
    from src.adf.metric import TaskSpec
    import pathlib as _pl

    if instance_path is None:
        paths = task.get_instance_paths()
        if not paths:
            raise FileNotFoundError(
                "No branching instances found. "
                "Run: python -m src.harness.gen_branching_instances"
            )
        inst_path = paths[0]
    else:
        inst_path = _pl.Path(instance_path)

    ground_truth = task.load_ground_truth(inst_path)

    BRANCHING_SPEC = TaskSpec(
        name="branching_ecl",
        nominal_steps=5,
        n_states=6,
        n_agents=3,
        n_tools=5,
        enum_arity=4,
        subset_size=2,
    )

    FULL_STATES = [
        "LOAD_DATA", "VALIDATE_DATA", "CALCULATE", "ESCALATION", "GENERATE_REPORT"
    ]
    LINEAR_STATES = ["LOAD_DATA", "VALIDATE_DATA", "CALCULATE", "GENERATE_REPORT"]

    STATE_TOOL = {
        "LOAD_DATA":       "data_loader",
        "VALIDATE_DATA":   "validation_tool",
        "CALCULATE":       "calculator_tool",
        "ESCALATION":      "escalation_tool",
        "GENERATE_REPORT": "report_generator",
    }
    ALL_TOOLS = [
        "data_loader", "validation_tool", "calculator_tool",
        "escalation_tool", "report_generator",
    ]
    DESCRIPTIONS = {
        "data_loader":      "Loads the loan portfolio into the working context.",
        "validation_tool":  "Validates loans: balance>0, pd in [0,1], lgd in [0,1].",
        "calculator_tool":  "Computes ECL=balance*pd*lgd per valid loan.",
        "escalation_tool":  (
            "Stress-test review for high-risk loans (PD > 0.15). "
            "Call this if any valid loan has PD > 0.15."
        ),
        "report_generator": "Assembles final report. Call last.",
    }
    STATE_SUBSETS = {
        "LOAD_DATA":       ["data_loader", "validation_tool"],
        "VALIDATE_DATA":   ["data_loader", "validation_tool"],
        "CALCULATE":       ["validation_tool", "calculator_tool"],
        "ESCALATION":      ["calculator_tool", "escalation_tool"],
        "GENERATE_REPORT": ["escalation_tool", "report_generator"],
    }
    WORKER_MAP_BR = {
        "LOAD_DATA":       "data_worker",
        "VALIDATE_DATA":   "data_worker",
        "CALCULATE":       "compute_worker",
        "ESCALATION":      "compute_worker",
        "GENERATE_REPORT": "report_worker",
    }
    VALIDATORS_BR = {
        "LOAD_DATA": lambda r: (
            isinstance(r, list)
            and all(isinstance(x, dict) and {"loan_id","balance","pd","lgd"} <= x.keys()
                    for x in r)
        ),
        "VALIDATE_DATA": lambda r: (
            isinstance(r, dict)
            and isinstance(r.get("valid"), list)
            and isinstance(r.get("excluded"), list)
        ),
        "CALCULATE": lambda r: (
            isinstance(r, dict)
            and isinstance(r.get("per_loan"), list)
            and isinstance(r.get("total_ecl"), (int, float))
        ),
        "ESCALATION": lambda r: (
            isinstance(r, dict) and "high_risk_count" in r
            and isinstance(r.get("stress_results"), list)
        ),
        "GENERATE_REPORT": lambda r: (
            isinstance(r, dict)
            and {"total_ecl","valid_count","excluded","per_loan","requires_escalation"}
            <= r.keys()
        ),
    }
    SKILL_TEXTS_BR = {
        "none": "",
        "vague": (
            "You are a credit risk specialist. Use domain knowledge to execute each "
            "step correctly and determine if escalation review is needed."
        ),
        "precise": (
            "Branching IFRS-9 ECL pipeline:\n"
            "1. LOAD_DATA: call data_loader.\n"
            "2. VALIDATE_DATA: call validation_tool.\n"
            "3. CALCULATE: call calculator_tool (ECL=balance×pd×lgd).\n"
            "4. ESCALATION (CONDITIONAL): call escalation_tool IF AND ONLY IF "
            "any valid loan has PD > 0.15. REQUIRED for high-risk portfolios, "
            "MUST NOT be skipped when PD > 0.15 is present.\n"
            "5. GENERATE_REPORT: call report_generator (include escalation if done).\n"
            "Do NOT call escalation_tool when all loans have PD ≤ 0.15."
        ),
    }

    def make_raw_tools(session: dict) -> dict:
        def data_loader_fn():
            session["rows"] = task.load_data(inst_path)
            return session["rows"]

        def validation_tool_fn():
            if "rows" not in session:
                raise ValueError("data_loader must be called first")
            valid, excluded = task.validate_data(session["rows"])
            session["valid"] = valid
            session["excluded"] = excluded
            return {"valid": valid, "excluded": excluded}

        def calculator_tool_fn():
            if "valid" not in session:
                raise ValueError("validation_tool must be called first")
            calc = task.calculate_ecl(session["valid"])
            session["calc"] = calc
            return calc

        def escalation_tool_fn():
            if "valid" not in session:
                raise ValueError("validation_tool must be called first")
            esc = task.escalation_review(session["valid"])
            session["escalation"] = esc
            return esc

        def report_generator_fn():
            if "calc" not in session or "excluded" not in session:
                raise ValueError("calculator_tool must be called first")
            return task.generate_report(
                session["calc"], session["excluded"], session.get("escalation")
            )

        return {
            "data_loader":      data_loader_fn,
            "validation_tool":  validation_tool_fn,
            "calculator_tool":  calculator_tool_fn,
            "escalation_tool":  escalation_tool_fn,
            "report_generator": report_generator_fn,
        }

    def codegen_env_factory_br(session: dict) -> dict:
        return {
            "load_data":            lambda: task.load_data(inst_path),
            "validate_data":        task.validate_data,
            "calculate_ecl":        task.calculate_ecl,
            "escalation_review":    task.escalation_review,
            "generate_report":      task.generate_report,
            "ESCALATION_THRESHOLD": task.ESCALATION_THRESHOLD,
            "session":              session,
        }

    resolved_prompt = task.TASK_PROMPT.replace("{data_path}", str(inst_path))

    return TaskDefinition(
        name="branching_ecl",
        task_spec=BRANCHING_SPEC,
        states=FULL_STATES,
        state_tool_map=STATE_TOOL,
        state_tool_subsets=STATE_SUBSETS,
        all_tool_names=ALL_TOOLS,
        tool_descriptions=DESCRIPTIONS,
        make_raw_tools=make_raw_tools,
        validators=VALIDATORS_BR,
        report_state="GENERATE_REPORT",
        worker_map=WORKER_MAP_BR,
        all_workers=["data_worker", "compute_worker", "report_worker"],
        skill_texts=SKILL_TEXTS_BR,
        ground_truth=ground_truth,
        task_prompt=resolved_prompt,
        codegen_env_factory=codegen_env_factory_br,
        linear_states=LINEAR_STATES,  # ADF≈0 configs use the 4-state linear FSM
    )


# Registry of built-in task definitions
_TASK_DEF_REGISTRY: dict[str, Callable[[], TaskDefinition]] = {
    "finance_ecl":   _make_f1_finance_task_def,
    "legal_clause":  _make_f1_legal_task_def,
    "branching_ecl": _make_f2_branching_task_def,
}


def get_task_definition(
    task_name: str,
    instance_path: Optional[str] = None,
) -> TaskDefinition:
    """Look up a built-in task definition by name.

    instance_path is passed to factory functions that support per-instance
    configuration (currently branching_ecl).
    """
    if task_name not in _TASK_DEF_REGISTRY:
        raise KeyError(
            f"Unknown task '{task_name}'. "
            f"Available: {list(_TASK_DEF_REGISTRY.keys())}"
        )
    factory = _TASK_DEF_REGISTRY[task_name]
    try:
        return factory(instance_path)  # type: ignore[call-arg]
    except TypeError:
        return factory()  # F1 factories take no args


def register_task_definition(name: str, factory: Callable[[], TaskDefinition]) -> None:
    """Register a custom task definition (used by F2/F3 when built)."""
    _TASK_DEF_REGISTRY[name] = factory


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_configurable(
    task_name: str,
    cfg: HarnessConfig,
    model: str,
    run_index: int,
    condition: str = "configurable",
    temperature: float = 0.0,
    llm: Optional[Any] = None,
    task_def: Optional[TaskDefinition] = None,
    unbounded_proxy: int = 1024,
) -> RunResult:
    """Execute one run of a task under the given HarnessConfig.

    Parameters
    ----------
    task_name:
        Task family identifier; looked up in _TASK_DEF_REGISTRY unless
        task_def is supplied directly.
    cfg:
        HarnessConfig specifying every layer.  cfg.normalized() is applied
        before execution starts.
    model:
        OpenRouter model slug.
    run_index:
        Zero-based index within this condition's N runs (used in run_id).
    condition:
        Free-form label for this experimental condition (e.g. "L3_T0").
    temperature:
        Passed to make_llm().  Primary grid uses T=0.
    llm:
        Injection point for test stubs.  If None, make_llm() is called.
    task_def:
        If supplied, used directly; overrides task_name lookup.
    unbounded_proxy:
        UNBOUNDED_PROXY value for this run's ADF computation.
    """
    cfg = cfg.normalized()

    if task_def is None:
        task_def = get_task_definition(task_name)

    # Compute ADF for this (cfg, task_spec) pair
    adf_result: ADFResult = compute_adf(cfg, task_def.task_spec, unbounded_proxy)

    logger = TraceLogger(
        task=task_def.name,
        model=model,
        condition=condition,
        run_index=run_index,
        task_input_hash=hash_obj(str(task_def.ground_truth)),
    )

    # Log ADF configuration immediately after run_start
    logger.log_adf_config(
        cfg_dict={
            "plan_mode": cfg.plan_mode,
            "state_mode": cfg.state_mode,
            "routing_mode": cfg.routing_mode,
            "tool_mode": cfg.tool_mode,
            "arg_mode": cfg.arg_mode,
            "action_substrate": cfg.action_substrate,
            "skill_level": cfg.skill_level,
            "retry_policy": cfg.retry_policy,
        },
        adf_total=adf_result.total,
        adf_rate=adf_result.rate,
        n_decision_points=adf_result.n_decision_points,
        per_layer=adf_result.per_layer,
        proxy=adf_result.proxy,
    )

    llm = llm or make_llm(model, temperature=temperature)

    totals: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    start = time.time()

    steps: list = []
    final_output: Any = None
    task_success = False
    run_error: Optional[str] = None
    plan_content: Any = None
    plan_valid: bool = True
    plan_retries: int = 0

    try:
        # ---- Plan phase ------------------------------------------------
        plan_content, plan_valid, plan_retries = _plan_phase(
            cfg, task_def, llm, logger, totals
        )  # always assigns all three; no dir() guard needed below

        if not plan_valid and cfg.plan_mode == "schema_validated":
            run_error = (
                f"schema_validated plan failed after {MAX_PLAN_RETRIES} retries"
            )

        # ---- Determine state execution sequence ------------------------
        import pathlib as _pl2
        _dr = _pl2.Path(__file__).resolve().parents[2] / "data"
        try:
            base_prompt = task_def.task_prompt.format(data_path=str(_dr))
        except KeyError:
            base_prompt = task_def.task_prompt
        skill_txt = task_def.skill_texts.get(cfg.skill_level, "")
        full_prompt = base_prompt + (
            f"\n\nSkill guidance ({cfg.skill_level}):\n{skill_txt}" if skill_txt else ""
        )
        history: list = [SystemMessage(content=full_prompt)]
        session: dict = {}

        if run_error is None:
            if cfg.state_mode == "fsm_fixed":
                # ---- FSM-fixed execution loop --------------------------
                # Branching tasks supply a `linear_states` field: the FSM
                # uses the shorter linear path (no ESCALATION), which cannot
                # satisfy branching instances — the H2 failure mechanism.
                fsm_states = (
                    task_def.linear_states
                    if task_def.linear_states is not None
                    else task_def.states
                )
                last_state = "START"
                for step_i, state in enumerate(fsm_states):
                    agent = _routing_phase(
                        cfg, task_def, state, step_i, history, llm, logger, totals
                    )
                    logger.log_state_transition(last_state, state)
                    last_state = state

                    step_trace = _execute_step(
                        cfg, task_def, session, state, step_i,
                        agent, history, llm, logger, totals,
                    )
                    steps.append(step_trace)

                    if step_trace.step_success and step_trace.final_result is not None:
                        history.append(
                            HumanMessage(
                                content=f"{state} result: "
                                + json.dumps(step_trace.final_result, default=str)
                            )
                        )
                        if state == task_def.report_state:
                            final_output = step_trace.final_result

                    if not step_trace.step_success:
                        run_error = (
                            f"step '{state}' failed after retries: {step_trace.error}"
                        )
                        break

            else:  # model_chosen state sequence
                # ---- Model-chosen execution loop -----------------------
                # The model decides the next state after each step.  We
                # cap at _MAX_MODEL_CHOSEN_STEPS to prevent infinite loops.
                all_states = list(task_def.states)
                current = all_states[0] if all_states else "END"
                prev_state = "START"
                step_i = 0
                while current != "END" and step_i < _MAX_MODEL_CHOSEN_STEPS:
                    agent = _routing_phase(
                        cfg, task_def, current, step_i, history, llm, logger, totals
                    )
                    logger.log_state_transition(prev_state, current)
                    prev_state = current

                    step_trace = _execute_step(
                        cfg, task_def, session, current, step_i,
                        agent, history, llm, logger, totals,
                    )
                    steps.append(step_trace)

                    if step_trace.step_success and step_trace.final_result is not None:
                        history.append(
                            HumanMessage(
                                content=f"{current} result: "
                                + json.dumps(step_trace.final_result, default=str)
                            )
                        )
                        if current == task_def.report_state:
                            final_output = step_trace.final_result

                    if not step_trace.step_success:
                        run_error = f"step '{current}' failed: {step_trace.error}"
                        break

                    step_i += 1
                    current = _get_next_state_model_chosen(
                        step_i, history, all_states, task_def.report_state,
                        llm, logger, totals,
                    )

        # ---- Score success ---------------------------------------------
        if run_error is None and final_output is not None:
            task_success = hash_obj(final_output) == hash_obj(task_def.ground_truth)

    except Exception as exc:  # noqa: BLE001
        run_error = f"run-level exception: {exc}"

    finally:
        latency_ms = (time.time() - start) * 1000
        realized_T = 1 + 5 * len(steps)  # matches nominal formula |T|=1+5S
        logger.log_run_end(
            final_output=final_output,
            success=task_success,
            error=run_error,
            total_tokens=totals["total_tokens"],
            prompt_tokens=totals["prompt_tokens"],
            completion_tokens=totals["completion_tokens"],
            latency_ms=latency_ms,
        )

    return RunResult(
        run_id=logger.run_id,
        harness_config=cfg,
        adf_total=adf_result.total,
        adf_rate=adf_result.rate,
        plan_mode=cfg.plan_mode,
        plan_content=plan_content,
        plan_valid=plan_valid,
        plan_retries=plan_retries,
        steps=steps,
        final_output=final_output,
        task_success=task_success,
        error=run_error,
        realized_T=realized_T,
        tokens=dict(totals),
        latency_ms=latency_ms,
    )


if __name__ == "__main__":
    import sys

    task_arg = sys.argv[1] if len(sys.argv) > 1 else "finance_ecl"
    model_arg = sys.argv[2] if len(sys.argv) > 2 else "qwen/qwen-2.5-7b-instruct"
    rung_arg = sys.argv[3] if len(sys.argv) > 3 else "L0"

    _RUNG_CFGS = {
        "L0prime": HarnessConfig(plan_mode="schema_validated", state_mode="fsm_fixed",
                                 routing_mode="fixed_map", tool_mode="forced_single",
                                 arg_mode="fixed", action_substrate="tool_call"),
        "L0": HarnessConfig(plan_mode="schema_validated", state_mode="fsm_fixed",
                            routing_mode="fixed_map", tool_mode="forced_single",
                            arg_mode="enum_strict", action_substrate="tool_call"),
        "L3": HarnessConfig(plan_mode="free_text", state_mode="fsm_fixed",
                            routing_mode="model_chosen", tool_mode="subset",
                            arg_mode="typed", action_substrate="tool_call"),
        "L5": HarnessConfig(plan_mode="free_text", state_mode="model_chosen",
                            routing_mode="model_chosen", tool_mode="free_choice",
                            arg_mode="free_form", action_substrate="tool_call"),
    }
    cfg_arg = _RUNG_CFGS.get(rung_arg, _RUNG_CFGS["L0"])

    result = run_configurable(task_arg, cfg_arg, model_arg, run_index=0,
                              condition=rung_arg)
    print(json.dumps(
        {
            "run_id": result.run_id,
            "task_success": result.task_success,
            "adf_rate": round(result.adf_rate, 3),
            "adf_total": round(result.adf_total, 3),
            "error": result.error,
            "steps": len(result.steps),
        },
        indent=2,
    ))
