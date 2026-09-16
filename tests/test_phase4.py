"""Phase 4 validation suite — loop_eng.md §5 gates.

All 10 gates that must pass before any real API spend.  Run with:
    pytest tests/ -v

Gates implemented here:
  1. Fake-model control-flow (every HarnessConfig layer independently togglable)
  2. ADF monotonicity (tightening any single layer never increases ADF)
  3. ADF hand-check (3 configs from docs/adf_definition.md match compute_adf exactly)
  4. Trace completeness (every run emits all required events; failed run = complete trace)
  5. Ground-truth test (scorers agree with hand-computed answers for ≥5 instances/family)
  6. Branching-task sanity (ADF≈0 config MUST fail branching instances)
  7. Difficulty calibration check (both extremes score < 100% — proxy for < 90% with real model)
  8. (F3 loop termination — not applicable to F1/F2; skipped)
  9. (F4 fault injection — out of scope; skipped)
 10. (F5 judge stability — out of scope; skipped)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import pytest

# ---------------------------------------------------------------------------
# Shared stub model (imported by multiple tests)
# ---------------------------------------------------------------------------

from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration


_F1_PLAN_FINANCE = (
    '[{"state":"LOAD_DATA","intended_tool":"data_loader"},'
    '{"state":"VALIDATE_DATA","intended_tool":"validation_tool"},'
    '{"state":"CALCULATE","intended_tool":"calculator_tool"},'
    '{"state":"GENERATE_REPORT","intended_tool":"report_generator"}]'
)
_F1_PLAN_LEGAL = (
    '[{"state":"LOAD_DOC","intended_tool":"document_loader"},'
    '{"state":"EXTRACT_CLAUSES","intended_tool":"clause_extractor"},'
    '{"state":"CLASSIFY","intended_tool":"clause_classifier"},'
    '{"state":"GENERATE_REPORT","intended_tool":"report_generator"}]'
)
_F2_PLAN_FULL = (
    '[{"state":"LOAD_DATA","intended_tool":"data_loader"},'
    '{"state":"VALIDATE_DATA","intended_tool":"validation_tool"},'
    '{"state":"CALCULATE","intended_tool":"calculator_tool"},'
    '{"state":"ESCALATION","intended_tool":"escalation_tool"},'
    '{"state":"GENERATE_REPORT","intended_tool":"report_generator"}]'
)
_F2_PLAN_LINEAR = (
    '[{"state":"LOAD_DATA","intended_tool":"data_loader"},'
    '{"state":"VALIDATE_DATA","intended_tool":"validation_tool"},'
    '{"state":"CALCULATE","intended_tool":"calculator_tool"},'
    '{"state":"GENERATE_REPORT","intended_tool":"report_generator"}]'
)

_ALL_TOOL_MAPS = {
    "LOAD_DATA":       "data_loader",
    "VALIDATE_DATA":   "validation_tool",
    "CALCULATE":       "calculator_tool",
    "ESCALATION":      "escalation_tool",
    "GENERATE_REPORT": "report_generator",
    "LOAD_DOC":        "document_loader",
    "EXTRACT_CLAUSES": "clause_extractor",
    "CLASSIFY":        "clause_classifier",
}


class _FullStub(BaseChatModel):
    """Scripted fake model; drives every task family through the correct path."""

    @property
    def _llm_type(self) -> str:
        return "full_stub"

    def _find_state(self, messages: list) -> Optional[str]:
        for m in reversed(messages):
            if not hasattr(m, "content") or not m.content:
                continue
            c = m.content
            m2 = re.search(r"executing step '([A-Z_]+)'", c)
            if m2 and m2.group(1) in _ALL_TOOL_MAPS:
                return m2.group(1)
            m3 = re.search(r"Tool to use: '([a-z_]+)'", c)
            if m3:
                for s, t in _ALL_TOOL_MAPS.items():
                    if t == m3.group(1):
                        return s
            m4 = re.search(r"Choose from: \['([a-z_]+)'\]", c)
            if m4:
                for s, t in _ALL_TOOL_MAPS.items():
                    if t == m4.group(1):
                        return s
        return None

    def _generate(self, messages: list, stop=None, run_manager=None, **kwargs):
        combined = " ".join(
            m.content for m in messages if hasattr(m, "content") and m.content
        )
        last = ""
        for m in reversed(messages):
            if hasattr(m, "content") and m.content:
                last = m.content
                break

        # Schema-validated plan
        if "intended_tool" in combined:
            if "ESCALATION" in combined:
                c = _F2_PLAN_FULL
            elif "LOAD_DOC" in combined:
                c = _F1_PLAN_LEGAL
            else:
                c = _F1_PLAN_FINANCE
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=c))])

        # Free-text plan
        if "short step names" in last:
            if "LOAD_DOC" in combined:
                c = '["LOAD_DOC","EXTRACT_CLAUSES","CLASSIFY","GENERATE_REPORT"]'
            else:
                c = '["LOAD_DATA","VALIDATE_DATA","CALCULATE","GENERATE_REPORT"]'
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=c))])

        # Routing
        if '"agent"' in last or "specialists" in last:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content='{"agent":"data_worker"}'))]
            )

        # State choice (model_chosen)
        if "next_state" in last or "Available states" in last:
            # Walk through states in order
            state_order = [
                "LOAD_DATA","VALIDATE_DATA","CALCULATE","ESCALATION","GENERATE_REPORT","END",
                "LOAD_DOC","EXTRACT_CLAUSES","CLASSIFY",
            ]
            for s in state_order:
                if s not in combined and s in _ALL_TOOL_MAPS:
                    return ChatResult(
                        generations=[ChatGeneration(message=AIMessage(
                            content=f'{{"next_state":"{s}"}}'
                        ))]
                    )
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content='{"next_state":"END"}'))]
            )

        # Tool call
        state = self._find_state(messages)
        if state and state in _ALL_TOOL_MAPS:
            tc = {
                "name": _ALL_TOOL_MAPS[state],
                "args": {},
                "id": f"call_{state}",
                "type": "tool_call",
            }
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[tc]))]
            )

        tc = {"name": "data_loader", "args": {}, "id": "call_fb", "type": "tool_call"}
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[tc]))]
        )

    def bind_tools(self, tools: list, **kwargs) -> "_FullStub":
        return self


STUB = _FullStub()


# ---------------------------------------------------------------------------
# Gate 1 — Fake-model control-flow: every layer independently togglable
# ---------------------------------------------------------------------------

class TestControlFlow:
    """Gate 1: every HarnessConfig layer independently togglable; the harness
    runs with any combination and enforces what the config claims."""

    from src.adf.metric import HarnessConfig as _HC

    @pytest.mark.parametrize("plan_mode", ["schema_validated", "free_text"])
    @pytest.mark.parametrize("state_mode", ["fsm_fixed", "model_chosen"])
    @pytest.mark.parametrize("routing_mode", ["fixed_map", "model_chosen"])
    @pytest.mark.parametrize("tool_mode", ["forced_single", "subset"])
    @pytest.mark.parametrize("arg_mode", ["fixed", "enum_strict", "typed", "free_form"])
    def test_all_combinations_run(
        self, plan_mode, state_mode, routing_mode, tool_mode, arg_mode
    ):
        """Every combination of the first 5 layers must complete without crash."""
        from src.harness.configurable import get_task_definition, run_configurable
        from src.adf.metric import HarnessConfig

        cfg = HarnessConfig(
            plan_mode=plan_mode,
            state_mode=state_mode,
            routing_mode=routing_mode,
            tool_mode=tool_mode,
            arg_mode=arg_mode,
            action_substrate="tool_call",
        )
        # Use finance_ecl as the reference task (simplest, fewest tools)
        result = run_configurable(
            "finance_ecl", cfg, "stub", 0,
            f"gate1_{plan_mode}_{state_mode}_{routing_mode}_{tool_mode}_{arg_mode}",
            llm=STUB,
        )
        # Run must not crash (error is allowed — it's data)
        assert result is not None
        assert hasattr(result, "run_id")

    def test_forced_tool_really_forces(self):
        """tool_mode='forced_single' must bind exactly one tool per step."""
        from src.harness.configurable import get_task_definition, _build_lc_tools
        from src.adf.metric import HarnessConfig

        td = get_task_definition("finance_ecl")
        session: dict = {}
        cfg = HarnessConfig(tool_mode="forced_single", arg_mode="fixed")
        for state in td.states:
            tools = _build_lc_tools(cfg, td, session, state)
            assert len(tools) == 1, (
                f"forced_single must bind exactly 1 tool for state {state}, "
                f"got {[t.name for t in tools]}"
            )
            assert tools[0].name == td.state_tool_map[state]

    def test_subset_tool_mode(self):
        """tool_mode='subset' must bind the per-state subset, not all tools."""
        from src.harness.configurable import get_task_definition, _build_lc_tools
        from src.adf.metric import HarnessConfig

        td = get_task_definition("finance_ecl")
        session: dict = {}
        cfg = HarnessConfig(tool_mode="subset", arg_mode="fixed")
        for state in td.states:
            tools = _build_lc_tools(cfg, td, session, state)
            expected = set(td.state_tool_subsets[state])
            actual = {t.name for t in tools}
            assert actual == expected, (
                f"subset must bind {expected} for state {state}, got {actual}"
            )

    def test_free_choice_tool_mode(self):
        """tool_mode='free_choice' must bind all M tools."""
        from src.harness.configurable import get_task_definition, _build_lc_tools
        from src.adf.metric import HarnessConfig

        td = get_task_definition("finance_ecl")
        session: dict = {}
        cfg = HarnessConfig(tool_mode="free_choice", arg_mode="fixed")
        for state in td.states:
            tools = _build_lc_tools(cfg, td, session, state)
            assert {t.name for t in tools} == set(td.all_tool_names), (
                f"free_choice must expose all tools for state {state}"
            )

    def test_schema_plan_rejects_bad_plan(self):
        """schema_validated plan gate must reject a plan with wrong states."""
        from src.harness.configurable import _plan_phase
        from src.adf.metric import HarnessConfig

        # Stub that always returns a wrong plan
        class BadPlanStub(BaseChatModel):
            @property
            def _llm_type(self): return "bad_plan_stub"
            def _generate(self, messages, **kwargs):
                c = '[{"state":"WRONG","intended_tool":"data_loader"}]'
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=c))])
            def bind_tools(self, tools, **kwargs): return self

        import sys; sys.path.insert(0, ".")
        from src.harness.configurable import get_task_definition
        td = get_task_definition("finance_ecl")
        cfg = HarnessConfig(plan_mode="schema_validated")
        totals: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        from src.tracing.logger import TraceLogger
        from src.tracing.util import hash_obj
        logger = TraceLogger("finance_ecl", "bad_stub", "test", 0,
                              hash_obj("test"))

        _, plan_valid, plan_retries = _plan_phase(
            cfg, td, BadPlanStub(), logger, totals
        )
        assert not plan_valid, "Schema plan must reject a plan with wrong states"

    def test_fsm_blocks_escalation_at_linear_rung(self):
        """For branching_ecl, fsm_fixed must NOT visit ESCALATION (linear_states).
        This is the core Gate 6 / H2 mechanism test."""
        from src.harness.configurable import get_task_definition, run_configurable
        from src.adf.metric import HarnessConfig
        from src.tasks.branching_ecl import get_instance_paths, load_instance_manifest

        paths = get_instance_paths()
        if not paths:
            pytest.skip("No branching instances generated — run gen_branching_instances.py")

        manifest = load_instance_manifest()
        branching_path = next(
            (p for p in paths if manifest["instances"][p.stem]["required_path"] == "branching"),
            None,
        )
        if branching_path is None:
            pytest.skip("No branching instance found in manifest")

        cfg = HarnessConfig(
            plan_mode="schema_validated", state_mode="fsm_fixed",
            routing_mode="fixed_map", tool_mode="forced_single",
            arg_mode="fixed", action_substrate="tool_call",
        )
        td = get_task_definition("branching_ecl", instance_path=str(branching_path))
        result = run_configurable(
            "branching_ecl", cfg, "stub", 0, "gate1_fsm_linear_branching", llm=STUB,
            task_def=td,
        )
        # With linear FSM, ESCALATION is never called → report lacks stress test
        # → hash comparison fails → task_success must be False
        assert not result.task_success, (
            "ADF≈0 config (linear FSM) must FAIL on a branching instance. "
            "If it passes, the branching task has no real branching — Gate 6 violation."
        )
        # Verify no step has state='ESCALATION'
        visited = [s.state for s in result.steps]
        assert "ESCALATION" not in visited, (
            f"Linear FSM must not visit ESCALATION, but visited: {visited}"
        )


# ---------------------------------------------------------------------------
# Gate 2 — ADF monotonicity
# ---------------------------------------------------------------------------

class TestADFMonotonicity:
    """Gate 2: tightening any single layer must never increase computed ADF."""

    def test_ladder_monotone_at_all_proxies(self):
        """The 8 ladder rungs (L0' through L6) must be monotone at all three proxies."""
        from src.adf.metric import HarnessConfig, FINANCE_ECL, compute_adf, SENSITIVITY_BAND

        rungs = [
            HarnessConfig(plan_mode="schema_validated", state_mode="fsm_fixed",
                          routing_mode="fixed_map", tool_mode="forced_single",
                          arg_mode="fixed", action_substrate="tool_call"),
            HarnessConfig(plan_mode="schema_validated", state_mode="fsm_fixed",
                          routing_mode="fixed_map", tool_mode="forced_single",
                          arg_mode="enum_strict", action_substrate="tool_call"),
            HarnessConfig(plan_mode="free_text", state_mode="fsm_fixed",
                          routing_mode="fixed_map", tool_mode="forced_single",
                          arg_mode="enum_strict", action_substrate="tool_call"),
            HarnessConfig(plan_mode="free_text", state_mode="fsm_fixed",
                          routing_mode="model_chosen", tool_mode="forced_single",
                          arg_mode="enum_strict", action_substrate="tool_call"),
            HarnessConfig(plan_mode="free_text", state_mode="fsm_fixed",
                          routing_mode="model_chosen", tool_mode="subset",
                          arg_mode="typed", action_substrate="tool_call"),
            HarnessConfig(plan_mode="free_text", state_mode="model_chosen",
                          routing_mode="model_chosen", tool_mode="subset",
                          arg_mode="typed", action_substrate="tool_call"),
            HarnessConfig(plan_mode="free_text", state_mode="model_chosen",
                          routing_mode="model_chosen", tool_mode="free_choice",
                          arg_mode="free_form", action_substrate="tool_call"),
            HarnessConfig(plan_mode="free_text", state_mode="model_chosen",
                          routing_mode="model_chosen", tool_mode="free_choice",
                          arg_mode="free_form", action_substrate="hybrid"),
        ]

        for proxy in SENSITIVITY_BAND:
            rates = [compute_adf(cfg, FINANCE_ECL, proxy).rate for cfg in rungs]
            for i in range(1, len(rates)):
                assert rates[i] >= rates[i - 1] - 1e-9, (
                    f"Ladder not monotone at proxy={proxy}: "
                    f"rung {i} rate={rates[i]:.4f} < rung {i-1} rate={rates[i-1]:.4f}"
                )

    @pytest.mark.parametrize("layer,tight,free", [
        ("plan_mode",       "schema_validated",  "free_text"),
        ("state_mode",      "fsm_fixed",         "model_chosen"),
        ("routing_mode",    "fixed_map",         "model_chosen"),
        ("tool_mode",       "forced_single",     "free_choice"),
        ("arg_mode",        "fixed",             "free_form"),
        ("action_substrate","tool_call",         "hybrid"),
    ])
    def test_single_layer_tightening_never_increases_adf(self, layer, tight, free):
        """Tightening one layer while holding all others constant must not increase ADF."""
        from src.adf.metric import HarnessConfig, FINANCE_ECL, compute_adf

        base_kwargs = {
            "plan_mode": "free_text",
            "state_mode": "model_chosen",
            "routing_mode": "model_chosen",
            "tool_mode": "free_choice",
            "arg_mode": "free_form",
            "action_substrate": "hybrid",
        }
        free_cfg = HarnessConfig(**{**base_kwargs, layer: free})
        tight_cfg = HarnessConfig(**{**base_kwargs, layer: tight})

        for proxy in (64, 1024, 2**14):
            free_rate = compute_adf(free_cfg, FINANCE_ECL, proxy).rate
            tight_rate = compute_adf(tight_cfg, FINANCE_ECL, proxy).rate
            assert tight_rate <= free_rate + 1e-9, (
                f"Tightening {layer} ({free} → {tight}) INCREASED ADF_rate "
                f"at proxy={proxy}: {free_rate:.4f} → {tight_rate:.4f}"
            )


# ---------------------------------------------------------------------------
# Gate 3 — ADF hand-check
# ---------------------------------------------------------------------------

class TestADFHandCheck:
    """Gate 3: three configs whose ADF is computed by hand in docs/adf_definition.md
    must match compute_adf() exactly."""

    def test_example_A_L0_fully_constrained(self):
        """Example A (L0): schema/fsm/fixed/forced/enum/tool_call on finance_ecl.
        Expected from docs/adf_definition.md §4.1: total=8.000, rate=8/21=0.381."""
        from src.adf.metric import HarnessConfig, FINANCE_ECL, compute_adf

        cfg = HarnessConfig(
            plan_mode="schema_validated", state_mode="fsm_fixed",
            routing_mode="fixed_map", tool_mode="forced_single",
            arg_mode="enum_strict", action_substrate="tool_call",
        )
        result = compute_adf(cfg, FINANCE_ECL, 1024)
        assert abs(result.total - 8.000) < 1e-6, f"total={result.total}"
        assert abs(result.rate - 8 / 21) < 1e-6, f"rate={result.rate}"

    def test_example_B_L3_intermediate(self):
        """Example B (L3): schema/fsm/model_chosen/subset/typed/tool_call.
        Expected: total=30.340, rate=30.340/21=1.445."""
        import math
        from src.adf.metric import HarnessConfig, FINANCE_ECL, compute_adf

        cfg = HarnessConfig(
            plan_mode="schema_validated", state_mode="fsm_fixed",
            routing_mode="model_chosen", tool_mode="subset",
            arg_mode="typed", action_substrate="tool_call",
        )
        result = compute_adf(cfg, FINANCE_ECL, 1024)
        # routing: 4*log2(3)=6.340, tool: 4*log2(2)=4, args: 4*log2(32)=20 → total=30.340
        assert abs(result.total - 30.340) < 0.001, f"total={result.total}"
        assert abs(result.rate - 30.340 / 21) < 0.001, f"rate={result.rate}"

    def test_example_C_L6_unconstrained(self):
        """Example C (L6): free/model_chosen/model_chosen/free_choice/free_form/codegen.
        Expected: total=74.915, rate=74.915/21=3.567 (codegen normalises arg to free_form).
        Note: substrate=codegen adds 0 bits (single forced choice), not hybrid's log2(2)."""
        from src.adf.metric import HarnessConfig, FINANCE_ECL, compute_adf

        cfg = HarnessConfig(
            plan_mode="free_text", state_mode="model_chosen",
            routing_mode="model_chosen", tool_mode="free_choice",
            arg_mode="free_form", action_substrate="codegen",
        )
        result = compute_adf(cfg, FINANCE_ECL, 1024)
        # plan(10) + state(4*log2(5)=9.288) + routing(4*log2(3)=6.340)
        # + tool(4*log2(5)=9.288) + args(4*10=40) + substrate(0) = 74.915
        assert abs(result.total - 74.915) < 0.01, f"total={result.total}"

    def test_L0prime_is_exactly_zero(self):
        """L0' (arg=fixed, all else forced) must have ADF_total = ADF_rate = 0.000."""
        from src.adf.metric import HarnessConfig, FINANCE_ECL, compute_adf, SENSITIVITY_BAND

        cfg = HarnessConfig(
            plan_mode="schema_validated", state_mode="fsm_fixed",
            routing_mode="fixed_map", tool_mode="forced_single",
            arg_mode="fixed", action_substrate="tool_call",
        )
        for proxy in SENSITIVITY_BAND:
            result = compute_adf(cfg, FINANCE_ECL, proxy)
            assert result.total == 0.0, f"L0' total != 0 at proxy={proxy}: {result.total}"
            assert result.rate == 0.0, f"L0' rate != 0 at proxy={proxy}: {result.rate}"

    def test_codegen_normalises_arg_mode(self):
        """HarnessConfig.normalized() must override arg_mode to free_form for codegen."""
        from src.adf.metric import HarnessConfig

        cfg = HarnessConfig(action_substrate="codegen", arg_mode="enum_strict")
        norm = cfg.normalized()
        assert norm.arg_mode == "free_form", (
            f"codegen must normalise arg_mode to free_form, got {norm.arg_mode}"
        )

    def test_H5_pair_matched_adf(self):
        """H5 substrate pair (L5-tool_call vs L5-codegen) must have identical ADF
        at all three proxy values."""
        from src.adf.metric import HarnessConfig, FINANCE_ECL, compute_adf, SENSITIVITY_BAND

        base = dict(
            plan_mode="free_text", state_mode="model_chosen",
            routing_mode="model_chosen", tool_mode="free_choice",
            arg_mode="free_form",
        )
        cfg_tool = HarnessConfig(**base, action_substrate="tool_call")
        cfg_code = HarnessConfig(**base, action_substrate="codegen")

        for proxy in SENSITIVITY_BAND:
            r_tool = compute_adf(cfg_tool, FINANCE_ECL, proxy)
            r_code = compute_adf(cfg_code, FINANCE_ECL, proxy)
            assert abs(r_tool.rate - r_code.rate) < 1e-9, (
                f"H5 pair ADF mismatch at proxy={proxy}: "
                f"tool={r_tool.rate:.4f}, codegen={r_code.rate:.4f}"
            )


# ---------------------------------------------------------------------------
# Gate 4 — Trace completeness
# ---------------------------------------------------------------------------

class TestTraceCompleteness:
    """Gate 4: every run emits all required per-layer events; a failed run still
    produces a complete trace with the failure recorded (never dropped)."""

    REQUIRED_EVENTS = {
        "run_start", "adf_config", "plan", "state_transition",
        "step_attempt", "tool_call", "run_end",
    }

    def _collect_events(self, run_id: str) -> set[str]:
        logs_dir = Path(__file__).resolve().parents[1] / "logs" / "raw"
        log_file = logs_dir / f"{run_id}.jsonl"
        if not log_file.exists():
            return set()
        events = set()
        for line in log_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.add(json.loads(line)["event"])
        return events

    def test_successful_run_has_all_events(self):
        """A successful run must emit all required event types."""
        from src.harness.configurable import run_configurable
        from src.adf.metric import HarnessConfig

        cfg = HarnessConfig(
            plan_mode="schema_validated", state_mode="fsm_fixed",
            routing_mode="fixed_map", tool_mode="forced_single",
            arg_mode="enum_strict", action_substrate="tool_call",
        )
        result = run_configurable(
            "finance_ecl", cfg, "stub", 0, "gate4_success", llm=STUB
        )
        events = self._collect_events(result.run_id)
        missing = self.REQUIRED_EVENTS - events
        assert not missing, f"Successful run missing events: {missing}"

    def test_failed_run_has_all_events(self):
        """A run that fails at a step must still emit a complete trace,
        including the failure recorded in step_attempt and run_end."""
        from src.harness.configurable import run_configurable
        from src.adf.metric import HarnessConfig

        # Stub that always returns a wrong tool name → step fails
        class WrongToolStub(BaseChatModel):
            @property
            def _llm_type(self): return "wrong_tool_stub"
            def _generate(self, messages, **kwargs):
                if "intended_tool" in " ".join(
                    m.content for m in messages if hasattr(m,"content") and m.content
                ):
                    c = '[{"state":"LOAD_DATA","intended_tool":"data_loader"},{"state":"VALIDATE_DATA","intended_tool":"validation_tool"},{"state":"CALCULATE","intended_tool":"calculator_tool"},{"state":"GENERATE_REPORT","intended_tool":"report_generator"}]'
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=c))])
                # Always call wrong tool
                tc = {"name":"nonexistent_tool","args":{},"id":"call_bad","type":"tool_call"}
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="",tool_calls=[tc]))])
            def bind_tools(self, tools, **kwargs): return self

        cfg = HarnessConfig(plan_mode="schema_validated", state_mode="fsm_fixed",
                            routing_mode="fixed_map", tool_mode="forced_single",
                            arg_mode="fixed")
        result = run_configurable(
            "finance_ecl", cfg, "wrong_stub", 0, "gate4_failure", llm=WrongToolStub()
        )
        assert result.error is not None, "Run should have failed"
        events = self._collect_events(result.run_id)
        # Must still have run_start, adf_config, run_end, and step_attempt
        for required in ("run_start", "adf_config", "step_attempt", "run_end"):
            assert required in events, (
                f"Failed run missing '{required}' event — failure was dropped"
            )

    def test_adf_config_event_has_correct_fields(self):
        """The adf_config event must contain harness_config, adf_total, adf_rate."""
        from src.harness.configurable import run_configurable
        from src.adf.metric import HarnessConfig

        cfg = HarnessConfig(arg_mode="enum_strict")
        result = run_configurable(
            "finance_ecl", cfg, "stub", 0, "gate4_adf_fields", llm=STUB
        )
        logs_dir = Path(__file__).resolve().parents[1] / "logs" / "raw"
        log_file = logs_dir / f"{result.run_id}.jsonl"
        adf_event = None
        for line in log_file.read_text().splitlines():
            if line.strip():
                ev = json.loads(line)
                if ev["event"] == "adf_config":
                    adf_event = ev
                    break
        assert adf_event is not None
        for field in ("harness_config", "adf_total", "adf_rate", "per_layer", "proxy"):
            assert field in adf_event, f"adf_config missing field: {field}"
        assert abs(adf_event["adf_total"] - 8.0) < 0.01  # enum_strict → 8 bits


# ---------------------------------------------------------------------------
# Gate 5 — Ground-truth test
# ---------------------------------------------------------------------------

class TestGroundTruth:
    """Gate 5: deterministic scorers agree with hand-computed answers for
    ≥5 instances per family."""

    def test_finance_ecl_ground_truth_scorer(self):
        """finance_ecl reference_run must match the scorer's expected output."""
        from src.tasks.finance_ecl import reference_run, GROUND_TRUTH
        from src.tracing.util import hash_obj

        computed = reference_run()
        assert hash_obj(computed) == hash_obj(GROUND_TRUTH), (
            "reference_run output doesn't match GROUND_TRUTH — scorer/reference drift"
        )
        assert computed["valid_count"] > 0
        assert computed["total_ecl"] > 0.0
        assert len(computed["excluded"]) > 0  # portfolio has invalid rows

    def test_legal_clause_ground_truth_scorer(self):
        """legal_clause reference_run must match GROUND_TRUTH."""
        from src.tasks.legal_clause import reference_run, GROUND_TRUTH
        from src.tracing.util import hash_obj

        computed = reference_run()
        assert hash_obj(computed) == hash_obj(GROUND_TRUTH)
        assert computed["clause_count"] > 0

    def test_branching_ecl_linear_instances(self):
        """For branching-ECL linear instances, reference_run must produce
        requires_escalation=False and no 'escalation' field."""
        from src.tasks.branching_ecl import (
            get_instance_paths, load_instance_manifest,
            load_ground_truth, reference_run,
        )
        from src.tracing.util import hash_obj

        paths = get_instance_paths()
        if not paths:
            pytest.skip("No branching instances — run gen_branching_instances.py")

        manifest = load_instance_manifest()
        linear_paths = [
            p for p in paths
            if manifest["instances"][p.stem]["required_path"] == "linear"
        ][:5]
        assert len(linear_paths) >= 5, (
            f"Need ≥5 linear instances, found {len(linear_paths)}"
        )
        for p in linear_paths:
            gt = load_ground_truth(p)
            computed = reference_run(p)
            assert hash_obj(computed) == hash_obj(gt), f"GT mismatch for {p.name}"
            assert not computed["requires_escalation"]
            assert "escalation" not in computed

    def test_branching_ecl_branching_instances(self):
        """For branching-ECL branching instances, reference_run must produce
        requires_escalation=True and a correct 'escalation' field."""
        from src.tasks.branching_ecl import (
            get_instance_paths, load_instance_manifest,
            load_ground_truth, reference_run, ESCALATION_THRESHOLD,
        )
        from src.tracing.util import hash_obj

        paths = get_instance_paths()
        if not paths:
            pytest.skip("No branching instances — run gen_branching_instances.py")

        manifest = load_instance_manifest()
        branching_paths = [
            p for p in paths
            if manifest["instances"][p.stem]["required_path"] == "branching"
        ][:5]
        assert len(branching_paths) >= 5, (
            f"Need ≥5 branching instances, found {len(branching_paths)}"
        )
        for p in branching_paths:
            gt = load_ground_truth(p)
            computed = reference_run(p)
            assert hash_obj(computed) == hash_obj(gt), f"GT mismatch for {p.name}"
            assert computed["requires_escalation"]
            assert "escalation" in computed
            esc = computed["escalation"]
            assert esc["high_risk_count"] > 0
            # All stress results must be for loans with PD > threshold
            for sr in esc["stress_results"]:
                assert sr["pd_original"] > ESCALATION_THRESHOLD


# ---------------------------------------------------------------------------
# Gate 6 — Branching-task sanity
# ---------------------------------------------------------------------------

class TestBranchingSanity:
    """Gate 6: ADF≈0 config MUST fail branching instances; if it passes, the
    task has no real branching (loop_eng.md §4.0 / §5 gate 6)."""

    def test_L0prime_fails_on_all_branching_instances(self):
        """The fully-constrained linear FSM config must FAIL on every branching
        instance in the generated dataset (not just one)."""
        from src.harness.configurable import get_task_definition, run_configurable
        from src.tasks.branching_ecl import get_instance_paths, load_instance_manifest
        from src.adf.metric import HarnessConfig

        paths = get_instance_paths()
        if not paths:
            pytest.skip("No branching instances generated")

        manifest = load_instance_manifest()
        branching_paths = [
            p for p in paths
            if manifest["instances"][p.stem]["required_path"] == "branching"
        ]
        assert branching_paths, "No branching instances in manifest"

        cfg = HarnessConfig(
            plan_mode="schema_validated", state_mode="fsm_fixed",
            routing_mode="fixed_map", tool_mode="forced_single",
            arg_mode="fixed", action_substrate="tool_call",
        )
        fails = 0
        for p in branching_paths[:10]:  # test first 10 branching instances
            td = get_task_definition("branching_ecl", instance_path=str(p))
            result = run_configurable(
                "branching_ecl", cfg, "stub", 0, f"gate6_br_{p.stem}", llm=STUB,
                task_def=td,
            )
            if not result.task_success:
                fails += 1
            # ESCALATION must not appear in visited states
            visited = [s.state for s in result.steps]
            assert "ESCALATION" not in visited, (
                f"Linear FSM visited ESCALATION for {p.name}: {visited}"
            )

        fail_rate = fails / len(branching_paths[:10])
        assert fail_rate == 1.0, (
            f"Gate 6 FAIL: L0prime succeeded on {10 - fails}/10 branching instances. "
            f"Success rate = {1-fail_rate:.1%}. "
            "The branching task is not actually branching — fix the task."
        )

    def test_L0prime_succeeds_on_linear_instances(self):
        """The fully-constrained config must SUCCEED on linear instances
        (it has the right path for them)."""
        from src.harness.configurable import get_task_definition, run_configurable
        from src.tasks.branching_ecl import get_instance_paths, load_instance_manifest
        from src.adf.metric import HarnessConfig

        paths = get_instance_paths()
        if not paths:
            pytest.skip("No branching instances generated")

        manifest = load_instance_manifest()
        linear_paths = [
            p for p in paths
            if manifest["instances"][p.stem]["required_path"] == "linear"
        ][:5]

        cfg = HarnessConfig(
            plan_mode="schema_validated", state_mode="fsm_fixed",
            routing_mode="fixed_map", tool_mode="forced_single",
            arg_mode="fixed", action_substrate="tool_call",
        )
        passes = 0
        for p in linear_paths:
            td = get_task_definition("branching_ecl", instance_path=str(p))
            result = run_configurable(
                "branching_ecl", cfg, "stub", 0, f"gate6_lin_{p.stem}", llm=STUB,
                task_def=td,
            )
            if result.task_success:
                passes += 1

        pass_rate = passes / len(linear_paths)
        assert pass_rate == 1.0, (
            f"L0prime only passed {passes}/{len(linear_paths)} linear instances "
            f"({pass_rate:.0%}). Linear-instance scoring appears broken."
        )


# ---------------------------------------------------------------------------
# Gate 7 — Difficulty calibration proxy
# ---------------------------------------------------------------------------

class TestDifficultyCalibration:
    """Gate 7 proxy (stub-model level): verify the task structure allows both
    failure modes to occur, even if we can't run the real model without API spend.

    With the stub: at ADF≈0 (linear FSM), branching instances must fail; linear
    instances must succeed.  At ADF≈max (model_chosen), the stub correctly follows
    the full path, so it succeeds — but this is a stub limitation, not a real model.
    The real calibration test (§4.4) runs after the Phase 4 gate, with the actual
    model in pilot mode.

    What we CAN test structurally:
    - Instances exist that require different paths (15 linear, 15 branching).
    - The failure set at ADF≈0 is exclusively branching instances (not linear).
    - Ground truth correctly reflects both paths.
    """

    def test_failure_sets_are_path_specific(self):
        """At L0prime, only branching instances fail; linear instances pass.
        This demonstrates the failure sets are disjoint (loop_eng.md §4.4)."""
        from src.harness.configurable import get_task_definition, run_configurable
        from src.tasks.branching_ecl import get_instance_paths, load_instance_manifest
        from src.adf.metric import HarnessConfig

        paths = get_instance_paths()
        if not paths:
            pytest.skip("No branching instances generated")

        manifest = load_instance_manifest()
        cfg = HarnessConfig(
            plan_mode="schema_validated", state_mode="fsm_fixed",
            routing_mode="fixed_map", tool_mode="forced_single",
            arg_mode="fixed",
        )

        linear_successes, branching_failures = 0, 0
        n_linear = n_branching = 0

        for p in paths[:20]:  # first 20 instances
            kind = manifest["instances"][p.stem]["required_path"]
            td = get_task_definition("branching_ecl", instance_path=str(p))
            result = run_configurable(
                "branching_ecl", cfg, "stub", 0, f"gate7_{p.stem}", llm=STUB,
                task_def=td,
            )
            if kind == "linear":
                n_linear += 1
                if result.task_success:
                    linear_successes += 1
            else:
                n_branching += 1
                if not result.task_success:
                    branching_failures += 1

        assert n_linear > 0 and n_branching > 0, "Need both types in first 20 instances"
        assert linear_successes == n_linear, (
            f"Not all linear instances succeeded at L0prime: "
            f"{linear_successes}/{n_linear}"
        )
        assert branching_failures == n_branching, (
            f"Not all branching instances failed at L0prime: "
            f"{branching_failures}/{n_branching}"
        )
        # Failure sets are disjoint: linear always passes, branching always fails
