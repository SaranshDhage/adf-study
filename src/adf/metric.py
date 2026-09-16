"""Agent Degrees of Freedom (ADF) metric.

Spec: docs/adf_definition.md. ADF is a property of (harness_config, task_spec)
only -- never of a realized trace (see spec section 1.1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

UNBOUNDED_PROXY_DEFAULT = 2 ** 10
SENSITIVITY_BAND = (2 ** 6, 2 ** 10, 2 ** 14)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    nominal_steps: int      # S
    n_states: int           # N, including END
    n_agents: int           # K
    n_tools: int            # M
    enum_arity: int         # A_enum
    subset_size: int        # m_phase


@dataclass(frozen=True)
class HarnessConfig:
    plan_mode: str = "schema_validated"      # free_text | schema_validated
    state_mode: str = "fsm_fixed"            # model_chosen | fsm_fixed
    routing_mode: str = "fixed_map"          # model_chosen | fixed_map
    tool_mode: str = "forced_single"         # free_choice | subset | forced_single
    arg_mode: str = "enum_strict"            # free_form | typed | enum_strict | fixed
    action_substrate: str = "tool_call"      # codegen | tool_call | hybrid
    skill_level: str = "precise"             # none | vague | precise  (0 bits)
    retry_policy: str = "bounded"            # none | bounded | bounded_with_feedback (0 bits)

    def normalized(self) -> "HarnessConfig":
        """Apply spec section 2.1: codegen overrides arg_mode to free_form."""
        if self.action_substrate == "codegen" and self.arg_mode != "free_form":
            return replace(self, arg_mode="free_form")
        return self


@dataclass(frozen=True)
class ADFResult:
    total: float
    rate: float
    n_decision_points: int
    per_layer: dict
    proxy: int


def compute_adf(cfg: HarnessConfig, task: TaskSpec,
                unbounded_proxy: int = UNBOUNDED_PROXY_DEFAULT) -> ADFResult:
    cfg = cfg.normalized()
    S, P = task.nominal_steps, unbounded_proxy
    typed = math.isqrt(P) if math.isqrt(P) ** 2 == P else math.sqrt(P)

    plan = {"schema_validated": 1, "free_text": P}[cfg.plan_mode]
    state = {"fsm_fixed": 1, "model_chosen": task.n_states}[cfg.state_mode]
    route = {"fixed_map": 1, "model_chosen": task.n_agents}[cfg.routing_mode]
    tool = {"forced_single": 1, "subset": task.subset_size,
            "free_choice": task.n_tools + 1}[cfg.tool_mode]
    arg = {"fixed": 1, "enum_strict": task.enum_arity,
           "typed": typed, "free_form": P}[cfg.arg_mode]
    substrate = {"tool_call": 1, "codegen": 1, "hybrid": 2}[cfg.action_substrate]

    per_layer = {
        "plan":      math.log2(plan),
        "state":     S * math.log2(state),
        "routing":   S * math.log2(route),
        "tool":      S * math.log2(tool),
        "arguments": S * math.log2(arg),
        "substrate": S * math.log2(substrate),
        # skill_level and retry_policy contribute 0 bits by definition (spec 2.3)
        "skill":     0.0,
        "retry":     0.0,
    }
    total = sum(per_layer.values())
    n_dp = 1 + 5 * S
    return ADFResult(total=total, rate=total / n_dp, n_decision_points=n_dp,
                     per_layer=per_layer, proxy=P)


FINANCE_ECL = TaskSpec(name="finance_ecl", nominal_steps=4, n_states=5,
                       n_agents=3, n_tools=4, enum_arity=4, subset_size=2)
