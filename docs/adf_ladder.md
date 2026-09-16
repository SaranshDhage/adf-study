# ADF Ladder — Rung Definitions

Status: **FROZEN before grid runs** (required by `PREREGISTRATION.md` §5).
Implements: `docs/adf_definition.md` §4 and `PREREGISTRATION.md` §5 ("≥ 6 rungs
spanning L0′ to L6, fixed in `docs/adf_ladder.md` before any grid run").

All ADF values are **machine-computed** by `src/adf/metric.py :: compute_adf()` against
`FINANCE_ECL` (S=4, N=5, K=3, M=4, A_enum=4, m_phase=2, |T|=21).  Primary proxy
`UNBOUNDED_PROXY = 2^10`; sensitivity band at `2^6` and `2^14`.

---

## 1. Rung table

Eight rungs (L0′ through L6) span 0.000 to 3.758 bits/dp at the primary proxy.
Each rung loosens at most two layers from the previous rung to keep the confound
surface small.

| Rung | plan | state | routing | tool | arg | substrate | ADF_total | ADF_rate |
|------|------|-------|---------|------|-----|-----------|----------:|---------:|
| **L0′** | schema | fsm | fixed | forced | **fixed** | tool_call | 0.000 | **0.000** |
| **L0** | schema | fsm | fixed | forced | **enum** | tool_call | 8.000 | **0.381** |
| **L1** | **free** | fsm | fixed | forced | enum | tool_call | 18.000 | **0.857** |
| **L2** | free | fsm | **model** | forced | enum | tool_call | 24.340 | **1.159** |
| **L3** | free | fsm | model | **subset** | **typed** | tool_call | 40.340 | **1.921** |
| **L4** | free | **model** | model | subset | typed | tool_call | 49.628 | **2.363** |
| **L5** | free | model | model | **free** | **free_form** | tool_call | 74.915 | **3.567** |
| **L6** | free | model | model | free | free_form | **hybrid** | 78.915 | **3.758** |

Bold cells = the layer(s) loosened from the previous rung.

---

## 2. Sensitivity band (ADF_rate at all three proxy values)

| Rung | rate @2^6 | **rate @2^10** *(primary)* | rate @2^14 |
|------|----------:|---------------------------:|-----------:|
| L0′ | 0.000 | **0.000** | 0.000 |
| L0 | 0.381 | **0.381** | 0.381 |
| L1 | 0.667 | **0.857** | 1.048 |
| L2 | 0.969 | **1.159** | 1.350 |
| L3 | 1.350 | **1.921** | 2.492 |
| L4 | 1.792 | **2.363** | 2.935 |
| L5 | 2.615 | **3.567** | 4.520 |
| L6 | 2.805 | **3.758** | 4.710 |

**Monotonicity holds at all three proxy values** (verified by `tests/test_adf.py`).
Rungs that are proxy-independent (L0′, L0): contain no `free_text`, `typed`, or
`free_form` layers, so their ADF is a fixed integer regardless of `UNBOUNDED_PROXY`.

---

## 3. Per-layer ADF contributions (primary proxy, finance_ecl)

| Layer | L0′ | L0 | L1 | L2 | L3 | L4 | L5 | L6 |
|-------|----:|---:|---:|---:|---:|---:|---:|---:|
| plan | 0 | 0 | **10** | 10 | 10 | 10 | 10 | 10 |
| state | 0 | 0 | 0 | 0 | 0 | **9.288** | 9.288 | 9.288 |
| routing | 0 | 0 | 0 | **6.340** | 6.340 | 6.340 | 6.340 | 6.340 |
| tool | 0 | 0 | 0 | 0 | **4.000** | 4.000 | **9.288** | 9.288 |
| arguments | 0 | **8** | 8 | 8 | **20** | 20 | **40** | 40 |
| substrate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **4.000** |
| skill | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| retry | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **total** | **0** | **8** | **18** | **24.340** | **40.340** | **49.628** | **74.915** | **78.915** |
| **rate** | **0.000** | **0.381** | **0.857** | **1.159** | **1.921** | **2.363** | **3.567** | **3.758** |

---

## 4. Full HarnessConfig for each rung

These are the exact configs passed to `run_configurable()`.  `skill_level`
is held at `"precise"` in the main grid (see §6 for why it is excluded from
the ladder).  `retry_policy` is `"bounded"` throughout (0 bits; contributes
to `realized_T` only).

### L0′ — maximally constrained (true ADF = 0)

```python
HarnessConfig(
    plan_mode      = "schema_validated",
    state_mode     = "fsm_fixed",
    routing_mode   = "fixed_map",
    tool_mode      = "forced_single",
    arg_mode       = "fixed",          # <-- no argument choice
    action_substrate = "tool_call",
    skill_level    = "precise",
    retry_policy   = "bounded",
)
```

ADF_rate = 0.000 bits/dp at all proxy values.  This is the true origin of the
ADF axis.  The FSM forces the state sequence, routing is predetermined, the
tool is forced, and the tool has no arguments.  The model has zero degrees of
freedom in the harness; determinism is determined entirely by the model's own
sampling temperature.

### L0 — enum arguments (lowest non-zero rung)

```python
HarnessConfig(
    plan_mode      = "schema_validated",
    state_mode     = "fsm_fixed",
    routing_mode   = "fixed_map",
    tool_mode      = "forced_single",
    arg_mode       = "enum_strict",    # 4-way enum; model picks one
    action_substrate = "tool_call",
    skill_level    = "precise",
    retry_policy   = "bounded",
)
```

ADF_rate = 0.381 bits/dp.  The only freedom is choosing one of 4 enum values
per tool call.  This rung tests whether enum-argument choice alone introduces
measurable non-determinism.

### L1 — free plan added

```python
HarnessConfig(
    plan_mode      = "free_text",      # <-- free text plan
    state_mode     = "fsm_fixed",
    routing_mode   = "fixed_map",
    tool_mode      = "forced_single",
    arg_mode       = "enum_strict",
    action_substrate = "tool_call",
    skill_level    = "precise",
    retry_policy   = "bounded",
)
```

ADF_rate = 0.857 bits/dp.  The plan is now free text (logged but does not
constrain execution — the FSM still dictates state order).  This isolates the
plan-layer contribution found in the prior paper.

### L2 — model-chosen routing added

```python
HarnessConfig(
    plan_mode      = "free_text",
    state_mode     = "fsm_fixed",
    routing_mode   = "model_chosen",   # <-- manager picks agent per step
    tool_mode      = "forced_single",
    arg_mode       = "enum_strict",
    action_substrate = "tool_call",
    skill_level    = "precise",
    retry_policy   = "bounded",
)
```

ADF_rate = 1.159 bits/dp.  Routing choice is now model-driven (K=3 options
per step), adding 6.340 bits total.  The FSM still determines the state
sequence; only the agent assignment per step is free.

### L3 — tool subset + typed arguments

```python
HarnessConfig(
    plan_mode      = "free_text",
    state_mode     = "fsm_fixed",
    routing_mode   = "model_chosen",
    tool_mode      = "subset",         # <-- 2-tool window per step
    arg_mode       = "typed",          # <-- typed string (no enum)
    action_substrate = "tool_call",
    skill_level    = "precise",
    retry_policy   = "bounded",
)
```

ADF_rate = 1.921 bits/dp.  Tool selection is now from a per-phase 2-tool
subset (m_phase=2); arguments are typed strings.  State order is still
FSM-fixed.

### L4 — model-chosen state sequence

```python
HarnessConfig(
    plan_mode      = "free_text",
    state_mode     = "model_chosen",   # <-- model picks next state
    routing_mode   = "model_chosen",
    tool_mode      = "subset",
    arg_mode       = "typed",
    action_substrate = "tool_call",
    skill_level    = "precise",
    retry_policy   = "bounded",
)
```

ADF_rate = 2.363 bits/dp.  The model now decides what state to execute next
(from N=5 options, including END).  This is the first rung where the model can
choose a state order different from the FSM's, which is the mechanism that
makes branching tasks solvable or not.

### L5 — fully free tool selection and arguments

```python
HarnessConfig(
    plan_mode      = "free_text",
    state_mode     = "model_chosen",
    routing_mode   = "model_chosen",
    tool_mode      = "free_choice",    # <-- any of M+1 tools
    arg_mode       = "free_form",      # <-- unconstrained free text
    action_substrate = "tool_call",
    skill_level    = "precise",
    retry_policy   = "bounded",
)
```

ADF_rate = 3.567 bits/dp.  All constraint layers are relaxed except substrate
(still `tool_call`).  This is the baseline for the H5 substrate-equivalence
test.

### L6 — hybrid substrate (maximum ADF)

```python
HarnessConfig(
    plan_mode      = "free_text",
    state_mode     = "model_chosen",
    routing_mode   = "model_chosen",
    tool_mode      = "free_choice",
    arg_mode       = "free_form",
    action_substrate = "hybrid",       # <-- model chooses tool_call or codegen
    skill_level    = "precise",
    retry_policy   = "bounded",
)
```

ADF_rate = 3.758 bits/dp.  Adding `hybrid` substrate gives the model the
additional choice of writing Python code instead of calling a tool — one
additional bit of freedom per step.  This is the highest-ADF rung in the
confirmed grid.

---

## 5. H5 substrate-equivalence test pair

H5 tests whether two configs with **equal ADF** but different *mechanisms*
yield indistinguishable determinism.

| Config | plan | state | routing | tool | arg | substrate | ADF_rate |
|--------|------|-------|---------|------|-----|-----------|----------|
| **L5-tool** | free | model | model | free | free_form | **tool_call** | 3.567 |
| **L5-codegen** | free | model | model | free | free_form | **codegen** | 3.567 |

`codegen` normalises `arg_mode` to `free_form` (spec §2.1), so both configs
have identical computed ADF at all three proxy values.  The mechanism differs:
L5-tool calls LangChain tools; L5-codegen executes model-generated Python.
If H5 holds, their Determinism Index distributions should overlap; if H5
fails, mechanism matters beyond ADF.

```python
L5_CODEGEN = HarnessConfig(
    plan_mode      = "free_text",
    state_mode     = "model_chosen",
    routing_mode   = "model_chosen",
    tool_mode      = "free_choice",
    arg_mode       = "free_form",       # normalised from any value by codegen
    action_substrate = "codegen",
    skill_level    = "precise",
    retry_policy   = "bounded",
)
```

---

## 6. What is NOT in the ladder (and why)

**`skill_level`** is held at `"precise"` across the main grid.  It contributes
0 bits to ADF (spec §2.3) and would confound the constraint-determinism
relationship if varied together with ADF level.  It is run as a separate
2-factor orthogonal sweep at L0, L3, L5 after the main grid completes.

**`retry_policy`** is held at `"bounded"` throughout.  It contributes 0 bits to
ADF_rate; it moves `ADF_total` through `realized_T` and is logged and analysed
as a horizon effect.

**`action_substrate = "codegen"` alone** does not add a dedicated rung because
it is equivalent in ADF to L5 (tool_call).  It appears only as the H5 test
counterpart to L5.

**`arg_mode = "typed"` at maximum freedom** (L5 moves directly to `free_form`):
adding a separate rung between L4 (typed) and L5 (free_form) would space the
ladder unevenly at the high end.  The L3→L4→L5 sequence already exercises the
typed→free_form transition within-family.

---

## 7. Ladder construction rules (for F2 and future families)

When a new task family is added, the *same eight HarnessConfig objects* are
reused.  `compute_adf()` is called with the family's own `TaskSpec` to produce
family-specific ADF_total and ADF_rate values.  The ladder rungs are defined
by HarnessConfig, not by absolute ADF values — families that differ in horizon
or tool count will have different total/rate values at each rung, which is
expected and correct (that's why ADF_rate is the cross-family metric and
ADF_total is within-family only).

The monotonicity gate in the Phase 4 test suite (`tests/test_adf.py`) must
pass for every registered family with the 8 rungs above.
