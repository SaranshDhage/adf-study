# Agent Degrees of Freedom (ADF): Formal Definition

Status: **DRAFT — pending sign-off. Not yet preregistered.**
Implements: `src/adf/metric.py :: compute_adf(harness_config, task_spec) -> ADFResult`

---

## 1. Core formula

ADF measures how much *legal choice* a harness leaves the model. For a run with
decision points `T`:

```
ADF_total = Σ_{t ∈ T} log2( |legal_choices(t)| )        # bits per run
ADF_rate  = ADF_total / |T|                             # bits per decision point
```

`ADF_total` is the plan's original formula, unchanged. `ADF_rate` is an addition;
§3 explains why it is required and which of the two each hypothesis is tested on.

### 1.1 `T` is taken from the task spec, never from the trace

`T` is enumerated from the task's **nominal decision schedule** — a committed
per-family constant — not from what a run actually did.

This is not a stylistic choice. If `T` came from the realized trace, ADF would be
an *outcome*: a model that drifts emits more turns, which inflates its measured
ADF, and "high ADF predicts low determinism" becomes partly circular. Freezing
`T` a priori makes ADF a pure property of `(harness_config, task_spec)` — a
treatment that can be set before a run, which is what an independent variable has
to be.

Realized decision-point count is logged separately as `realized_T`, an outcome
measure. Any run where `realized_T` diverges from nominal by >2× is flagged in
diagnostics (it means the nominal schedule is mis-specified for that family).

---

## 2. Decision points and their cardinalities

For a task with nominal horizon `S` steps, state space `N` states (including
`END`), `K` specialist agents, and `M` exposed tools:

| # | Decision point | Occurrences | Mode | `\|legal_choices\|` |
|---|---|---|---|---|
| 1 | **Plan structure** | 1 | `schema_validated` | 1 |
| | | | `free_text` | `UNBOUNDED_PROXY` |
| 2 | **State transition** | S | `fsm_fixed` | 1 |
| | | | `model_chosen` | `N` |
| 3 | **Agent routing** | S | `fixed_map` | 1 |
| | | | `model_chosen` | `K` |
| 4 | **Tool selection** | S | `forced_single` | 1 |
| | | | `subset` | `m_phase` (per-phase legal subset) |
| | | | `free_choice` | `M + 1` (+1 = answer without a tool) |
| 5 | **Tool arguments** | S | `enum_strict` | `A_enum` (task constant, counted) |
| | | | `typed` | `sqrt(UNBOUNDED_PROXY)` |
| | | | `free_form` | `UNBOUNDED_PROXY` |
| 6 | **Action substrate** | S | `tool_call` / `codegen` | 1 (forced) |
| | | | `hybrid` | 2 |

So `|T| = 1 + 5S` — one plan decision plus five per-step decisions.

### 2.1 Interaction rule: codegen overrides arg_mode

Under `action_substrate="codegen"` the emitted code *is* the argument, so
`arg_mode` is overridden to `free_form`. Config combinations asserting otherwise
(`codegen` + `enum_strict`) are normalized with a logged warning, and the
validation suite asserts the override fires. Without this rule the same
mechanism would be counted at two different cardinalities.

### 2.2 `UNBOUNDED_PROXY` and the sensitivity band

Free text and arbitrary code have no countable cardinality. A single documented
constant stands in, applied identically everywhere it appears. Primary value
`2^10`; every result is recomputed at `2^6` and `2^14` and reported as a band.

`typed` is defined as `sqrt(UNBOUNDED_PROXY)` so it moves coherently with the
band instead of sitting at a fixed value while its neighbours shift:

| `UNBOUNDED_PROXY` | free_text / free_form | typed |
|---|---|---|
| 2^6 | 6 bits | 3 bits |
| 2^10 *(primary)* | 10 bits | 5 bits |
| 2^14 | 14 bits | 7 bits |

### 2.3 What is deliberately NOT in ADF

**`skill_level` contributes zero bits.** A skill file changes the model's *prior*
over choices; it does not remove any choice from the legal set. Counting it would
make ADF a measure of two different things at once.

This has a direct consequence for the experiment design: the ADF ladder in the
plan's §6.2 currently moves `skill_level` together with the constraint layers
(`precise` at ADF≈0, `none` at ADF≈max). Since skills are not part of ADF, that
ladder confounds *less freedom* with *more guidance* — a determinism drop could
be caused by either. **Recommendation: hold `skill_level="precise"` fixed across
the main grid and run skill level as a separate orthogonal factor at 2–3 ADF
levels.** That costs one extra small sweep and turns a confound into a second
publishable result ("can guidance substitute for constraint?" — the open question
the plan's §4.1 already names).

**`retry_policy` contributes zero bits to `ADF_rate`.** A retry does not widen the
legal set at any decision point; it adds decision points. Retries therefore move
`ADF_total` through `realized_T`, and are logged and analysed as horizon effects.

---

## 3. Why `ADF_rate` is needed as well as `ADF_total`

`ADF_total` sums over decision points, so it scales with horizon. A 30-step task
has a larger `ADF_total` than a 4-step task *at identical per-decision
constraint*, purely because it is longer.

That breaks two preregistered hypotheses as stated:

- **H3** compares the success-maximising ADF across task families. Families differ
  in horizon (F1 linear ≈ 4 steps; F3 regulatory ≈ 15–30). On `ADF_total`, H3
  would be confirmed by horizon alone, with no freedom effect present at all.
- **H4** compares that optimum across models on the same families — safe on either
  measure, since horizon is held constant within a family.

Resolution, to be frozen at preregistration:

| Hypothesis | Tested on | Rationale |
|---|---|---|
| H1 (determinism ↓ ADF) | both, reported separately | within-family; both valid, `ADF_rate` primary |
| H2 (non-monotonic success) | `ADF_rate` | within-family; optimum must be horizon-free |
| **H3 (task-dependence)** | **`ADF_rate` only** | cross-family — `ADF_total` is confounded by horizon |
| H4 (model-dependence) | `ADF_rate` | within-family, cross-model |
| H5 (substrate equivalence) | matched on **both** | pairs must match on rate *and* total to be a fair test |

`ADF_total` remains reported throughout; it is the honest measure of total freedom
in a run and is the right x-axis for within-family cost curves (T2).

---

## 4. Worked examples

All three use **F1 `finance_ecl`**: `S=4`, `N=5` (4 pipeline states + END),
`K=3` (data/compute/report), `M=4` tools, `A_enum=4`, `UNBOUNDED_PROXY=2^10`.
`|T| = 1 + 5(4) = 21`.

### 4.1 Example A — rung L0, maximally constrained

`plan=schema_validated, state=fsm_fixed, routing=fixed_map, tool=forced_single,`
`arg=enum_strict, substrate=tool_call, skill=precise, retry=bounded`

| Decision point | Count | `\|choices\|` | bits each | Subtotal |
|---|---|---|---|---|
| Plan | 1 | 1 | 0 | 0 |
| State | 4 | 1 | 0 | 0 |
| Routing | 4 | 1 | 0 | 0 |
| Tool | 4 | 1 | 0 | 0 |
| Arguments | 4 | 4 | 2 | **8** |
| Substrate | 4 | 1 | 0 | 0 |

**ADF_total = 8.000 bits · ADF_rate = 8/21 = 0.381 bits/dp**

Note this is not zero. Even fully constrained, the model still picks argument
values from a 4-way enum. A genuine ADF=0 rung requires arguments fully
determined by state (`arg=fixed`), which is **added as rung L0′** so the ladder
actually reaches the origin the plan's §6.2 diagram claims.

### 4.2 Example B — rung L3, intermediate

`plan=schema_validated, state=fsm_fixed, routing=model_chosen, tool=subset,`
`arg=typed, substrate=tool_call, skill=precise, retry=bounded`

| Decision point | Count | `\|choices\|` | bits each | Subtotal |
|---|---|---|---|---|
| Plan | 1 | 1 | 0 | 0 |
| State | 4 | 1 | 0 | 0 |
| Routing | 4 | 3 | 1.585 | **6.340** |
| Tool | 4 | 2 | 1.000 | **4.000** |
| Arguments | 4 | 32 | 5 | **20.000** |
| Substrate | 4 | 1 | 0 | 0 |

**ADF_total = 30.340 bits · ADF_rate = 1.445 bits/dp**

### 4.3 Example C — rung L6, unconstrained

`plan=free_text, state=model_chosen, routing=model_chosen, tool=free_choice,`
`arg=free_form, substrate=codegen, skill=none, retry=none`

| Decision point | Count | `\|choices\|` | bits each | Subtotal |
|---|---|---|---|---|
| Plan | 1 | 1024 | 10 | **10.000** |
| State | 4 | 5 | 2.322 | **9.288** |
| Routing | 4 | 3 | 1.585 | **6.340** |
| Tool | 4 | 5 | 2.322 | **9.288** |
| Arguments | 4 | 1024 | 10 | **40.000** |
| Substrate | 4 | 1 (forced codegen) | 0 | 0 |

**ADF_total = 74.915 bits · ADF_rate = 3.567 bits/dp**

### 4.4 Ladder and sensitivity band

| Rung | ADF_rate @2^6 | **@2^10** | @2^14 |
|---|---|---|---|
| L0′ (args fixed) | 0.000 | **0.000** | 0.000 |
| L0 | 0.381 | **0.381** | 0.381 |
| L3 | 1.064 | **1.445** | 1.826 |
| L6 | 2.615 | **3.567** | 4.520 |

Monotone in ADF at every proxy value, which is the property the Phase 4
monotonicity gate tests.
