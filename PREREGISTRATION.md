# Preregistration — Agent Degrees of Freedom (ADF)

**Owner:** Saransh Dhage · **Date frozen:** 2026-09-16
**Prior work:** arXiv:2608.26197 (harness engineering, FSM + Structured Planning)
**Plan:** `loop_eng.md` v2 · **ADF spec:** `docs/adf_definition.md` · **Impl:** `src/adf/metric.py`

Once this file is committed, nothing in sections 1-5 may be added, removed, reworded, or
reinterpreted in light of observed results. New hypotheses go to `POSTHOC.md` and are
reported as exploratory. Engineering (bugs, instrumentation, harness implementation, task
generators, retry logic, provider config, sample size upward) remains free to change.

---

## 1. Hypotheses

**H1 (determinism-ADF).** Determinism Index decreases monotonically as ADF increases,
across all models and task families.
*Tested on both ADF_rate (primary) and ADF_total (secondary), within family.*

**H2 (non-monotonic success).** Task Success Rate is non-monotonic in ADF - low at ADF~0 on
branching tasks (over-constrained, cannot express data-dependent paths), low at high ADF
(under-constrained, drifts), with an interior maximum.
*Tested on ADF_rate, within family.*

**H3 (task-dependence of the optimum).** The ADF value maximizing task success is higher for
branching/judgment tasks than for linear tasks.
*Tested on ADF_rate ONLY. ADF_total is confounded by task horizon (see spec section 3) and
may not be used for any cross-family comparison.*

**H4 (model-dependence of the optimum).** The ADF value maximizing task success is higher for
more capable models (they need less scaffolding).
*Tested on ADF_rate, within family, across models. Primary contrast: Gemma-3-27B vs
Qwen3.8-27B (parameter class held constant, generation varies).*

**H5 (substrate equivalence).** Two harness configurations with equal ADF but different
mechanisms (e.g. constrained code-gen vs. a set of bound tools) yield statistically
indistinguishable determinism. This is the strong form of the claim: ADF, not mechanism, is
what matters. If H5 fails, the finding becomes "mechanism matters beyond freedom," which is
equally publishable and must be reported as such.
*Pairs must match on BOTH ADF_rate and ADF_total to count as matched.*

A null or negative result on any hypothesis is a valid outcome and will be reported as such.

---

## 2. Primary metrics

**Determinism Index (DI)** - composite in [0,1], carried forward unchanged from the prior
paper (`Harness_Engg-1/docs/determinism_index.md`): equal-weighted (0.25 each) mean of
pairwise Plan Stability, Tool Path Consistency, State Transition Stability, and Output
Consistency, each a normalized edit similarity averaged over all C(N,2) run pairs.
**All four components are reported separately in every result table**, never only the
composite - the prior paper's plan-text artifact was a single component dominating DI.

**Reproducibility Rate** - fraction of run pairs whose full trace signature matches exactly.
Headline strict metric.

**Task Success Rate** - deterministic scoring only. For F0, tau2-bench's database-state
comparison against the annotated goal state. For F1/F2, the existing Python scorers. No LLM
judge is used for any confirmatory metric.

**pass^k** - tau2-bench's reliability-across-trials metric, reported for F0. pass^k measures
*outcome* consistency; DI measures *process* consistency. They are complementary and are
never substituted for one another.

Determinism and success are reported as two separate axes and are never blended.

---

## 3. ADF formula (frozen)

```
ADF_total = sum_t log2( |legal_choices(t)| )        ADF_rate = ADF_total / |T|
```

Decision points, cardinalities, and the codegen/arg_mode interaction rule are as specified in
`docs/adf_definition.md` sections 2-2.3. `|T| = 1 + 5S`, enumerated from each task family's
committed nominal decision schedule and never from a realized trace.

`UNBOUNDED_PROXY = 2^10` primary; every conclusion recomputed at `2^6` and `2^14` and
reported as a sensitivity band. `typed` arguments = `sqrt(UNBOUNDED_PROXY)`.
`skill_level` and `retry_policy` contribute exactly 0 bits.

---

## 4. Statistical plan

- alpha = 0.05, **Holm-corrected across the family of 5 hypotheses**.
- Bootstrap 95% CIs, >= 10,000 resamples. No bare point estimates anywhere.
- Permutation tests for rate differences; Cohen's d for headline contrasts.
- **Minimum N = 50 runs per cell** (cell = family x model x ADF rung x temperature).
  N is set by a power calculation from pilot variance targeting 80% power; if the required N
  exceeds budget, the number of ADF rungs is reduced, never N per cell.
- Primary grid at T=0; reduced grid at T=1.0 reported separately, never conflated.

---

## 5. Scope frozen at preregistration

**Families:** F0 (tau2-bench airline/retail/telecom, v1.0.1), F1 (linear: finance_ecl,
legal_clause), F2 (branching). F3/F4/F5 are **out of scope for confirmatory claims**; if
built later they are reported as a separate exploratory extension.

**Models:** `qwen/qwen-2.5-7b-instruct`, `google/gemma-3-27b-it`, `qwen/qwen3.8-27b`,
`meta-llama/llama-3.3-70b-instruct`, provider- and quantization-pinned with
`allow_fallbacks=false`. One frontier model on a reduced grid if budget allows.

**ADF rungs:** >= 6 spanning L0' (0.000 bits/dp) to L6, fixed in `docs/adf_ladder.md`
before any grid run. `skill_level` is held at `precise` across the main grid and swept as a
separate orthogonal factor.

**Confirmatory analysis begins only after** the Phase 4 validation gates and the section 4.4
difficulty calibration pass. Calibration tunes tasks; it is complete before any preregistered
test is run and its results are never used to adjust hypotheses.
