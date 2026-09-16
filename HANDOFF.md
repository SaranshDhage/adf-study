# Session handoff prompt

Paste the block below into a new Claude Code session started in
`/home/saransh/Projects/Agentic-Research/adf-study`.

---

You are continuing a multi-session research project. Read these three files before
doing anything else, in this order:

1. `../loop_eng.md` — the full research plan (the spec you are executing)
2. `checkpoint.md` — the running log; the last entry is where we stopped
3. `PREREGISTRATION.md` — frozen hypotheses and metrics

Then read HANDOFF.md in this repo, which is the detailed state summary. Follow it.

## What this project is

Testing whether execution determinism in an agentic system is a predictable function
of one measurable quantity — Agent Degrees of Freedom (ADF), the log2-sum of legal
choices the harness leaves the model per decision point — while task success is
*non-monotonic* in ADF with a task-dependent optimum. Deliverable is a curve with an
identifiable knee, not a two-condition comparison. Prior work: arXiv:2608.26197.

## Hard rules (from loop_eng.md section 1 — these override your defaults)

- **`PREREGISTRATION.md` is frozen at commit `34b1a37`.** The hypothesis set, primary
  metric definitions, significance threshold (alpha=0.05 Holm), and the ADF formula
  may NOT change. New hypotheses go in `POSTHOC.md`, reported as exploratory only.
- The loop iterates on **engineering, never on hypotheses**. Bugs, instrumentation,
  task generators, retry logic, provider config, and sample size (upward only) are
  free to change.
- **A null or negative result is a successful outcome.** Do not tune conditions until
  a curve looks good.
- **Never silently drop runs.** Discards go to `discards.jsonl` with a reason code.
  Only HTTP 429/5xx and connection resets are valid discard reasons. Model failures,
  validation failures, and timeouts are DATA.
- **Log everything, every run** — model slug, resolved provider, temperature, seed,
  full request/response, ADF config, per-layer trace. A run that isn't logged didn't
  happen.
- **If something is ambiguous, stop and ask.** Write it into `checkpoint.md` under
  "Blocked on decision" and halt that branch.
- Append to `checkpoint.md` after every batch and every loop iteration.

## State: Phase 2 in progress (6 commits, clean tree)

### Completed artifacts
- `docs/adf_definition.md` — ADF formula, 6 decision points, worked examples.
- `src/adf/metric.py` — `compute_adf()`, `HarnessConfig`, `TaskSpec`, `ADFResult`.
  All 3 worked examples machine-verified.
- `PREREGISTRATION.md` — frozen at commit `34b1a37`.
- `docs/provider_pinning.md` — provider AND quantization pinned per model.
- `NOTICE.md` — 1,823 lines vendored from `../Harness_Engg-1`.
- **`src/harness/configurable.py`** — full configurable executor (NEW this session).
  Every HarnessConfig combination runs; per-layer trace emitted; failed steps
  retained. See docs below.
- **`src/tracing/logger.py`** — extended with 6 new event types (backward-compat).
- **`docs/adf_ladder.md`** — 8 rungs L0′–L6, **FROZEN** per prereg §5.

### ADF ladder (finance_ecl, primary proxy 2^10)
```
L0'  0.000  schema/fsm/fixed/forced/fixed/tool_call
L0   0.381  +enum_strict args
L1   0.857  +free_text plan
L2   1.159  +model_chosen routing
L3   1.921  +subset tools + typed args
L4   2.363  +model_chosen state
L5   3.567  +free_choice tools + free_form args
L6   3.758  +hybrid substrate
H5 pair: L5-tool_call vs L5-codegen (ADF identical, mechanism differs)
```

### Decisions already made — do not relitigate
- **Scope:** F0 (tau2-bench v1.0.1) + F1 (linear) + F2 (branching). F3/F4/F5
  out of confirmatory scope.
- **Models:** 4 models, provider/quant-pinned with `allow_fallbacks=false`.
- **ADF_rate is primary for H1/H2/H3/H4; ADF_total secondary.**
- **H3 on ADF_rate ONLY** (cross-family; total is confounded by horizon).
- **`skill_level` and `retry_policy` contribute 0 bits.** Hold `skill_level=precise`
  fixed in main grid; run as separate orthogonal factor at L0/L3/L5.
- **`T` from nominal schedule, never from trace.**
- **ADF ladder:** 8 rungs defined in `docs/adf_ladder.md`, frozen.

## Budget
OpenRouter balance ~$4.45. Hard ceiling $4.00. Spent ~$0.0005. Phases 2–4 are
zero-spend. Full grid (~3,600 runs at N≥50) is NOT affordable at $4 — flag for
human decision before Phase 6.2. If N must give: reduce ADF rungs, never N/cell.

## Next steps (in priority order)

### 1. F2 branching task family (IMMEDIATE — gates Phase 4)
File: `src/tasks/branching_ecl.py`

The branching task must satisfy two constraints (loop_eng.md §4.0):
- At ADF≈0 (forced path), the agent MUST FAIL on instances that require the
  alternate path. If it passes at ADF≈0, the task has no real branching — fix it.
- At ADF≈max (unconstrained), the agent MUST ALSO FAIL (drifts/skips steps).

Design: a loan portfolio where some loans are flagged for ESCALATION before REPORT
(e.g. high-risk loans with PD > 0.15). The correct state sequence is DATA-DEPENDENT
and NOT INFERABLE from the task description — only from reading the data. The
branching condition (PD > 0.15) must be in the data, not the prompt.

States needed:
- Linear path: LOAD_DATA → VALIDATE_DATA → CALCULATE → GENERATE_REPORT
- Branching path: LOAD_DATA → VALIDATE_DATA → CALCULATE → ESCALATION → GENERATE_REPORT
  (triggered when any loan has PD > 0.15 after validation)

The ESCALATION state performs additional review (e.g., compute stress-test ECL at
2× PD). The final report must reflect which path was taken.

**Instance generation rules (loop_eng.md §4.5):**
- Generate with strong LLM using known param distributions; compute ground truth with
  deterministic Python ONLY (never LLM).
- Commit generators + RNG seeds so dataset is exactly reproducible.
- Include at controlled rates: high-PD loans (need ESCALATION), low-PD loans (linear
  path). Target: 50% branching, 50% linear.
- Target ≥20 instances (prereg says ≥20 per family).
- Record per-instance "required_path" as metadata.

TaskDefinition: register with `register_task_definition('branching_ecl')`.

### 2. Phase 4 validation suite (GATING — no API spend until all pass)
Files: `tests/test_harness.py`, `tests/test_adf.py`, `tests/test_branching.py`

All 10 gates from loop_eng.md §5 as real pytest tests:
1. Fake-model control-flow: every HarnessConfig combination; assert harness enforces
   what config claims (forced tool really forces; FSM blocks illegal transitions;
   schema validation rejects bad plans).
2. ADF monotonicity: tightening any single layer never increases computed ADF.
3. ADF hand-check: 3 configs whose ADF is computed by hand in docs/adf_definition.md
   match compute_adf() exactly.
4. Trace completeness: every run emits all per-layer fields; failed run still produces
   complete trace with failure recorded.
5. Ground-truth test: deterministic scorers agree with hand-computed answers on ≥5
   instances per family.
6. **Branching-task sanity**: ADF≈0 config MUST fail branching instances requiring the
   alternate path. If it passes, the task isn't actually branching — fix the task.
7. **Difficulty calibration** (§4.4): for F2, both ADF≈0 and ADF≈max score below 90%;
   failure sets are disjoint.
8. Loop termination: (not applicable to F1/F2 — relevant for F3 if built).
9. Deterministic fault injection: (F4, out of scope for now).
10. Judge stability: (F5, out of scope for now).

**Gate 6 (branching sanity) and Gate 7 (difficulty calibration) are the critical
gates that make H2 measurable. Do not proceed to API spend without them.**

### 3. tau2-bench F0 integration (after Phase 4 passes)
- Clone `github.com/sierra-research/tau2-bench` at tag v1.0.1.
- Pin user simulator at T=0 (see loop_eng.md §4.3).
- Wrap as a TaskDefinition or use tau2-bench's native runner with ADF config injection.
- Note: tau2-bench requires `uv` and Python 3.12–3.13.

### 4. Instance generation for F1 families (≥20 per family)
- Current: only 2 instances (portfolio.csv, contract.txt).
- Generate diverse instances with known difficulty parameters; commit generators + seeds.

## Known gaps and gotchas (carry forward)

- **`data/` has only 2 instances** — ≥20 per family required before pilot.
- **F2 branching task does not exist yet** — Phase 4 gates 6 and 7 cannot pass.
- **Phase 4 test suite is empty** — `tests/` has only `__init__.py`.
- **tau2-bench not cloned** — F0 not yet integrated.
- **`qwen-2.5-7b-instruct` single provider (Phala)** — re-verify JSON correctness
  with `tool_choice="required"` before full grid (known prior issue, cleared
  2026-09-16; may regress).
- **Context floor 32k** (Qwen-2.5-7B) — caps prompt size for whole study.
- **`loop_eng.md` has 9 known defects** — all documented in checkpoint Session 1.
  Most important: §4.3 user-simulator check is unsatisfiable as written; restate
  as prefix-replay determinism test.
- **Full grid (~3,600 runs at N≥50) is not affordable at $4** — human budget
  decision required before Phase 6.2. Do not silently shrink N.
- **codegen substrate** — implemented with exec() sandbox in configurable.py;
  not hardened but correct for research purposes. Verify behavior in Phase 4 tests.
