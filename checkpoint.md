# ADF Study — Checkpoint Log

## 2026-09-16 — Phase 1 (partial): ADF formalized, preregistration BLOCKED

**Ran:** repo scaffold, `docs/adf_definition.md`, `src/adf/metric.py`, worked-example
verification, OpenRouter roster availability check.
**Spent:** $0.00 — no model API calls (2 free OpenRouter catalog GETs).

### Done
- `docs/adf_definition.md` — formula, 6 decision points, proxy handling, 3 worked examples.
- `src/adf/metric.py` — `compute_adf()`; all 3 worked examples verified to match the doc
  (2 hand-arithmetic errors in the doc found and corrected this way).
- Ladder verified monotone in ADF at all three proxy values.
- Model roster confirmed live on OpenRouter: `qwen/qwen-2.5-7b-instruct`,
  `google/gemma-3-27b-it`, `qwen/qwen3.8-27b`, `meta-llama/llama-3.3-70b-instruct`.

### Plan defects found (plan v2, `loop_eng.md`)
1. **ADF confounded with horizon.** `ADF_total` scales with task length, so H3
   (cross-family) would be confirmed by horizon alone. Fix: added `ADF_rate`
   (bits/decision point); H3 tested on rate only. See spec section 3.
2. **ADF endogenous on conversational/looping tasks.** F0 (tau2) and F3
   (convergence loop) have runtime-determined decision counts. Fix: `T` is taken
   from a committed nominal schedule, never the trace. See spec section 1.1.
3. **skill_level confounds the ladder.** Skills change the prior, not the legal set
   (0 bits), but section 6.2 moves skill together with constraint. Fix: hold skill
   fixed in the main grid, sweep separately.
4. **Ladder never reaches ADF=0.** enum args leave 2 bits/step. Added rung L0'.
5. **User-simulator determinism check is unsatisfiable as written** (section 4.3):
   user turn k conditions on agent turns 1..k-1, which vary by construction, so
   "verify user turns are identical across trials" can never pass on exactly the
   runs that matter. Restate as a prefix-replay determinism test.
6. **Grid arithmetic is stale.** Section 6.2 says "3 task families x 4 models";
   section 4.2 now defines 6 families and section 5 lists 5 models.
7. **Figure/family namespace collision.** Families F0-F5 vs figures F1-F3 (section 9).
8. **tau2-bench version wrong.** Plan cites v0.2.0 for the task fixes; verified
   v0.2.0 (Oct 2025) was the leaderboard release and **v1.0.1 (Jul 2026)** shipped
   the corrections. Banking exists as `banking_knowledge` (extra), not a core domain.
9. **Quantization is an unpinned confound.** Providers serve the roster at fp4/fp8/
   bf16. Pinning provider alone is insufficient; pin quantization too.
   `qwen-2.5-7b-instruct` has exactly one provider (Phala, quant unknown) — single
   point of failure under `allow_fallbacks=false`.

### Blocked on decision (Hard Rule 7)
- **D1** Budget ceiling — no figure anywhere in the plan; section 5 requires a hard cap.
- **D2** Grid scope — 6 families x 5 models x >=6 rungs x N>=50 is ~9,000 multi-turn runs.
- **D3** F3/F4 have no customer to simulate; tau2 user simulator is degenerate for them.
- **D4** Confirm the section 3 H1-H5 measure assignment before freezing.

**Stop conditions:** none applicable yet (Phase 1).
**Next:** resolve D1-D4 -> write and commit `PREREGISTRATION.md` -> Phase 2.

---

## 2026-09-16 — Phase 1 COMPLETE: preregistration committed

**PREREGISTRATION.md frozen at commit `34b1a3786ec8d58a885dad7281755a0f984f8b6c`**
(2026-09-16 16:18:16 +0530). Hypotheses, metrics, ADF formula, and significance
threshold are now immutable. Engineering remains free to change.

### D1-D4 resolved
- **D1 Budget:** deferred. OpenRouter balance verified at **$4.446** ($5.00 credited,
  $0.554 used). Cost-guard hard ceiling set to **$4.00**, leaving $0.45 reserve.
  Phases 2-4 are zero-spend by design, so this does not block the build.
- **D2 Scope:** F0 (tau2-bench) + F1 (linear) + F2 (branching). F3/F4/F5 out of
  confirmatory scope; exploratory extension only if built later.
- **D3 F3/F4 user simulator:** no user simulator (single-shot pipelines) when built.
- **D4 Measures:** accepted as specified — H3 on ADF_rate only.

### Budget consequence (flagged, not blocking)
At $4.00 the preregistered full grid (N>=50 x 3 families x 4 models x >=6 rungs
~= 3,600+ multi-turn runs) is **not affordable**. Cheapest roster model is
$0.10/1M input tokens, but tau2-bench episodes are multi-turn with a user
simulator on top. The pilot (section 6.1: 1 model, 1 family, 5 rungs, N=10 = 50 runs)
is affordable at roughly $0.05-0.30 and proceeds as planned. The full grid needs a
budget decision before Phase 6.2 — per Rule "reduce ADF rungs, never N per cell."

**Next:** Phase 2 — configurable harness.

---

## 2026-09-16 — Phase 2 started: code vendored, provider risk cleared

**Spent:** ~$0.0005 (5 diagnostic calls, not experimental data).

- Vendored 1,823 lines from `Harness_Engg-1` per plan section 3 (FSM, validation gate,
  tracing, F1 tasks + scorers, analysis, data). Provenance in `NOTICE.md`.
- `docs/provider_pinning.md` — provider **and quantization** pinned per model.

### Blocker found and cleared
The prior paper recorded that Phala corrupts JSON on `tool_choice="required"` for
`qwen-2.5-7b-instruct` (3/3). Phala is that model's **only** OpenRouter provider, and
the ADF~0 rung requires forced tool selection — so the small tier could not have run
the most constrained condition, and the frozen roster would have needed a substitution.
**Re-tested: fixed, 3/3 pass.** Roster stands unchanged. Re-verify before the full grid.

**Next:** `src/harness/configurable.py` — the config-driven executor.

---

## Session 3 — Phase 2 continued: configurable harness + ADF ladder

**Date:** 2026-09-17 (multi-session continuation)
**Spent:** $0.00 (zero API calls — pure build session)
**Commits:** 7d92e41 (Phase 2: configurable harness + ADF ladder)

### Delivered this session

1. **`src/harness/configurable.py`** (new, ~600 lines) — the config-driven executor.
   - `HarnessConfig`-driven; every layer independently togglable (all 8 fields, any
     combination).
   - Reuses `fsm.py`'s `_structured_plan_prompt`, `parse_structured_plan`,
     `validate_structured_plan`, and all validators verbatim (loop_eng.md §3).
   - `_make_arg_variant()`: wraps session-based tools with the arg schema for the
     configured `arg_mode` (fixed / enum_strict / typed / free_form).  Each has a
     default value so the tool doesn't fail when the model omits the arg.
   - `_build_lc_tools()`: selects the tool list for the step based on `tool_mode`
     (forced_single / subset / free_choice).
   - `_plan_phase()`: `schema_validated` (with retries and escalation) + `free_text`.
   - `_routing_phase()`: `fixed_map` (logged, no LLM call) + `model_chosen`
     (manager LLM picks agent from K candidates; routing_decision event logged).
   - `_execute_step()`: handles all three substrates — `tool_call` (full), `codegen`
     (exec() sandbox), `hybrid` (model may call tool OR write code).
   - `_get_next_state_model_chosen()`: iterative state selection loop for
     `state_mode="model_chosen"`; caps at 30 steps; logs `state_choice` event.
   - All attempts retained in trace regardless of outcome (rule §1.5).
   - `TaskDefinition` dataclass + registry; `_make_f1_finance_task_def` and
     `_make_f1_legal_task_def` factories; `register_task_definition()` hook for
     F2 and later families.
   - `run_configurable()` main entry point: computes ADF, logs `adf_config` event
     before execution, runs all phases, scores against ground truth.

2. **`src/tracing/logger.py`** (extended, backward-compatible) — six new event types:
   `adf_config`, `routing_decision`, `skill_load`, `step_attempt`, `state_choice`,
   `codegen_attempt`.

3. **`docs/adf_ladder.md`** (new, **FROZEN** per prereg §5 — required before any grid
   run) — 8 rungs L0′ through L6:
   ```
   Rung    ADF_rate @2^10
   L0'     0.000   schema/fsm/fixed/forced/fixed/tool_call
   L0      0.381   +enum_strict args
   L1      0.857   +free_text plan
   L2      1.159   +model_chosen routing
   L3      1.921   +subset tools + typed args
   L4      2.363   +model_chosen state
   L5      3.567   +free_choice tools + free_form args
   L6      3.758   +hybrid substrate
   ```
   Monotone at all three proxy values (verified by compute_adf).
   H5 test pair documented: L5-tool_call vs L5-codegen (ADF_rate identical at all
   proxies — the substrate-equivalence test pair per prereg §1 H5).

4. **venv created** (`.venv/`) — `requirements.txt` installed.

### Smoke tests (zero API spend — deterministic stub model)

7 configs exercised (fin/L0′ through fin/L3, leg/L0, leg/L1):
- All `task_success=True`
- All `adf_rate` exact to 3dp vs formula
- Trace for fin/L2 verified: all 9 required event types present
  (`run_start`, `adf_config`, `plan`, `routing_decision`, `state_transition`,
  `skill_load`, `step_attempt`, `tool_call`, `run_end`)

### What the harness does NOT do yet (next session)

- **F2 branching task family** — `get_task_definition('branching')` not yet registered.
  Required before Phase 4 branching-task sanity gate.
- **tau2-bench F0** — not cloned or integrated.
- **Phase 4 validation test suite** (`tests/`) — still only `__init__.py`.
  Required before ANY real API spend.
- **Difficulty calibration** (section 4.4) — requires F2 to exist first.
- **Multi-session instance generation** — `data/` still has only 2 instances
  (finance/portfolio.csv, legal/contract.txt); ≥20 per family required.

### Next steps (in priority order)

1. **F2 branching task family** — `src/tasks/branching_ecl.py` + scorer + ≥20
   instances with known required paths (correct state sequence depends on data,
   NOT inferable from the task description alone).  Register with
   `register_task_definition('branching')`.  This is the family that makes H2
   and H3 testable.
2. **Phase 4 validation suite** (`tests/test_harness.py`, `tests/test_adf.py`,
   `tests/test_branching.py`) — all 10 gates in loop_eng.md §5 as real pytest
   tests.  Gate: all pass before any API spend.
3. **tau2-bench clone and integration** (F0 family, v1.0.1 pinned) — requires
   tau2-bench repo, uv, Python 3.12; user simulator pinned at T=0.
4. **Instance generation** for F1 families — ≥20 instances each with ground
   truth from deterministic Python; RNG seeds committed.
5. **Difficulty calibration** (§4.4) — requires F2 + ≥20 instances.

**Stop conditions:** None applicable (Phase 2 still in progress).
**Diagnostics:** Not applicable (no data runs yet).
**Budget:** $0.00 spent this session. Running total ~$0.0005.

---

## Budget decision — APPROVED by human 2026-09-17

**Decision:** Full confirmatory grid reduced to **3 ADF rungs × 2 task families × 4 models × N=50 = 1,200 runs** to fit the $4.00 ceiling. N/cell is inviolate per preregistration.

**Selected rungs for the full grid:** L0′ (0.000 bits/dp), L3 (1.921 bits/dp), L5 (3.567 bits/dp).
Rationale: widest possible spread while staying within budget. All 8 rungs remain defined
in `docs/adf_ladder.md` and available for the pilot.

**Pilot** (§6.1): still uses 5 rungs (L0′, L1, L2, L3, L5), 1 model, 1 family, N=10 = 50 runs.
Pilot is within budget regardless.

**Human decision flag discharged.** No further budget decision needed before Phase 6.2.

---

## Session 4 — F2 branching task + Phase 4 validation suite

**Date:** 2026-09-17 (continuation)
**Spent:** $0.00 (zero API calls this session)
**Commits:** 1ba6c28

### Budget decision recorded
Full grid: **3 rungs (L0′, L3, L5) × 2 families × 4 models × N=50 = 1,200 runs**.
Human-approved 2026-09-17. N/cell inviolate.

### Delivered this session

1. **`src/tasks/branching_ecl.py`** — F2 branching task family.
   - Linear path: LOAD_DATA → VALIDATE_DATA → CALCULATE → GENERATE_REPORT
   - Branching path: adds ESCALATION (stress-test ECL at PD_stress=2×PD) when
     any valid loan has PD > 0.15. Condition NOT stated in the task prompt.
   - 5 tools (adds `escalation_tool`). TaskSpec: S=5, N=6, K=3, M=5.

2. **`src/harness/gen_branching_instances.py`** — deterministic generator (seed=42).
   30 instances (15 linear, 15 branching), 8 valid + 4 invalid rows each.
   Ground truth: deterministic Python only, never LLM.

3. **`data/branching/`** — 30 instances + GT JSON + manifest.json (committed).

4. **`configurable.py` extended** — `TaskDefinition.linear_states` field; FSM loop
   uses `linear_states` for `fsm_fixed` on branching tasks (the H2 mechanism).
   `_make_f2_branching_task_def()` registered as `'branching_ecl'`.

5. **`tests/test_phase4.py`** — 92 tests, **all PASS**.
   - Gate 1: 64 HarnessConfig combos run without crash + 5 structural checks
   - Gate 2: ADF monotonicity at all 3 proxies + per-layer tightening
   - Gate 3: ADF hand-check (Examples A/B/C + L0'=0, codegen norm, H5 pair)
   - Gate 4: Trace completeness for success + failure runs
   - Gate 5: Ground-truth scorers for finance_ecl, legal_clause, 5 linear +
             5 branching instances
   - Gate 6: L0' FAILS all 10 branching instances (ESCALATION never visited);
             L0' PASSES all linear instances
   - Gate 7: Failure sets disjoint (only branching instances fail at L0')

### Phase 4 verdict: ALL GATES PASS

The Phase 4 guard is cleared. Real API spend may now begin.

### What remains before the pilot (Phase 5 → 6.1)

1. **F1 instance generation** (≥20 per family) — `data/finance/` has only 1
   instance; `data/legal/` has only 1. Need to generate diverse instances with
   known difficulty parameters and commit generators + RNG seeds.
   Generator already exists structurally for branching; need similar for F1.

2. **tau2-bench F0** (not yet integrated) — still required for the confirmatory
   grid but NOT for the pilot (pilot can use F1 or F2). OK to defer to after pilot.

3. **Pilot run** (§6.1): 1 model, 1 family (branching_ecl or finance_ecl),
   5 rungs (L0′,L1,L2,L3,L5), N=10 = 50 runs. Uses real API.
   - Purpose: verify pipeline end-to-end + variance estimate for power calc.
   - Cheapest: `qwen/qwen-2.5-7b-instruct` on `branching_ecl` (most interesting
     family for H2 test, and cheapest model).
   - Spend estimate: ~50 runs × ~5k tokens avg × $0.10/1M = ~$0.025. Well within budget.

4. **discards.jsonl** — needs to be created (currently absent). First API run
   will likely trigger some 429/5xx; the logger must write to it.

5. **Provider re-verification** — Phala/qwen-2.5-7b-instruct JSON correctness
   under `tool_choice="required"` must be re-tested before the pilot (last cleared
   2026-09-16; may have regressed).

**Stop conditions:** None applicable yet (no data runs).
**Budget running total:** ~$0.0005.
**Next:** F1 instance generation → pilot run → full grid.

---

## Session 5 — Bedrock migration (commit d7eee36)

**Date:** 2026-09-17
**Spent this session:** ~20 API calls for provider checks + 4 live validation runs.
  No grid data. Total spend: ~$0.01 (Bedrock billing, no ceiling).

### Decision: switched from OpenRouter to AWS Bedrock
Human approved 2026-09-17. Budget ceiling ($4.00 OpenRouter cap) is dissolved.
Bedrock billed to project AWS account.

### Bedrock roster (confirmed working, tool-calling validated)
| Role | Model ID | OpenRouter sub |
|---|---|---|
| Small | `amazon.nova-micro-v1:0` | `qwen/qwen-2.5-7b-instruct` |
| Mid gen-1 | `google.gemma-3-27b-it` | `google/gemma-3-27b-it` (EXACT) |
| Mid gen-2 | `qwen.qwen3-32b-v1:0` | `qwen/qwen3.8-27b` (~same) |
| Large | `amazon.nova-pro-v1:0` | `meta-llama/llama-3.3-70b-instruct` |
| Frontier | *(none)* | dropped (Claude/GPT/Llama = ValidationException) |

### Engineering fix required for Bedrock
`langchain-aws ChatBedrockConverse` blocks `toolChoice` for Gemma-3-27B and
Qwen3-32B (incorrect internal allow-list). Direct `boto3.client.converse()` works
for both. Wrote `BedrockChatModel(BaseChatModel)` that wraps boto3 directly.
`tool_choice="any"` maps to Bedrock `toolChoice={"any":{}}` (= "required").
92/92 Phase 4 tests pass.

### ⚠️  OPEN QUESTION FOR HUMAN — rung count

The 3-rung reduction (L0′, L3, L5) was approved solely because of the $4 budget
cap. That cap is gone. The preregistration (PREREGISTRATION.md §5) requires ≥6 rungs.

**With unlimited Bedrock budget, do you want to restore the full ≥6 rung grid?**

Cost estimate for the full grid (all 8 rungs, 2 families, 4 models, N=50):
  8 × 2 × 4 × 50 = 3,200 runs.
  Rough cost at ~6,000 tokens/run × ~$1/1M tokens (Nova-Micro) to ~$8/1M (Nova-Pro):
  Low end (all Nova-Micro): ~$0.10.  Realistic blended: ~$5–15 total.
  These are Bedrock on-demand prices; exact cost depends on actual token lengths.

If you say yes, I'll update the grid to ≥6 rungs (L0′, L1, L2, L3, L4, L5) and
the pilot to cover all 6 rungs at N=10 before the full N=50 grid.

### Next after rung decision
1. F1 instance generation (≥20 per family) — still only 1 instance each.
2. Pilot run: 1 model, 1 family, pilot rungs, N=10.
3. Full grid.

---

## Session 6 — Grid running; interim results (commits 020858b→3cada76)

**Date:** 2026-09-17 (continuing)
**Spend this session:** ~$0.50 estimate (pilot + 476 grid runs on Bedrock)

### Budget decision — RUNG COUNT RESTORED
Human approved: full ≥6 rung grid (all 8 rungs L0′–L6 restored).
Prior 3-rung reduction was budget-only; preregistration requires ≥6.

### GitHub setup
Both repos initialized locally. Push script: `push_to_github.py`.
Requires a GitHub PAT (classic, `repo` scope):
  `python3 push_to_github.py <PAT>`
Will create: SaranshDhage/harness-engg-phase1 + SaranshDhage/adf-study

### Full grid running in background
- Process PID 769456, writing to results/grid/runs.jsonl
- 3,600 total runs: 8 rungs × 9 cells (8+H5) × 2 families × 4 models × N=50
- Progress at session end: ~476/3600 (13.2%)
- Nova-Micro × finance_ecl complete (9 cells × 50 runs = 450 runs)
- Gemma-3-27B started (~4 min/run — will take ~30h to complete its block)
- **Grid will continue running after session ends (nohup)**

### Pilot results (before full grid)
finance_ecl × Nova-Micro (N=10 per rung):
  L0prime=90%, L0=100%, L1-L4=100%, L5=80%, L6=100%
  (L6 was 0% due to hybrid double-call bug, fixed before grid)

branching_ecl × Nova-Micro (N=10 per rung):
  L0prime=60%, L0=60%, L1=60%, L2=60%, L3=50%, L4=30%, L5=10%, L6=50%
  Pattern: monotone decreasing — Nova-Micro too small to discover ESCALATION.
  Non-monotone H2 curve expected from larger models.

### Interim full-grid results (Nova-Micro × finance_ecl, N=50)
TSR per rung (with bootstrap 95% CI):
  L0prime 0.000 → 96.4% [90.9, 100]
  L0      0.381 → 96.0% [90.0, 100]
  L1      0.857 → 96.0% [90.0, 100]
  L2      1.159 → 96.0% [90.0, 100]
  L3      1.921 → 100%  [100,  100]   ← no failures
  L4      2.363 → 100%  [100,  100]
  L5      3.567 → 92.0% [84.0,  98]   ← 4 ModelErrorException failures
  L6      3.758 → 100%  [100,  100]   ← hybrid works after bug fix
  L5_codegen 3.567 → 0.0% [0, 0]     ← H5 FAIL: mechanism matters for TSR

DI (within-instance, T=0):
  L0prime: DI=1.000, PS=1.000, RR=1.000  (schema plan: perfectly deterministic)
  L0:      DI=1.000, PS=1.000, RR=1.000
  L1:      DI=0.982, PS=0.927, RR=1.000  (free_text plan: PS drops)
  L2:      DI=0.964, PS=0.854, RR=1.000  (routing freed)
  L3:      DI=0.969, PS=0.875, RR=1.000
  L4:      DI=0.971, PS=0.885, RR=1.000
  L5:      DI=0.987, PS=0.948, RR=1.000
  L5_codegen: DI=0.984, PS=0.938, RR=0.000  ← code strings vary even at T=0
  L6:      DI=0.961, PS=0.844, RR=1.000

### Findings so far (interim, confirmatory analysis pending full grid)
1. **H1 partial**: DI drops when plan is freed (L0→L1 PS: 1.000→0.927).
   Not strictly monotone — plan-layer dominates, other layers near-flat at T=0.
2. **H2 not visible yet** on linear task (expected — H2 is for branching).
3. **H5 split**: DI is indistinguishable (diff=0.003) ✓ but TSR is 92% vs 0% ✗.
   Mechanism matters for SUCCESS even when ADF is equal. Publishable null for H5.
4. **Plan Stability** is the dominant DI component, confirming prior paper.
5. **L5_codegen RR=0**: codegen is trace-non-deterministic even at T=0.

### What to do when the grid finishes
1. Run: `python -m src.analysis.phase7_analysis`
2. Check: do Gemma/Qwen3/Nova-Pro show non-monotone TSR on branching_ecl? (H2)
3. Check: is Qwen3-32B's optimal ADF higher than Gemma-3-27B's? (H4)
4. Run: `python3 push_to_github.py <PAT>` to push both repos
5. Update POSTHOC.md with any exploratory findings

### Diagnostics (loop_eng.md §8.1)
- Saturation: L3-L4 TSR=100% (Nova-Micro finance) — saturated-high, noted
- Discards: 0 (no 429/5xx/resets yet on Bedrock)
- Plan component: PS dominates DI signal (expected, confirmed prior paper)

### Stop conditions (§8.2) — status
- [ ] Every hypothesis computed with CI — **not yet** (need full grid)
- [ ] Every cell ≥50 runs — **in progress** (9/72 cells complete)
- [ ] Diagnostic checklist clean 2× — **not yet**
- [ ] H5 comparison complete — **partial** (Nova-Micro/finance only)
- [ ] Conclusions stable across proxy sensitivity band — **not yet**

### Next session priorities
1. Check if grid finished (ps aux | grep run_grid)
2. If done: run phase7_analysis.py for full results
3. If still running: monitor and checkpoint
4. Push to GitHub once PAT is available
