# Session handoff prompt

Paste the block below into a new Claude Code session started in
`/home/saransh/Projects/Agentic-Research/adf-study`.

---

You are continuing a multi-session research project. Read these three files before
doing anything else, in this order:

1. `../loop_eng.md` — the full research plan (the spec you are executing)
2. `checkpoint.md` — the running log; the last entry is where we stopped
3. `PREREGISTRATION.md` — frozen hypotheses and metrics

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
- **Never silently drop runs.** Discards go to `discards.jsonl` with a reason code and
  are counted in the final analysis. Only HTTP 429/5xx and connection resets are valid
  discard reasons. Model failures, validation failures, and timeouts are DATA.
- **Log everything, every run** — model slug, resolved provider, temperature, seed,
  full request/response, ADF config, per-layer trace. A run that isn't logged didn't happen.
- **If something is ambiguous, stop and ask.** Write the question into `checkpoint.md`
  under "Blocked on decision" and halt that branch rather than guessing.
- Append to `checkpoint.md` after every batch and every loop iteration. Assume the next
  session starts with no memory beyond it.

## State: Phase 1 complete, Phase 2 started (4 commits, clean tree)

- `docs/adf_definition.md` — ADF formalized. Three worked examples, all machine-verified
  against `src/adf/metric.py`. Ladder monotone at all three UNBOUNDED_PROXY values.
- `src/adf/metric.py` — `compute_adf(cfg, task, proxy) -> ADFResult`. Working.
- `PREREGISTRATION.md` — frozen (commit `34b1a37`).
- `docs/provider_pinning.md` — provider AND quantization pinned per model.
- `NOTICE.md` — 1,823 lines vendored from the published repo `../Harness_Engg-1`
  (FSM, validation gate, tracing, F1 tasks + scorers, analysis, data).

### Decisions already made — do not relitigate
- **Scope:** F0 (tau2-bench v1.0.1) + F1 (linear: finance_ecl, legal_clause) + F2
  (branching). F3/F4/F5 are out of confirmatory scope.
- **Models:** `qwen/qwen-2.5-7b-instruct` (Phala), `google/gemma-3-27b-it` (DeepInfra),
  `qwen/qwen3.8-27b` (DeepInfra), `meta-llama/llama-3.3-70b-instruct` (DeepInfra).
  All with `allow_fallbacks=false`.
- **ADF_rate (bits/decision point) is primary; ADF_total is secondary.** H3 is tested on
  ADF_rate ONLY — ADF_total is confounded by task horizon. See `docs/adf_definition.md` section 3.
- **`skill_level` and `retry_policy` contribute 0 bits to ADF.** Skills change the
  model's prior, not the legal choice set. Consequence: hold `skill_level="precise"`
  fixed across the main grid and sweep it as a separate orthogonal factor — the plan's
  section 6.2 ladder moves skill together with constraint, which would confound H1.
- **`T` is enumerated from each family's committed nominal schedule, never from a
  realized trace** — otherwise ADF becomes an outcome and H1 goes circular.

## Budget — the binding constraint

OpenRouter balance **~$4.45**. Hard cost-guard ceiling **$4.00**. Spent so far ~$0.0005
(diagnostics only). Phases 2-4 are zero-spend by design. The pilot (1 model, 1 family,
5 rungs, N=10 = 50 runs) is affordable at ~$0.05-0.30.

**The preregistered full grid (~3,600 multi-turn runs at N>=50) is NOT affordable at $4.**
This needs a human budget decision before Phase 6.2. The rule if N must give: reduce the
number of ADF rungs, **never** N per cell. Flag it, don't silently shrink N.

## Next steps, in order

1. **`src/harness/configurable.py`** — the config-driven executor. `HarnessConfig` already
   exists in `src/adf/metric.py`; the executor must honor every layer independently and run
   with ANY combination. Reuse `src/harness/fsm.py` — do not rewrite it. Must emit a
   per-layer trace (plan, routing, skill loaded, state sequence, tool sequence, args,
   output, retries, failures); failed steps are retained, never dropped.
   - Interaction rule to enforce: `action_substrate="codegen"` overrides `arg_mode` to
     `free_form` (`HarnessConfig.normalized()` already does this; assert it in tests).
2. **`docs/adf_ladder.md`** — define the >=6 rungs (L0' through L6) with computed
   ADF_rate/ADF_total per rung. Required by the prereg BEFORE any grid run. Does not exist yet.
3. **Multi-agent layer** (manager + `data`/`compute`/`report` workers, routing logged and
   constrainable) and **skills** at none/vague/precise.
4. **F2 branching family** — correct state sequence must be data-dependent and NOT
   inferable from the task description, only from the data.
5. **tau2-bench (F0)** — clone `github.com/sierra-research/tau2-bench`, **pin v1.0.1**
   (the plan's citation of v0.2.0 is wrong; v0.2.0 was the leaderboard release, v1.0.1
   shipped the task corrections). Needs `uv`, Python 3.12-3.13. Pin the user-simulator
   model at temperature 0 — it is a second non-determinism source.
6. **Phase 4 validation suite** in `tests/` — all gates in loop_eng.md section 5 must pass
   before ANY grid API spend.
7. **Difficulty calibration** (section 4.4) before the grid.

## Known gaps and gotchas

- **`data/` has only 2 instances** (`finance/portfolio.csv`, `legal/contract.txt`) against
  the plan's >=20 per family. Instance generation is still to do: generate with an LLM
  using known parameter distributions, compute ground truth with **deterministic Python,
  never an LLM**; commit generators and RNG seeds.
- **Dependencies are not installed** in this repo. `requirements.txt` is copied from the
  prior project; there is no venv here yet.
- **`qwen-2.5-7b-instruct` has exactly one provider (Phala)** and quantization reports as
  `unknown`. Under `allow_fallbacks=false` its cells stall if Phala is down — no silent
  re-route. The prior paper recorded Phala corrupting JSON on `tool_choice="required"`;
  re-tested 2026-09-16 and it no longer reproduces (3/3 pass), so the roster stands.
  **Re-verify this before the full grid** — a silent provider regression would corrupt
  the ADF~0 cells specifically.
- **Context floor is 32k** (Qwen-2.5-7B). Prompts are held identical across models, so
  that caps maximum prompt size for the whole study.
- **`loop_eng.md` has known defects** — all nine are catalogued in the first `checkpoint.md`
  entry. Notably: its section 4.3 user-simulator determinism check is unsatisfiable as
  written (user turn k conditions on agent turns 1..k-1, which vary by construction);
  restate it as a prefix-replay determinism test. Its section 6.2 grid arithmetic is stale,
  and task families F0-F5 collide with figure names F1-F3 in section 9.
