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
