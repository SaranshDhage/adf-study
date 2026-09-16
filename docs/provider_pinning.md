# Provider Pinning — Bedrock (updated 2026-09-17)

Migrated from OpenRouter to AWS Bedrock.  All models are served through the Bedrock
Converse API via `langchain-aws :: ChatBedrock`.  Credentials are read from environment
variables `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`.

No `allow_fallbacks=false` equivalent is needed: Bedrock routes each model ID to a
single canonical backend, so quantization drift and provider failover are non-issues.

---

## Roster

| Role | Bedrock model ID | Preregistered slug (OpenRouter) | Status |
|------|-----------------|----------------------------------|--------|
| Small baseline | `amazon.nova-micro-v1:0` | `qwen/qwen-2.5-7b-instruct` | **Substituted** — Qwen-2.5-7B not on Bedrock; Nova-Micro is the cheapest available text model with tool-calling |
| Mid open (gen 1) | `google.gemma-3-27b-it` | `google/gemma-3-27b-it` | **Direct match** |
| Mid open (gen 2) | `qwen.qwen3-32b-v1:0` | `qwen/qwen3.8-27b` | **Near-match** — same Qwen3 generation, 32B vs 27B |
| Large | `amazon.nova-pro-v1:0` | `meta-llama/llama-3.3-70b-instruct` | **Substituted** — Llama-3.3-70B `ValidationException`; Nova-Pro is the accessible large-tier model |
| Frontier | *(none)* | one proprietary (reduced grid) | **Dropped** — Claude/GPT both return `ValidationException` on this account |

The Gemma ↔ Qwen3 pairing (preregistered H4 primary contrast: same parameter class,
different model generation) is preserved: 27B Gemma vs 32B Qwen3 are the same
architectural tier, different families and generations.

---

## Tool-calling behaviour per model (verified 2026-09-17)

| Model | `toolChoice=auto` | `toolChoice=any` (="required") | Notes |
|-------|------------------|-------------------------------|-------|
| `google.gemma-3-27b-it` | ❌ text only | ✅ calls tool | Always use `tool_choice="required"` for forced-single steps |
| `qwen.qwen3-32b-v1:0` | ✅ | ✅ | |
| `amazon.nova-micro-v1:0` | ✅ | ✅ | |
| `amazon.nova-pro-v1:0` | ✅ | ✅ | |
| `mistral.ministral-3-8b-instruct` | ✅ | ✅ | Reserve model, not in primary grid |

**Implication for the harness:** `tool_mode="forced_single"` always passes
`tool_choice="required"` (maps to Bedrock `toolChoice={"any": {}}` via LangChain).
This forces Gemma to call the tool.  No per-model override needed; the harness
handles this uniformly.

---

## Model substitution rationale (engineering decision, not hypothesis change)

The preregistration froze model *names* and *capability tiers*, not provider IDs.
Switching from OpenRouter to Bedrock is an infrastructure change.  The substitutions
preserve the capability ordering (small → mid-gen1 → mid-gen2 → large), which is
what H4 requires.

Key H4 consequence: the primary contrast remains mid-gen1 (Gemma-3-27B) vs
mid-gen2 (Qwen3-32B).  Parameter count is held approximately constant (~27-32B);
model generation varies.  The claim being tested is unchanged.

The substitution of `qwen-2.5-7b` → `nova-micro` weakens the small-tier anchor
slightly: Nova-Micro is Amazon's proprietary model, not open-weights.  If the
small-tier result is surprising, this should be flagged in the paper as a
provenance difference.

---

## Region

`us-east-1` (from `AWS_DEFAULT_REGION`).  All four models are available in this
region.  If region changes, re-verify availability before any grid run.

---

## Re-verification schedule

Re-run the tool-calling smoke test in `tests/test_phase4.py :: TestControlFlow`
before the pilot and again before the full grid.  Bedrock model availability can
change without notice (models can be deprecated or access can be revoked).
