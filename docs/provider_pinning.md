# Provider and quantization pinning

Auto-routing is a confound for a determinism study. Every request pins
`provider.order` to exactly one backend with `allow_fallbacks=false`.

**Quantization must be pinned alongside provider.** OpenRouter serves the roster at
fp4/fp8/bf16 depending on backend; numeric precision is a plausible determinism
factor, so it is held constant rather than left to chance.

| Tier | Slug | Provider | Quant | ctx | $/1M in |
|---|---|---|---|---|---|
| Small open | `qwen/qwen-2.5-7b-instruct` | Phala | unknown | 32k | 0.10 |
| Mid open gen1 | `google/gemma-3-27b-it` | DeepInfra | fp8 | 131k | 0.08 |
| Mid open gen2 | `qwen/qwen3.8-27b` | DeepInfra | bf16 | 262k | 0.15 |
| Large open | `meta-llama/llama-3.3-70b-instruct` | DeepInfra | fp8 | 131k | 0.10 |

DeepInfra serves three of four, which keeps backend variance low across tiers.

## Known risks

**`qwen-2.5-7b-instruct` has exactly one provider (Phala).** Under
`allow_fallbacks=false` there is no alternate backend; if Phala is unavailable, that
model's cells stall rather than silently re-route. Quantization is reported as
`unknown`, so it cannot be pinned for this model — documented as a limitation.

**Context asymmetry.** 32k (Qwen-2.5-7B) to 262k (Qwen3.8-27B). Prompts are held
identical across models so context length is not a confound; the 32k floor caps
maximum prompt size for the whole study.

## Resolved: forced tool calls on Phala

The prior paper recorded (`fsm.py` docstring) that Phala corrupted its own JSON
output when `tool_choice="required"` was set for `qwen-2.5-7b-instruct`, reproduced
3/3. This matters because the ADF~0 rung requires forced tool selection — if it
still held, the small model could not run the most constrained condition at all.

**Re-tested 2026-09-16: no longer reproduces, 3/3 valid tool calls with correct enum
arguments.** The provider-side bug has been fixed. `tool_choice="required"` is used
uniformly across all models, which removes the prior paper's per-model calibration
knob and is itself a small methodological improvement.

Re-verify this before the full grid; a silent provider regression would corrupt the
ADF~0 cells specifically.
