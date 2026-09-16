"""Small helpers shared by baseline_agent.py and fsm.py."""

import json
import os
import re

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Without an explicit max_tokens, OpenRouter reserves against the model's
# theoretical max completion length (tens of thousands of tokens) before
# checking affordability against account balance — free-tier balance can't
# cover that worst-case reservation even though our actual completions (a
# short plan array or a single tool call) never come close to using it.
# Confirmed via the raw API: unset max_tokens -> 402 "requested up to 28912
# tokens, but can only afford 12710"; capping it resolves this at zero cost.
DEFAULT_MAX_TOKENS = 1024


def make_llm(model: str, temperature: float = 0.7, max_tokens: int = DEFAULT_MAX_TOKENS) -> ChatOpenAI:
    """Builds a ChatOpenAI client pointed at OpenRouter. `model` is an
    OpenRouter model slug, e.g. "qwen/qwen-2.5-7b-instruct" or
    "google/gemma-2-9b-it". Requires OPENROUTER_API_KEY in the environment
    or a .env file (see .env.example) — deliberately not defaulted, so a
    missing key fails loudly here rather than as a confusing 401 later.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def accumulate_usage(response: AIMessage, totals: dict[str, int]) -> None:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return
    totals["prompt_tokens"] += usage.get("input_tokens", 0) or 0
    totals["completion_tokens"] += usage.get("output_tokens", 0) or 0
    totals["total_tokens"] += usage.get("total_tokens", 0) or 0


def parse_plan(content: str) -> list[str]:
    """Best-effort JSON-array extraction — local models often wrap the array
    in prose or markdown fences despite being told not to."""
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        return [content.strip()[:200]] if content.strip() else []
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, list):
            return [str(step) for step in parsed]
    except json.JSONDecodeError:
        pass
    return [content.strip()[:200]]
