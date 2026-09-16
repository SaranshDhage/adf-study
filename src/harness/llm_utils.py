"""LLM client helpers — Bedrock backend.

Migrated from OpenRouter (ChatOpenAI) to AWS Bedrock after confirming that
the OpenRouter credit ceiling is no longer the binding constraint.  Bedrock
is billed to the project AWS account.

LangChain-AWS issue
-------------------
`ChatBedrockConverse` has an internal model allow-list that incorrectly blocks
`toolChoice` for `google.gemma-3-27b-it` and `qwen.qwen3-32b-v1:0`.  Direct
boto3 calls with `toolChoice={"any": {}}` work for both models (verified
2026-09-17).  `BedrockChatModel` below wraps boto3 directly so every roster
model gets correct `toolChoice` support without patching LangChain.

Design
------
`BedrockChatModel` is a LangChain `BaseChatModel` subclass that:
  - Calls the Bedrock Converse API via boto3 (bypasses LangChain's allow-list).
  - Implements `bind_tools(tools, tool_choice)` returning a bound clone.
  - Converts LangChain messages to Bedrock Converse format and vice versa.
  - Records `usage_metadata` on the AIMessage so `accumulate_usage` works.

Supported tool_choice values:
  "auto"  → toolChoice={"auto": {}}
  "any"   → toolChoice={"any": {}}    (Bedrock's "required")
  None    → no toolChoice field
"""
from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Iterator, Optional, Sequence

import boto3
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import BaseModel as PydanticBaseModel, ConfigDict

_AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
DEFAULT_MAX_TOKENS = 2048


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------

def _lc_to_bedrock_messages(
    messages: list[BaseMessage],
) -> tuple[Optional[str], list[dict]]:
    """Convert LangChain messages to Bedrock Converse format.

    Returns (system_prompt_str, converse_messages_list).
    Bedrock requires the system prompt to be passed separately.
    """
    system: Optional[str] = None
    bedrock_msgs: list[dict] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system = msg.content
        elif isinstance(msg, HumanMessage):
            bedrock_msgs.append({
                "role": "user",
                "content": [{"text": msg.content or " "}],
            })
        elif isinstance(msg, AIMessage):
            content_blocks = []
            # Text content
            if msg.content:
                content_blocks.append({"text": msg.content})
            # Tool use blocks
            for tc in (msg.tool_calls or []):
                content_blocks.append({
                    "toolUse": {
                        "toolUseId": tc.get("id") or str(uuid.uuid4()),
                        "name": tc["name"],
                        "input": tc.get("args", {}),
                    }
                })
            if not content_blocks:
                content_blocks = [{"text": " "}]
            bedrock_msgs.append({"role": "assistant", "content": content_blocks})
        elif isinstance(msg, ToolMessage):
            # Tool results must follow the assistant message that called the tool
            bedrock_msgs.append({
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": msg.tool_call_id,
                        "content": [{"text": str(msg.content)}],
                        "status": "success",
                    }
                }],
            })

    return system, bedrock_msgs


def _tool_to_bedrock_spec(tool: BaseTool) -> dict:
    """Convert a LangChain BaseTool to a Bedrock toolSpec dict."""
    schema = {}
    if tool.args_schema:
        schema = tool.args_schema.model_json_schema()
        # Remove $defs and title (Bedrock doesn't need them)
        schema.pop("title", None)
        schema.pop("$defs", None)
        # Ensure type is present
        schema.setdefault("type", "object")
        # Ensure properties is present
        schema.setdefault("properties", {})
        # Remove 'required' if empty (some models reject it)
        if "required" in schema and not schema["required"]:
            schema.pop("required")
    else:
        schema = {"type": "object", "properties": {}}

    return {
        "toolSpec": {
            "name": tool.name,
            "description": tool.description or tool.name,
            "inputSchema": {"json": schema},
        }
    }


def _bedrock_to_lc_message(response: dict) -> AIMessage:
    """Convert a Bedrock Converse response to a LangChain AIMessage."""
    output_msg = response["output"]["message"]
    content_blocks = output_msg.get("content", [])
    usage = response.get("usage", {})

    text_parts = [b["text"] for b in content_blocks if "text" in b]
    tool_calls = []
    for b in content_blocks:
        if "toolUse" in b:
            tu = b["toolUse"]
            tool_calls.append({
                "id": tu.get("toolUseId", str(uuid.uuid4())),
                "name": tu["name"],
                "args": tu.get("input", {}),
                "type": "tool_call",
            })

    ai_msg = AIMessage(
        content="\n".join(text_parts),
        tool_calls=tool_calls,
    )
    # Attach usage so accumulate_usage() works
    ai_msg.usage_metadata = {
        "input_tokens":  usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
        "total_tokens":  usage.get("inputTokens", 0) + usage.get("outputTokens", 0),
    }
    return ai_msg


# ---------------------------------------------------------------------------
# Custom Bedrock chat model
# ---------------------------------------------------------------------------

class BedrockChatModel(BaseChatModel):
    """Thin LangChain wrapper around the Bedrock Converse API via boto3.

    Bypasses `langchain-aws` so that `toolChoice={"any":{}}` works for all
    roster models, including Gemma-3-27B and Qwen3-32B which langchain-aws
    incorrectly blocks.
    """

    model_id: str
    region_name: str = _AWS_REGION
    temperature: float = 0.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    _bound_tools: list = []
    _tool_choice: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return f"bedrock-converse-{self.model_id}"

    def _get_client(self):
        return boto3.client("bedrock-runtime", region_name=self.region_name)

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> "BedrockChatModel":
        """Return a clone with tools bound and optional tool_choice set."""
        clone = self.model_copy()
        clone._bound_tools = list(tools)
        clone._tool_choice = tool_choice
        return clone

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = self._get_client()
        system_prompt, bedrock_msgs = _lc_to_bedrock_messages(messages)

        invoke_kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": bedrock_msgs,
            "inferenceConfig": {
                "temperature": self.temperature,
                "maxTokens":   self.max_tokens,
            },
        }

        if system_prompt:
            invoke_kwargs["system"] = [{"text": system_prompt}]

        if self._bound_tools:
            tool_config: dict[str, Any] = {
                "tools": [_tool_to_bedrock_spec(t) for t in self._bound_tools]
            }
            if self._tool_choice == "any":
                tool_config["toolChoice"] = {"any": {}}
            elif self._tool_choice == "auto":
                tool_config["toolChoice"] = {"auto": {}}
            # None: omit toolChoice (model decides)
            invoke_kwargs["toolConfig"] = tool_config

        response = client.converse(**invoke_kwargs)
        ai_msg = _bedrock_to_lc_message(response)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])


# ---------------------------------------------------------------------------
# Public API (unchanged interface from prior llm_utils.py)
# ---------------------------------------------------------------------------

def make_llm(
    model: str,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> BedrockChatModel:
    """Build a BedrockChatModel for the given Bedrock model ID.

    model: e.g. "google.gemma-3-27b-it", "qwen.qwen3-32b-v1:0",
                "amazon.nova-micro-v1:0", "amazon.nova-pro-v1:0"
    """
    return BedrockChatModel(
        model_id=model,
        region_name=_AWS_REGION,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def accumulate_usage(response: AIMessage, totals: dict[str, int]) -> None:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return
    totals["prompt_tokens"]     += usage.get("input_tokens", 0) or 0
    totals["completion_tokens"] += usage.get("output_tokens", 0) or 0
    totals["total_tokens"]      += usage.get("total_tokens", 0) or 0


def parse_plan(content: str) -> list[str]:
    """Best-effort JSON-array extraction from a free-text plan response."""
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
