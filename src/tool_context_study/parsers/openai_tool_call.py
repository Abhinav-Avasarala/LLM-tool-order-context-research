"""Parse an OpenAI-compatible chat completion response into a tool-call decision.

Used against vLLM/SGLang's OpenAI-compatible `/v1/chat/completions` endpoint.
Deliberately model-agnostic: it only assumes the standard
`choices[0].message.tool_calls[i].function.{name,arguments}` shape.
"""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

ParseStatus = Literal["ok", "no_tool_call", "malformed_tool_call", "inference_error"]


class ParsedToolCall(TypedDict):
    status: ParseStatus
    selected_tool: str | None
    arguments: dict[str, Any] | None
    n_tool_calls: int


def parse_chat_completion(response_json: dict[str, Any]) -> ParsedToolCall:
    choices = response_json.get("choices")
    if not choices:
        return {"status": "inference_error", "selected_tool": None, "arguments": None, "n_tool_calls": 0}

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls") or []
    n = len(tool_calls)

    if n == 0:
        return {"status": "no_tool_call", "selected_tool": None, "arguments": None, "n_tool_calls": 0}

    if n > 1:
        # Ambiguous which call is "the" selection; brief section 10 scores this
        # as incorrect regardless of whether one call happens to be right, so
        # score.py decides the "multiple_tool_calls" category from n_tool_calls
        # rather than us picking one call here.
        return {"status": "ok", "selected_tool": None, "arguments": None, "n_tool_calls": n}

    call = tool_calls[0]
    function = call.get("function", {})
    name = function.get("name")
    raw_arguments = function.get("arguments")

    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError:
        return {"status": "malformed_tool_call", "selected_tool": name, "arguments": None, "n_tool_calls": 1}

    if name is None:
        return {"status": "malformed_tool_call", "selected_tool": None, "arguments": None, "n_tool_calls": 1}

    return {"status": "ok", "selected_tool": name, "arguments": arguments, "n_tool_calls": 1}
