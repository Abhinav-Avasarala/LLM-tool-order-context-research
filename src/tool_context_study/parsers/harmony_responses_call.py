"""Parse a vLLM `/v1/responses` (Harmony) response into a tool-call decision.

GPT-OSS-only (brief section 7): decision_log.md's "Phase 3 (GPT-OSS smoke
test)" entry records why -- vLLM 0.10.1.1's `/v1/chat/completions` endpoint
crashes with an empty-body HTTP 500 whenever gpt-oss-20b actually emits a
real tool call (reproduced deterministically, independent of our own catalog
schemas; matches open upstream issues vllm-project/vllm#23837 and #22578).
`/v1/responses` is the reliable path those issues point to instead, but it
uses the OpenAI Responses API's own request/response shapes rather than
Chat Completions' `choices[0].message.tool_calls`, hence a separate
converter + parser pair instead of reusing `openai_tool_call.py`.
"""

from __future__ import annotations

import json
from typing import Any

from tool_context_study.parsers.openai_tool_call import ParsedToolCall


def to_responses_tool(chat_tool: dict[str, Any]) -> dict[str, Any]:
    """Convert one Chat-Completions-shaped tool (`{type, function: {...}}`)
    into the Responses API's flat shape (`{type, name, description,
    parameters}`)."""
    function = chat_tool["function"]
    return {
        "type": "function",
        "name": function["name"],
        "description": function.get("description", ""),
        "parameters": function.get("parameters", {}),
    }


def parse_responses_output(response_json: dict[str, Any]) -> ParsedToolCall:
    output = response_json.get("output")
    if output is None:
        return {"status": "inference_error", "selected_tool": None, "arguments": None, "n_tool_calls": 0}

    calls = [item for item in output if item.get("type") == "function_call"]
    n = len(calls)

    if n == 0:
        return {"status": "no_tool_call", "selected_tool": None, "arguments": None, "n_tool_calls": 0}

    if n > 1:
        # Same policy as parse_chat_completion: score.py's "multiple_tool_calls"
        # category is decided from n_tool_calls, not by us picking one call here.
        return {"status": "ok", "selected_tool": None, "arguments": None, "n_tool_calls": n}

    call = calls[0]
    name = call.get("name")
    raw_arguments = call.get("arguments")

    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError:
        return {"status": "malformed_tool_call", "selected_tool": name, "arguments": None, "n_tool_calls": 1}

    if name is None:
        return {"status": "malformed_tool_call", "selected_tool": None, "arguments": None, "n_tool_calls": 1}

    return {"status": "ok", "selected_tool": name, "arguments": arguments, "n_tool_calls": 1}
