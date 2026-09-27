"""Client-side reimplementation of vLLM's gpt-oss harmony prompt
construction, used only to compute an accurate `total_input_tokens`.

GPT-OSS-only. Needed because vLLM 0.10.1.1's `/v1/responses` endpoint
always reports all-zero `usage` (see docs/decision_log.md's "Phase 3
(GPT-OSS smoke test)" entry). This mirrors
`vllm.entrypoints.openai.serving_responses.py`'s
`_construct_input_messages_with_harmony` (new-conversation case only --
this project never continues a prior `response_id`) and
`vllm.entrypoints.harmony_utils.render_for_completion`, using the
`openai_harmony` library directly rather than importing vLLM itself, so
the (CPU-only) client env doesn't need the CUDA-oriented `vllm` package
installed alongside it.
"""

from __future__ import annotations

import datetime
from typing import Any

from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
)

_REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}

_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _encoding


def _system_message(reasoning_effort: str | None) -> Message:
    content = SystemContent.new()
    if reasoning_effort is not None:
        content = content.with_reasoning_effort(_REASONING_EFFORT[reasoning_effort])
    content = content.with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
    return Message.from_role_and_content(Role.SYSTEM, content)


def _developer_message(responses_tools: list[dict[str, Any]]) -> Message:
    content = DeveloperContent.new()
    function_tools = [t for t in responses_tools if t["type"] == "function"]
    if function_tools:
        descriptions = [
            ToolDescription.new(name=t["name"], description=t["description"], parameters=t["parameters"])
            for t in function_tools
        ]
        content = content.with_function_tools(descriptions)
    return Message.from_role_and_content(Role.DEVELOPER, content)


def count_harmony_input_tokens(
    messages: list[dict[str, Any]],
    responses_tools: list[dict[str, Any]],
    reasoning_effort: str | None,
) -> int:
    """`responses_tools` are already Responses-API-shaped (see
    `harmony_responses_call.to_responses_tool`). Only handles this project's
    actual request shape: a fresh conversation (no prior `response_id`),
    plain user/assistant/system text turns -- never a `function_call_output`
    turn, since we never send prior tool results back to the model.
    """
    harmony_messages = [_system_message(reasoning_effort), _developer_message(responses_tools)]
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            # Matches vllm.entrypoints.harmony_utils.parse_response_input:
            # a caller-supplied system message becomes a developer message.
            role = "developer"
            content = "Instructions:\n" + content
        harmony_messages.append(Message.from_role_and_content(role, content))

    conversation = Conversation.from_messages(harmony_messages)
    token_ids = _get_encoding().render_conversation_for_completion(conversation, Role.ASSISTANT)
    return len(token_ids)
