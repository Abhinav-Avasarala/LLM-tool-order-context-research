"""Send materialized conditions to a running OpenAI-compatible model server
(vLLM/SGLang) and save raw + parsed results (brief section 8).

Does not start or manage the server -- that's a separate, explicit step
(see docs/decision_log.md for the exact vLLM invocation used in Phase 2),
since server lifecycle/GPU placement is an operational concern, not part
of the scored pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests

from tool_context_study.config import PROJECT_ROOT
from tool_context_study.parsers.harmony_responses_call import parse_responses_output, to_responses_tool
from tool_context_study.parsers.harmony_token_count import count_harmony_input_tokens
from tool_context_study.parsers.openai_tool_call import parse_chat_completion
from tool_context_study.schemas import ConditionRecord, ResultRecord

RESULTS_RAW_DIR = PROJECT_ROOT / "results" / "raw"


def call_condition(
    base_url: str,
    model: str,
    condition: ConditionRecord,
    *,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: float = 120.0,
    reasoning_effort: str | None = None,
    api_format: str = "chat",
) -> ResultRecord:
    """`api_format="responses"` targets vLLM's `/v1/responses` endpoint
    instead of `/v1/chat/completions`. GPT-OSS-only: see
    `docs/decision_log.md`'s "Phase 3 (GPT-OSS smoke test)" entry --
    `/v1/chat/completions` crashes (empty-body HTTP 500) whenever gpt-oss-20b
    actually emits a real tool call under vLLM 0.10.1.1 (matches open
    upstream issues vllm-project/vllm#23837, #22578); `/v1/responses` is the
    reliable path those issues point to, but needs its own request/response
    shapes (`parsers/harmony_responses_call.py`), hence this branch.
    """
    if api_format == "responses":
        responses_tools = [to_responses_tool(t) for t in condition.tools]
        payload: dict[str, Any] = {
            "model": model,
            "input": condition.messages,
            "tools": responses_tools,
            "tool_choice": "auto",
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
        endpoint = f"{base_url}/v1/responses"
    else:
        payload = {
            "model": model,
            "messages": condition.messages,
            "tools": condition.tools,
            "tool_choice": "auto",
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if reasoning_effort is not None:
            # GPT-OSS-only knob (brief section 7: reasoning effort fixed to
            # "low" for this direct-selection task); vLLM's harmony code path
            # reads this field, non-harmony models like Qwen never see it since
            # we only pass it when the caller explicitly asks for it.
            payload["reasoning_effort"] = reasoning_effort
        endpoint = f"{base_url}/v1/chat/completions"

    start = time.monotonic()
    try:
        response = requests.post(endpoint, json=payload, timeout=timeout)
        latency = time.monotonic() - start
        response.raise_for_status()
        response_json = response.json()
    except (requests.RequestException, ValueError) as exc:
        latency = time.monotonic() - start
        return ResultRecord(
            condition_id=condition.condition_id,
            model=model,
            model_revision="unknown",
            status="inference_error",
            raw_response={"error": str(exc)},
            history_tokens=0 if condition.history_condition == "H0" else -1,
            total_input_tokens=-1,
            latency_seconds=latency,
        )

    if api_format == "responses":
        parsed = parse_responses_output(response_json)
        # vLLM 0.10.1.1's /v1/responses always reports all-zero `usage`
        # (see docs/decision_log.md's "Phase 3 (GPT-OSS smoke test)" entry) --
        # count locally instead, using the same openai_harmony encoding vLLM
        # uses internally, rather than trust that unreliable field.
        total_input_tokens = count_harmony_input_tokens(condition.messages, responses_tools, reasoning_effort)
    else:
        parsed = parse_chat_completion(response_json)
        usage = response_json.get("usage") or {}
        total_input_tokens = usage.get("prompt_tokens", -1)

    return ResultRecord(
        condition_id=condition.condition_id,
        model=model,
        model_revision=response_json.get("model", model),
        status=parsed["status"],
        selected_tool=parsed["selected_tool"],
        arguments=parsed["arguments"],
        raw_response=response_json,
        history_tokens=0 if condition.history_condition == "H0" else -1,
        total_input_tokens=total_input_tokens,
        latency_seconds=latency,
    )


def load_conditions(path: Path) -> list[ConditionRecord]:
    conditions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                conditions.append(ConditionRecord.model_validate_json(line))
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", required=True, type=Path, help="Path to a conditions .jsonl file")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", required=True, help="Model name as registered with the server")
    parser.add_argument("--output", required=True, type=Path, help="Path to write result .jsonl to")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["low", "medium", "high"],
        help="GPT-OSS only (brief section 7 fixes this to 'low'); omitted for non-harmony models.",
    )
    parser.add_argument(
        "--api-format",
        default="chat",
        choices=["chat", "responses"],
        help="'responses' targets vLLM's /v1/responses (GPT-OSS only; see docs/decision_log.md).",
    )
    args = parser.parse_args()

    conditions = load_conditions(args.conditions)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with open(args.output, "w") as out_f:
        for i, condition in enumerate(conditions, start=1):
            result = call_condition(
                args.base_url,
                args.model,
                condition,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                api_format=args.api_format,
            )
            results.append(result)
            out_f.write(result.model_dump_json() + "\n")
            out_f.flush()
            print(
                f"[{i}/{len(conditions)}] {condition.condition_id}: "
                f"status={result.status} selected={result.selected_tool!r} "
                f"({result.latency_seconds:.2f}s)"
            )

    n_ok = sum(1 for r in results if r.status == "ok")
    print(f"Done: {n_ok}/{len(results)} produced a parseable single tool call.")


if __name__ == "__main__":
    main()
