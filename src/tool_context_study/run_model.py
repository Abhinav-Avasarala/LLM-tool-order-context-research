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

import requests

from tool_context_study.config import PROJECT_ROOT
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
) -> ResultRecord:
    payload = {
        "model": model,
        "messages": condition.messages,
        "tools": condition.tools,
        "tool_choice": "auto",
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    start = time.monotonic()
    try:
        response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
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

    parsed = parse_chat_completion(response_json)
    usage = response_json.get("usage", {})

    return ResultRecord(
        condition_id=condition.condition_id,
        model=model,
        model_revision=response_json.get("model", model),
        status=parsed["status"],
        selected_tool=parsed["selected_tool"],
        arguments=parsed["arguments"],
        raw_response=response_json,
        history_tokens=0 if condition.history_condition == "H0" else -1,
        total_input_tokens=usage.get("prompt_tokens", -1),
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
    args = parser.parse_args()

    conditions = load_conditions(args.conditions)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with open(args.output, "w") as out_f:
        for i, condition in enumerate(conditions, start=1):
            result = call_condition(args.base_url, args.model, condition, max_tokens=args.max_tokens)
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
