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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    timeout: float = 300.0,
    reasoning_effort: str | None = None,
    api_format: str = "chat",
    session: requests.Session | None = None,
    max_retries: int = 2,
    retry_backoff: float = 2.0,
) -> ResultRecord:
    """`api_format="responses"` targets vLLM's `/v1/responses` endpoint
    instead of `/v1/chat/completions`. GPT-OSS-only: see
    `docs/decision_log.md`'s "Phase 3 (GPT-OSS smoke test)" entry --
    `/v1/chat/completions` crashes (empty-body HTTP 500) whenever gpt-oss-20b
    actually emits a real tool call under vLLM 0.10.1.1 (matches open
    upstream issues vllm-project/vllm#23837, #22578); `/v1/responses` is the
    reliable path those issues point to, but needs its own request/response
    shapes (`parsers/harmony_responses_call.py`), hence this branch.

    `session`, when given, is a shared `requests.Session` reused across
    concurrent calls (see `main`'s `--concurrency`) so they pool HTTP
    connections instead of each opening its own; falls back to the plain
    `requests` module for single-call use (e.g. tests) when omitted.
    `timeout` defaults higher than a single sequential call would need
    because under concurrency a request can sit queued behind other
    in-flight requests before vLLM's scheduler admits it.

    `max_retries` retries transient failures (connection errors, timeouts,
    5xx) with a short linear backoff before giving up and recording
    `inference_error` -- a single flaky request or a server momentarily
    overloaded shouldn't cost an entire condition, let alone (in
    `phase5_run_baseline.py`'s coarser retry loop) an entire server
    replica cycle. 4xx responses are never retried -- those mean the
    request itself is malformed, which retrying can't fix.
    """
    requester = session if session is not None else requests
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
    last_exc: Exception | None = None
    response_json = None
    for retry_num in range(max_retries + 1):
        try:
            response = requester.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            response_json = response.json()
            last_exc = None
            break
        except requests.HTTPError as exc:
            last_exc = exc
            if exc.response is not None and exc.response.status_code < 500:
                break  # client error (malformed request) -- retrying can't help
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
        if last_exc is not None and retry_num < max_retries:
            time.sleep(retry_backoff * (retry_num + 1))
    latency = time.monotonic() - start

    if last_exc is not None:
        return ResultRecord(
            condition_id=condition.condition_id,
            model=model,
            model_revision="unknown",
            status="inference_error",
            raw_response={"error": str(last_exc)},
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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Requests to keep in flight at once, so vLLM's continuous batching actually has "
        "multiple requests to batch instead of processing one at a time. vLLM's own scheduler "
        "admits as many as its KV cache allows (it logs a 'maximum concurrency for N tokens per "
        "request' figure at server startup) and queues the rest, so this is a ceiling, not a "
        "guarantee that many run simultaneously -- safe to leave higher than that figure.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per request on transient failure (connection error, timeout, 5xx) before "
        "recording inference_error. 4xx responses are never retried.",
    )
    args = parser.parse_args()

    conditions = load_conditions(args.conditions)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # One shared Session (rather than the bare `requests` module) so concurrent
    # calls pool HTTP connections instead of each opening its own; pool sized
    # to `--concurrency` so none of them block waiting for a free connection.
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.concurrency, pool_maxsize=args.concurrency)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    write_lock = threading.Lock()
    n_done = 0
    n_ok = 0

    with open(args.output, "w") as out_f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                call_condition,
                args.base_url,
                args.model,
                condition,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                api_format=args.api_format,
                session=session,
                max_retries=args.max_retries,
            ): condition
            for condition in conditions
        }
        # Results are written as they complete, not in input order -- safe
        # because score.py joins results back to conditions by condition_id,
        # never by line position.
        for future in as_completed(futures):
            condition = futures[future]
            result = future.result()
            with write_lock:
                n_done += 1
                if result.status == "ok":
                    n_ok += 1
                out_f.write(result.model_dump_json() + "\n")
                out_f.flush()
                print(
                    f"[{n_done}/{len(conditions)}] {condition.condition_id}: "
                    f"status={result.status} selected={result.selected_tool!r} "
                    f"({result.latency_seconds:.2f}s)"
                )

    print(f"Done: {n_ok}/{len(conditions)} produced a parseable single tool call.")


if __name__ == "__main__":
    main()
