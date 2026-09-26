"""Project-owned record schemas.

These are the canonical, project-owned shapes referenced throughout the
research brief (sections 5, 6, and 9). BFCL's own JSON shapes are read only
by `ingest_bfcl.py` and converted into `RawBFCLQuestion` / `RawBFCLAnswer`
immediately; nothing downstream depends on BFCL's raw field names.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

HistoryConditionId = Literal["H0", "H16N", "H64N", "H64I"]
GoldPosition = Literal[1, 16, 32]


class RawBFCLQuestion(BaseModel):
    """One question record as read from a BFCL `*.json` (JSONL) file."""

    id: str
    question: list[list[dict[str, Any]]]
    function: list[dict[str, Any]]
    source_file: str
    source_revision: str


class RawBFCLAnswer(BaseModel):
    """One ground-truth record as read from a BFCL `possible_answer/*.json` file."""

    id: str
    ground_truth: list[dict[str, Any]]
    source_file: str
    source_revision: str


class OpenAIFunctionSchema(BaseModel):
    """The canonical OpenAI-style function schema (brief section 5)."""

    type: Literal["function"] = "function"
    function: dict[str, Any]


class NormalizedToolSchema(BaseModel):
    """One catalog entry: a normalized, hashable tool schema with provenance."""

    schema_id: str = Field(description="sha256 of the canonicalized schema JSON")
    name: str = Field(description="Normalized name, charset [A-Za-z0-9_-]")
    original_name: str = Field(description="Name as it appeared in the source")
    domain: str | None = Field(default=None, description="Assigned in Phase 3")
    group: str | None = Field(default=None, description="Assigned in Phase 3")
    source: str = Field(description="e.g. BFCL_v3_multiple")
    source_revision: str
    schema_: dict[str, Any] = Field(alias="schema", description="OpenAI-style function schema")

    model_config = {"populate_by_name": True}


class FinalTaskRecord(BaseModel):
    """One frozen final-set (or development-set) task (brief section 9)."""

    task_id: str
    request: str
    gold_tool: str
    gold_arguments: dict[str, Any]
    authoring_source: str
    review_status: Literal["pending", "approved", "rejected"] = "pending"


class ConditionRecord(BaseModel):
    """One fully-materialized model input (brief section 9)."""

    condition_id: str
    task_id: str
    history_condition: HistoryConditionId
    history_id: str | None
    gold_tool: str
    gold_position: GoldPosition
    permutation_seed: int
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]


class ResultRecord(BaseModel):
    """One model generation result (brief section 9)."""

    condition_id: str
    model: str
    model_revision: str
    status: Literal["ok", "no_tool_call", "malformed_tool_call", "inference_error"]
    selected_tool: str | None = None
    arguments: dict[str, Any] | None = None
    raw_response: Any = None
    history_tokens: int
    total_input_tokens: int
    latency_seconds: float
