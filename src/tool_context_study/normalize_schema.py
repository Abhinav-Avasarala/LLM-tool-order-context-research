"""Normalize a raw BFCL function definition into the canonical OpenAI-style
tool schema described in the research brief, section 5.

Rules implemented here:
  - Map BFCL's parameter type vocabulary onto JSON Schema types (e.g. "dict" -> "object").
  - Convert names to a conservative charset ([A-Za-z0-9_-]), preserving a
    mapping back to the original name.
  - Canonicalize JSON (sorted keys, no incidental whitespace) before hashing,
    so `schema_id` is stable and comparable across runs.
  - Drop non-standard per-property keys BFCL uses in place of JSON Schema's
    `required` array (namely `optional`), so the emitted schema is valid
    JSON Schema.

Unknown/unmapped parameter types raise `UnknownParameterTypeError` rather
than being silently guessed, since the brief requires every unmapped
construct to be checked manually before it enters the frozen catalog.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from tool_context_study.schemas import NormalizedToolSchema

_NAME_CHARSET_RE = re.compile(r"[^A-Za-z0-9_-]")

_TYPE_MAP = {
    "dict": "object",
    "object": "object",
    "float": "number",
    "number": "number",
    "integer": "integer",
    "int": "integer",
    "string": "string",
    "str": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "list": "array",
    "tuple": "array",
    "array": "array",
}


class UnknownParameterTypeError(ValueError):
    """Raised when a BFCL parameter type has no known JSON Schema mapping."""


def normalize_name(original_name: str) -> str:
    """Convert e.g. `country_info.capital` -> `country_info__capital`."""
    replaced = original_name.replace(".", "__")
    return _NAME_CHARSET_RE.sub("_", replaced)


def _normalize_type(raw_type: str) -> str:
    try:
        return _TYPE_MAP[raw_type]
    except KeyError as exc:
        raise UnknownParameterTypeError(
            f"No JSON Schema mapping for BFCL parameter type {raw_type!r}"
        ) from exc


def _normalize_node(node: Any) -> Any:
    """Recursively normalize a parameter (sub)schema node."""
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "optional":
            # JSON Schema expresses optionality via the parent's `required`
            # array, not a per-property flag; BFCL's `optional` is redundant
            # with that and not valid JSON Schema, so it is dropped here.
            continue
        if key == "type":
            out[key] = _normalize_type(value)
        elif key == "properties" and isinstance(value, dict):
            out[key] = {name: _normalize_node(sub) for name, sub in value.items()}
        elif key == "items":
            out[key] = _normalize_node(value)
        else:
            out[key] = value
    return out


def canonicalize_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def compute_schema_id(schema: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonicalize_json(schema).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def normalize_function(
    raw_function: dict[str, Any],
    *,
    source: str,
    source_revision: str,
    domain: str | None = None,
    group: str | None = None,
) -> NormalizedToolSchema:
    """Convert one raw BFCL `function` entry into a `NormalizedToolSchema`."""
    original_name = raw_function["name"]
    name = normalize_name(original_name)

    parameters = _normalize_node(raw_function.get("parameters", {"type": "object", "properties": {}}))
    if "required" not in parameters:
        parameters["required"] = []

    function_schema = {
        "name": name,
        "description": raw_function.get("description", ""),
        "parameters": parameters,
    }
    full_schema = {"type": "function", "function": function_schema}

    return NormalizedToolSchema(
        schema_id=compute_schema_id(full_schema),
        name=name,
        original_name=original_name,
        domain=domain,
        group=group,
        source=source,
        source_revision=source_revision,
        schema=full_schema,
    )
