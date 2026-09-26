import json

import pytest

from tool_context_study.normalize_schema import (
    UnknownParameterTypeError,
    canonicalize_json,
    compute_schema_id,
    normalize_function,
    normalize_name,
)


def test_normalize_name_replaces_dot_and_invalid_chars():
    assert normalize_name("country_info.capital") == "country_info__capital"
    assert normalize_name("weird name!!") == "weird_name__"


def test_canonicalize_json_is_key_order_independent():
    a = canonicalize_json({"b": 1, "a": 2})
    b = canonicalize_json({"a": 2, "b": 1})
    assert a == b
    assert a == '{"a":2,"b":1}'


def test_normalize_function_basic_shape():
    raw = {
        "name": "country_info.capital",
        "description": "Find the capital city of a country.",
        "parameters": {
            "type": "dict",
            "properties": {"country": {"type": "string", "description": "Country name."}},
            "required": ["country"],
        },
    }
    result = normalize_function(raw, source="BFCL_v3_multiple", source_revision="abc123")

    assert result.name == "country_info__capital"
    assert result.original_name == "country_info.capital"
    assert result.source == "BFCL_v3_multiple"
    assert result.source_revision == "abc123"
    assert result.schema_["type"] == "function"
    assert result.schema_["function"]["name"] == "country_info__capital"
    assert result.schema_["function"]["parameters"]["type"] == "object"
    assert result.schema_id.startswith("sha256:")


def test_normalize_function_drops_optional_key_and_maps_dict_to_object():
    raw = {
        "name": "triangle_properties.get",
        "description": "Get triangle properties.",
        "parameters": {
            "type": "dict",
            "properties": {
                "side1": {"type": "integer", "description": "side 1"},
                "get_area": {
                    "type": "boolean",
                    "description": "whether to compute area",
                    "default": True,
                    "optional": True,
                },
            },
            "required": ["side1"],
        },
    }
    result = normalize_function(raw, source="BFCL_v3_multiple", source_revision="abc123")
    params = result.schema_["function"]["parameters"]

    assert params["type"] == "object"
    assert "optional" not in params["properties"]["get_area"]
    assert params["properties"]["get_area"]["default"] is True


def test_normalize_function_rejects_unknown_type():
    raw = {
        "name": "mystery.tool",
        "description": "A tool with an unmappable parameter type.",
        "parameters": {"type": "dict", "properties": {"x": {"type": "any", "description": "?"}}, "required": []},
    }
    with pytest.raises(UnknownParameterTypeError):
        normalize_function(raw, source="BFCL_v3_multiple", source_revision="abc123")


def test_compute_schema_id_is_deterministic_and_sensitive_to_content():
    schema_a = {"type": "function", "function": {"name": "f", "description": "d", "parameters": {}}}
    schema_b = json.loads(json.dumps(schema_a))  # structurally identical copy
    schema_c = {**schema_a, "function": {**schema_a["function"], "description": "different"}}

    assert compute_schema_id(schema_a) == compute_schema_id(schema_b)
    assert compute_schema_id(schema_a) != compute_schema_id(schema_c)


def test_two_distinct_functions_get_distinct_names_and_schema_ids():
    raw_a = {
        "name": "country_info.capital",
        "description": "Find the capital city of a country.",
        "parameters": {"type": "dict", "properties": {"country": {"type": "string", "description": "d"}}, "required": ["country"]},
    }
    raw_b = {
        "name": "country_info.population",
        "description": "Find the population of a country.",
        "parameters": {"type": "dict", "properties": {"country": {"type": "string", "description": "d"}}, "required": ["country"]},
    }
    a = normalize_function(raw_a, source="s", source_revision="r")
    b = normalize_function(raw_b, source="s", source_revision="r")

    assert a.name != b.name
    assert a.schema_id != b.schema_id
