import pytest
from pydantic import ValidationError

from tool_context_study.schemas import ConditionRecord, FinalTaskRecord, NormalizedToolSchema


def test_normalized_tool_schema_accepts_schema_alias():
    tool = NormalizedToolSchema(
        schema_id="sha256:deadbeef",
        name="country_info__capital",
        original_name="country_info.capital",
        source="BFCL_v3_multiple",
        source_revision="abc123",
        schema={"type": "function", "function": {"name": "country_info__capital"}},
    )
    assert tool.schema_["type"] == "function"
    # round-trips through the public alias too
    assert tool.model_dump(by_alias=True)["schema"]["type"] == "function"


def test_final_task_record_requires_core_fields():
    task = FinalTaskRecord(
        task_id="final_0042",
        request="Look up the capital of Peru.",
        gold_tool="country_info__capital",
        gold_arguments={"country": "Peru"},
        authoring_source="fresh_from_schema",
    )
    assert task.review_status == "pending"


def test_condition_record_rejects_invalid_gold_position():
    with pytest.raises(ValidationError):
        ConditionRecord(
            condition_id="c1",
            task_id="t1",
            history_condition="H0",
            history_id=None,
            gold_tool="country_info__capital",
            gold_position=17,  # not one of 1, 16, 32
            permutation_seed=0,
            messages=[],
            tools=[],
        )


def test_condition_record_rejects_invalid_history_condition():
    with pytest.raises(ValidationError):
        ConditionRecord(
            condition_id="c1",
            task_id="t1",
            history_condition="H32N",  # not a defined condition
            history_id=None,
            gold_tool="country_info__capital",
            gold_position=1,
            permutation_seed=0,
            messages=[],
            tools=[],
        )
