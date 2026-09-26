import pytest

from tool_context_study.build_conditions import build_condition, derive_permutation_seed, validate_condition
from tool_context_study.schemas import FinalTaskRecord, NormalizedToolSchema


def _make_catalog(n: int = 32) -> list[NormalizedToolSchema]:
    return [
        NormalizedToolSchema(
            schema_id=f"sha256:fake{i}",
            name=f"tool_{i}",
            original_name=f"tool.{i}",
            source="fixture",
            source_revision="rev0",
            schema={"type": "function", "function": {"name": f"tool_{i}", "description": "d", "parameters": {}}},
        )
        for i in range(n)
    ]


def _make_task(gold_tool: str = "tool_7") -> FinalTaskRecord:
    return FinalTaskRecord(
        task_id="dev_test_0",
        request="Do the thing.",
        gold_tool=gold_tool,
        gold_arguments={},
        authoring_source="fixture",
    )


def test_build_condition_places_gold_tool_at_requested_position():
    catalog = _make_catalog()
    task = _make_task()

    cond_p1 = build_condition(
        task, catalog, gold_position=1, history_condition="H0", history_messages=[], base_seed=42
    )
    cond_p16 = build_condition(
        task, catalog, gold_position=16, history_condition="H0", history_messages=[], base_seed=42
    )
    cond_p32 = build_condition(
        task, catalog, gold_position=32, history_condition="H0", history_messages=[], base_seed=42
    )

    assert cond_p1.tools[0]["function"]["name"] == "tool_7"
    assert cond_p16.tools[15]["function"]["name"] == "tool_7"
    assert cond_p32.tools[31]["function"]["name"] == "tool_7"


def test_build_condition_preserves_full_tool_set_and_uniqueness():
    catalog = _make_catalog()
    task = _make_task()
    cond = build_condition(task, catalog, gold_position=16, history_condition="H0", history_messages=[], base_seed=42)

    names = [t["function"]["name"] for t in cond.tools]
    assert len(names) == 32
    assert len(set(names)) == 32
    assert set(names) == {t.name for t in catalog}


def test_build_condition_different_positions_shuffle_differently():
    catalog = _make_catalog()
    task = _make_task()
    cond_p1 = build_condition(
        task, catalog, gold_position=1, history_condition="H0", history_messages=[], base_seed=42
    )
    cond_p32 = build_condition(
        task, catalog, gold_position=32, history_condition="H0", history_messages=[], base_seed=42
    )

    # Distractor ordering shouldn't be identical (e.g. a naive "reverse the list" trick)
    # just because only the gold tool moved.
    distractors_p1 = [t["function"]["name"] for t in cond_p1.tools if t["function"]["name"] != "tool_7"]
    distractors_p32 = [t["function"]["name"] for t in cond_p32.tools if t["function"]["name"] != "tool_7"]
    assert distractors_p1 != distractors_p32


def test_build_condition_is_reproducible_given_same_seed():
    catalog = _make_catalog()
    task = _make_task()
    cond_a = build_condition(
        task, catalog, gold_position=16, history_condition="H0", history_messages=[], base_seed=42
    )
    cond_b = build_condition(
        task, catalog, gold_position=16, history_condition="H0", history_messages=[], base_seed=42
    )
    assert cond_a.tools == cond_b.tools
    assert cond_a.permutation_seed == cond_b.permutation_seed


def test_build_condition_appends_system_and_user_messages():
    catalog = _make_catalog()
    task = _make_task()
    cond = build_condition(task, catalog, gold_position=1, history_condition="H0", history_messages=[], base_seed=1)

    assert cond.messages[0]["role"] == "system"
    assert cond.messages[-1] == {"role": "user", "content": "Do the thing."}


def test_build_condition_rejects_gold_tool_not_in_catalog():
    catalog = _make_catalog()
    task = _make_task(gold_tool="not_in_catalog")
    with pytest.raises(ValueError):
        build_condition(task, catalog, gold_position=1, history_condition="H0", history_messages=[], base_seed=1)


def test_derive_permutation_seed_is_deterministic_and_varies_with_inputs():
    a = derive_permutation_seed(42, "task_1", 1, "H0")
    b = derive_permutation_seed(42, "task_1", 1, "H0")
    c = derive_permutation_seed(42, "task_1", 32, "H0")
    d = derive_permutation_seed(42, "task_2", 1, "H0")

    assert a == b
    assert a != c
    assert a != d


def test_validate_condition_passes_for_well_formed_condition():
    catalog = _make_catalog()
    task = _make_task()
    cond = build_condition(task, catalog, gold_position=16, history_condition="H0", history_messages=[], base_seed=1)
    validate_condition(cond, catalog)  # should not raise
